"""Tests for CONCEPT:ORCH-1.36 — Declarative Resilience Policy.

Covers the policy primitive in isolation (unit tests) and a live-path
integration test proving it is actually invoked from the specialist execution
path in ``agent_utilities.graph.executor``.
"""

from __future__ import annotations

import random

import pytest

from agent_utilities.orchestration.resilience import (
    DEFAULT_POLICY,
    DEFAULT_RETRYABLE,
    ResiliencePolicy,
    compute_backoff,
    run_with_resilience,
    run_with_resilience_sync,
)


# --------------------------------------------------------------------------- #
# compute_backoff
# --------------------------------------------------------------------------- #
def test_compute_backoff_deterministic_without_jitter():
    policy = ResiliencePolicy(
        backoff_base_s=1.0, backoff_factor=2.0, max_backoff_s=100.0, jitter=False
    )
    # 1-indexed attempts: base * factor**(attempt-1)
    assert compute_backoff(1, policy) == pytest.approx(1.0)
    assert compute_backoff(2, policy) == pytest.approx(2.0)
    assert compute_backoff(3, policy) == pytest.approx(4.0)
    assert compute_backoff(4, policy) == pytest.approx(8.0)


def test_compute_backoff_capped_at_max():
    policy = ResiliencePolicy(
        backoff_base_s=1.0, backoff_factor=10.0, max_backoff_s=5.0, jitter=False
    )
    # 1 -> 1, 2 -> 10 (capped to 5), 3 -> 100 (capped to 5)
    assert compute_backoff(1, policy) == pytest.approx(1.0)
    assert compute_backoff(2, policy) == pytest.approx(5.0)
    assert compute_backoff(3, policy) == pytest.approx(5.0)


def test_compute_backoff_jitter_is_seedable_and_bounded():
    policy = ResiliencePolicy(
        backoff_base_s=4.0, backoff_factor=2.0, max_backoff_s=100.0, jitter=True
    )
    # Same seed -> identical result (deterministic under injected rng).
    v1 = compute_backoff(1, policy, rng=random.Random(42))
    v2 = compute_backoff(1, policy, rng=random.Random(42))
    assert v1 == v2
    # Jitter multiplies the capped delay by a factor in [0.5, 1.0].
    capped = 4.0
    assert 0.5 * capped <= v1 <= capped


# --------------------------------------------------------------------------- #
# retry_on filtering
# --------------------------------------------------------------------------- #
def test_should_retry_filters_exceptions():
    policy = ResiliencePolicy(retry_on=DEFAULT_RETRYABLE)
    assert policy.should_retry(TimeoutError("slow")) is True
    assert policy.should_retry(ConnectionError("down")) is True
    # ValueError / permission errors are never retried even though they are
    # not in retry_on.
    assert policy.should_retry(ValueError("bad input")) is False
    assert policy.should_retry(PermissionError("nope")) is False


def test_should_retry_never_retries_non_retryable_even_if_broad_allowlist():
    # A caller passing Exception as the allow-list must still NOT retry a
    # deterministic ValueError.
    policy = ResiliencePolicy(retry_on=(Exception,))
    assert policy.should_retry(ConnectionError()) is True
    assert policy.should_retry(ValueError()) is False


def test_should_retry_predicate():
    policy = ResiliencePolicy(retry_on=lambda e: "retry-me" in str(e))
    assert policy.should_retry(RuntimeError("please retry-me")) is True
    assert policy.should_retry(RuntimeError("nope")) is False


def test_invalid_max_attempts_rejected():
    with pytest.raises(ValueError):
        ResiliencePolicy(max_attempts=0)


# --------------------------------------------------------------------------- #
# run_with_resilience — async
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_retry_count_honored_and_sleep_injected():
    calls = {"n": 0}
    slept: list[float] = []

    async def fake_sleep(d: float) -> None:
        slept.append(d)

    async def flaky():
        calls["n"] += 1
        raise TimeoutError("transient")

    policy = ResiliencePolicy(
        max_attempts=3, backoff_base_s=0.1, jitter=False, retry_on=DEFAULT_RETRYABLE
    )
    with pytest.raises(TimeoutError):
        await run_with_resilience(flaky, policy, sleep=fake_sleep)

    # Primary attempted exactly max_attempts times.
    assert calls["n"] == 3
    # Slept between attempts (max_attempts - 1 times), never actually waited.
    assert len(slept) == 2


@pytest.mark.asyncio
async def test_primary_success_short_circuits():
    calls = {"n": 0}

    async def fake_sleep(d: float) -> None:  # pragma: no cover - must not run
        raise AssertionError("should not sleep on first success")

    async def ok():
        calls["n"] += 1
        return "result"

    policy = ResiliencePolicy(max_attempts=5)
    out = await run_with_resilience(ok, policy, sleep=fake_sleep)
    assert out == "result"
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_value_error_not_retried():
    calls = {"n": 0}

    async def fake_sleep(d: float) -> None:  # pragma: no cover
        raise AssertionError("ValueError must not trigger a retry/backoff")

    async def bad():
        calls["n"] += 1
        raise ValueError("deterministic")

    policy = ResiliencePolicy(max_attempts=5, retry_on=DEFAULT_RETRYABLE)
    with pytest.raises(ValueError):
        await run_with_resilience(bad, policy, sleep=fake_sleep)
    # No retries for a non-retryable error.
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_fallback_invoked_only_after_primary_exhaustion():
    order: list[str] = []

    async def fake_sleep(d: float) -> None:
        order.append("sleep")

    async def primary():
        order.append("primary")
        raise ConnectionError("down")

    async def fb_success():
        order.append("fallback")
        return "from-fallback"

    policy = ResiliencePolicy(
        max_attempts=2,
        backoff_base_s=0.01,
        jitter=False,
        retry_on=DEFAULT_RETRYABLE,
        fallbacks=[fb_success],
    )
    out = await run_with_resilience(primary, policy, sleep=fake_sleep)
    assert out == "from-fallback"
    # Primary ran twice, then the fallback ran exactly once, last.
    assert order == ["primary", "sleep", "primary", "fallback"]


@pytest.mark.asyncio
async def test_all_fail_raises_last_exception():
    async def fake_sleep(d: float) -> None:
        return None

    async def primary():
        raise ConnectionError("primary-down")

    async def fb():
        raise TimeoutError("fallback-down")

    policy = ResiliencePolicy(
        max_attempts=1,
        retry_on=DEFAULT_RETRYABLE,
        fallbacks=[fb],
    )
    with pytest.raises(TimeoutError, match="fallback-down"):
        await run_with_resilience(primary, policy, sleep=fake_sleep)


@pytest.mark.asyncio
async def test_per_attempt_timeout_enforced():
    import asyncio

    slept: list[float] = []

    async def fake_sleep(d: float) -> None:
        slept.append(d)

    async def slow():
        await asyncio.sleep(10)  # would block without the policy timeout

    policy = ResiliencePolicy(
        max_attempts=2,
        backoff_base_s=0.5,
        jitter=False,
        timeout_s=0.01,
        retry_on=DEFAULT_RETRYABLE,
    )
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await run_with_resilience(slow, policy, sleep=fake_sleep)
    # Timeout is a transient error -> retried, so one backoff sleep happened.
    assert len(slept) == 1


@pytest.mark.asyncio
async def test_default_policy_values():
    assert DEFAULT_POLICY.max_attempts == 3
    assert DEFAULT_POLICY.backoff_factor == 2.0
    assert DEFAULT_POLICY.retry_on == DEFAULT_RETRYABLE


# --------------------------------------------------------------------------- #
# run_with_resilience_sync
# --------------------------------------------------------------------------- #
def test_sync_retry_then_success():
    calls = {"n": 0}
    slept: list[float] = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("transient")
        return "ok"

    policy = ResiliencePolicy(
        max_attempts=3, backoff_base_s=0.0, jitter=False, retry_on=DEFAULT_RETRYABLE
    )
    out = run_with_resilience_sync(flaky, policy, sleep=slept.append)
    assert out == "ok"
    assert calls["n"] == 2


def test_sync_fallback():
    def primary():
        raise ConnectionError("down")

    def fb():
        return "fb"

    policy = ResiliencePolicy(
        max_attempts=1, retry_on=DEFAULT_RETRYABLE, fallbacks=[fb]
    )
    assert run_with_resilience_sync(primary, policy, sleep=lambda _d: None) == "fb"


# --------------------------------------------------------------------------- #
# LIVE-PATH integration: prove the policy is invoked from the real
# specialist-execution path in agent_utilities.graph.executor.
# --------------------------------------------------------------------------- #
class _FakeUsage:
    def __init__(self) -> None:
        self.total_tokens = 0


class _FakeRunResult:
    def __init__(self, output: str) -> None:
        self.output = output
        self._usage = _FakeUsage()

    def all_messages(self):
        return []

    @property
    def usage(self):  # graph executor reads getattr(res, "usage", None)
        return self._usage


class _FlakyAgent:
    """Stand-in for pydantic_ai.Agent whose ``run`` fails transiently once.

    Records every ``run`` invocation so the test can assert the resilience
    policy retried on the live path. Construction args are ignored — the
    executor builds the Agent twice (early + inside the AsyncExitStack); both
    share the class-level counters.
    """

    calls: list[int] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def run(self, *args, **kwargs):
        _FlakyAgent.calls.append(1)
        if len(_FlakyAgent.calls) == 1:
            # Transient error — must be retried by the declarative policy.
            raise ConnectionError("simulated transient model/tool error")
        return _FakeRunResult("specialist produced a real answer")


@pytest.mark.asyncio
async def test_resilience_wired_into_execute_specialist_live_path(monkeypatch):
    """LIVE-PATH: exercise ``_execute_dynamic_mcp_agent`` (the real per-specialist
    execution path that ``execute_specialist_node`` / ``agent_package_step``
    funnel into) with a primary that fails transiently once then succeeds, and
    assert it retried and ultimately succeeded — proving CONCEPT:ORCH-1.36 is
    invoked in production, not just unit-tested in isolation.
    """
    from types import SimpleNamespace

    from agent_utilities.graph import executor as executor_mod
    from agent_utilities.graph.state import GraphDeps, GraphState
    from agent_utilities.models import MCPAgent

    _FlakyAgent.calls = []

    # Patch the Agent class the executor constructs so the LLM call is the flaky
    # stand-in (no real LLM / engine needed). compute_backoff jitter is real but
    # the backoff is tiny; patch sleep so the test does not actually wait.
    monkeypatch.setattr(executor_mod, "Agent", _FlakyAgent)

    slept: list[float] = []

    async def _no_wait(d: float) -> None:
        slept.append(d)

    monkeypatch.setattr(executor_mod.asyncio, "sleep", _no_wait)

    agent_info = MCPAgent(
        name="probe_specialist",
        description="probe specialist for resilience live-path test",
        system_prompt="You are a probe.",
        mcp_server="",  # no server -> no toolset/circuit-breaker gating
        tools=[],
    )

    state = GraphState(query="do the thing", mode="ask", topology="basic")
    deps = GraphDeps(tag_prompts={}, tag_env_vars={}, mcp_toolsets=[])

    ctx = SimpleNamespace(deps=deps, state=state, inputs=None, node_id="probe")

    result = await executor_mod._execute_dynamic_mcp_agent(ctx, agent_info)

    # The specialist ultimately succeeded and routed to the joiner.
    assert result == "execution_joiner"
    # The flaky agent was called at least twice -> the policy retried after the
    # transient ConnectionError instead of bubbling it straight up.
    assert len(_FlakyAgent.calls) >= 2
    # The real result made it into the registry.
    assert any("real answer" in v for v in state.results_registry.values())
