"""FIX 2 (delegation resilience) + CONCEPT:AU-KG.retrieval.fail-closed-grounding-contract.

The mandatory evidence compilation at the model-transport boundary runs OFF the
asyncio event loop and is BOUNDED, so a slow/contended retrieval can never block
the loop (the graph-os liveness crash) or overrun the delegation's wall clock.

What happens on a timeout/error/quality-gate-failure is governed by the ambient
:data:`GroundingPolicy` (:func:`contextual_model.use_grounding_policy`):

- ``"required"`` (the DEFAULT — every caller who never opens the scope gets this):
  FAILS CLOSED. :class:`GroundingUnavailableError` is raised; the request never
  reaches the model.
- ``"best_effort"``/``"none"`` (explicit per-run opt-in): the request proceeds,
  but the messages carry an explicit degraded-grounding marker, and
  :func:`contextual_model.grounding_snapshot` reports the degradation so the
  caller can avoid recording the run as a plain success.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agent_utilities.core import contextual_model
from agent_utilities.core import contextual_model as cm
from agent_utilities.core.contextual_model import (
    GroundingUnavailableError,
    _compile_messages_bounded,
    _compiled_evidence_and_bundle_bounded,
    grounding_snapshot,
    use_grounding_policy,
)


class _Msg:
    """A stand-in message object (the helper only passes the list through)."""


class _FakeBundle:
    """Duck-typed stand-in for ``ContextBundle`` — the wrapper only ever reads
    ``retrieval_quality_gate_failed``/``retrieval_quality_reason`` via ``getattr``,
    so this avoids pulling in the full retrieval stack (and its heavy numeric-kernel
    dependency) just to exercise the model-transport boundary's policy contract.
    """

    def __init__(
        self,
        *,
        retrieval_quality_gate_failed: bool = False,
        retrieval_quality_reason: str = "",
    ) -> None:
        self.retrieval_quality_gate_failed = retrieval_quality_gate_failed
        self.retrieval_quality_reason = retrieval_quality_reason


@pytest.fixture
def messages() -> list[object]:
    return [_Msg(), _Msg()]


@pytest.fixture(autouse=True)
def reset_context_compile_breaker():
    """Keep the process-wide context-compile breaker AND grounding tracking hermetic per test."""
    contextual_model._ctx_compile_degradation_streak = 0
    contextual_model._ctx_compile_breaker_reopen_at = 0.0
    contextual_model._grounding_policy.set("required")
    contextual_model._grounding_outcome.set(None)
    try:
        yield
    finally:
        contextual_model._ctx_compile_degradation_streak = 0
        contextual_model._ctx_compile_breaker_reopen_at = 0.0
        contextual_model._grounding_policy.set("required")
        contextual_model._grounding_outcome.set(None)


def _marker_text(governed: list) -> str:
    """Extract the leading system-prompt content the wrapper prepends, if any."""
    if not governed:
        return ""
    first = governed[0]
    parts = list(getattr(first, "parts", ()) or ())
    if not parts:
        return ""
    content = getattr(parts[0], "content", "")
    return content if isinstance(content, str) else ""


# ---------------------------------------------------------------------------
# Default policy ("required") — FAILS CLOSED
# ---------------------------------------------------------------------------


async def test_bounded_fails_closed_on_timeout_by_default(monkeypatch, messages):
    """A compile that blocks past the budget is abandoned; by DEFAULT (grounding=
    'required') the request is refused rather than silently sent ungrounded."""
    monkeypatch.setattr(contextual_model, "_CONTEXT_COMPILE_TIMEOUT_S", 0.2)

    def _slow_blocking(_messages, _model_name):  # simulates future.result() stall
        time.sleep(5.0)
        raise AssertionError("compile should have been abandoned at the budget")

    monkeypatch.setattr(
        contextual_model, "_compiled_evidence_and_bundle", _slow_blocking
    )

    start = time.perf_counter()
    with pytest.raises(GroundingUnavailableError, match="timeout"):
        await _compiled_evidence_and_bundle_bounded(messages, "m")
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0  # refused at the budget, not after the 5s block
    degraded, reason = grounding_snapshot()
    assert degraded is True
    assert reason == "timeout"


async def test_bounded_fails_closed_on_error_by_default(monkeypatch, messages):
    """Any compilation error fails closed by default rather than propagating an
    ungrounded passthrough."""

    def _boom(_messages, _model_name):
        raise RuntimeError("retrieval exploded")

    monkeypatch.setattr(contextual_model, "_compiled_evidence_and_bundle", _boom)

    with pytest.raises(GroundingUnavailableError, match="error:RuntimeError"):
        await _compiled_evidence_and_bundle_bounded(messages, "m")

    degraded, reason = grounding_snapshot()
    assert degraded is True
    assert reason == "error:RuntimeError"


async def test_bounded_fails_closed_when_breaker_open(monkeypatch, messages):
    """An OPEN compile-latency circuit breaker also fails closed by default —
    skipping the to_thread+wait_for round trip but still refusing the model call."""
    contextual_model._ctx_compile_degradation_streak = (
        contextual_model._CTX_COMPILE_BREAKER_THRESHOLD
    )
    contextual_model._ctx_compile_breaker_reopen_at = time.monotonic() + 30.0

    with pytest.raises(GroundingUnavailableError, match="circuit_breaker_open"):
        await _compiled_evidence_and_bundle_bounded(messages, "m")


async def test_bounded_fails_closed_on_quality_gate_failure(monkeypatch, messages):
    """A genuine compile that yields a quality-gate-rejected (empty) bundle is
    ALSO governed by the grounding policy, not treated as an ordinary result."""
    bundle = _FakeBundle(
        retrieval_quality_gate_failed=True,
        retrieval_quality_reason="low_relevance_topk",
    )

    def _quality_failed(_messages, _model_name):
        return messages, bundle

    monkeypatch.setattr(
        contextual_model, "_compiled_evidence_and_bundle", _quality_failed
    )

    with pytest.raises(GroundingUnavailableError, match="quality_gate"):
        await _compiled_evidence_and_bundle_bounded(messages, "m")

    degraded, reason = grounding_snapshot()
    assert degraded is True
    assert reason.startswith("quality_gate:low_relevance_topk")


# ---------------------------------------------------------------------------
# Explicit opt-in ("best_effort" / "none") — proceeds, but visibly marked
# ---------------------------------------------------------------------------


async def test_bounded_best_effort_degrades_with_marker_on_timeout(
    monkeypatch, messages
):
    """An explicit best_effort opt-in still proceeds on timeout, but the messages
    carry a visible degraded-grounding marker and the outcome is tracked."""
    monkeypatch.setattr(contextual_model, "_CONTEXT_COMPILE_TIMEOUT_S", 0.2)

    def _slow_blocking(_messages, _model_name):
        time.sleep(5.0)
        raise AssertionError("compile should have been abandoned at the budget")

    monkeypatch.setattr(
        contextual_model, "_compiled_evidence_and_bundle", _slow_blocking
    )

    with use_grounding_policy("best_effort"):
        governed, bundle = await _compiled_evidence_and_bundle_bounded(messages, "m")
        degraded, reason = grounding_snapshot()

    assert bundle is None
    assert governed[1:] == messages  # original turn preserved, prefixed with a marker
    marker = _marker_text(governed)
    assert "degraded-evidence" in marker
    assert "grounding: degraded" in marker
    assert "reason: timeout" in marker
    assert degraded is True
    assert reason == "timeout"


async def test_bounded_none_degrades_with_marker_on_error(monkeypatch, messages):
    """An explicit ``none`` opt-in proceeds on a compile error, marked ``grounding: none``."""

    def _boom(_messages, _model_name):
        raise RuntimeError("nope")

    monkeypatch.setattr(contextual_model, "_compiled_evidence_and_bundle", _boom)

    with use_grounding_policy("none"):
        governed, bundle = await _compiled_evidence_and_bundle_bounded(messages, "m")
        degraded, reason = grounding_snapshot()

    assert bundle is None
    marker = _marker_text(governed)
    assert "grounding: none" in marker
    assert "reason: error:RuntimeError" in marker
    assert degraded is True
    assert reason == "error:RuntimeError"


async def test_bounded_best_effort_degrades_on_quality_gate_failure(
    monkeypatch, messages
):
    """A quality-gate-rejected bundle under best_effort proceeds marked degraded,
    and the (empty, but real) bundle is still returned for TTFT/observability."""
    bundle = _FakeBundle(
        retrieval_quality_gate_failed=True,
        retrieval_quality_reason="low_relevance_topk,drift",
    )

    def _quality_failed(_messages, _model_name):
        return messages, bundle

    monkeypatch.setattr(
        contextual_model, "_compiled_evidence_and_bundle", _quality_failed
    )

    with use_grounding_policy("best_effort"):
        governed, returned_bundle = await _compiled_evidence_and_bundle_bounded(
            messages, "m"
        )
        degraded, reason = grounding_snapshot()

    assert returned_bundle is bundle  # real (empty) bundle preserved, not discarded
    marker = _marker_text(governed)
    assert "grounding: degraded" in marker
    assert "reason: quality_gate:low_relevance_topk,drift" in marker
    assert degraded is True
    assert reason == "quality_gate:low_relevance_topk,drift"


# ---------------------------------------------------------------------------
# Unaffected paths — fast success + event-loop liveness
# ---------------------------------------------------------------------------


async def test_bounded_fast_path_returns_full_context(monkeypatch, messages):
    """A quick, quality-gate-passing retrieval still returns the FULL compiled
    result (fast path intact) — the grounding contract never touches success."""
    sentinel_bundle = _FakeBundle()
    compiled = [_Msg(), *messages]

    def _quick(_messages, _model_name):
        return compiled, sentinel_bundle

    monkeypatch.setattr(contextual_model, "_compiled_evidence_and_bundle", _quick)

    governed, bundle = await _compiled_evidence_and_bundle_bounded(messages, "m")
    assert governed is compiled
    assert bundle is sentinel_bundle
    degraded, _ = grounding_snapshot()
    assert degraded is False


async def test_bounded_does_not_block_the_event_loop(monkeypatch, messages):
    """While a slow compile runs, the event loop stays responsive: a concurrent
    ticker keeps advancing (it would not if the loop were blocked inline) — even
    though the default policy ultimately refuses the call."""
    monkeypatch.setattr(contextual_model, "_CONTEXT_COMPILE_TIMEOUT_S", 0.5)

    def _slow_blocking(_messages, _model_name):
        time.sleep(0.5)
        raise AssertionError("compile should have been abandoned at the budget")

    monkeypatch.setattr(
        contextual_model, "_compiled_evidence_and_bundle", _slow_blocking
    )

    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker_task = asyncio.create_task(_ticker())
    try:
        with pytest.raises(GroundingUnavailableError):
            await _compiled_evidence_and_bundle_bounded(messages, "m")
    finally:
        ticker_task.cancel()

    # A blocked loop would have advanced the ticker ~0 times during the compile;
    # off-loop, it advances many times across the ~0.5s window.
    assert ticks >= 5


async def test_compile_messages_bounded_fails_closed_by_default(monkeypatch, messages):
    """The messages-only bounded variant (count_tokens/compact_messages) also
    fails closed by default on a compilation error."""

    def _boom(_messages, _model_name):
        raise RuntimeError("nope")

    monkeypatch.setattr(contextual_model, "_compiled_evidence_and_bundle", _boom)

    with pytest.raises(GroundingUnavailableError):
        await _compile_messages_bounded(messages, "m")


async def test_compile_messages_bounded_best_effort_degrades(monkeypatch, messages):
    """Under an explicit best_effort opt-in, the messages-only variant returns the
    marker-prefixed messages instead of raising."""

    def _boom(_messages, _model_name):
        raise RuntimeError("nope")

    monkeypatch.setattr(contextual_model, "_compiled_evidence_and_bundle", _boom)

    with use_grounding_policy("best_effort"):
        governed = await _compile_messages_bounded(messages, "m")

    assert governed[1:] == messages
    assert "degraded-evidence" in _marker_text(governed)


@pytest.mark.asyncio
async def test_degraded_outcome_survives_a_child_task_boundary(monkeypatch, messages):
    """Regression (wave-0 gate, D-33): the run-scoped degraded outcome must be
    visible to the frame that opened the scope even when the model call that
    discovered the degradation ran in a CHILD asyncio task.

    ``run_agent`` reads ``grounding_snapshot()`` to fold a degraded-grounding run
    into the same ``degraded`` flag that gates RunTrace status, the reward EMA,
    ARPO step credit and shape-policy learning. The model request routinely sits a
    task boundary below that scope (pydantic-ai/anyio task groups, and every
    ``asyncio.to_thread`` hop the event-loop-isolation work added), and a
    ContextVar WRITE inside a child context copy is invisible to the parent — so
    with plain-value ContextVars the degradation was silently dropped and the run
    was recorded as a plain success, which is the exact defect this contract
    exists to prevent.
    """

    def _boom(_messages, _model_name):
        raise RuntimeError("compile down")

    monkeypatch.setattr(cm, "_compiled_evidence_and_bundle", _boom)
    monkeypatch.setattr(cm, "_ctx_compile_breaker_reopen_at", 0.0)
    monkeypatch.setattr(cm, "_ctx_compile_degradation_streak", 0)

    async def _model_call():
        return await cm._compiled_evidence_and_bundle_bounded(messages, "m")

    with cm.use_grounding_policy("best_effort"):
        await asyncio.create_task(_model_call())
        degraded, reason = cm.grounding_snapshot()

    assert degraded is True
    assert reason == "error:RuntimeError"


@pytest.mark.asyncio
async def test_degraded_outcome_does_not_leak_across_sibling_scopes(
    monkeypatch, messages
):
    """A fresh scope starts clean — the mutable outcome container is per-scope."""

    def _boom(_messages, _model_name):
        raise RuntimeError("compile down")

    monkeypatch.setattr(cm, "_compiled_evidence_and_bundle", _boom)
    monkeypatch.setattr(cm, "_ctx_compile_breaker_reopen_at", 0.0)
    monkeypatch.setattr(cm, "_ctx_compile_degradation_streak", 0)

    with cm.use_grounding_policy("best_effort"):
        await cm._compiled_evidence_and_bundle_bounded(messages, "m")
        assert cm.grounding_snapshot()[0] is True

    monkeypatch.setattr(cm, "_ctx_compile_breaker_reopen_at", 0.0)
    monkeypatch.setattr(cm, "_ctx_compile_degradation_streak", 0)
    with cm.use_grounding_policy("best_effort"):
        assert cm.grounding_snapshot() == (False, "")
