"""Tests for the ResiliencePolicy backoff/jitter modes and retry-delay hints.

CONCEPT:ORCH-1.36 — Declarative Resilience Policy.

These modes exist so hand-rolled retry loops across the codebase (linear
backoff in the PulseLink X-search tool, additive lock-contention jitter in the
LadybugDB backend, zero-delay schema-heal retries in the PostgreSQL backend)
can migrate onto the ONE shared policy with byte-identical per-site delays.
"""

from __future__ import annotations

import random

import pytest

from agent_utilities.orchestration.resilience import (
    ResiliencePolicy,
    RetryableError,
    compute_backoff,
    run_with_resilience,
    run_with_resilience_sync,
)

# --------------------------------------------------------------------------- #
# linear backoff strategy
# --------------------------------------------------------------------------- #


def test_linear_backoff_grows_with_attempt():
    policy = ResiliencePolicy(
        backoff_base_s=1.5,
        backoff_strategy="linear",
        max_backoff_s=5.0,
        jitter=False,
    )
    # base * attempt, capped — the historical x_search delays (1.5, 3.0, 4.5, 5.0).
    assert compute_backoff(1, policy) == pytest.approx(1.5)
    assert compute_backoff(2, policy) == pytest.approx(3.0)
    assert compute_backoff(3, policy) == pytest.approx(4.5)
    assert compute_backoff(4, policy) == pytest.approx(5.0)  # capped


def test_invalid_backoff_strategy_rejected():
    with pytest.raises(ValueError, match="backoff_strategy"):
        ResiliencePolicy(backoff_strategy="fibonacci")


# --------------------------------------------------------------------------- #
# additive jitter strategy
# --------------------------------------------------------------------------- #


def test_additive_jitter_adds_bounded_offset():
    policy = ResiliencePolicy(
        backoff_base_s=0.1,
        backoff_factor=2.0,
        max_backoff_s=100.0,
        jitter=True,
        jitter_strategy="additive",
    )
    # The historical LadybugDB lock backoff: (2**n)*0.1 + rand()*0.1.
    for attempt, capped in ((1, 0.1), (2, 0.2), (3, 0.4)):
        value = compute_backoff(attempt, policy, rng=random.Random(7))
        assert capped <= value < capped + 0.1


def test_additive_jitter_is_seedable():
    policy = ResiliencePolicy(
        backoff_base_s=0.1, jitter=True, jitter_strategy="additive"
    )
    v1 = compute_backoff(1, policy, rng=random.Random(42))
    v2 = compute_backoff(1, policy, rng=random.Random(42))
    assert v1 == v2


def test_invalid_jitter_strategy_rejected():
    with pytest.raises(ValueError, match="jitter_strategy"):
        ResiliencePolicy(jitter_strategy="gaussian")


# --------------------------------------------------------------------------- #
# RetryableError delay hint
# --------------------------------------------------------------------------- #


async def test_retryable_error_hint_overrides_policy_backoff():
    sleeps: list[float] = []

    async def _record_sleep(delay: float) -> None:
        sleeps.append(delay)

    calls = {"n": 0}

    async def _flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableError("heal and retry now", backoff_s=0.0)
        return "ok"

    policy = ResiliencePolicy(
        max_attempts=3,
        backoff_base_s=10.0,  # would be a huge sleep without the hint
        jitter=False,
        retry_on=(RetryableError,),
    )
    result = await run_with_resilience(_flaky, policy, sleep=_record_sleep)
    assert result == "ok"
    assert calls["n"] == 3
    # backoff_s=0.0 means "retry immediately": the runner never sleeps.
    assert sleeps == []


def test_retryable_error_hint_sync_uses_exact_delay():
    sleeps: list[float] = []
    calls = {"n": 0}

    def _flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RetryableError("server says wait", backoff_s=0.25)
        return "ok"

    policy = ResiliencePolicy(
        max_attempts=2, backoff_base_s=99.0, jitter=False, retry_on=(RetryableError,)
    )
    result = run_with_resilience_sync(_flaky, policy, sleep=sleeps.append)
    assert result == "ok"
    assert sleeps == [0.25]


def test_retryable_error_without_hint_uses_policy_backoff():
    sleeps: list[float] = []
    calls = {"n": 0}

    def _flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RetryableError("transient")  # no backoff_s hint
        return "ok"

    policy = ResiliencePolicy(
        max_attempts=2, backoff_base_s=0.5, jitter=False, retry_on=(RetryableError,)
    )
    result = run_with_resilience_sync(_flaky, policy, sleep=sleeps.append)
    assert result == "ok"
    assert sleeps == [0.5]


def test_retryable_error_still_subject_to_policy_retry_on():
    calls = {"n": 0}

    def _flaky() -> str:
        calls["n"] += 1
        raise RetryableError("never allowed", backoff_s=0.0)

    # Policy does not list RetryableError -> no retry despite the hint.
    policy = ResiliencePolicy(max_attempts=3, jitter=False, retry_on=(OSError,))
    with pytest.raises(RetryableError):
        run_with_resilience_sync(_flaky, policy, sleep=lambda _s: None)
    assert calls["n"] == 1
