"""Timeout/retry/cancel/DLQ recovery chaos — SCALE-P2-1 soak scenario 4.

Built directly against WorkItem's CAS state machine + a fresh
:class:`FakeScaleEngine` (see ``test_chaos_worker_and_delivery.py``'s module
docstring for why: exact state construction, not a rate schedule).

Covers: a turn that times out/fails repeatedly retries with backoff and then
dead-letters once retries are exhausted (never silently dropped), and a turn
cancelled mid-flight lands ``cancelled`` and can never later be resurrected
into ``succeeded`` by a stray late commit (no falsely-completed item).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _engine(loadgen):
    return loadgen.build_mock_engine(
        latency=loadgen.LatencyModel(
            write_mean_s=0.0, write_jitter_s=0.0, query_mean_s=0.0, query_jitter_s=0.0
        )
    )


def test_retry_then_dead_letter_after_max_attempts(loadgen):
    """A turn that keeps timing out/failing retries with backoff, then DLQs."""
    engine = _engine(loadgen)
    wi = loadgen.wi

    item_id = wi.submit_work_item(
        engine, kind="agent_turn", payload_ref="sess-4", tenant="acme", max_attempts=3
    )

    # Attempt 1: claim, then it "times out" (fails, retryable).
    claim = wi.claim_and_start(engine, item_id=item_id, token="w1", now=0.0)
    assert claim is not None and claim["attempt"] == 1
    outcome = wi.commit_result(
        engine,
        item_id,
        claim,
        outcome="failed",
        retryable=True,
        error_ref="timeout",
        now=1.0,
    )
    assert outcome == "retry_scheduled"
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.READY.value
    assert item["next_retry_at"] > 1.0  # backoff window — not immediately reclaimable

    # Backoff not yet elapsed: an immediate reclaim attempt correctly fails.
    too_soon = wi.claim_specific(engine, item_id, token="w2", now=1.1)
    assert too_soon is None

    # Attempt 2, after the backoff window: claim again, times out again.
    claim2 = wi.claim_and_start(engine, item_id=item_id, token="w2", now=60.0)
    assert claim2 is not None and claim2["attempt"] == 2
    outcome2 = wi.commit_result(
        engine,
        item_id,
        claim2,
        outcome="failed",
        retryable=True,
        error_ref="timeout",
        now=61.0,
    )
    assert outcome2 == "retry_scheduled"

    # Attempt 3 (== max_attempts): one more failure exhausts retries -> dead_letter.
    claim3 = wi.claim_and_start(engine, item_id=item_id, token="w3", now=200.0)
    assert claim3 is not None and claim3["attempt"] == 3
    outcome3 = wi.commit_result(
        engine,
        item_id,
        claim3,
        outcome="failed",
        retryable=True,
        error_ref="timeout",
        now=201.0,
    )
    assert outcome3 == "dead_letter"
    final = wi.get_work_item(engine, item_id)
    assert final["status"] == wi.WorkItemStatus.DEAD_LETTER.value
    # Never lost: it has a terminal, inspectable outcome — not silently vanished.
    assert final["error_ref"] == "timeout"


def test_cancel_mid_flight_is_terminal_and_never_falsely_completes(loadgen):
    """Cancelling a running turn lands ``cancelled``; a late stray commit cannot flip it."""
    engine = _engine(loadgen)
    wi = loadgen.wi

    item_id = wi.submit_work_item(
        engine, kind="agent_turn", payload_ref="sess-5", tenant="acme"
    )
    claim = wi.claim_and_start(engine, item_id=item_id, token="w1", now=0.0)
    assert claim is not None

    cancelled = wi.cancel_work_item(engine, item_id, reason="user_abort", now=5.0)
    assert cancelled is True
    assert (
        wi.get_work_item(engine, item_id)["status"] == wi.WorkItemStatus.CANCELLED.value
    )

    # The worker that was still "in flight" eventually tries to commit success —
    # must NOT resurrect a cancelled item into succeeded (falsely-completed guard).
    late = wi.commit_result(
        engine, item_id, claim, outcome="succeeded", result_ref="late", now=6.0
    )
    assert late in ("fenced", "conflict", "noop")
    assert (
        wi.get_work_item(engine, item_id)["status"] == wi.WorkItemStatus.CANCELLED.value
    )

    # Cancelling an already-cancelled item is idempotent (redelivered cancel request).
    assert wi.cancel_work_item(engine, item_id, reason="user_abort", now=7.0) is True
    # But a DIFFERENT terminal item cannot be cancelled into "cancelled" retroactively.
    other_id = wi.submit_work_item(
        engine, kind="agent_turn", payload_ref="sess-6", tenant="acme"
    )
    other_claim = wi.claim_and_start(engine, item_id=other_id, token="w2", now=0.0)
    wi.commit_result(
        engine, other_id, other_claim, outcome="succeeded", result_ref="ok", now=1.0
    )
    assert wi.cancel_work_item(engine, other_id, now=2.0) is False
    assert (
        wi.get_work_item(engine, other_id)["status"]
        == wi.WorkItemStatus.SUCCEEDED.value
    )
