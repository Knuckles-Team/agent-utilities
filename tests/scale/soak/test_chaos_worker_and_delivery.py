"""Worker loss + at-least-once delivery chaos — SCALE-P2-1 soak scenarios 2-3.

Built directly against the engine-native WorkItem CAS state machine
(:mod:`agent_utilities.orchestration.work_item`) and a fresh
:class:`FakeScaleEngine`, not through the rate-based load generator — these
scenarios need to construct an EXACT state ("this item's lease just expired
mid-execution", "the queue redelivered this exact claim") that a Poisson-rate
schedule cannot express precisely. Time is driven via explicit ``now=``
timestamps (the same seam ``tests/unit/orchestration/test_work_item.py`` and
``tests/scale/soak/test_chaos_lifecycle_and_dlq.py`` use) rather than real or
simulated sleeping, so every scenario is exact and instant.

Covers (CI-runnable): worker/host loss mid-lease + crash-recovery reclaim,
duplicate/redelivered claim rejection (queue at-least-once with a live lease),
and idempotent double-commit (a redelivered ack after the original already
landed). Each scenario asserts: no lost item, no duplicate side effect, no
falsely-completed item.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def _make_loadgen_engine(loadgen):
    return loadgen.build_mock_engine(
        latency=loadgen.LatencyModel(
            write_mean_s=0.0, write_jitter_s=0.0, query_mean_s=0.0, query_jitter_s=0.0
        )
    )


def test_worker_crash_mid_lease_is_reclaimed_and_completes_exactly_once(loadgen):
    """A worker claims a turn then crashes before committing (host/worker loss).

    The lease expires; ``reap_expired_leases`` (the crash-recovery sweep every
    dispatch-worker fleet runs) requeues it to ``ready`` with a bumped fencing
    epoch; a second (healthy) worker claims and completes it. No item is lost,
    and — because the crashed worker never got far enough to record its side
    effect before "dying" — there is exactly ONE side-effect execution, not
    two, once a real worker actually finishes the turn.
    """
    engine = _make_loadgen_engine(loadgen)
    wi = loadgen.wi
    side_effects: list[str] = []

    item_id = wi.submit_work_item(
        engine, kind="agent_turn", payload_ref="sess-1", tenant="acme"
    )

    # Worker A claims at t=0 with a short lease, then crashes (never commits).
    claim_a = wi.claim_and_start(
        engine, item_id=item_id, token="worker-a", now=0.0, lease_ttl_s=30.0
    )
    assert claim_a is not None
    assert (
        wi.get_work_item(engine, item_id)["status"] == wi.WorkItemStatus.RUNNING.value
    )

    # 90s later (well past the 30s lease), the reaper sweeps expired leases —
    # exactly what a live dispatch-worker fleet's periodic reap does.
    reaped = wi.reap_expired_leases(engine, now=90.0)
    assert item_id in reaped["reaped_ready"]
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.READY.value
    assert item["lease_epoch"] == 2  # fenced past worker A's epoch (1)

    # Worker B claims the reclaimed item and completes it for real.
    claim_b = wi.claim_and_start(engine, item_id=item_id, token="worker-b", now=91.0)
    assert claim_b is not None
    assert claim_b["fence_token"] == 3
    side_effects.append(f"executed:{item_id}:{claim_b['attempt']}")
    outcome = wi.commit_result(
        engine, item_id, claim_b, outcome="succeeded", result_ref="ok", now=92.0
    )
    assert outcome == "committed"

    # Worker A's crash never let it record a side effect — exactly one execution total.
    assert side_effects == [f"executed:{item_id}:{claim_b['attempt']}"]
    assert (
        wi.get_work_item(engine, item_id)["status"] == wi.WorkItemStatus.SUCCEEDED.value
    )

    # Worker A's belated commit attempt (it "wakes up" and tries to finish anyway)
    # must be rejected, never silently overwriting the real completion.
    late_outcome = wi.commit_result(
        engine, item_id, claim_a, outcome="succeeded", result_ref="stale", now=200.0
    )
    assert late_outcome in ("fenced", "noop")
    assert (
        wi.get_work_item(engine, item_id)["result_ref"] == "ok"
    )  # untouched by the stale commit


def test_redelivered_claim_while_lease_is_live_is_rejected(loadgen):
    """At-least-once queue redelivery while the original claim's lease is still live.

    A queue's at-least-once guarantee means the SAME envelope can be handed to
    a second worker even though the first is still (or already) processing it
    — e.g. an ack got lost on the wire. The engine-native lease is what
    prevents a duplicate EXECUTION: the redelivered worker's claim attempt
    must be rejected outright (a live lease elsewhere), so a disciplined
    worker (only executes after winning the claim) never runs the body twice.
    """
    engine = _make_loadgen_engine(loadgen)
    wi = loadgen.wi

    item_id = wi.submit_work_item(
        engine, kind="agent_turn", payload_ref="sess-2", tenant="acme"
    )
    claim_1 = wi.claim_and_start(
        engine, item_id=item_id, token="worker-1", now=0.0, lease_ttl_s=60.0
    )
    assert claim_1 is not None

    # Redelivery: a second worker gets the SAME envelope while worker-1's lease is live.
    claim_2 = wi.claim_specific(
        engine, item_id, token="worker-2", now=1.0, lease_ttl_s=60.0
    )
    assert claim_2 is None  # correctly rejected — no second execution follows

    # The original worker finishes normally; exactly one commit lands.
    outcome = wi.commit_result(
        engine, item_id, claim_1, outcome="succeeded", result_ref="ok", now=2.0
    )
    assert outcome == "committed"


def test_redelivered_ack_after_completion_is_idempotent_noop(loadgen):
    """A redelivered/duplicate commit for an ALREADY-terminal item is a safe no-op.

    Models the queue redelivering the completion envelope itself (e.g. the
    worker's ack was lost after it had already committed) — the second commit
    attempt must not double-apply the result or re-release downstream
    dependents a second time.
    """
    engine = _make_loadgen_engine(loadgen)
    wi = loadgen.wi

    item_id = wi.submit_work_item(
        engine, kind="agent_turn", payload_ref="sess-3", tenant="acme"
    )
    claim = wi.claim_and_start(engine, item_id=item_id, token="worker-1", now=0.0)
    assert claim is not None

    first = wi.commit_result(
        engine, item_id, claim, outcome="succeeded", result_ref="ok", now=1.0
    )
    assert first == "committed"
    second = wi.commit_result(
        engine, item_id, claim, outcome="succeeded", result_ref="ok-again", now=1.5
    )
    assert second == "noop"

    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.SUCCEEDED.value
    assert (
        item["result_ref"] == "ok"
    )  # the redelivered ack never overwrote the first result
