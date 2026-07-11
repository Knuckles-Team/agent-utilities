"""Hot-tenant isolation, full restart/cold-activation, and rolling-upgrade
continuity chaos — SCALE-P2-1 soak scenarios 5-7.

Built directly against WorkItem's CAS state machine + :class:`FakeScaleEngine`
(see ``test_chaos_worker_and_delivery.py``'s module docstring for why).
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


def test_hot_tenant_quota_does_not_starve_other_tenants(loadgen):
    """The elephant tenant hitting its in-flight quota must not block other tenants.

    Mirrors the workload contract's elephant tenant (5% of residents, 10% of
    active load, CONCEPT AU-P1-1 ``tenant_in_flight_count``): a noisy neighbor
    saturating its own quota is contained to itself, not a fleet-wide stall.
    """
    engine = _engine(loadgen)
    wi = loadgen.wi

    # Elephant tenant fills its quota (cap=2).
    wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref="e1",
        tenant="elephant",
        max_tenant_in_flight=2,
    )
    wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref="e2",
        tenant="elephant",
        max_tenant_in_flight=2,
    )
    with pytest.raises(wi.TenantQuotaExceeded):
        wi.submit_work_item(
            engine,
            kind="agent_turn",
            payload_ref="e3",
            tenant="elephant",
            max_tenant_in_flight=2,
        )

    # An ordinary tenant is completely unaffected — same call pattern succeeds.
    ordinary_id = wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref="o1",
        tenant="tenant-7",
        max_tenant_in_flight=2,
    )
    claim = wi.claim_and_start(engine, item_id=ordinary_id, token="w1", now=0.0)
    assert claim is not None
    outcome = wi.commit_result(
        engine, ordinary_id, claim, outcome="succeeded", result_ref="ok", now=1.0
    )
    assert outcome == "committed"

    # Once the elephant's in-flight items complete, its quota frees up again
    # (it is a live cap, not a permanent ban).
    e1_claim = wi.claim_and_start(
        engine, item_id=None, tenant="elephant", token="w2", now=2.0
    )
    assert e1_claim is not None
    wi.commit_result(
        engine,
        e1_claim["work_item_id"],
        e1_claim,
        outcome="succeeded",
        result_ref="ok",
        now=3.0,
    )
    wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref="e3-retry",
        tenant="elephant",
        max_tenant_in_flight=2,
    )  # no longer raises — one slot freed


def test_full_restart_cold_activation_recovers_in_flight_work(loadgen):
    """Process-local state (locks, worker registry) is gone after a restart; durable
    WorkItem state is not — snapshot/restore models the same guarantee the real
    tiered engine gives via its L3 durable mirror.
    """
    engine = _engine(loadgen)
    wi = loadgen.wi

    done_id = wi.submit_work_item(
        engine, kind="agent_turn", payload_ref="d1", tenant="acme"
    )
    done_claim = wi.claim_and_start(
        engine, item_id=done_id, token="pre-restart-w1", now=0.0
    )
    wi.commit_result(
        engine, done_id, done_claim, outcome="succeeded", result_ref="ok", now=1.0
    )

    in_flight_id = wi.submit_work_item(
        engine, kind="agent_turn", payload_ref="d2", tenant="acme"
    )
    in_flight_claim = wi.claim_and_start(
        engine, item_id=in_flight_id, token="pre-restart-w2", now=2.0, lease_ttl_s=30.0
    )
    assert in_flight_claim is not None
    # The process dies here — never commits. Simulate "restart": a fresh engine
    # instance rehydrated from the durable snapshot (process-local locks/registries
    # gone; every WorkItem node intact).
    snapshot = engine.snapshot()
    restarted = loadgen.FakeScaleEngine.from_snapshot(
        snapshot,
        latency=loadgen.LatencyModel(
            write_mean_s=0.0, write_jitter_s=0.0, query_mean_s=0.0, query_jitter_s=0.0
        ),
    )

    # Everything durable survived the restart, including the in-flight item's state.
    assert (
        wi.get_work_item(restarted, done_id)["status"]
        == wi.WorkItemStatus.SUCCEEDED.value
    )
    assert (
        wi.get_work_item(restarted, in_flight_id)["status"]
        == wi.WorkItemStatus.RUNNING.value
    )

    # A post-restart reap sweep (the fleet's crash-recovery pass, run on any
    # surviving/replacement host) reclaims the in-flight item — it is not lost.
    reaped = wi.reap_expired_leases(restarted, now=1000.0)
    assert in_flight_id in reaped["reaped_ready"]

    # A post-restart worker claims and finishes it for real — exactly once.
    post_claim = wi.claim_and_start(
        restarted, item_id=in_flight_id, token="post-restart-w1", now=1001.0
    )
    assert post_claim is not None
    outcome = wi.commit_result(
        restarted,
        in_flight_id,
        post_claim,
        outcome="succeeded",
        result_ref="ok",
        now=1002.0,
    )
    assert outcome == "committed"
    assert (
        wi.get_work_item(restarted, in_flight_id)["status"]
        == wi.WorkItemStatus.SUCCEEDED.value
    )

    # The pre-restart claim (now long dead) cannot resurrect/overwrite the outcome.
    stale = wi.commit_result(
        restarted,
        in_flight_id,
        in_flight_claim,
        outcome="succeeded",
        result_ref="stale",
        now=1003.0,
    )
    assert stale in ("fenced", "noop")
    assert wi.get_work_item(restarted, in_flight_id)["result_ref"] == "ok"


def test_rolling_upgrade_worker_pool_replacement_has_no_gap(loadgen):
    """Old-generation workers retire (stop heartbeating) while new-generation
    workers come up; every in-flight item still completes exactly once, with
    no window where nothing owns it and no double-completion from a stale
    old-generation ack landing after a new-generation worker already finished it.
    """
    engine = _engine(loadgen)
    wi = loadgen.wi

    # Three turns claimed by the OLD worker generation, all mid-flight when the
    # upgrade begins (rolling: old workers are drained, not force-killed, but we
    # model the worst case — they vanish without committing).
    old_claims = {}
    for i in range(3):
        item_id = wi.submit_work_item(
            engine, kind="agent_turn", payload_ref=f"u{i}", tenant="acme"
        )
        claim = wi.claim_and_start(
            engine, item_id=item_id, token="old-gen", now=0.0, lease_ttl_s=20.0
        )
        assert claim is not None
        old_claims[item_id] = claim

    # Upgrade completes; old generation is gone. New-generation workers come up
    # and a reap sweep reclaims everything the old generation left in flight.
    reaped = wi.reap_expired_leases(engine, now=100.0)
    assert set(reaped["reaped_ready"]) == set(old_claims)

    # New generation claims + completes every item — continuity maintained.
    for item_id in old_claims:
        new_claim = wi.claim_and_start(
            engine, item_id=item_id, token="new-gen", now=101.0
        )
        assert new_claim is not None
        outcome = wi.commit_result(
            engine, item_id, new_claim, outcome="succeeded", result_ref="ok", now=102.0
        )
        assert outcome == "committed"

    # Every stale old-generation ack arriving late is rejected, never double-completing.
    for item_id, old_claim in old_claims.items():
        late = wi.commit_result(
            engine,
            item_id,
            old_claim,
            outcome="succeeded",
            result_ref="stale",
            now=200.0,
        )
        assert late in ("fenced", "noop")
        assert wi.get_work_item(engine, item_id)["result_ref"] == "ok"
