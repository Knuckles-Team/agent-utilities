"""NE-181 session-boundary contract fixtures.

These fixtures are deliberately transport/domain-local.  They prove the
carrier and lifecycle invariants without starting a broker, a graph engine,
or an MCP server; root validation owns the live WorkItem/queue and chaos gates.
"""

from __future__ import annotations

import pytest

from agent_utilities.orchestration import agent_dispatch
from agent_utilities.orchestration.agent_dispatch import (
    KIND_GOAL_LOOP,
    AgentTurnEnvelope,
    DispatchCarrier,
    DispatchCarrierError,
    SessionLockCapacityError,
)
from agent_utilities.orchestration.agent_dispatch_worker import (
    DispatchWorkerLifecycle,
)

_CARRIER_SECRET = "unit-only-dispatch-carrier-secret"


def _carrier(*, now: float = 100.0, tenant: str = "tenant-a") -> DispatchCarrier:
    return DispatchCarrier.mint(
        tenant=tenant,
        session_id="session-a",
        job_id="dispatch-job-a",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-a",
        now=now,
        ttl_seconds=30.0,
        nonce="nonce-a",
        secret=_CARRIER_SECRET,
    )


def test_carrier_round_trip_binds_versioned_tenant_session_job_and_expiry():
    carrier = _carrier()
    envelope = AgentTurnEnvelope(
        job_id="dispatch-job-a",
        session_id="session-a",
        tenant="tenant-a",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-a",
        carrier=carrier,
    )

    assert envelope.authenticate_carrier(now=101.0, secret=_CARRIER_SECRET) == carrier
    assert carrier.version == 1
    assert carrier.expires_at == 130.0


def test_forged_and_cross_tenant_carriers_fail_closed():
    carrier = _carrier()
    forged = carrier.model_copy(update={"tenant": "tenant-attacker"})
    forged_envelope = AgentTurnEnvelope(
        job_id="dispatch-job-a",
        session_id="session-a",
        tenant="tenant-attacker",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-a",
        carrier=forged,
    )
    with pytest.raises(DispatchCarrierError):
        forged_envelope.authenticate_carrier(now=101.0, secret=_CARRIER_SECRET)

    cross_tenant = AgentTurnEnvelope(
        job_id="dispatch-job-a",
        session_id="session-a",
        tenant="tenant-b",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-a",
        carrier=carrier,
    )
    with pytest.raises(DispatchCarrierError):
        cross_tenant.authenticate_carrier(now=101.0, secret=_CARRIER_SECRET)


def test_expired_carrier_and_missing_carrier_are_rejected():
    expired = AgentTurnEnvelope(
        job_id="dispatch-job-a",
        session_id="session-a",
        tenant="tenant-a",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-a",
        carrier=_carrier(),
    )
    with pytest.raises(DispatchCarrierError, match="expired"):
        expired.authenticate_carrier(now=130.0, secret=_CARRIER_SECRET)

    missing = AgentTurnEnvelope(
        job_id="dispatch-job-b",
        session_id="session-b",
        tenant="tenant-a",
        kind=KIND_GOAL_LOOP,
    )
    with pytest.raises(DispatchCarrierError, match="no carrier"):
        missing.authenticate_carrier(now=101.0, secret=_CARRIER_SECRET)


def test_duplicate_delivery_keeps_one_identity_for_native_workitem_idempotency():
    carrier = _carrier()
    first = AgentTurnEnvelope(
        job_id="dispatch-job-a",
        session_id="session-a",
        tenant="tenant-a",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-a",
        carrier=carrier,
    )
    duplicate = AgentTurnEnvelope(
        job_id="dispatch-job-a",
        session_id="session-a",
        tenant="tenant-a",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-a",
        carrier=carrier,
    )

    first.authenticate_carrier(now=101.0, secret=_CARRIER_SECRET)
    duplicate.authenticate_carrier(now=101.0, secret=_CARRIER_SECRET)
    assert f"workitem:dispatch:{first.job_id}" == (
        f"workitem:dispatch:{duplicate.job_id}"
    )


def test_high_cardinality_session_churn_reclaims_local_lock_entries():
    for index in range(512):
        with agent_dispatch.session_execution_guard(f"session-churn-{index}"):
            pass
    assert agent_dispatch.session_lock_registry_size() == 0


def test_session_lock_registry_refuses_unbounded_simultaneous_cardinality(monkeypatch):
    monkeypatch.setattr(agent_dispatch, "MAX_SESSION_LOCK_ENTRIES", 1)
    with agent_dispatch.session_execution_guard("session-held"):
        with pytest.raises(SessionLockCapacityError):
            with agent_dispatch.session_execution_guard("session-over-cap"):
                pass


def test_scale_down_drain_waits_for_active_session_then_reconnects_generation():
    lifecycle = DispatchWorkerLifecycle("worker-a")
    assert lifecycle.begin_session("session-a") is True

    drain = lifecycle.request_drain(reason="scale_down")
    assert drain["state"] == lifecycle.DRAINING
    assert lifecycle.should_claim() is False
    assert lifecycle.begin_session("session-b") is False
    assert lifecycle.wait_drained(timeout=0.0) is False

    lifecycle.end_session("session-a")
    assert lifecycle.wait_drained(timeout=0.0) is True
    lifecycle.mark_stopped()
    assert lifecycle.state == lifecycle.DRAINED
    assert lifecycle.reconnect() == 2
    assert lifecycle.state == lifecycle.RUNNING
    assert lifecycle.begin_session("session-b") is True
    lifecycle.end_session("session-b")


def test_reconnect_cannot_cut_off_an_active_session():
    lifecycle = DispatchWorkerLifecycle("worker-a")
    assert lifecycle.begin_session("session-a") is True
    lifecycle.request_drain(reason="worker_replacement")
    with pytest.raises(RuntimeError, match="active"):
        lifecycle.reconnect()
    lifecycle.end_session("session-a")
