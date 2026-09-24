"""EH-407: a guardrail loosening is approved only by a person at the operator console.

Every other ``action.approval`` can still be decided through the agent
governance tool; a ``guardrail.loosen`` one cannot, whatever the session holds.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.orchestration.action_policy import ACTION_APPROVAL_KIND
from agent_utilities.orchestration.approval import (
    CONSOLE_ONLY_KINDS,
    ApprovalSurface,
    decide_action_approval,
)
from agent_utilities.security.guardrail_evolution import LOOSEN_ACTION_KIND
from tests.unit.fleet_autonomy_fakes import (
    FakeControlLeaseClient,
    verified_fleet_session,
)

HOUR_MS = 3_600_000


def _engine() -> tuple[Any, FakeControlLeaseClient]:
    leases = FakeControlLeaseClient()
    graph = SimpleNamespace(client=SimpleNamespace(control_leases=leases))
    graph.for_graph = lambda _name: graph
    return SimpleNamespace(graph_compute=graph), leases


def _queue(leases: FakeControlLeaseClient, tenant: str, kind: str) -> str:
    approval_id = f"action_approval:{kind}"
    leases.issue(
        tenant=tenant,
        lease_id=approval_id,
        kind=ACTION_APPROVAL_KIND,
        grant={"kind": kind, "target": "fleet/child/github", "request_digest": "d"},
        issued_at_ms=1,
        expires_at_ms=HOUR_MS,
        hard_expires_at_ms=HOUR_MS,
        idempotency_key=approval_id,
    )
    return approval_id


def test_an_agent_tool_cannot_grant_a_guardrail_loosening() -> None:
    assert LOOSEN_ACTION_KIND in CONSOLE_ONLY_KINDS
    engine, leases = _engine()
    with verified_fleet_session() as session:
        approval_id = _queue(leases, session.tenant, LOOSEN_ACTION_KIND)
        with pytest.raises(PermissionError):
            decide_action_approval(engine, approval_id, "approved")
        lease = leases.get(tenant=session.tenant, lease_id=approval_id)
        assert lease is not None and lease["status"] == "active", "still pending"
        decided = decide_action_approval(
            engine, approval_id, "approved", ApprovalSurface.OPERATOR_CONSOLE
        )
    assert decided == {"approval_id": approval_id, "decision": "approved"}


def test_other_approvals_keep_their_agent_tool_path() -> None:
    engine, leases = _engine()
    with verified_fleet_session() as session:
        approval_id = _queue(leases, session.tenant, "restart_service")
        decided = decide_action_approval(engine, approval_id, "approved")
    assert decided["decision"] == "approved"
