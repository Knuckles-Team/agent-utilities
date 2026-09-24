"""Governed decisions for ActionPolicy approval records.

The pending/decided approval queue lives on typed EG ``action.approval``
ControlLease records (eg-workitem WRAPUP §3d) — a generic
``compare_and_set_node_fields`` against an ``ActionApproval`` node is refused
by the connected engine's native row guard. Deciding a pending approval is
therefore a lease transition: ``active`` (pending) → ``consumed`` (approved)
or ``revoked`` (denied), CAS'd on the lease's own ``revision``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class ApprovalSurface(StrEnum):
    """Where an approval decision came from."""

    AGENT_TOOL = "agent_tool"
    OPERATOR_CONSOLE = "operator_console"


#: Approval kinds only a person at the operator console may decide -- never an
#: agent through a tool, whatever scopes its session carries. A guardrail
#: loosening (EH-407) is the case: learned signals may only tighten.
CONSOLE_ONLY_KINDS = frozenset({"guardrail.loosen"})


def _refuse_console_only(current: Mapping[str, Any], surface: ApprovalSurface) -> None:
    grant = current.get("grant")
    kind = grant.get("kind") if isinstance(grant, Mapping) else None
    if kind in CONSOLE_ONLY_KINDS and surface is not ApprovalSurface.OPERATOR_CONSOLE:
        raise PermissionError("this approval is decided only at the operator console")


def decide_action_approval(
    engine: Any,
    approval_id: str,
    decision: str,
    surface: ApprovalSurface = ApprovalSurface.AGENT_TOOL,
) -> dict[str, str]:
    """Atomically decide one pending ``action.approval`` lease through the verified graph session.

    ``surface`` defaults to the agent-tool surface; only the console route
    passes :attr:`ApprovalSurface.OPERATOR_CONSOLE`, which alone may decide a
    :data:`CONSOLE_ONLY_KINDS` approval.
    """
    if not str(approval_id).startswith("action_approval:"):
        raise ValueError("approval_id must identify an ActionApproval")
    normalized = str(decision).strip().lower()
    if normalized in {"approve", "approved"}:
        status, target = "approved", "consumed"
    elif normalized in {"deny", "denied", "reject", "rejected"}:
        status, target = "denied", "revoked"
    else:
        raise ValueError("decision must be approved or denied")

    from agent_utilities.knowledge_graph.core.session import (
        resolve_session,
        use_session,
    )
    from agent_utilities.orchestration.action_policy import (
        approval_lease_client,
        approval_lease_tenant,
    )
    from agent_utilities.security.brain_context import use_actor

    session = resolve_session(required_scope="kg:write")
    graph = engine.graph_compute
    if session.graph:
        graph = graph.for_graph(session.graph)
    with use_session(session), use_actor(session.actor):
        leases = approval_lease_client(graph)
        tenant = approval_lease_tenant()
        current = leases.get(tenant=tenant, lease_id=str(approval_id))
        if not isinstance(current, dict) or current.get("status") != "active":
            raise LookupError("approval is missing or no longer pending")
        _refuse_console_only(current, surface)
        result = leases.transition(
            tenant=tenant,
            lease_id=str(approval_id),
            expected_revision=current["revision"],
            to=target,
            idempotency_key=f"decide:{approval_id}",
        )
    if not isinstance(result, dict) or result.get("outcome") != "applied":
        raise LookupError("approval is missing or no longer pending")
    return {"approval_id": str(approval_id), "decision": status}
