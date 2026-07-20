"""Governed decisions for ActionPolicy approval records."""

from __future__ import annotations

import time
from typing import Any


def decide_action_approval(
    engine: Any, approval_id: str, decision: str
) -> dict[str, str]:
    """Atomically decide one pending ``ActionApproval`` through the verified graph session."""
    if not str(approval_id).startswith("action_approval:"):
        raise ValueError("approval_id must identify an ActionApproval")
    normalized = str(decision).strip().lower()
    if normalized in {"approve", "approved"}:
        status = "approved"
    elif normalized in {"deny", "denied", "reject", "rejected"}:
        status = "denied"
    else:
        raise ValueError("decision must be approved or denied")

    from agent_utilities.knowledge_graph.core.session import (
        resolve_session,
        use_session,
    )
    from agent_utilities.security.brain_context import use_actor

    session = resolve_session(required_scope="kg:write")
    graph = engine.graph_compute
    if session.graph:
        graph = graph.for_graph(session.graph)
    with use_session(session), use_actor(session.actor):
        applied = graph.compare_and_set_node_fields(
            str(approval_id),
            {"status": "pending"},
            {"status": status, "decided_unix": time.time()},
        )
    if not applied:
        raise LookupError("approval is missing or no longer pending")
    return {"approval_id": str(approval_id), "decision": status}
