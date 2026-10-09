"""Shared registry-test fixtures for tests/unit/gateway."""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext


def registry_reader(actor_id: str, tenant_id: str):
    """One authenticated registry:read service actor and its tenant session."""
    actor = ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("registry:read",),
        tenant_id=tenant_id,
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=tenant_id,
        scopes=frozenset({"kg:read"}),
        graph=tenant_id,
        policy_version="test",
        audience="test",
    )
    return actor, session
