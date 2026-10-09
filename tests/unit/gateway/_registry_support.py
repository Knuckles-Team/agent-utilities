"""Shared registry-test fixtures for tests/unit/gateway."""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


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


class BindAuthority:
    """ASGI wrapper that binds one actor and graph session around each request."""

    def __init__(self, app, actor: ActorContext, session: GraphSession):
        self.app = app
        self.actor = actor
        self.session = session

    async def __call__(self, scope, receive, send):
        with use_actor(self.actor), use_session(self.session):
            await self.app(scope, receive, send)
