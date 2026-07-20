"""Adversarial tests for authenticated GraphSession query boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    use_session,
)
from agent_utilities.knowledge_graph.facade import KnowledgeGraph
from agent_utilities.security.brain_context import ActorContext, use_actor
from agent_utilities.security.request_identity import mint_graph_session


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        auth_jwt_jwks_uri="https://identity.invalid/jwks",
        auth_jwt_issuer=None,
        auth_jwt_audience="agent-services",
        auth_jwt_algorithms=["RS256"],
        mcp_jwt_audience=None,
        kg_policy_version="policy-v1",
        graph_service_endpoints=[],
    )


def test_guarded_facade_read_fails_closed_when_tenant_policy_is_unavailable(
    monkeypatch,
):
    from agent_utilities.knowledge_graph.core import secured_reads

    store = mock.Mock()
    facade = KnowledgeGraph()
    facade._store = store
    actor = ActorContext(
        actor_id="principal-policy",
        roles=("kg:read",),
        tenant_id="tenant-alpha",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant-alpha",
        scopes=frozenset({"kg:read"}),
        graph="tenant__tenant-alpha____commons__",
    )

    def unavailable_policy():
        raise RuntimeError("policy store unavailable")

    monkeypatch.setattr(secured_reads, "get_company_brain", unavailable_policy)

    with use_session(session), pytest.raises(
        PermissionError, match="Tenant query scoping failed"
    ):
        facade.query(
            "MATCH (n:Record) RETURN n",
            session=session,
        )
    store.execute_read.assert_not_called()


def test_guarded_facade_read_fails_closed_when_acl_filter_is_unavailable(monkeypatch):
    from agent_utilities.knowledge_graph.core import secured_reads

    class PolicyBrain:
        tenancy = SimpleNamespace(scope_cypher_query=lambda query, _tenant: query)

        @property
        def permissions(self):
            raise RuntimeError("permission store unavailable")

    store = mock.Mock()
    store.execute_read.return_value = [{"id": "record-1"}]
    facade = KnowledgeGraph()
    facade._store = store
    actor = ActorContext(
        actor_id="principal-filter",
        roles=("kg:read",),
        tenant_id="tenant-alpha",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor, tenant="tenant-alpha", scopes=frozenset({"kg:read"})
    )

    monkeypatch.setattr(secured_reads, "get_company_brain", PolicyBrain)

    with use_session(session), pytest.raises(
        PermissionError, match="Node permission evaluation failed"
    ):
        facade.query(
            "MATCH (n:Record) RETURN n",
            session=session,
        )


def test_guarded_facade_read_fails_closed_when_audit_is_unavailable(monkeypatch):
    from agent_utilities.knowledge_graph.core import secured_reads

    class PolicyBrain:
        tenancy = SimpleNamespace(scope_cypher_query=lambda query, _tenant: query)

        @property
        def provenance(self):
            raise RuntimeError("audit store unavailable")

    store = mock.Mock()
    store.execute_read.return_value = []
    facade = KnowledgeGraph()
    facade._store = store
    actor = ActorContext(
        actor_id="principal-audit",
        roles=("kg:read",),
        tenant_id="tenant-alpha",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor, tenant="tenant-alpha", scopes=frozenset({"kg:read"})
    )

    monkeypatch.setattr(secured_reads, "get_company_brain", PolicyBrain)

    with use_session(session), pytest.raises(
        PermissionError, match="Read audit recording failed"
    ):
        facade.query(
            "MATCH (n:Record) RETURN n",
            session=session,
        )


@pytest.mark.asyncio
async def test_authenticated_mcp_dispatch_uses_the_middleware_minted_currency():
    from agent_utilities.mcp import kg_server

    captured = {}

    def tool():
        captured["session"] = current_session()
        return "ok"

    actor = ActorContext(
        actor_id="principal-mcp",
        roles=("kg:read",),
        tenant_id="tenant-alpha",
        authenticated=True,
    )
    cfg = _config()
    with mock.patch("agent_utilities.core.config.config", cfg):
        session = mint_graph_session(actor)
    with (
        mock.patch.dict(kg_server.REGISTERED_TOOLS, {"test_read": tool}),
        use_actor(actor),
        use_session(session),
    ):
        assert await kg_server._execute_tool("test_read") == "ok"

    captured_session = captured["session"]
    assert captured_session is session
    assert captured_session.actor is actor
    assert captured_session.scopes == frozenset({"kg:read"})
    assert captured_session.graph == "tenant__tenant-alpha____commons__"
    assert current_session() is None
