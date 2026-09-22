"""Focused contracts for the AU GraphOS catalog read ports."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, cast

import pytest

from agent_utilities.api.catalog import (
    AgentCatalogRecord,
    CatalogReadAuthority,
    CatalogReadError,
    WorkflowCatalogRecord,
    catalog_read_ports,
)
from agent_utilities.control_plane.catalogs import (
    catalog_read_ports as control_plane_catalog_read_ports,
)
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    ScopeError,
    SessionRequiredError,
    use_session,
)
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext


def _session() -> GraphSession:
    actor = ActorContext(
        actor_id="principal-a",
        actor_type=ActorType.AUTOMATED_SERVICE,
        tenant_id="tenant-a",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="tenant-a",
        scopes=frozenset({"kg:read"}),
        graph="tenant-a",
        audience="graph-os",
        policy_version="policy-a",
    )


class _Engine:
    def __init__(self, agents: list[dict[str, Any]], workflows: list[dict[str, Any]]):
        self.agents = agents
        self.workflows = workflows
        self.calls: list[tuple[str, GraphSession]] = []

    def query_cypher(
        self, query: str, *, session: GraphSession
    ) -> list[dict[str, Any]]:
        self.calls.append((query, session))
        if "WorkflowDefinition" in query:
            return self.workflows
        return self.agents


def test_factory_exposes_one_authority_with_typed_async_and_sync_ports() -> None:
    workflow_digest = hashlib.sha256(b"workflow").hexdigest()
    engine = _Engine(
        agents=[
            {
                "id": "agent:beta",
                "agent_id": None,
                "name": "Beta",
                "description": "B",
                "system_prompt": "prompt-b",
                "status": "active",
                "tool_id": "tool:z",
                "tool_name": "zeta",
            },
            {
                "id": "agent:alpha",
                "agent_id": "public:alpha",
                "name": "Alpha",
                "description": "A",
                "system_prompt": "",
                "status": None,
                "tool_id": None,
                "tool_name": None,
            },
            {
                "id": "agent:beta",
                "agent_id": None,
                "name": "Beta",
                "description": "B",
                "system_prompt": "prompt-b",
                "status": "active",
                "tool_id": "tool:a",
                "tool_name": "alpha",
            },
            {
                "id": "agent:retired",
                "agent_id": None,
                "name": "Retired",
                "description": "old",
                "system_prompt": None,
                "status": "retired",
                "tool_id": None,
                "tool_name": None,
            },
        ],
        workflows=[
            {
                "id": "workflow:z",
                "name": "Zeta",
                "description": "Z",
                "status": "published",
                "version": 2,
                "content_hash": workflow_digest,
            },
            {
                "id": "workflow:a",
                "name": "Alpha",
                "description": "A",
                "status": "retired",
                "version": 3,
                "content_hash": f"sha256:{workflow_digest}",
            },
        ],
    )
    session = _session()

    with use_session(session):
        workflows, agents = catalog_read_ports(engine, session)
        workflow_authority = cast(CatalogReadAuthority, workflows)
        agent_authority = cast(CatalogReadAuthority, agents)
        assert workflows is agents
        assert workflow_authority.list_current_workflows_sync() == (
            WorkflowCatalogRecord(
                workflow_id="workflow:a",
                name="Alpha",
                description="A",
                status="retired",
                revision=3,
                definition_digest=f"sha256:{workflow_digest}",
            ),
            WorkflowCatalogRecord(
                workflow_id="workflow:z",
                name="Zeta",
                description="Z",
                status="active",
                revision=2,
                definition_digest=f"sha256:{workflow_digest}",
            ),
        )
        assert agent_authority.list_authorized_agents_sync() == (
            AgentCatalogRecord(
                agent_id="public:alpha",
                name="Alpha",
                description="A",
                system_prompt=None,
                tools=None,
            ),
            AgentCatalogRecord(
                agent_id="agent:beta",
                name="Beta",
                description="B",
                system_prompt="prompt-b",
                tools=("alpha", "zeta"),
            ),
        )
        assert asyncio.run(workflows.list_current_workflows()) == (
            workflow_authority.list_current_workflows_sync()
        )
        assert asyncio.run(agents.list_authorized_agents()) == (
            agent_authority.list_authorized_agents_sync()
        )

    assert len(engine.calls) == 6
    assert all(call_session is session for _, call_session in engine.calls)
    assert any("MATCH (a:Agent)" in query for query, _ in engine.calls)
    assert any("MATCH (w:WorkflowDefinition)" in query for query, _ in engine.calls)
    assert control_plane_catalog_read_ports is catalog_read_ports


def test_reads_require_verified_ambient_session_and_never_fallback_to_empty() -> None:
    engine = _Engine([], [])
    authority = CatalogReadAuthority(engine)

    from agent_utilities.knowledge_graph.core.session import suspend_session

    with suspend_session(), pytest.raises(SessionRequiredError):
        authority.list_authorized_agents_sync()
    with suspend_session(), pytest.raises(SessionRequiredError):
        authority.list_current_workflows_sync()


def test_reads_require_the_kg_read_scope() -> None:
    actor = ActorContext(
        actor_id="principal-a",
        actor_type=ActorType.AUTOMATED_SERVICE,
        tenant_id="tenant-a",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant-a",
        scopes=frozenset(),
        graph="tenant-a",
        audience="graph-os",
        policy_version="policy-a",
    )
    with use_session(session), pytest.raises(ScopeError):
        CatalogReadAuthority(_Engine([], [])).list_authorized_agents_sync()


def test_authoritative_query_failure_is_propagated() -> None:
    class BrokenEngine:
        def query_cypher(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("backend unavailable")

    session = _session()
    with use_session(session):
        authority = CatalogReadAuthority(BrokenEngine())
        with pytest.raises(RuntimeError, match="backend unavailable"):
            authority.list_current_workflows_sync()


def test_malformed_workflow_state_fails_closed() -> None:
    engine = _Engine(
        [],
        [
            {
                "id": "workflow:bad",
                "name": "Bad",
                "description": "bad",
                "status": "mystery",
                "version": 1,
            }
        ],
    )
    session = _session()
    with use_session(session):
        with pytest.raises(CatalogReadError, match="unsupported status"):
            CatalogReadAuthority(engine).list_current_workflows_sync()


def test_missing_authoritative_workflow_revision_or_digest_fails_closed() -> None:
    engine = _Engine(
        [],
        [
            {
                "id": "workflow:incomplete",
                "name": "Incomplete",
                "description": "missing version and content hash",
                "status": "active",
            }
        ],
    )
    session = _session()
    with use_session(session), pytest.raises(CatalogReadError, match="no revision"):
        CatalogReadAuthority(engine).list_current_workflows_sync()

    engine.workflows = [
        {
            "id": "workflow:incomplete",
            "name": "Incomplete",
            "description": "missing content hash",
            "status": "active",
            "version": 1,
        }
    ]
    with (
        use_session(session),
        pytest.raises(CatalogReadError, match="no workflow definition digest"),
    ):
        CatalogReadAuthority(engine).list_current_workflows_sync()
