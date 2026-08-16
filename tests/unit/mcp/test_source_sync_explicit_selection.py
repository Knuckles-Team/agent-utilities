"""U-37 — ``source_sync`` must honor independent ``connection``/``graph``
selectors exactly like ``graph_query``/``graph_write``/``graph_ingest``: an
explicit graph never defaults, never fans out, and an unknown graph fails
closed BEFORE any sync (and therefore any watermark/checkpoint read or
advance) happens.

Root cause this closes: ``source_sync`` had no `connection`/`graph`
parameters at all -- `kg_server._get_engine()` always returned the
process-default root engine, so an unbounded multi-source import could
never give each source its own isolated physical graph and watermark; every
source's ChangeEnvelope cursor lived in the same default graph regardless of
how differently trusted/scoped the sources were.

Distinct from `source='all'` (fans out across CONNECTORS/sources, unrelated
to this axis): `connection='all'`/a list here means fan out across BACKENDS,
which `source_sync` deliberately rejects — a sync always targets exactly
one backend.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession, current_session
from agent_utilities.mcp import kg_server
from agent_utilities.security.brain_context import ActorContext, ActorType


def _register_tools():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.ontology_tools import register_ontology_tools

    mcp = FastMCP("test")
    register_ontology_tools(mcp)


def _session(*, tenant: str = "test-tenant") -> GraphSession:
    actor = ActorContext(
        actor_id="test-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id=tenant,
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=tenant,
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="test-policy",
        audience="test-audience",
    )


class _FakeTenants:
    def __init__(self, names):
        self._names = list(names)

    def list(self):
        return [{"name": n} for n in self._names]


class _FakeComputeClient:
    def __init__(self, names):
        self.tenants = _FakeTenants(names)


class _FakeGraphCompute:
    def __init__(self, names):
        self.client = _FakeComputeClient(names)


class _FakeBackend:
    cypher_support = "full"
    supports_sparql = False

    def close(self):
        pass


class _FakeEngine:
    def __init__(self, catalog_graphs):
        self.graph_compute = _FakeGraphCompute(catalog_graphs)
        self.backend = _FakeBackend()


@pytest.fixture(autouse=True)
def _reset_state():
    saved_registry = kg_server._CONNECTION_REGISTRY
    saved_session = kg_server._PROCESS_SESSION
    kg_server._CONNECTION_REGISTRY = None
    kg_server._PROCESS_SESSION = _session()
    yield
    kg_server._CONNECTION_REGISTRY = saved_registry
    kg_server._PROCESS_SESSION = saved_session


def _install_engine(engine) -> None:
    from agent_utilities.knowledge_graph.core.connection_registry import (
        ConnectionRegistry,
    )

    kg_server._CONNECTION_REGISTRY = ConnectionRegistry(
        default_engine_provider=lambda: engine
    )


def _fake_sync_source(calls):
    def _sync(engine, source, *, mode="delta", ids=None):
        calls.append(
            {
                "engine": engine,
                "source": source,
                "mode": mode,
                "ids": ids,
                "session_graph": current_session().graph if current_session() else None,
            }
        )
        return {"status": "success", "source": source, "watermark": "wm-1"}

    return _sync


async def test_source_sync_unknown_graph_fails_closed_before_any_sync(monkeypatch):
    _register_tools()
    engine = _FakeEngine(["graph-a"])
    _install_engine(engine)
    calls: list[dict] = []
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.source_sync.sync_source",
        _fake_sync_source(calls),
    )

    out = await kg_server._execute_tool(
        "source_sync", source="leanix", graph="ghost-graph"
    )
    payload = json.loads(out)
    assert payload["error"]["code"] == "graph_not_found"
    assert "ghost-graph" not in out
    assert calls == []


async def test_source_sync_explicit_graph_plus_fanout_connection_is_rejected(
    monkeypatch,
):
    _register_tools()
    engine = _FakeEngine(["graph-a"])
    _install_engine(engine)
    calls: list[dict] = []
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.source_sync.sync_source",
        _fake_sync_source(calls),
    )

    out = await kg_server._execute_tool(
        "source_sync", source="leanix", connection="all", graph="graph-a"
    )
    payload = json.loads(out)
    assert payload["error"]["code"] == "graph_selection_conflict"
    assert calls == []


async def test_source_sync_narrows_the_session_for_the_duration_of_the_call(
    monkeypatch,
):
    """Positive proof: an explicit `graph` narrows the ambient verified
    session for the sync call (the SAME `bound_to_graph` primitive R-02
    uses), so the watermark/checkpoint the sync reads/writes lands on the
    selected physical graph, never the caller's own default. Restored after."""
    _register_tools()
    engine = _FakeEngine(["graph-a"])
    _install_engine(engine)
    calls: list[dict] = []
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.source_sync.sync_source",
        _fake_sync_source(calls),
    )
    own_default_graph = current_session().graph

    out = await kg_server._execute_tool("source_sync", source="leanix", graph="graph-a")
    payload = json.loads(out)
    assert payload["status"] == "success"
    assert payload["connection"] == "default"
    assert payload["graph"] == "graph-a"
    assert len(calls) == 1
    assert calls[0]["session_graph"] == "graph-a"

    # The caller's own session is restored after the call.
    assert current_session().graph == own_default_graph


async def test_source_sync_with_no_explicit_graph_is_unchanged_default_behavior(
    monkeypatch,
):
    _register_tools()
    engine = _FakeEngine(["graph-a"])
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    calls: list[dict] = []
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.source_sync.sync_source",
        _fake_sync_source(calls),
    )
    own_default_graph = current_session().graph

    out = await kg_server._execute_tool("source_sync", source="leanix")
    payload = json.loads(out)
    assert payload["status"] == "success"
    assert "connection" not in payload
    assert "graph" not in payload
    assert len(calls) == 1
    assert calls[0]["engine"] is engine
    assert calls[0]["session_graph"] == own_default_graph
