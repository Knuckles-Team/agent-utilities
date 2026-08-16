"""U-06 — ``graph_ingest``'s action='ingest' codebase/document path must
honor independent ``connection``/``graph`` selectors exactly like
``graph_query``/``graph_write`` (see ``test_graph_explicit_selection.py``):
an explicit graph never defaults, never fans out, and an unknown graph
fails closed BEFORE any job is submitted or content is written.

Root cause this closes: ``graph_ingest`` had no `connection`/`graph`
parameters at all -- `kg_server._get_engine()` always returned the
process-default root engine regardless of any explicit graph the caller
might have selected elsewhere, so a codebase/document ingest job could only
ever land on the default/registry-wide graph.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.mcp import kg_server
from agent_utilities.security.brain_context import ActorContext, ActorType


def _register_tools():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.write_ingest_tools import register_write_ingest_tools

    mcp = FastMCP("test")
    register_write_ingest_tools(mcp)


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


class _MultiGraphIngestEngine:
    """Fake ``IntelligenceGraphEngine`` exposing just enough of the surface
    ``graph_ingest``'s action='ingest' codebase path touches: the catalog
    probe `resolve_explicit_graph` consults, plus `submit_task` (recording
    exactly what it was called with, never actually running a worker)."""

    def __init__(self, catalog_graphs):
        self.graph_compute = _FakeGraphCompute(catalog_graphs)
        self.backend = _FakeBackend()
        self.submit_calls: list[dict] = []

    def submit_task(self, **kwargs):
        self.submit_calls.append(kwargs)
        return "job-fake-1"


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


async def test_ingest_unknown_graph_fails_closed_before_any_job_is_submitted():
    _register_tools()
    engine = _MultiGraphIngestEngine(catalog_graphs=["graph-a"])
    _install_engine(engine)

    out = await kg_server._execute_tool(
        "graph_ingest",
        action="ingest",
        target_path="checkout/some-repo",
        content_type="codebase",
        graph="ghost-graph",
    )
    payload = json.loads(out)
    assert payload["error"]["code"] == "graph_not_found"
    assert "ghost-graph" not in out  # no raw name leak in the public payload
    # Fail closed happens BEFORE the call — no job was ever submitted.
    assert engine.submit_calls == []


async def test_ingest_explicit_graph_plus_fanout_connection_is_rejected():
    _register_tools()
    engine = _MultiGraphIngestEngine(catalog_graphs=["graph-a"])
    _install_engine(engine)

    out = await kg_server._execute_tool(
        "graph_ingest",
        action="ingest",
        target_path="checkout/some-repo",
        content_type="codebase",
        connection="all",
        graph="graph-a",
    )
    payload = json.loads(out)
    assert payload["error"]["code"] == "graph_selection_conflict"
    assert engine.submit_calls == []


async def test_ingest_resolved_graph_is_threaded_into_submit_task_not_dropped():
    """Positive proof the plumbing actually connects: a VALID explicit graph
    reaches `submit_task`'s own `graph` kwarg (which — see
    `test_submit_task_explicit_graph.py` — is what makes it survive onto the
    WorkItem's durable metadata for the async worker to re-narrow onto)."""
    _register_tools()
    engine = _MultiGraphIngestEngine(catalog_graphs=["graph-a"])
    _install_engine(engine)

    out = await kg_server._execute_tool(
        "graph_ingest",
        action="ingest",
        target_path="checkout/some-repo",
        content_type="codebase",
        graph="graph-a",
    )
    assert "graph-a" in out
    assert len(engine.submit_calls) == 1
    assert engine.submit_calls[0]["graph"] == "graph-a"


async def test_ingest_explicit_graph_on_document_content_fails_closed_not_silently_dropped():
    """BUG-120: the async worker only re-narrows onto an explicit `graph`
    for `task_type='codebase'` (`_bound_to_explicit_ingest_graph`'s call
    site lives only in the codebase branch of `_run_background_task`) --
    document ingestion never reads the WorkItem's `graph` metadata. Silently
    accepting `graph=` for a document path and echoing it as resolved would
    be a fabricated success (the caller's selection is validated, echoed,
    and then quietly never honored). This must be a typed denial instead,
    with no job submitted, until document ingest gets the same worker-side
    wiring codebase already has.
    """
    _register_tools()
    engine = _MultiGraphIngestEngine(catalog_graphs=["graph-a"])
    _install_engine(engine)

    out = await kg_server._execute_tool(
        "graph_ingest",
        action="ingest",
        target_path="checkout/some-notes.md",
        content_type="document",
        graph="graph-a",
    )
    payload = json.loads(out)
    assert payload["error"]["code"] == "graph_selection_conflict"
    assert engine.submit_calls == []


async def test_ingest_with_no_explicit_graph_is_unchanged_default_behavior(monkeypatch):
    """No `graph`/`connection` given at all -> the resolution block is a
    complete no-op and the pre-existing `kg_server._get_engine()` path is
    untouched (the process-default engine, exactly as before this fix)."""
    _register_tools()
    engine = _MultiGraphIngestEngine(catalog_graphs=["graph-a"])
    # Unlike the explicit-selector tests above, the no-selector path never
    # consults the connection registry (`_install_engine`) -- it is the SAME
    # `kg_server._get_engine()` singleton lookup `graph_ingest` always used;
    # stub it directly to observe that this fix left it alone.
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    out = await kg_server._execute_tool(
        "graph_ingest",
        action="ingest",
        target_path="checkout/some-repo",
        content_type="codebase",
    )
    assert "Started ingestion job" in out
    assert len(engine.submit_calls) == 1
    assert engine.submit_calls[0]["graph"] == ""
