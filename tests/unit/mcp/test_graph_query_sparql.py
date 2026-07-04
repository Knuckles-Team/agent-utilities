"""graph_query scope='sparql' — native SPARQL-on-the-KG over MCP (CONCEPT:AU-KG.ingest.mirror-inbound).

The `sparql` scope routes the `cypher` arg (the SPARQL text) to ``engine.sparql()``,
which bridges to the epistemic-graph engine's native RDF surface
(``backend.graph.sparql`` → ``client.rdf.sparql``). The REST twin ``/graph/query``
dispatches the same tool, so surface-parity holds with no second handler.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.mcp import kg_server

pytestmark = pytest.mark.concept("AU-KG.ingest.mirror-inbound")


def _register_graph_query():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools

    register_query_tools(FastMCP("test"))
    return kg_server.REGISTERED_TOOLS["graph_query"]


class _FakeEngine:
    def __init__(self, rows):
        self._rows = rows
        self.seen = None

    def sparql(self, query):
        self.seen = query
        return self._rows


def test_sparql_scope_routes_to_engine_sparql(monkeypatch):
    graph_query = _register_graph_query()
    rows = [{"s": "n1", "p": "rdf:type", "o": "Agent"}]
    engine = _FakeEngine(rows)
    monkeypatch.setattr(
        kg_server,
        "_resolve_target_engines",
        lambda target: ([("kg", engine)], {}, False),
    )
    out = graph_query(
        cypher="SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 5",
        scope="sparql",
        params="{}",
    )
    assert json.loads(out) == rows
    assert engine.seen.startswith("SELECT ?s ?p ?o")


def test_sparql_scope_ask_roundtrip(monkeypatch):
    graph_query = _register_graph_query()
    engine = _FakeEngine([{"boolean": True}])
    monkeypatch.setattr(
        kg_server,
        "_resolve_target_engines",
        lambda target: ([("kg", engine)], {}, False),
    )
    out = json.loads(graph_query(cypher="ASK { ?s a ?o }", scope="sparql", params="{}"))
    assert out == [{"boolean": True}]


def test_sparql_scope_fans_out(monkeypatch):
    graph_query = _register_graph_query()
    e1, e2 = _FakeEngine([{"x": "a"}]), _FakeEngine([{"x": "b"}])
    monkeypatch.setattr(
        kg_server,
        "_resolve_target_engines",
        lambda target: ([("k1", e1), ("k2", e2)], {}, True),
    )
    out = json.loads(
        graph_query(
            cypher="SELECT ?x WHERE { ?x a ?t }",
            scope="sparql",
            target="all",
            params="{}",
        )
    )
    assert out["targets"] == {"k1": [{"x": "a"}], "k2": [{"x": "b"}]}


# ── engine.sparql() bridge (CONCEPT:AU-KG.ingest.mirror-inbound) ─────────────────────────────────


class _GraphCompute:
    def __init__(self, rows):
        self._rows = rows
        self.seen = None

    def sparql(self, query, base_iri="", type_convention=""):
        self.seen = (query, base_iri, type_convention)
        return self._rows


class _Backend:
    """Mimics backend.graph.sparql access path."""

    def __init__(self, rows):
        self.graph = _GraphCompute(rows)


class _Engine(QueryMixin):
    def __init__(self, backend):
        self.backend = backend


def test_engine_sparql_bridges_to_graph_compute():
    rows = [{"name": "alice"}]
    eng = _Engine(_Backend(rows))
    assert eng.sparql("SELECT ?name WHERE { ?s :name ?name }") == rows
    assert eng.backend.graph.seen[0].startswith("SELECT ?name")


def test_engine_sparql_no_surface_raises():
    class _NoSparql:
        graph = object()  # no .sparql

    eng = _Engine(_NoSparql())
    with pytest.raises(RuntimeError, match="no epistemic-graph SPARQL surface"):
        eng.sparql("SELECT ?s WHERE { ?s ?p ?o }")
