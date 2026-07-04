"""graph_query scope='sql' — native SQL-on-the-KG over MCP (CONCEPT:AU-KG.query.read-only-sql-over).

The `sql` scope routes the `cypher` arg (the SQL text) to ``engine.sql()``, which
bridges to the epistemic-graph engine's DataFusion SQL surface (the same path the
pg-wire listener uses). RLS is engine-side; read-path-first (SELECT/WITH/EXPLAIN
only). The REST twin ``/graph/query`` dispatches the same tool, so surface-parity
holds with no second handler.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.mcp import kg_server

pytestmark = pytest.mark.concept("AU-KG.query.read-only-sql-over")


def _register_graph_query():
    """Register the query tools onto a throwaway FastMCP and return graph_query."""
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools

    register_query_tools(FastMCP("test"))
    return kg_server.REGISTERED_TOOLS["graph_query"]


class _FakeEngine:
    """Stand-in engine exposing .sql() like IntelligenceGraphEngine."""

    def __init__(self, rows):
        self._rows = rows
        self.seen = None

    def sql(self, query):
        self.seen = query
        return self._rows


def test_sql_scope_routes_to_engine_sql(monkeypatch):
    graph_query = _register_graph_query()
    rows = [{"id": "n1", "label": "Agent"}]
    engine = _FakeEngine(rows)
    monkeypatch.setattr(
        kg_server,
        "_resolve_target_engines",
        lambda target: ([("kg", engine)], {}, False),
    )

    out = graph_query(
        cypher="SELECT id, label FROM nodes LIMIT 5", scope="sql", params="{}"
    )
    assert json.loads(out) == rows
    assert engine.seen == "SELECT id, label FROM nodes LIMIT 5"


def test_sql_scope_fans_out(monkeypatch):
    graph_query = _register_graph_query()
    e1, e2 = _FakeEngine([{"id": "a"}]), _FakeEngine([{"id": "b"}])
    monkeypatch.setattr(
        kg_server,
        "_resolve_target_engines",
        lambda target: ([("k1", e1), ("k2", e2)], {}, True),
    )

    out = json.loads(
        graph_query(
            cypher="SELECT id FROM nodes", scope="sql", target="all", params="{}"
        )
    )
    assert out["targets"] == {"k1": [{"id": "a"}], "k2": [{"id": "b"}]}
    assert out["errors"] == {}


def test_sql_scope_surfaces_engine_error(monkeypatch):
    graph_query = _register_graph_query()

    class _Boom:
        def sql(self, query):
            raise ValueError("only SELECT/WITH/EXPLAIN are allowed")

    monkeypatch.setattr(
        kg_server,
        "_resolve_target_engines",
        lambda target: ([("kg", _Boom())], {}, False),
    )
    out = json.loads(graph_query(cypher="DELETE FROM nodes", scope="sql", params="{}"))
    assert "error" in out and "SELECT" in out["error"]


# ── engine.sql() read-only guard + bridge (CONCEPT:AU-KG.query.read-only-sql-over) ──────────────────


class _SqlNamespace:
    def __init__(self, rows):
        self._rows = rows

    def sql(self, query):
        return self._rows


class _Client:
    def __init__(self, rows):
        self.query = _SqlNamespace(rows)


class _Backend:
    """Mimics EpistemicGraphBackend.graph._client.query.sql access path."""

    def __init__(self, rows):
        self.graph = type("G", (), {"_client": _Client(rows)})()


class _Engine(QueryMixin):
    def __init__(self, backend):
        self.backend = backend


def test_engine_sql_bridges_to_client():
    rows = [{"id": "n1"}]
    eng = _Engine(_Backend(rows))
    assert eng.sql("SELECT id FROM nodes LIMIT 1") == rows


def test_engine_sql_rejects_writes():
    eng = _Engine(_Backend([]))
    for stmt in (
        "DELETE FROM nodes",
        "UPDATE nodes SET x=1",
        "INSERT INTO nodes VALUES (1)",
    ):
        with pytest.raises(ValueError, match="read-only"):
            eng.sql(stmt)


def test_engine_sql_allows_with_and_explain():
    rows = [{"n": 1}]
    eng = _Engine(_Backend(rows))
    assert eng.sql("WITH t AS (SELECT 1) SELECT * FROM t") == rows
    assert eng.sql("EXPLAIN SELECT id FROM nodes") == rows


def test_engine_sql_no_surface_raises():
    class _NoSql:
        graph = type("G", (), {"_client": type("C", (), {"query": object()})()})()

    eng = _Engine(_NoSql())
    with pytest.raises(RuntimeError, match="no epistemic-graph SQL surface"):
        eng.sql("SELECT id FROM nodes")
