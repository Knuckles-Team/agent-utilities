"""PostgreSQL durable backend serves the *unanchored* relationship count.

``MATCH ()-[r]->() RETURN count(r)`` has no node anchor for the Cypher
transpiler to plan from, so it would transpile to UNKNOWN/[]. The tiered backend
routes unanchored relationship reads to the durable tier, so PostgreSQLBackend
answers the global edge metric directly from ``kg_edges`` — and defers (so the
transpiler still handles them) the moment the pattern constrains its endpoints.
(CONCEPT:KG-2.7 P1)
"""

from __future__ import annotations

import contextlib

import pytest

from agent_utilities.knowledge_graph.backends.postgresql_backend import (
    PostgreSQLBackend,
)


class _FakeCursor:
    def __init__(self, count: int):
        self._count = count
        self.executed: tuple | None = None

    def execute(self, sql, params=None):
        self.executed = (sql, params)

    def fetchone(self):
        return (self._count,)


class _FakeConn:
    def __init__(self, cur):
        self._cur = cur

    def cursor(self):
        cur = self._cur

        @contextlib.contextmanager
        def _cm():
            yield cur

        return _cm()


@pytest.fixture()
def backend(monkeypatch) -> PostgreSQLBackend:
    b = PostgreSQLBackend(dsn="postgresql://unused/test")
    cur = _FakeCursor(count=42)

    @contextlib.contextmanager
    def _fake_conn():
        yield _FakeConn(cur)

    monkeypatch.setattr(b, "_conn", _fake_conn)
    b._last_cur = cur  # type: ignore[attr-defined]
    return b


def test_global_edge_count_served_from_kg_edges(backend):
    handled, rows = backend._try_global_edge_count(
        "MATCH ()-[r]->() RETURN count(r) AS edges"
    )
    assert handled is True
    assert rows == [{"edges": 42}]
    assert "FROM kg_edges" in backend._last_cur.executed[0]


def test_named_label_free_endpoints_also_served(backend):
    handled, rows = backend._try_global_edge_count("MATCH (a)-[r]->(b) RETURN count(r)")
    assert handled is True
    assert rows == [{"count": 42}]


def test_rel_type_filter_adds_where_clause(backend):
    handled, rows = backend._try_global_edge_count(
        "MATCH ()-[r:KNOWS]->() RETURN count(r) AS c"
    )
    assert handled is True
    assert rows == [{"c": 42}]
    sql, params = backend._last_cur.executed
    assert "rel_type = %s" in sql and params == ("KNOWS",)


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a:Account)-[r]->(b) RETURN count(r)",  # node label
        "MATCH (a {id:'x'})-[r]->(b) RETURN count(r)",  # id anchor
        "MATCH (a)-[r]->(b) WHERE a.kind = 'x' RETURN count(r)",  # WHERE filter
        "MATCH (n) RETURN count(n)",  # not a relationship pattern
        "MATCH ()-[r]->() RETURN r",  # not a count
    ],
)
def test_defers_when_endpoints_constrained(backend, query):
    handled, rows = backend._try_global_edge_count(query)
    assert handled is False and rows == []
