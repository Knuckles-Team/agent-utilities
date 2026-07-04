"""Tests for symbol→symbol path navigation (CONCEPT:EG-KG.compute.handled-outside-single-anchor, action='connects').

Exercises ``code_connects`` offline via a fake engine that resolves symbol names
to :Code ids, runs a canned shortest path, and annotates each hop's edge.
"""

from __future__ import annotations

from agent_utilities.mcp.tools.query_tools import code_connects

_AUTH = "code:auth.py::AuthService"
_POOL = "code:database.py::DatabasePool"
_GETCONN = "code:database.py::get_connection"


class FakeEngine:
    def __init__(self, path: dict[tuple[str, str], list[str]]):
        self._path = path
        self._ids = {
            "AuthService": _AUTH,
            "DatabasePool": _POOL,
            "get_connection": _GETCONN,
        }

    def query_cypher(self, cypher, params):
        if "c.name = $symbol" in cypher:
            nid = self._ids.get(params.get("symbol"))
            return [{"id": nid, "name": params.get("symbol")}] if nid else []
        if "c.id = $node_id" in cypher:
            nid = params.get("node_id")
            return [{"id": nid, "name": nid.rsplit("::", 1)[-1]}]
        if "[r]-(y" in cypher:  # hop edge annotation
            return [{"rel": "calls", "confidence": 1.0}]
        return []

    def get_shortest_path(self, a, b):
        return self._path.get((a, b))


def test_connects_returns_annotated_path():
    eng = FakeEngine({(_AUTH, _POOL): [_AUTH, _GETCONN, _POOL]})
    out = code_connects(eng, symbol="AuthService", target_symbol="DatabasePool")
    assert out["connected"] is True
    assert out["path"] == [_AUTH, _GETCONN, _POOL]
    assert out["length"] == 2
    assert len(out["hops"]) == 2
    assert out["hops"][0]["rel"] == "calls"
    assert out["hops"][0]["from"] == _AUTH


def test_connects_tries_reverse_direction():
    # Only the B→A path exists; connects must find it (undirected intent).
    eng = FakeEngine({(_POOL, _AUTH): [_POOL, _GETCONN, _AUTH]})
    out = code_connects(eng, symbol="AuthService", target_symbol="DatabasePool")
    assert out["connected"] is True
    assert out["path"][0] == _POOL


def test_connects_no_path():
    eng = FakeEngine({})
    out = code_connects(eng, symbol="AuthService", target_symbol="DatabasePool")
    assert out["connected"] is False
    assert out["path"] == []


def test_connects_unresolved_source():
    eng = FakeEngine({})
    out = code_connects(eng, symbol="Nonexistent", target_symbol="DatabasePool")
    assert "error" in out
    assert "source" in out["error"]


def test_connects_same_symbol():
    eng = FakeEngine({})
    out = code_connects(eng, symbol="AuthService", target_symbol="AuthService")
    assert "error" in out
    assert "same symbol" in out["error"]
