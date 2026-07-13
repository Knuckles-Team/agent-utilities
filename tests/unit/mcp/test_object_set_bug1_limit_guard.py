"""BUG-1 (kg-exhaustive-smoke.md, CRITICAL): ``object_set(action="of_type"/"from_ids")``
used to call ``ObjectSet.ids()`` with NO bound at all — for the DYNAMIC set every
``of_type``/``from_ids``/``union``/``intersect``/``subtract`` call materializes,
that scans and returns EVERY matching node in the graph. Over the live
deployment (13,793 ``Concept`` nodes out of 139,655 total) this OOM-killed the
graph-os pod (see ``agent_utilities/mcp/tools/ontology_tools.py``:
``object_set()``).

These exercise the REAL ``object_set`` MCP tool dispatch (via
``kg_server._execute_tool``, the same path the live server uses) against a
large FAKE object set (thousands of matching nodes) so an unbounded
materialization would be obviously wrong if the guard regressed — no live
engine, no real (potentially crash-prone) unbounded call is ever made.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.ontology.object_set import (
    object_set_from_ids,
    object_set_of_type,
)
from agent_utilities.mcp import kg_server


class _FakeGraph:
    """Minimal duck-typed graph: N nodes all of the same type."""

    def __init__(self, n: int, node_type: str = "Concept") -> None:
        self._nodes = {f"concept:{i}": {"id": f"concept:{i}", "type": node_type} for i in range(n)}

    def node_ids(self) -> list[str]:
        return list(self._nodes.keys())

    def _get_node_properties(self, node_id: str) -> dict:
        return dict(self._nodes.get(node_id, {}))

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes


class _OntologyStub:
    """Stand-in for ``kg_server._ontology_system()``: real ObjectSet factories
    bound to a large fake graph, so ``.ids()``/``.count()`` behave exactly like
    the production ``OntologySystem.object_set``/``object_set_of_type``."""

    def __init__(self, graph: _FakeGraph) -> None:
        self._graph = graph

    def object_set(self, ids):
        return object_set_from_ids(self._graph, ids)

    def object_set_of_type(self, type_or_interface: str):
        return object_set_of_type(self._graph, type_or_interface)


@pytest.fixture
def _ensure_registered():
    kg_server.ensure_tools_registered()


def _patch_ontology(monkeypatch, n: int = 20_000) -> _FakeGraph:
    graph = _FakeGraph(n)
    monkeypatch.setattr(kg_server, "_ontology_system", lambda: _OntologyStub(graph))
    return graph


async def test_of_type_never_returns_more_than_the_requested_limit(
    monkeypatch, _ensure_registered
):
    _patch_ontology(monkeypatch, n=20_000)
    out = json.loads(
        await kg_server._execute_tool(
            "object_set", action="of_type", type_or_interface="Concept", limit=3
        )
    )
    assert out["count"] == 3
    assert len(out["ids"]) == 3
    assert out["limited"] is True


async def test_of_type_default_limit_is_bounded_not_unbounded(
    monkeypatch, _ensure_registered
):
    # No explicit `limit` at all -> the tool's own Field default (50) applies,
    # NOT an unbounded scan of all 20,000 matching nodes.
    _patch_ontology(monkeypatch, n=20_000)
    out = json.loads(
        await kg_server._execute_tool(
            "object_set", action="of_type", type_or_interface="Concept"
        )
    )
    assert out["count"] == 50
    assert len(out["ids"]) == 50


async def test_of_type_rejects_an_absurd_caller_supplied_limit(
    monkeypatch, _ensure_registered
):
    # A caller passing an enormous explicit limit must still be clamped to the
    # hard cap, not honored verbatim (defense against "just ask for everything").
    _patch_ontology(monkeypatch, n=20_000)
    out = json.loads(
        await kg_server._execute_tool(
            "object_set",
            action="of_type",
            type_or_interface="Concept",
            limit=10_000_000,
        )
    )
    assert out["count"] <= 10_000
    assert len(out["ids"]) <= 10_000


async def test_from_ids_respects_limit(monkeypatch, _ensure_registered):
    _patch_ontology(monkeypatch, n=100)
    ids = [f"concept:{i}" for i in range(100)]
    out = json.loads(
        await kg_server._execute_tool(
            "object_set",
            action="from_ids",
            ids_json=json.dumps(ids),
            limit=7,
        )
    )
    assert out["count"] == 7
    assert len(out["ids"]) == 7


async def test_union_respects_limit(monkeypatch, _ensure_registered):
    # `other` for union/intersect/subtract can itself be an unbounded of_type()
    # DYNAMIC set (built internally when no explicit ids are given) — bound it too.
    _patch_ontology(monkeypatch, n=20_000)
    out = json.loads(
        await kg_server._execute_tool(
            "object_set",
            action="union",
            type_or_interface="Concept",
            ids_json=json.dumps(["concept:0", "concept:1"]),
            limit=5,
        )
    )
    assert out["count"] == 5
    assert len(out["ids"]) == 5
