"""A label-scoped MATCH with no real predicate uses the engine's bounded labeled
fetch, not a full scan; a MATCH carrying a real WHERE predicate routes to the
native Cypher engine instead of a client-side scan-and-eval.

CONCEPT:EG-KG.txn.per-graph-write-isolation — `_exec_node_match` previously materialized the WHOLE graph
(`_get_all_nodes_with_properties`) for any `MATCH (n:Label) … LIMIT k`; it now
pushes the label (and the LIMIT, for a pure read) down to the engine. A WHERE
predicate beyond the trivial id equality no longer gets a client-side
scan-and-`_eval_groups` pass — CONCEPT:AU-P0-2 moved that to
`GraphComputeEngine.query_cypher` (the native engine's own WHERE evaluator), so
`_exec_node_match` now defers (`handled=False`) for that shape.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)


class _FakeGraph:
    def __init__(self) -> None:
        self.by_label_calls: list[tuple[str, int]] = []
        self.full_scan_calls = 0
        self.cypher_calls: list[str] = []

    def get_nodes_by_label(
        self, label: str, limit: int
    ) -> list[tuple[str, dict[str, Any]]]:
        self.by_label_calls.append((label, limit))
        return [("a1", {"type": "Agent", "name": "A"})]

    def _get_all_nodes_with_properties(self) -> list[tuple[str, dict[str, Any]]]:
        self.full_scan_calls += 1
        return [("z1", {"type": "Other"})]

    def has_node(self, _nid: str) -> bool:
        return False

    def _get_node_properties(self, _nid: str) -> dict[str, Any]:
        return {}

    def query_cypher(self, query: str) -> list[dict[str, Any]]:
        self.cypher_calls.append(query)
        return [{"n": "a1"}]


def _backend(g: _FakeGraph) -> EpistemicGraphBackend:
    b = EpistemicGraphBackend.__new__(EpistemicGraphBackend)
    b._graph = g
    b._embeddings = {}
    return b


def test_label_query_pushes_label_and_limit_down() -> None:
    g = _FakeGraph()
    b = _backend(g)
    rows = b.execute("MATCH (n:Agent) RETURN n LIMIT 5")
    assert g.by_label_calls == [("Agent", 5)]  # pure read -> LIMIT pushed down
    assert g.full_scan_calls == 0  # NO full-graph scan
    assert g.cypher_calls == []  # no predicate -> stays off the native engine
    assert rows  # returned the agent


def test_label_query_with_where_routes_to_native_engine() -> None:
    g = _FakeGraph()
    b = _backend(g)
    rows = b.execute("MATCH (n:Agent) WHERE n.name = 'A' RETURN n LIMIT 5")
    # A real WHERE predicate defers the label/full scan entirely and routes the
    # whole (literal-inlined) query to the native Cypher engine instead.
    assert g.by_label_calls == []
    assert g.full_scan_calls == 0
    assert g.cypher_calls == ["MATCH (n:Agent) WHERE n.name = 'A' RETURN n LIMIT 5"]
    assert rows == [{"n": "a1"}]


def test_bare_match_without_label_still_full_scans() -> None:
    g = _FakeGraph()
    b = _backend(g)
    b.execute("MATCH (n) RETURN n LIMIT 3")
    assert g.by_label_calls == []
    assert g.full_scan_calls == 1
    assert g.cypher_calls == []
