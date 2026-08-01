"""Every ``EpistemicGraphBackend.execute()`` MATCH — label-scoped or bare —
routes straight to the native engine's ``query_cypher``; there is no
client-side dispatch layer left in front of it.

CONCEPT:EG-KG.txn.per-graph-write-isolation — a prior revision of
``EpistemicGraphBackend`` had a client-side ``_exec_node_match`` that
special-cased ``MATCH (n:Label) … LIMIT k`` into a bounded
``get_nodes_by_label`` call (avoiding ``_get_all_nodes_with_properties``'s
whole-graph materialization) and only fell through to the native engine for a
real WHERE predicate. CONCEPT:AU-P0-2 removed that dispatch layer entirely:
``execute_read``/``execute_write`` now render params inline and hand the
WHOLE statement to ``self._graph.query_cypher()``/``query_cypher_write()``
unconditionally — label and LIMIT pushdown (and WHERE evaluation) all happen
server-side, inside the native engine, for every shape. ``nodes_by_label()``
still exists on ``EpistemicGraphBackend`` as an explicit, separate,
non-Cypher API for callers who want to bypass Cypher parsing outright — but
routing a MATCH *through* ``execute()`` no longer reaches it.
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
    """A label-scoped MATCH with no WHERE predicate still routes to the native
    engine's query_cypher — label + LIMIT pushdown is the native engine's own
    job now (CONCEPT:AU-P0-2), not a client-side get_nodes_by_label dispatch."""
    g = _FakeGraph()
    b = _backend(g)
    rows = b.execute("MATCH (n:Agent) RETURN n LIMIT 5")
    assert g.by_label_calls == []  # no client-side label dispatch anymore
    assert g.full_scan_calls == 0  # NO full-graph scan
    assert g.cypher_calls == ["MATCH (n:Agent) RETURN n LIMIT 5"]
    assert rows == [{"n": "a1"}]


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
    """A label-less MATCH also routes to the native engine's query_cypher —
    there is no client-side _get_all_nodes_with_properties full-scan fallback
    left in ``execute()`` to exercise; the engine decides how to satisfy an
    unlabeled scan server-side."""
    g = _FakeGraph()
    b = _backend(g)
    rows = b.execute("MATCH (n) RETURN n LIMIT 3")
    assert g.by_label_calls == []
    assert g.full_scan_calls == 0
    assert g.cypher_calls == ["MATCH (n) RETURN n LIMIT 3"]
    assert rows == [{"n": "a1"}]
