"""L1 (epistemic_graph) must never silently return a *global* edge count for a
relationship aggregate whose WHERE filter it cannot honor (CONCEPT:KG-2.9h).

Live symptom: ``MATCH (d)-[r]->(c) WHERE d.doc_type='news_article' RETURN
count(c)`` returned the GLOBAL chunk/edge count (e.g. 822419) because the
unanchored aggregate path dropped the WHERE. Silent-wrong is worse than an
error, so the L1 backend now FAILS LOUD for this shape, and the tiered backend
routes such queries to L1 (fail loud) rather than L3 (which would return an
empty/degenerate result).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)


class _GlobalCountOnlyGraph:
    """A graph that ONLY exposes a global ``edge_count`` (no scan support).

    This is the worst case: the where-anchored traversal path can't resolve the
    anchor (no node scan), so the backend MUST fail loud rather than fall back to
    the global edge count. (CONCEPT:KG-2.9h)
    """

    def __init__(self, edges: int):
        self._edges = edges

    def edge_count(self) -> int:  # what the buggy path used to return
        return self._edges

    def has_node(self, _n) -> bool:
        return False

    def has_edge(self, _a, _b) -> bool:
        return False

    def _get_edge_properties(self, _a, _b):
        return {}

    def get_successors(self, _n):
        return []

    def get_predecessors(self, _n):
        return []

    def get_triples(self):
        return []

    # No get_nodes_by_label / _get_all_nodes_with_properties → the where-anchored
    # path can't resolve the anchor and defers, so the backend fails loud.


class _ScanGraph(_GlobalCountOnlyGraph):
    """A graph that supports a node scan, so the where-anchored path can honor
    the WHERE filter and return the CORRECT filtered count."""

    def __init__(
        self, edges: int, nodes: dict[str, dict], edges_map: dict[str, list[str]]
    ):
        super().__init__(edges)
        self._nodes = nodes
        self._edges_map = edges_map

    def get_nodes_by_label(self, _label, _limit):
        return list(self._nodes.items())

    def _get_node_properties(self, nid):
        return self._nodes.get(nid, {})

    def get_successors(self, nid):
        return list(self._edges_map.get(nid, []))


def _backend(graph) -> EpistemicGraphBackend:
    # Build without __init__ (which would connect to the live engine) and inject
    # a fake graph — we only exercise the pure Cypher-subset interpreter.
    b = EpistemicGraphBackend.__new__(EpistemicGraphBackend)
    b._graph = graph
    b._embeddings = {}
    b._node_counter = 0
    return b


class TestWhereHasUnhonoredFilter:
    def test_node_property_filter_is_unhonored(self):
        assert EpistemicGraphBackend._where_has_unhonored_filter(
            "MATCH (d)-[r]->(c) WHERE d.doc_type='news_article' RETURN count(c)"
        )

    def test_no_where_is_honorable(self):
        assert not EpistemicGraphBackend._where_has_unhonored_filter(
            "MATCH ()-[r]->() RETURN count(r)"
        )

    def test_pure_id_anchor_is_honorable(self):
        assert not EpistemicGraphBackend._where_has_unhonored_filter(
            "MATCH (s)-[r]->(t) WHERE s.id=$x RETURN count(r)"
        )

    def test_id_anchor_plus_extra_prop_is_unhonored(self):
        assert EpistemicGraphBackend._where_has_unhonored_filter(
            "MATCH (s)-[r]->(t) WHERE s.id=$x AND t.kind='X' RETURN count(r)"
        )


class TestExecuteNeverReturnsGlobalCountForFilteredQuery:
    def test_filtered_aggregate_fails_loud_when_no_scan_support(self):
        # The graph can only answer the GLOBAL edge count. The backend must NOT
        # return it for a filtered query — it raises instead. (the cardinal sin
        # eliminated: silent-wrong global count.)
        b = _backend(_GlobalCountOnlyGraph(edges=822419))
        with pytest.raises(ValueError, match="cannot honor"):
            b.execute(
                "MATCH (d)-[r]->(c) WHERE d.doc_type='news_article' RETURN count(c)",
                {},
            )

    def test_filtered_aggregate_returns_correct_count_when_scannable(self):
        # Two news_article docs, each linking to 1 chunk → filtered count == 2,
        # NOT the global edge count (4). The where-anchored path honors the WHERE.
        nodes = {
            "d1": {"doc_type": "news_article"},
            "d2": {"doc_type": "news_article"},
            "d3": {"doc_type": "blog"},
            "c1": {},
            "c2": {},
            "c3": {},
        }
        edges_map = {"d1": ["c1"], "d2": ["c2"], "d3": ["c3"]}
        b = _backend(_ScanGraph(edges=4, nodes=nodes, edges_map=edges_map))
        rows = b.execute(
            "MATCH (d)-[r]->(c) WHERE d.doc_type='news_article' RETURN count(c)",
            {},
        )
        # exactly the filtered count, never the global edge count (4)
        assert rows and list(rows[0].values())[0] == 2

    def test_unfiltered_global_count_still_works(self):
        b = _backend(_GlobalCountOnlyGraph(edges=42))
        rows = b.execute("MATCH ()-[r]->() RETURN count(r) AS edges", {})
        assert rows == [{"edges": 42}]
