"""Tests for CONCEPT:AU-KG.research.research-pipeline-runner — Formal Graph Theory Primitives."""

from typing import Any

import pytest

# The compiled epistemic_graph.numeric kernel must be built for these tests; skip the whole module cleanly when it isn't, rather than erroring out collection (CONCEPT:AU-KG.compute.numeric-kernel).
pytest.importorskip("epistemic_graph.numeric")

from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
    chromatic_schedule,
)
from agent_utilities.knowledge_graph.core.graph_primitives import PyGraph


def build_graph(nodes, edges):
    g = PyGraph()
    n2i = {}
    for n in nodes:
        n2i[n] = g.add_node(n)
    for src, tgt, data in edges:
        g.add_edge(n2i[src], n2i[tgt], data)
    return g


class TestChromaticScheduling:
    """Tests for Chromatic Scheduling (MCS §12.6)."""

    def test_bipartite_graph(self):
        nodes = ["u1", "u2", "u3", "v1", "v2", "v3"]
        edges: list[tuple[Any, Any, Any]] = [
            (u, v, {}) for u in ["u1", "u2", "u3"] for v in ["v1", "v2", "v3"]
        ]
        g = build_graph(nodes, edges)
        coloring = chromatic_schedule(g)
        assert max(coloring.values()) + 1 == 2  # Bipartite → 2 colors

    def test_no_conflicts(self):
        g = build_graph([1, 2, 3], [])
        coloring = chromatic_schedule(g)
        assert max(coloring.values()) + 1 == 1  # All independent

    def test_adjacent_nodes_different_colors(self):
        nodes = [1, 2, 3, 4, 5]
        edges: list[tuple[Any, Any, Any]] = [
            (1, 2, {}),
            (2, 3, {}),
            (3, 4, {}),
            (4, 5, {}),
            (5, 1, {}),
        ]
        g = build_graph(nodes, edges)
        coloring = chromatic_schedule(g)
        # chromatic_schedule returns dict[str, int] — keys are stringified node IDs,
        # so index with str(node-data), not the raw int.
        for _, (u, v, _) in g._edges.items():
            assert coloring[str(g[u])] != coloring[str(g[v])]
