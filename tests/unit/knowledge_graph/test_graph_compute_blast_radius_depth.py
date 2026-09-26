"""The native ID list needs real graph distance before AU exposes depth."""

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


class _Graph:
    def __init__(self, edges):
        self.edges = edges
        self.calls = []

    def blast_radius(self, node_id, max_depth):
        self.calls.append(("blast_radius", node_id, max_depth))
        return ["B", "C", "D"]

    def get_subgraph(self, node_ids):
        self.calls.append(("get_subgraph", node_ids))
        return {"nodes": [], "edges": self.edges}


def _engine(graph):
    engine = GraphComputeEngine.__new__(GraphComputeEngine)
    engine._client = SimpleNamespace(graph=graph)
    return engine


def test_siblings_have_the_same_shortest_path_depth():
    graph = _Graph(
        [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "B", "target": "D"},
        ]
    )
    rows = _engine(graph).get_blast_radius("A", 2)
    assert rows == [
        {"id": "B", "type": "Node", "depth": 1},
        {"id": "C", "type": "Node", "depth": 2},
        {"id": "D", "type": "Node", "depth": 2},
    ]
    assert graph.calls == [
        ("blast_radius", "A", 2),
        ("get_subgraph", ["A", "B", "C", "D"]),
    ]


def test_changed_subgraph_rejects_invented_depths():
    graph = _Graph([{"source": "A", "target": "B"}])
    with pytest.raises(ValueError, match="changed during depth projection"):
        _engine(graph).get_blast_radius("A", 2)
