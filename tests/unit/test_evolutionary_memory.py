"""Tests for Evolutionary Memory parity features: Topological Partitioning, Drift Tracker, EWC++.

CONCEPT:AU-KG.compute.spectral-cluster-navigator (Topological Mincut)
CONCEPT:AU-AHE.evaluation.backtest-harness (Temporal Drift)
CONCEPT:AU-AHE.evaluation.backtest-harness (EWC++)
"""

from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.topological_partition import (
    detect_communities,
    persist_stable_communities,
)
from agent_utilities.knowledge_graph.memory import (
    apply_ewc_synthesis,
    check_knowledge_drift,
    compute_fisher_diagonal_proxy,
)


class MockEngine:
    def __init__(self, nx_graph: Any):
        self._nx_graph = nx_graph
        self.graph = nx_graph
        self.upserted_nodes: list[Any] = []
        self.upserted_edges: list[Any] = []

    def upsert_node(self, node: Any) -> None:
        self.upserted_nodes.append(node)

    def upsert_edge(self, edge: Any) -> None:
        self.upserted_edges.append(edge)


@pytest.mark.concept("AU-KG.compute.spectral-cluster-navigator", "CONCEPT:AU-KG.compute.spectral-cluster-navigator")
def test_detect_communities():
    """Test Louvain community detection.

    CONCEPT:AU-KG.compute.spectral-cluster-navigator
    """
    # Create a graph with two distinct cliques connected by a single edge
    from agent_utilities.knowledge_graph.core.graph_primitives import PyGraph

    G = PyGraph()
    n = {}
    for i in range(1, 7):
        n[i] = G.add_node(i)
    for u, v in [(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6), (3, 4)]:
        G.add_edge(n[u], n[v], {"weight": 1.0})

    communities = detect_communities(G)
    assert len(communities) == 2

    # Validate the sets (can be in any order)
    c1, c2 = communities
    if 1 in c1:
        assert c1 == {1, 2, 3}
        assert c2 == {4, 5, 6}
    else:
        assert c1 == {4, 5, 6}
        assert c2 == {1, 2, 3}


@pytest.mark.concept("AU-KG.compute.spectral-cluster-navigator", "CONCEPT:AU-KG.compute.spectral-cluster-navigator")
def test_persist_stable_communities():
    """Test community persistence to Cypher backend.

    CONCEPT:AU-KG.compute.spectral-cluster-navigator
    """
    from agent_utilities.knowledge_graph.core.graph_primitives import PyGraph

    G = PyGraph()
    n = {}
    for i in range(1, 7):
        n[i] = G.add_node(i)
    for u, v in [(1, 2), (2, 3), (1, 3), (4, 5), (5, 6), (4, 6), (3, 4)]:
        G.add_edge(n[u], n[v], {"weight": 1.0})

    engine = MockEngine(G)
    count = persist_stable_communities(engine)

    assert count == 2
    assert len(engine.upserted_nodes) == 2
    assert len(engine.upserted_edges) == 6  # 3 members per community = 6 edges


@pytest.mark.concept("AU-AHE.harness.evolution-checkpoint", "CONCEPT:AU-AHE.evaluation.backtest-harness")
def test_drift_tracker():
    """Test temporal knowledge drift measurement.

    CONCEPT:AU-AHE.evaluation.backtest-harness
    """
    history = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]]
    current = [0.0, 1.0, 0.0]  # Orthogonal, huge shift

    report = check_knowledge_drift("node_1", history, current, drift_threshold=0.15)

    assert report.node_id == "node_1"
    assert report.has_drifted is True
    assert report.cosine_shift == 1.0  # Cosine distance between [1,0,0] and [0,1,0]
    assert report.coefficient_of_variation > 0.0


@pytest.mark.concept("AU-AHE.harness.evolution-checkpoint", "CONCEPT:AU-AHE.evaluation.backtest-harness")
def test_ewc_synthesis():
    """Test Fisher-proxy Elastic Weight Consolidation.

    CONCEPT:AU-AHE.evaluation.backtest-harness
    """
    # History with high variance in index 1, low variance in index 0
    history = [[1.0, -1.0], [1.0, 1.0], [1.0, -0.5], [1.0, 0.5]]

    fisher = compute_fisher_diagonal_proxy(history)
    assert fisher[0] > fisher[1]
    assert fisher[0] == 1.0

    old_emb = [1.0, 0.0]
    new_emb = [0.0, 1.0]

    synthesized = apply_ewc_synthesis(old_emb, new_emb, fisher, lambda_param=0.5)

    assert len(synthesized) == 2
    assert synthesized[0] > 0.0  # Index 0 preserved due to high fisher (was 1.0)
