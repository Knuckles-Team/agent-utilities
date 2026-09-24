"""Tests for CONCEPT:AU-KG.research.research-pipeline-runner — Probabilistic Knowledge Graph Reasoning."""

import pytest

# The compiled epistemic_graph.numeric kernel must be built for these tests; skip the whole module cleanly when it isn't, rather than erroring out collection (CONCEPT:AU-KG.compute.numeric-kernel).
pytest.importorskip("epistemic_graph.numeric")

from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
    BayesianBeliefPropagator,
    RandomWalkExplorer,
)
from agent_utilities.knowledge_graph.core.graph_primitives import PyDiGraph


def build_digraph(edges: list[tuple[str, str]]) -> PyDiGraph:
    g = PyDiGraph()
    n2i = {}

    def get_or_add(node_id: str) -> int:
        if node_id not in n2i:
            n2i[node_id] = g.add_node({"id": node_id})
        return n2i[node_id]

    for src, tgt in edges:
        s_idx = get_or_add(src)
        t_idx = get_or_add(tgt)
        g.add_edge(s_idx, t_idx, {})
    return g


class TestBayesianBeliefPropagation:
    """Tests for Bayesian belief updates (MCS §18.4)."""

    def test_basic_update(self):
        g = build_digraph([("cause", "effect")])
        prop = BayesianBeliefPropagator(g)
        prop.set_prior("cause", 0.5)

        result = prop.observe_evidence(
            "cause", likelihood_ratio=4.0, evidence_label="test"
        )
        assert result.posterior > result.prior
        assert result.posterior == pytest.approx(0.8, abs=0.01)

    def test_strong_evidence(self):
        g = build_digraph([("A", "B")])
        prop = BayesianBeliefPropagator(g)
        prop.set_prior("A", 0.1)
        result = prop.observe_evidence("A", likelihood_ratio=100.0)
        assert result.posterior > 0.9

    def test_disconfirming_evidence(self):
        g = build_digraph([("A", "B")])
        prop = BayesianBeliefPropagator(g)
        prop.set_prior("A", 0.9)
        result = prop.observe_evidence("A", likelihood_ratio=0.01)
        assert result.posterior < 0.5

    def test_belief_propagation(self):
        g = build_digraph([("A", "B"), ("B", "C")])
        prop = BayesianBeliefPropagator(g)
        prop.set_prior("A", 0.5)
        prop.observe_evidence("A", likelihood_ratio=5.0)
        updated = prop.propagate("A", max_hops=2)
        assert "B" in updated
        assert updated["B"].posterior > 0.5

    def test_edge_prior(self):
        g = build_digraph([("A", "B")])
        prop = BayesianBeliefPropagator(g)
        prop.set_prior("A", 0.0)
        result = prop.observe_evidence("A", likelihood_ratio=100.0)
        assert result.posterior == 0.0

    def test_ceiling_prior(self):
        g = build_digraph([("A", "B")])
        prop = BayesianBeliefPropagator(g)
        prop.set_prior("A", 1.0)
        result = prop.observe_evidence("A", likelihood_ratio=0.01)
        assert result.posterior == 1.0


class TestRandomWalkExplorer:
    """Tests for stochastic KG exploration (MCS Ch 21)."""

    def test_basic_exploration(self):
        g = build_digraph([("A", "B"), ("B", "C"), ("C", "A")])
        explorer = RandomWalkExplorer(g)
        freq = explorer.explore("A", n_steps=1000)
        assert len(freq) == 3
        assert sum(freq.values()) == pytest.approx(1.0)

    def test_start_node_dominance_with_restart(self):
        g = build_digraph([("A", "B"), ("B", "C"), ("C", "D")])
        explorer = RandomWalkExplorer(g)
        freq = explorer.explore("A", n_steps=1000, restart_prob=0.5)
        assert freq["A"] > freq["D"]

    def test_missing_node(self):
        g = build_digraph([("A", "B")])
        explorer = RandomWalkExplorer(g)
        assert explorer.explore("Z") == {}

    def test_unexpected_connections(self):
        g = build_digraph([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        explorer = RandomWalkExplorer(g)
        results = explorer.discover_unexpected_connections(
            "A", n_walks=5, walk_length=100
        )
        assert len(results) > 0
        assert all("surprise_score" in r for r in results)
