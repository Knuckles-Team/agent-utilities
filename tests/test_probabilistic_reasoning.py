"""Tests for CONCEPT:KG-2.6 — Probabilistic Knowledge Graph Reasoning."""

import math

import pytest

from agent_utilities.knowledge_graph.core.graph_primitives import PyDiGraph
from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
    BayesianBeliefPropagator,
    RandomWalkExplorer,
    birthday_collision_probability,
    conditional_independence_test,
    total_probability_aggregation,
)

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


class TestTotalProbabilityAggregation:
    """Tests for Law of Total Probability (MCS §18.5)."""

    def test_uniform_weights(self):
        scores = [(0.8, 1.0), (0.6, 1.0), (0.4, 1.0)]
        result = total_probability_aggregation(scores)
        assert result == pytest.approx(0.6, abs=0.01)

    def test_weighted_combination(self):
        scores = [(0.9, 3.0), (0.1, 1.0)]
        result = total_probability_aggregation(scores)
        expected = (0.9 * 3 + 0.1 * 1) / 4
        assert result == pytest.approx(expected, abs=0.01)

    def test_empty_sources(self):
        assert total_probability_aggregation([]) == 0.0

    def test_bounds(self):
        result = total_probability_aggregation([(1.5, 1.0)])
        assert result <= 1.0


class TestBirthdayCollision:
    """Tests for Birthday Paradox collision detection (MCS §17.4)."""

    def test_classic_birthday(self):
        result = birthday_collision_probability(23, 365)
        assert result.collision_probability > 0.5

    def test_small_space(self):
        result = birthday_collision_probability(10, 10)
        assert result.collision_probability > 0.9

    def test_large_space(self):
        result = birthday_collision_probability(10, 2**64)
        assert result.collision_probability < 0.001

    def test_safe_threshold(self):
        result = birthday_collision_probability(1, 1000)
        assert result.safe_threshold == int(1.2 * math.sqrt(1000))

    def test_zero_items(self):
        result = birthday_collision_probability(0, 100)
        assert result.collision_probability == 0.0


class TestConditionalIndependence:
    """Tests for d-separation (MCS §18.7)."""

    def test_fork_blocked(self):
        g = build_digraph([("B", "A"), ("B", "C")])
        result = conditional_independence_test(g, "A", "C", {"B"})
        assert result["independent"]

    def test_fork_unblocked(self):
        g = build_digraph([("B", "A"), ("B", "C")])
        result = conditional_independence_test(g, "A", "C")
        assert not result["independent"]

    def test_missing_node(self):
        g = build_digraph([("A", "B")])
        result = conditional_independence_test(g, "A", "Z")
        assert result["independent"]
