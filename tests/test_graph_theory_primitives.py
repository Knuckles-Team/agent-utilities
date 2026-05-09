"""Tests for CONCEPT:KG-2.41 — Formal Graph Theory Primitives."""

import networkx as nx
import numpy as np
import pytest

from agent_utilities.knowledge_graph.core.graph_theory_primitives import (
    chromatic_number_upper_bound,
    chromatic_schedule,
    count_paths_of_length,
    dag_critical_path,
    edge_connectivity,
    euler_tour,
    generate_math_foundation_seed,
    minimum_vertex_cut,
    personalized_pagerank,
    reachability_within_hops,
    vertex_connectivity,
)


class TestDAGCriticalPath:
    """Tests for DAG Critical Path Analysis (MCS §10.5)."""

    def test_linear_dag(self):
        g = nx.DiGraph()
        g.add_weighted_edges_from([("A", "B", 3), ("B", "C", 5), ("C", "D", 2)])
        result = dag_critical_path(g)
        assert result["makespan"] == 10.0
        assert result["critical_path"] == ["A", "B", "C", "D"]

    def test_parallel_dag(self):
        g = nx.DiGraph()
        g.add_weighted_edges_from(
            [("S", "A", 2), ("S", "B", 5), ("A", "T", 3), ("B", "T", 1)]
        )
        result = dag_critical_path(g)
        assert result["makespan"] == 6.0  # max(2+3, 5+1) = max(5,6) = 6
        assert result["node_slack"]["S"] == 0.0

    def test_empty_dag(self):
        g = nx.DiGraph()
        result = dag_critical_path(g)
        assert result["makespan"] == 0.0
        assert result["critical_path"] == []

    def test_single_node(self):
        g = nx.DiGraph()
        g.add_node("A")
        result = dag_critical_path(g)
        assert result["makespan"] == 0.0
        assert "A" in result["node_earliest_start"]

    def test_cycle_raises(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        with pytest.raises(nx.NetworkXUnfeasible):
            dag_critical_path(g)

    def test_slack_identifies_non_critical(self):
        g = nx.DiGraph()
        g.add_weighted_edges_from(
            [("S", "A", 1), ("S", "B", 5), ("A", "T", 1), ("B", "T", 1)]
        )
        result = dag_critical_path(g)
        assert result["node_slack"]["A"] > 0  # A is not on critical path
        assert result["node_slack"]["B"] == 0.0  # B is on critical path


class TestConnectivity:
    """Tests for Graph Connectivity Certificates (MCS §12.8–12.10)."""

    def test_complete_graph_connectivity(self):
        g = nx.complete_graph(5)
        assert vertex_connectivity(g) == 4
        assert edge_connectivity(g) == 4

    def test_path_graph_connectivity(self):
        g = nx.path_graph(5)
        assert vertex_connectivity(g) == 1
        assert edge_connectivity(g) == 1

    def test_disconnected_graph(self):
        g = nx.Graph()
        g.add_nodes_from([1, 2, 3])
        assert vertex_connectivity(g) == 0
        assert edge_connectivity(g) == 0

    def test_minimum_vertex_cut_bridge(self):
        g = nx.Graph()
        g.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 3)])
        g.add_edge(2, 6)
        cut = minimum_vertex_cut(g)
        assert len(cut) >= 1  # At least one cut vertex

    def test_empty_graph(self):
        g = nx.Graph()
        assert vertex_connectivity(g) == 0
        assert minimum_vertex_cut(g) == set()


class TestEulerTour:
    """Tests for Euler Tour Serialization (MCS §12.9)."""

    def test_eulerian_graph(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])
        tour = euler_tour(g)
        assert len(tour) == 4  # 3 edges + return to start
        assert tour[0] == tour[-1]  # Circuit

    def test_non_eulerian_fallback(self):
        g = nx.path_graph(4)  # Not Eulerian (endpoints have odd degree)
        tour = euler_tour(g)
        assert len(tour) == 4  # DFS fallback

    def test_empty_graph(self):
        assert euler_tour(nx.Graph()) == []

    def test_disconnected_graph(self):
        g = nx.Graph()
        g.add_edge(1, 2)
        g.add_edge(3, 4)
        assert euler_tour(g) == []


class TestChromaticScheduling:
    """Tests for Chromatic Scheduling (MCS §12.6)."""

    def test_bipartite_graph(self):
        g = nx.complete_bipartite_graph(3, 3)
        coloring = chromatic_schedule(g)
        assert max(coloring.values()) + 1 == 2  # Bipartite → 2 colors

    def test_complete_graph(self):
        g = nx.complete_graph(4)
        assert chromatic_number_upper_bound(g) == 4

    def test_no_conflicts(self):
        g = nx.Graph()
        g.add_nodes_from([1, 2, 3])
        coloring = chromatic_schedule(g)
        assert max(coloring.values()) + 1 == 1  # All independent

    def test_adjacent_nodes_different_colors(self):
        g = nx.cycle_graph(5)
        coloring = chromatic_schedule(g)
        for u, v in g.edges():
            assert coloring[u] != coloring[v]


class TestPersonalizedPageRank:
    """Tests for Personalized PageRank (MCS §21.2)."""

    def test_star_graph(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "center"), ("B", "center"), ("C", "center")])
        ranks = personalized_pagerank(g)
        assert ranks["center"] > ranks["A"]

    def test_uniform_seeds(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        ranks = personalized_pagerank(g)
        assert abs(sum(ranks.values()) - 1.0) < 1e-6

    def test_personalized_bias(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        ranks = personalized_pagerank(g, seed_nodes={"A": 1.0})
        # A should have higher rank due to teleport bias
        assert ranks["A"] > 0.2

    def test_empty_graph(self):
        assert personalized_pagerank(nx.DiGraph()) == {}


class TestPathCounting:
    """Tests for Generating Function Path Counter (MCS Ch 16)."""

    def test_linear_paths(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        assert count_paths_of_length(g, "A", "D", 3) == 1
        assert count_paths_of_length(g, "A", "D", 2) == 0
        assert count_paths_of_length(g, "A", "A", 0) == 1

    def test_diamond_paths(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
        assert count_paths_of_length(g, "A", "D", 2) == 2

    def test_missing_nodes(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        assert count_paths_of_length(g, "A", "Z", 1) == 0

    def test_reachability(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        reach = reachability_within_hops(g, "A", 2)
        assert "C" in reach
        assert "D" not in reach
        assert reach["A"] == 0
        assert reach["B"] == 1


class TestMathFoundationSeed:
    """Tests for MCS Reference Taxonomy seed data."""

    def test_seed_has_entries(self):
        seed = generate_math_foundation_seed()
        assert len(seed) >= 15

    def test_seed_structure(self):
        seed = generate_math_foundation_seed()
        for entry in seed:
            assert "id" in entry
            assert "name" in entry
            assert "definition" in entry
            assert "domain" in entry
            assert entry["id"].startswith("mcs_")

    def test_covers_key_domains(self):
        seed = generate_math_foundation_seed()
        domains = {e["domain"] for e in seed}
        assert "graph_theory" in domains
        assert "probability" in domains
        assert "formal_methods" in domains
