"""Tests for CONCEPT:KG-2.6 — Formal Graph Theory Primitives."""

from typing import Any

import pytest

from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
    chromatic_number_upper_bound,
    chromatic_schedule,
    count_paths_of_length,
    dag_critical_path,
    edge_connectivity,
    euler_tour,
    generate_math_foundation_seed,
    minimum_vertex_cut,
    reachability_within_hops,
    vertex_connectivity,
)
from agent_utilities.knowledge_graph.core.graph_primitives import PyDiGraph, PyGraph


def build_digraph(nodes, edges):
    g = PyDiGraph()
    n2i = {}
    for n in nodes:
        n2i[n] = g.add_node(n)
    for src, tgt, data in edges:
        g.add_edge(n2i[src], n2i[tgt], data)
    return g


def build_graph(nodes, edges):
    g = PyGraph()
    n2i = {}
    for n in nodes:
        n2i[n] = g.add_node(n)
    for src, tgt, data in edges:
        g.add_edge(n2i[src], n2i[tgt], data)
    return g


class TestDAGCriticalPath:
    """Tests for DAG Critical Path Analysis (MCS §10.5)."""

    def test_linear_dag(self):
        nodes = ["A", "B", "C", "D"]
        edges: list[tuple[Any, Any, Any]] = [
            ("A", "B", {"weight": 3}),
            ("B", "C", {"weight": 5}),
            ("C", "D", {"weight": 2}),
        ]
        g = build_digraph(nodes, edges)
        result = dag_critical_path(g)
        assert result["makespan"] == 10.0
        assert result["critical_path"] == ["A", "B", "C", "D"]

    def test_parallel_dag(self):
        nodes = ["S", "A", "B", "T"]
        edges: list[tuple[Any, Any, Any]] = [
            ("S", "A", {"weight": 2}),
            ("S", "B", {"weight": 5}),
            ("A", "T", {"weight": 3}),
            ("B", "T", {"weight": 1}),
        ]
        g = build_digraph(nodes, edges)
        result = dag_critical_path(g)
        assert result["makespan"] == 6.0  # max(2+3, 5+1) = max(5,6) = 6
        assert result["node_slack"]["S"] == 0.0

    def test_empty_dag(self):
        g = PyDiGraph()
        result = dag_critical_path(g)
        assert result["makespan"] == 0.0
        assert result["critical_path"] == []

    def test_single_node(self):
        g = PyDiGraph()
        g.add_node("A")
        result = dag_critical_path(g)
        assert result["makespan"] == 0.0
        assert "A" in result["node_earliest_start"]

    def test_cycle_raises(self):
        nodes = ["A", "B", "C"]
        edges: list[tuple[Any, Any, Any]] = [
            ("A", "B", {}),
            ("B", "C", {}),
            ("C", "A", {}),
        ]
        g = build_digraph(nodes, edges)
        with pytest.raises(ValueError):
            dag_critical_path(g)

    def test_slack_identifies_non_critical(self):
        nodes = ["S", "A", "B", "T"]
        edges: list[tuple[Any, Any, Any]] = [
            ("S", "A", {"weight": 1}),
            ("S", "B", {"weight": 5}),
            ("A", "T", {"weight": 1}),
            ("B", "T", {"weight": 1}),
        ]
        g = build_digraph(nodes, edges)
        result = dag_critical_path(g)
        assert result["node_slack"]["A"] > 0  # A is not on critical path
        assert result["node_slack"]["B"] == 0.0  # B is on critical path


class TestConnectivity:
    """Tests for Graph Connectivity Certificates (MCS §12.8–12.10)."""

    def test_complete_graph_connectivity(self):
        nodes = [1, 2, 3, 4, 5]
        edges: list[tuple[Any, Any, Any]] = [
            (u, v, {}) for i, u in enumerate(nodes) for v in nodes[i + 1 :]
        ]
        g = build_graph(nodes, edges)
        assert vertex_connectivity(g) == 4
        assert edge_connectivity(g) == 4

    def test_path_graph_connectivity(self):
        nodes = [1, 2, 3, 4, 5]
        edges: list[tuple[Any, Any, Any]] = [
            (1, 2, {}),
            (2, 3, {}),
            (3, 4, {}),
            (4, 5, {}),
        ]
        g = build_graph(nodes, edges)
        assert vertex_connectivity(g) == 1
        assert edge_connectivity(g) == 1

    def test_disconnected_graph(self):
        g = build_graph([1, 2, 3], [])
        assert vertex_connectivity(g) == 0
        assert edge_connectivity(g) == 0

    def test_minimum_vertex_cut_bridge(self):
        nodes = [1, 2, 3, 4, 5, 6]
        edges: list[tuple[Any, Any, Any]] = [
            (1, 2, {}),
            (2, 3, {}),
            (3, 4, {}),
            (4, 5, {}),
            (5, 3, {}),
            (2, 6, {}),
        ]
        g = build_graph(nodes, edges)
        cut = minimum_vertex_cut(g)
        assert len(cut) >= 1  # At least one cut vertex

    def test_empty_graph(self):
        g = PyGraph()
        assert vertex_connectivity(g) == 0
        assert minimum_vertex_cut(g) == set()


class TestEulerTour:
    """Tests for Euler Tour Serialization (MCS §12.9)."""

    def test_eulerian_graph(self):
        nodes = [0, 1, 2]
        edges: list[tuple[Any, Any, Any]] = [(0, 1, {}), (1, 2, {}), (2, 0, {})]
        g = build_graph(nodes, edges)
        tour = euler_tour(g)
        assert len(tour) == 4  # 3 edges + return to start
        assert tour[0] == tour[-1]  # Circuit

    def test_non_eulerian_fallback(self):
        nodes = [1, 2, 3, 4]
        edges: list[tuple[Any, Any, Any]] = [(1, 2, {}), (2, 3, {}), (3, 4, {})]
        g = build_graph(nodes, edges)
        tour = euler_tour(g)
        assert len(tour) == 4  # DFS fallback

    def test_empty_graph(self):
        assert euler_tour(PyGraph()) == []

    def test_disconnected_graph(self):
        nodes = [1, 2, 3, 4]
        edges: list[tuple[Any, Any, Any]] = [(1, 2, {}), (3, 4, {})]
        g = build_graph(nodes, edges)
        assert euler_tour(g) == []


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

    def test_complete_graph(self):
        nodes = [1, 2, 3, 4]
        edges: list[tuple[Any, Any, Any]] = [
            (u, v, {}) for i, u in enumerate(nodes) for v in nodes[i + 1 :]
        ]
        g = build_graph(nodes, edges)
        assert chromatic_number_upper_bound(g) == 4

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
        for _, (u, v, _) in g._edges.items():
            assert coloring[g[u]] != coloring[g[v]]


class TestPathCounting:
    """Tests for Generating Function Path Counter (MCS Ch 16)."""

    def test_linear_paths(self):
        nodes = ["A", "B", "C", "D"]
        edges: list[tuple[Any, Any, Any]] = [
            ("A", "B", {}),
            ("B", "C", {}),
            ("C", "D", {}),
        ]
        g = build_digraph(nodes, edges)
        assert count_paths_of_length(g, "A", "D", 3) == 1
        assert count_paths_of_length(g, "A", "D", 2) == 0
        assert count_paths_of_length(g, "A", "A", 0) == 1

    def test_diamond_paths(self):
        nodes = ["A", "B", "C", "D"]
        edges: list[tuple[Any, Any, Any]] = [
            ("A", "B", {}),
            ("A", "C", {}),
            ("B", "D", {}),
            ("C", "D", {}),
        ]
        g = build_digraph(nodes, edges)
        assert count_paths_of_length(g, "A", "D", 2) == 2

    def test_missing_nodes(self):
        nodes = ["A", "B"]
        edges: list[tuple[Any, Any, Any]] = [("A", "B", {})]
        g = build_digraph(nodes, edges)
        assert count_paths_of_length(g, "A", "Z", 1) == 0

    def test_reachability(self):
        nodes = ["A", "B", "C", "D"]
        edges: list[tuple[Any, Any, Any]] = [
            ("A", "B", {}),
            ("B", "C", {}),
            ("C", "D", {}),
        ]
        g = build_digraph(nodes, edges)
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
