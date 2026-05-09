#!/usr/bin/env python3
"""Formal Graph Theory Primitives.

CONCEPT:KG-2.41 — Formal Graph Theory Primitives

Implements mathematically rigorous graph-theoretic operations derived from
*Mathematics for Computer Science* (Lehman, Leighton, Meyer — MIT 6.042J).

- **DAG Critical Path Analysis** (MCS §10.5): Topological sort + longest-path.
- **Graph Connectivity Certificates** (MCS §12.8–12.10): k-vertex/k-edge connectivity.
- **Euler Tour Serialization** (MCS §12.9): O(E) full-graph traversal.
- **Chromatic Scheduling** (MCS §12.6): Greedy graph coloring for parallel scheduling.
- **Personalized PageRank** (MCS §21.2): Random walk with restart.
- **Generating Function Path Counter** (MCS Ch 16): A^k path counting.
- **MCS Reference Taxonomy**: Curated seed nodes for KG preloading.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def dag_critical_path(
    graph: nx.DiGraph,
    weight_attr: str = "weight",
    default_weight: float = 1.0,
) -> dict[str, Any]:
    """Compute the critical (longest) path in a weighted DAG.

    CONCEPT:KG-2.41 — DAG Critical Path Analysis (MCS §10.5)

    The critical path is the longest weighted path from any source to any
    sink.  Its length equals the minimum possible makespan.  Uses a single
    forward DP pass after topological sort: O(V + E).

    Args:
        graph: A directed acyclic graph.
        weight_attr: Edge attribute name for weights.
        default_weight: Fallback weight when edge lacks weight attribute.

    Returns:
        Dict with ``makespan``, ``critical_path``, ``node_earliest_start``,
        ``node_slack``.

    Raises:
        nx.NetworkXUnfeasible: If the graph contains a cycle.
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise nx.NetworkXUnfeasible("Graph contains a cycle — not a DAG.")

    if len(graph) == 0:
        return {
            "makespan": 0.0,
            "critical_path": [],
            "node_earliest_start": {},
            "node_slack": {},
        }

    topo_order = list(nx.topological_sort(graph))
    earliest: dict[Any, float] = {n: 0.0 for n in topo_order}
    predecessor: dict[Any, Any] = {n: None for n in topo_order}

    for node in topo_order:
        for succ in graph.successors(node):
            edge_data = graph.get_edge_data(node, succ) or {}
            w = float(edge_data.get(weight_attr, default_weight))
            candidate = earliest[node] + w
            if candidate > earliest[succ]:
                earliest[succ] = candidate
                predecessor[succ] = node

    sink = max(topo_order, key=lambda n: earliest[n])
    makespan = earliest[sink]

    latest: dict[Any, float] = {n: makespan for n in topo_order}
    for node in reversed(topo_order):
        for succ in graph.successors(node):
            edge_data = graph.get_edge_data(node, succ) or {}
            w = float(edge_data.get(weight_attr, default_weight))
            latest[node] = min(latest[node], latest[succ] - w)

    slack = {n: latest[n] - earliest[n] for n in topo_order}

    critical_path: list[Any] = []
    current: Any = sink
    while current is not None:
        critical_path.append(current)
        current = predecessor[current]
    critical_path.reverse()

    return {
        "makespan": makespan,
        "critical_path": critical_path,
        "node_earliest_start": earliest,
        "node_slack": slack,
    }


def vertex_connectivity(graph: nx.Graph) -> Any:
    """Compute vertex connectivity κ(G). CONCEPT:KG-2.41 (MCS §12.10)."""
    if len(graph) < 2 or not nx.is_connected(graph):
        return 0
    return nx.node_connectivity(graph)


def edge_connectivity(graph: nx.Graph) -> Any:
    """Compute edge connectivity λ(G). CONCEPT:KG-2.41 (MCS §12.10)."""
    if len(graph) < 2 or not nx.is_connected(graph):
        return 0
    return nx.edge_connectivity(graph)


def minimum_vertex_cut(graph: nx.Graph) -> set[str]:
    """Fiand minimum vertex cut set — critical chokepoint nodes. CONCEPT:KG-2.41."""
    if len(graph) < 2 or not nx.is_connected(graph):
        return set()
    return set(nx.minimum_node_cut(graph))


def euler_tour(graph: nx.Graph) -> list[Any]:
    """Compute Euler tour of an undirected graph. CONCEPT:KG-2.41 (MCS §12.9).

    An Euler tour traverses every edge exactly once.  Falls back to DFS
    traversal when the graph is not Eulerian.

    Args:
        graph: An undirected ``nx.Graph``.

    Returns:
        List of node IDs representing the tour.
    """
    if len(graph) == 0 or not nx.is_connected(graph):
        return []
    if nx.is_eulerian(graph):
        circuit = list(nx.eulerian_circuit(graph))
        return [circuit[0][0]] + [e[1] for e in circuit]
    logger.info("Graph is not Eulerian — falling back to DFS traversal.")
    return list(nx.dfs_preorder_nodes(graph, source=next(iter(graph.nodes()))))


def chromatic_schedule(conflict_graph: nx.Graph) -> dict[str, int]:
    """Assign colors (execution slots) via greedy graph coloring. CONCEPT:KG-2.41 (MCS §12.6).

    Args:
        conflict_graph: Edges represent conflicts between tasks/agents.

    Returns:
        Dict mapping node ID → color (0-indexed). Same-color nodes run concurrently.
    """
    if len(conflict_graph) == 0:
        return {}
    return nx.coloring.greedy_color(conflict_graph, strategy="largest_first")


def chromatic_number_upper_bound(conflict_graph: nx.Graph) -> int:
    """Upper bouand on χ(G) via greedy coloring. Satisfies χ(G) ≤ Δ(G) + 1."""
    if len(conflict_graph) == 0:
        return 0
    coloring = chromatic_schedule(conflict_graph)
    return max(coloring.values()) + 1 if coloring else 0


def personalized_pagerank(
    graph: nx.DiGraph,
    seed_nodes: dict[str, float] | None = None,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Compute personalized PageRank via power iteration. CONCEPT:KG-2.41 (MCS §21.2).

    At each step the walker follows an edge (prob ``damping``) or teleports
    to a seed node (prob ``1 - damping``).

    Args:
        graph: A directed graph.
        seed_nodes: Dict of node_id → teleport weight.  None = uniform.
        damping: Continuation probability (default 0.85).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Dict of node_id → PageRank score (sums to ~1.0).
    """
    if len(graph) == 0:
        return {}

    nodes = list(graph.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    M = np.zeros((n, n))
    for i, node in enumerate(nodes):
        successors = list(graph.successors(node))
        if successors:
            w = 1.0 / len(successors)
            for succ in successors:
                M[node_idx[succ], i] = w
        else:
            M[:, i] = 1.0 / n

    v = np.ones(n) / n
    if seed_nodes:
        v = np.zeros(n)
        total_w = sum(seed_nodes.values())
        if total_w > 0:
            for node_id, wt in seed_nodes.items():
                if node_id in node_idx:
                    v[node_idx[node_id]] = wt / total_w

    rank = np.ones(n) / n
    for _ in range(max_iter):
        new_rank = damping * M @ rank + (1 - damping) * v
        if np.linalg.norm(new_rank - rank, 1) < tol:
            break
        rank = new_rank

    total = rank.sum()
    if total > 0:
        rank = rank / total

    return {nodes[i]: float(rank[i]) for i in range(n)}


def count_paths_of_length(
    graph: nx.DiGraph, source: str, target: str, length: int
) -> int:
    """Count directed walks of exact length k via adjacency matrix power. CONCEPT:KG-2.41 (MCS §10.3, Ch 16).

    Uses the theorem: (A^k)[i][j] = number of walks of length k from i to j.

    Args:
        graph: A directed graph.
        source: Source node ID.
        target: Target node ID.
        length: Exact path length k.

    Returns:
        Number of distinct walks of length k.
    """
    if source not in graph or target not in graph or length < 0:
        return 0
    if length == 0:
        return 1 if source == target else 0

    nodes = list(graph.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=np.int64)
    for u, v_node in graph.edges():
        A[node_idx[u], node_idx[v_node]] += 1

    result = np.linalg.matrix_power(A, length)
    return int(result[node_idx[source], node_idx[target]])


def reachability_within_hops(
    graph: nx.DiGraph, source: str, max_hops: int
) -> dict[str, int]:
    """BFS reachability within max_hops. CONCEPT:KG-2.41 (MCS §10.4 Walk Relations).

    Args:
        graph: A directed graph.
        source: Starting node ID.
        max_hops: Maximum traversal depth.

    Returns:
        Dict of reachable node_id → shortest distance.
    """
    if source not in graph:
        return {}
    distances: dict[str, int] = {source: 0}
    frontier = [source]
    for depth in range(1, max_hops + 1):
        next_frontier: list[str] = []
        for node in frontier:
            for neighbor in graph.successors(node):
                if neighbor not in distances:
                    distances[neighbor] = depth
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return distances


def generate_math_foundation_seed() -> list[dict[str, Any]]:
    """Generate curated MCS reference taxonomy for KG seeding. CONCEPT:KG-2.41.

    Returns a list of mathematical concept definitions derived from MIT's
    *Mathematics for Computer Science*, structured as KG-persistable node dicts.

    Returns:
        List of node dicts with id, name, definition, chapter, domain, relevance.
    """
    return [
        {
            "id": "mcs_graph_isomorphism",
            "name": "Graph Isomorphism",
            "definition": "Two graphs G1, G2 are isomorphic iff ∃ bijection f: V(G1)→V(G2) preserving adjacency.",
            "chapter": "MCS §12.4",
            "domain": "graph_theory",
            "relevance": "Foundation for KG-2.15 Analogy Engine (VF2 isomorphism).",
        },
        {
            "id": "mcs_dag_scheduling",
            "name": "DAG Scheduling & Critical Path",
            "definition": "Critical path = longest weighted path in DAG. Length = minimum makespan regardless of parallelism.",
            "chapter": "MCS §10.5",
            "domain": "graph_theory",
            "relevance": "ORCH-1.4 Swarm Preset Engine task scheduling.",
        },
        {
            "id": "mcs_topological_sort",
            "name": "Topological Sort",
            "definition": "Linear ordering of DAG vertices such that for every edge (u,v), u precedes v. Exists iff graph is a DAG.",
            "chapter": "MCS §10.5",
            "domain": "graph_theory",
            "relevance": "Prerequisite for critical path and dependency resolution.",
        },
        {
            "id": "mcs_eulerian_circuit",
            "name": "Euler Tour",
            "definition": "Closed walk traversing every edge exactly once. Exists iff every vertex has even degree (Euler's Theorem).",
            "chapter": "MCS §12.9",
            "domain": "graph_theory",
            "relevance": "O(E) KG serialization for checkpointing.",
        },
        {
            "id": "mcs_graph_coloring",
            "name": "Graph Coloring",
            "definition": "Proper k-coloring: assign k colors so no adjacent vertices share a color. χ(G) = minimum such k.",
            "chapter": "MCS §12.6",
            "domain": "graph_theory",
            "relevance": "Conflict-free parallel agent scheduling.",
        },
        {
            "id": "mcs_vertex_connectivity",
            "name": "Vertex Connectivity κ(G)",
            "definition": "Min vertices whose removal disconnects G. Whitney: κ(G) ≤ λ(G) ≤ δ(G).",
            "chapter": "MCS §12.10",
            "domain": "graph_theory",
            "relevance": "KG structural resilience measurement.",
        },
        {
            "id": "mcs_random_walk",
            "name": "Random Walk on Graphs",
            "definition": "Sequence where each vertex chosen uniformly from neighbors. Stationary distribution π = Mπ.",
            "chapter": "MCS §21.2",
            "domain": "probability",
            "relevance": "Foundation for Personalized PageRank.",
        },
        {
            "id": "mcs_bayes_theorem",
            "name": "Bayes' Theorem",
            "definition": "P(A|B) = P(B|A)·P(A)/P(B). Updates prior to posterior given evidence.",
            "chapter": "MCS §18.4",
            "domain": "probability",
            "relevance": "Probabilistic KG reasoning and belief propagation.",
        },
        {
            "id": "mcs_law_total_probability",
            "name": "Law of Total Probability",
            "definition": "P(B) = Σ P(B|Aᵢ)·P(Aᵢ) where {Aᵢ} partitions sample space.",
            "chapter": "MCS §18.5",
            "domain": "probability",
            "relevance": "Multi-source retrieval score combination.",
        },
        {
            "id": "mcs_conditional_independence",
            "name": "Conditional Independence",
            "definition": "A,B conditionally independent given C iff P(A∩B|C) = P(A|C)·P(B|C).",
            "chapter": "MCS §18.7",
            "domain": "probability",
            "relevance": "d-separation in causal KG subgraphs.",
        },
        {
            "id": "mcs_birthday_paradox",
            "name": "Birthday Paradox",
            "definition": "Among n items from d possibilities, collision prob > 50% when n ≈ 1.2√d.",
            "chapter": "MCS §17.4",
            "domain": "probability",
            "relevance": "Probabilistic KG duplicate detection.",
        },
        {
            "id": "mcs_generating_functions",
            "name": "Generating Functions",
            "definition": "OGF of {aₙ} is G(x) = Σ aₙxⁿ. Enables algebraic solution of recurrences.",
            "chapter": "MCS Ch 16",
            "domain": "combinatorics",
            "relevance": "KG query cardinality estimation.",
        },
        {
            "id": "mcs_state_machines",
            "name": "State Machines & Invariants",
            "definition": "(States, Start, Transitions). Invariant P preserved by all transitions proves P for all reachable states.",
            "chapter": "MCS §6.1–6.2",
            "domain": "formal_methods",
            "relevance": "ORCH-1.0 HSM Router correctness proofs.",
        },
        {
            "id": "mcs_modular_arithmetic",
            "name": "Modular Arithmetic",
            "definition": "a ≡ b (mod m) iff m|(a-b). gcd via Euclid's algorithm in O(log min(a,b)).",
            "chapter": "MCS §9.2–9.6",
            "domain": "number_theory",
            "relevance": "OS-5.1 Security hash-based ID schemes.",
        },
        {
            "id": "mcs_rsa",
            "name": "RSA Encryption",
            "definition": "n=pq, e coprime to φ(n), d=e⁻¹ mod φ(n). Encrypt: c=mᵉ mod n. Decrypt: m=cᵈ mod n.",
            "chapter": "MCS §9.11",
            "domain": "number_theory",
            "relevance": "Formal basis for JWT authentication.",
        },
        {
            "id": "mcs_strong_induction",
            "name": "Strong Induction",
            "definition": "Prove P(b), then P(n+1) assuming P(k) for all b≤k≤n. Equivalent to ordinary induction.",
            "chapter": "MCS §5.2",
            "domain": "formal_methods",
            "relevance": "Correctness proofs for recursive KG operations.",
        },
        {
            "id": "mcs_linearity_expectation",
            "name": "Linearity of Expectation",
            "definition": "E[X+Y] = E[X]+E[Y] always, regardless of independence.",
            "chapter": "MCS §19.4",
            "domain": "probability",
            "relevance": "Aggregate KG metric estimation.",
        },
        {
            "id": "mcs_recurrences",
            "name": "Linear Recurrences",
            "definition": "aₙ = c₁aₙ₋₁ + ... + cₖaₙ₋ₖ. Solved via characteristic polynomial.",
            "chapter": "MCS Ch 22",
            "domain": "combinatorics",
            "relevance": "KG growth modeling and memory consolidation dynamics.",
        },
        {
            "id": "mcs_adjacency_power",
            "name": "Adjacency Matrix Power",
            "definition": "(A^k)[i][j] = walks of length k from i to j.",
            "chapter": "MCS §10.3",
            "domain": "graph_theory",
            "relevance": "Path counting queries in KG.",
        },
        {
            "id": "mcs_planar_graphs",
            "name": "Planar Graphs & Euler's Formula",
            "definition": "v - e + f = 2 for connected planar graphs. K₅ and K₃,₃ are not planar.",
            "chapter": "MCS §13.3",
            "domain": "graph_theory",
            "relevance": "KG visualization layout optimization.",
        },
        {
            "id": "mcs_equivalence_relations",
            "name": "Equivalence Relations",
            "definition": "A relation that is reflexive, symmetric, and transitive. Partitions a set into disjoint equivalence classes.",
            "chapter": "MCS §4.3",
            "domain": "formal_relations",
            "relevance": "Zero-shot entity resolution and KG deduplication.",
        },
        {
            "id": "mcs_partial_orders",
            "name": "Partial Orders",
            "definition": "A relation that is reflexive, antisymmetric, and transitive. Represented via Hasse diagrams.",
            "chapter": "MCS §4.5",
            "domain": "formal_relations",
            "relevance": "Strict dependency ordering and KG hierarchy validation.",
        },
        {
            "id": "mcs_markov_chains",
            "name": "Markov Chains",
            "definition": "A stochastic process where the next state depends only on the current state. Described by a transition matrix.",
            "chapter": "MCS §21.1",
            "domain": "probability",
            "relevance": "Predictive modeling of agent state transitions and failure forecasting.",
        },
    ]
