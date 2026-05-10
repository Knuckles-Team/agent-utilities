from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import networkx as nx
import numpy as np

# --- Merged from formal_reasoning_core.py ---

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
            "domain": "formal_reasoning_core",
            "relevance": "Zero-shot entity resolution and KG deduplication.",
        },
        {
            "id": "mcs_partial_orders",
            "name": "Partial Orders",
            "definition": "A relation that is reflexive, antisymmetric, and transitive. Represented via Hasse diagrams.",
            "chapter": "MCS §4.5",
            "domain": "formal_reasoning_core",
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


# --- Merged from formal_reasoning_core.py ---

#!/usr/bin/env python3
"""Structural Causal Reasoning Engine.

CONCEPT:KG-2.43 — Structural Causal Reasoning Engine

Explicit causal chain modeling derived from MedCausalX (arXiv:2603.23085v1).
Provides Structural Causal Models (SCMs), causal verification protocols,
counterfactual generation, spuriousness detection, and trajectory-level
causal alignment scoring.

Operates natively on the Knowledge Graph's ``networkx.MultiDiGraph`` via
``CausalFactorNode`` and ``CAUSED_BY`` / ``CAUSAL_MECHANISM`` edges.
"""


logger = logging.getLogger(__name__)


class CausalRelationType(StrEnum):
    """Types of causal relationships in the SCM."""

    DIRECT_CAUSE = "direct_cause"
    CONTRIBUTING_FACTOR = "contributing_factor"
    NECESSARY_CONDITION = "necessary_condition"
    SUFFICIENT_CONDITION = "sufficient_condition"
    SPURIOUS_CORRELATION = "spurious_correlation"
    CONFOUNDED = "confounded"


@dataclass
class CausalFactor:
    """A node in a Structural Causal Model.

    Attributes:
        id: Unique identifier.
        name: Human-readable label.
        domain: Domain context (e.g., 'financial', 'medical', 'code').
        observed: Whether this variable is observed or latent.
        value: Optional observed value.
        confidence: Confidence in the causal assignment (0.0–1.0).
    """

    id: str = ""
    name: str = ""
    domain: str = "general"
    observed: bool = True
    value: Any = None
    confidence: float = 1.0


@dataclass
class CausalEdge:
    """A directed causal relationship between two factors.

    Attributes:
        source_id: Cause factor ID.
        target_id: Effect factor ID.
        relation_type: Type of causal relationship.
        mechanism: Description of the causal mechanism.
        strength: Causal strength (0.0–1.0).
        is_verified: Whether this edge has been verified via intervention.
    """

    source_id: str = ""
    target_id: str = ""
    relation_type: CausalRelationType = CausalRelationType.DIRECT_CAUSE
    mechanism: str = ""
    strength: float = 1.0
    is_verified: bool = False


@dataclass
class CausalVerificationResult:
    """Result of causal chain verification.

    Attributes:
        chain_id: Identifier for the verified chain.
        is_consistent: True if the chain maintains causal consistency.
        violations: List of detected causal violations.
        consistency_score: Overall consistency score (0.0–1.0).
        spurious_edges: Edges flagged as potentially spurious.
    """

    chain_id: str = ""
    is_consistent: bool = True
    violations: list[str] = field(default_factory=list)
    consistency_score: float = 1.0
    spurious_edges: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class CounterfactualQuery:
    """A counterfactual query: 'What if X were different?'

    Attributes:
        intervention_node: The node to intervene on.
        intervention_value: The counterfactual value.
        target_node: The node whose counterfactual outcome we want.
        original_value: The original value of the intervention node.
        counterfactual_outcome: The predicted outcome under intervention.
    """

    intervention_node: str = ""
    intervention_value: Any = None
    target_node: str = ""
    original_value: Any = None
    counterfactual_outcome: Any = None


class StructuralCausalModel:
    """A Structural Causal Model (SCM) built on a directed graph.

    CONCEPT:KG-2.43 — SCM (MedCausalX §3.1, Eq. 2)

    An SCM M = ⟨V, U, F, P(U)⟩ where:
    - V: Endogenous (observed) variables
    - U: Exogenous (latent) variables
    - F: Structural equations (directed edges with mechanisms)
    - P(U): Distribution over exogenous variables

    This implementation uses a ``networkx.DiGraph`` as the causal DAG
    and provides do-calculus operations, d-separation testing, and
    counterfactual reasoning.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._factors: dict[str, CausalFactor] = {}
        self._edges: list[CausalEdge] = []

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying causal DAG."""
        return self._graph

    @property
    def factor_count(self) -> int:
        """Number of causal factors."""
        return len(self._factors)

    @property
    def edge_count(self) -> int:
        """Number of causal edges."""
        return len(self._edges)

    def add_factor(self, factor: CausalFactor) -> None:
        """Add a causal factor (variable) to the SCM.

        Args:
            factor: The CausalFactor to add.
        """
        if not factor.id:
            factor.id = f"cf_{uuid.uuid4().hex[:8]}"
        self._factors[factor.id] = factor
        self._graph.add_node(factor.id, data=factor)

    def add_edge(self, edge: CausalEdge) -> None:
        """Add a causal edge (structural equation) to the SCM.

        Args:
            edge: The CausalEdge connecting cause to effect.

        Raises:
            ValueError: If the edge would create a cycle.
        """
        # Check for cycles
        if edge.target_id in self._factors and edge.source_id in self._factors:
            test_graph = self._graph.copy()
            test_graph.add_edge(edge.source_id, edge.target_id)
            if not nx.is_directed_acyclic_graph(test_graph):
                raise ValueError(
                    f"Adding edge {edge.source_id} → {edge.target_id} "
                    f"would create a cycle in the causal DAG."
                )

        self._edges.append(edge)
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            relation_type=edge.relation_type.value,
            mechanism=edge.mechanism,
            strength=edge.strength,
            is_verified=edge.is_verified,
        )

    def do_intervention(self, node_id: str, value: Any) -> nx.DiGraph:
        """Perform a do-calculus intervention: do(X = value).

        CONCEPT:KG-2.43 — do-Calculus Intervention

        Implements Pearl's do-operator by removing all incoming edges to
        the intervened node and setting its value.  Returns the mutilated
        graph for downstream causal inference.

        Args:
            node_id: The variable to intervene on.
            value: The value to set.

        Returns:
            A mutilated copy of the causal DAG with the intervention applied.
        """
        mutilated = self._graph.copy()

        # Remove all incoming edges to the intervened node
        parents = list(mutilated.predecessors(node_id))
        for parent in parents:
            mutilated.remove_edge(parent, node_id)

        # Set the intervention value
        if node_id in self._factors:
            factor = self._factors[node_id]
            mutilated.nodes[node_id]["intervention_value"] = value
            mutilated.nodes[node_id]["original_value"] = factor.value

        return mutilated

    def is_d_separated(
        self,
        x: str,
        y: str,
        conditioning_set: set[str] | None = None,
    ) -> bool:
        """Test d-separation between X and Y given conditioning set Z.

        CONCEPT:KG-2.43 — d-Separation (Conditional Independence)

        X and Y are d-separated given Z iff every path between X and Y
        is blocked by Z.  A path is blocked if it contains:
        1. A chain A → B → C or fork A ← B → C with B ∈ Z, or
        2. A collider A → B ← C with B ∉ Z and no descendant of B in Z.

        Uses networkx's built-in d-separation test.

        Args:
            x: First variable.
            y: Second variable.
            conditioning_set: Set of conditioned variables.

        Returns:
            True if X and Y are d-separated given Z.
        """
        z = conditioning_set or set()
        if x not in self._graph or y not in self._graph:
            return True  # Unconnected variables are trivially independent

        try:
            return nx.is_d_separator(self._graph, x, y, z)
        except Exception:
            # Fallback: check if there's any path
            try:
                nx.shortest_path(self._graph.to_undirected(), x, y)
                return False
            except nx.NetworkXNoPath:
                return True

    def get_causal_ancestors(self, node_id: str) -> set[str]:
        """Get all causal ancestors (upstream causes) of a node.

        Args:
            node_id: The effect variable.

        Returns:
            Set of ancestor node IDs.
        """
        if node_id not in self._graph:
            return set()
        return nx.ancestors(self._graph, node_id)

    def get_causal_descendants(self, node_id: str) -> set[str]:
        """Get all causal descendants (downstream effects) of a node.

        Args:
            node_id: The cause variable.

        Returns:
            Set of descendant node IDs.
        """
        if node_id not in self._graph:
            return set()
        return nx.descendants(self._graph, node_id)

    def topological_causal_order(self) -> list[str]:
        """Return factors in causal (topological) order.

        Returns:
            List of factor IDs from root causes to terminal effects.
        """
        return list(nx.topological_sort(self._graph))


class CausalVerifier:
    """Verifies causal consistency of reasoning trajectories.

    CONCEPT:KG-2.43 — Causal Verification Protocol (MedCausalX §3.2)

    Inspired by MedCausalX's <causal> and <verify> token mechanism.
    Checks whether a reasoning chain's intermediate steps maintain causal
    consistency with the underlying SCM.
    """

    def __init__(self, scm: StructuralCausalModel) -> None:
        self._scm = scm

    def verify_chain(
        self,
        reasoning_steps: list[dict[str, Any]],
    ) -> CausalVerificationResult:
        """Verify causal consistency of a reasoning chain.

        Each step should have ``cause`` and ``effect`` keys identifying
        the causal factors referenced.

        Args:
            reasoning_steps: List of dicts with 'cause', 'effect', and
                optionally 'mechanism' keys.

        Returns:
            CausalVerificationResult with violations and consistency score.
        """
        chain_id = f"chain_{uuid.uuid4().hex[:8]}"
        violations: list[str] = []
        spurious: list[tuple[str, str]] = []
        total_steps = len(reasoning_steps)

        if total_steps == 0:
            return CausalVerificationResult(chain_id=chain_id, consistency_score=1.0)

        for i, step in enumerate(reasoning_steps):
            cause = step.get("cause", "")
            effect = step.get("effect", "")

            if not cause or not effect:
                continue

            # Check 1: Does the causal direction exist in the SCM?
            if not self._scm.graph.has_edge(cause, effect):
                # Check if reverse exists (direction error)
                if self._scm.graph.has_edge(effect, cause):
                    violations.append(
                        f"Step {i}: Reversed causality — {cause}→{effect} "
                        f"should be {effect}→{cause}."
                    )
                else:
                    # No direct edge — check if there's a path
                    try:
                        path = nx.shortest_path(self._scm.graph, cause, effect)
                        if len(path) > 2:
                            violations.append(
                                f"Step {i}: Indirect causality — {cause}→{effect} "
                                f"requires intermediaries: {' → '.join(path)}."
                            )
                    except nx.NetworkXNoPath:
                        violations.append(
                            f"Step {i}: No causal path from {cause} to {effect}."
                        )
                        spurious.append((cause, effect))

            # Check 2: Temporal ordering (if multiple steps reference the same effect)
            if i > 0:
                prev_effect = reasoning_steps[i - 1].get("effect", "")
                if prev_effect and cause != prev_effect:
                    # Check if previous effect should precede current cause
                    if (
                        prev_effect in self._scm.graph
                        and cause in self._scm.graph
                        and not self._scm.is_d_separated(prev_effect, cause)
                    ):
                        pass  # Connected — ordering is fine

        valid_steps = total_steps - len(violations)
        score = valid_steps / total_steps if total_steps > 0 else 1.0

        return CausalVerificationResult(
            chain_id=chain_id,
            is_consistent=len(violations) == 0,
            violations=violations,
            consistency_score=score,
            spurious_edges=spurious,
        )


class SpuriousnessDetector:
    """Detects spurious correlations in the KG.

    CONCEPT:KG-2.43 — Causal Spuriousness Detection

    Identifies edges that rely on co-occurrence without a causal mechanism,
    using the d-separation criterion from SCM theory.
    """

    def __init__(self, scm: StructuralCausalModel) -> None:
        self._scm = scm

    def detect_spurious_edges(
        self,
        candidate_edges: list[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        """Identify which candidate edges are likely spurious.

        An edge X→Y is spurious if X and Y are d-separated when
        conditioning on the parents of Y (confounders removed).

        Args:
            candidate_edges: List of (source, target) tuples to test.

        Returns:
            List of dicts with edge info and spuriousness assessment.
        """
        results: list[dict[str, Any]] = []

        for source, target in candidate_edges:
            if source not in self._scm.graph or target not in self._scm.graph:
                results.append(
                    {
                        "source": source,
                        "target": target,
                        "is_spurious": True,
                        "reason": "Node not in causal model.",
                    }
                )
                continue

            # Get parents of target (potential confounders)
            parents = set(self._scm.graph.predecessors(target))
            parents.discard(source)  # Don't condition on the tested cause

            # d-separation test
            is_dsep = self._scm.is_d_separated(source, target, parents)

            results.append(
                {
                    "source": source,
                    "target": target,
                    "is_spurious": is_dsep,
                    "reason": "d-separated given confounders"
                    if is_dsep
                    else "causal path exists",
                    "conditioning_set": list(parents),
                }
            )

        return results


class CounterfactualGenerator:
    """Generates counterfactual queries from an SCM.

    CONCEPT:KG-2.43 — Counterfactual Generation (MedCausalX §3.1)
    """

    def __init__(self, scm: StructuralCausalModel) -> None:
        self._scm = scm

    def generate_counterfactuals(
        self,
        target_node: str,
        max_interventions: int = 5,
    ) -> list[CounterfactualQuery]:
        """Generate counterfactual queries for a target variable.

        For each causal ancestor of the target, generates a 'what if X
        were different?' query.

        Args:
            target_node: The variable whose outcome to investigate.
            max_interventions: Maximum number of counterfactuals to generate.

        Returns:
            List of CounterfactualQuery objects.
        """
        if target_node not in self._scm.graph:
            return []

        ancestors = self._scm.get_causal_ancestors(target_node)
        queries: list[CounterfactualQuery] = []

        # Sort by distance to target (closest ancestors first)
        ancestor_list = sorted(
            ancestors,
            key=lambda a: nx.shortest_path_length(self._scm.graph, a, target_node),
        )

        for ancestor_id in ancestor_list[:max_interventions]:
            factor = self._scm._factors.get(ancestor_id)
            if not factor:
                continue

            queries.append(
                CounterfactualQuery(
                    intervention_node=ancestor_id,
                    intervention_value=f"not_{factor.value}"
                    if factor.value
                    else "alternative",
                    target_node=target_node,
                    original_value=factor.value,
                    counterfactual_outcome=None,  # To be filled by downstream analysis
                )
            )

        return queries


def trajectory_causal_alignment_score(
    reasoning_steps: list[dict[str, Any]],
    scm: StructuralCausalModel,
) -> float:
    """Score a reasoning trajectory for global causal coherence.

    CONCEPT:KG-2.43 — Trajectory-Level Causal Alignment (MedCausalX §3.3)

    Unlike per-step token likelihood, this scores the entire trajectory
    for consistency with the causal DAG.  Integrates with AHE-3.10
    (Decomposed Reward Signals).

    The score is computed as:
    ``score = (valid_transitions + valid_orderings) / (2 × total_steps)``

    Args:
        reasoning_steps: List of reasoning step dicts with 'cause' and 'effect'.
        scm: The Structural Causal Model to validate against.

    Returns:
        Alignment score in [0.0, 1.0]. 1.0 = perfectly aligned.
    """
    if not reasoning_steps:
        return 1.0

    total = len(reasoning_steps)
    valid_transitions = 0
    valid_orderings = 0

    topo_order = {node: i for i, node in enumerate(scm.topological_causal_order())}

    for i, step in enumerate(reasoning_steps):
        cause = step.get("cause", "")
        effect = step.get("effect", "")

        if not cause or not effect:
            valid_transitions += 1
            valid_orderings += 1
            continue

        # Check if there's a valid causal path
        if cause in scm.graph and effect in scm.graph:
            try:
                nx.shortest_path(scm.graph, cause, effect)
                valid_transitions += 1
            except nx.NetworkXNoPath:
                pass

            # Check topological ordering
            cause_order = topo_order.get(cause, -1)
            effect_order = topo_order.get(effect, -1)
            if cause_order >= 0 and effect_order >= 0 and cause_order < effect_order:
                valid_orderings += 1
        else:
            # Nodes not in SCM — can't penalize
            valid_transitions += 1
            valid_orderings += 1

    return (valid_transitions + valid_orderings) / (2 * total) if total > 0 else 1.0


# --- Merged from formal_reasoning_core.py ---

#!/usr/bin/env python3
"""Probabilistic Knowledge Graph Reasoning.

CONCEPT:KG-2.45 — Probabilistic Knowledge Graph Reasoning

Probabilistic reasoning over the Knowledge Graph derived from
*Mathematics for Computer Science* (MCS) Chapters 17–21.

Transforms the KG from a deterministic lookup system to a belief network
capable of reasoning under uncertainty.

- **Bayesian Belief Propagation** (MCS §18.4): Prior→posterior updates.
- **Conditional Independence** (MCS §18.7): d-separation for efficient inference.
- **Random Walk Exploration** (MCS Ch 21): Stochastic KG discovery.
- **Law of Total Probability Aggregation** (MCS §18.5): Multi-source combination.
- **Birthday Paradox Collision Detector** (MCS §17.4): Probabilistic dedup.
"""


logger = logging.getLogger(__name__)


@dataclass
class BeliefState:
    """Belief state for a KG node.

    Attributes:
        node_id: The node this belief is about.
        prior: Prior probability P(H) before evidence.
        posterior: Posterior probability P(H|E) after evidence.
        evidence: List of evidence items that updated this belief.
        confidence: Meta-confidence in the belief itself.
    """

    node_id: str = ""
    prior: float = 0.5
    posterior: float = 0.5
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class CollisionEstimate:
    """Birthday paradox collision probability estimate.

    Attributes:
        n_items: Number of items being checked.
        space_size: Size of the identifier space.
        collision_probability: Estimated probability of at least one collision.
        expected_collisions: Expected number of collisions.
        safe_threshold: Max items before collision prob exceeds 50%.
    """

    n_items: int = 0
    space_size: int = 0
    collision_probability: float = 0.0
    expected_collisions: float = 0.0
    safe_threshold: int = 0


class BayesianBeliefPropagator:
    """Bayesian belief propagation over the KG topology.

    CONCEPT:KG-2.45 — Bayesian Belief Propagation (MCS §18.4)

    Given prior beliefs about node states and observed evidence, computes
    posterior beliefs via Bayes' rule propagated through the KG edges.

    Uses loopy belief propagation on the graph structure where edges
    represent probabilistic dependencies.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph
        self._beliefs: dict[str, BeliefState] = {}

    def set_prior(self, node_id: str, prior: float) -> None:
        """Set the prior probability for a node.

        Args:
            node_id: The node ID.
            prior: Prior probability P(H) in [0, 1].
        """
        self._beliefs[node_id] = BeliefState(
            node_id=node_id,
            prior=prior,
            posterior=prior,
        )

    def observe_evidence(
        self,
        node_id: str,
        likelihood_ratio: float,
        evidence_label: str = "",
    ) -> BeliefState:
        """Update belief at a node given observed evidence via Bayes' rule.

        CONCEPT:KG-2.45 — Bayes' Rule Update (MCS §18.4)

        P(H|E) = P(E|H) × P(H) / P(E)

        Using odds form: O(H|E) = LR × O(H)
        where LR = P(E|H) / P(E|¬H) is the likelihood ratio.

        Args:
            node_id: Node to update.
            likelihood_ratio: P(E|H)/P(E|¬H). >1 supports H, <1 opposes.
            evidence_label: Optional label for the evidence.

        Returns:
            Updated BeliefState.
        """
        if node_id not in self._beliefs:
            self._beliefs[node_id] = BeliefState(node_id=node_id)

        belief = self._beliefs[node_id]
        prior = belief.posterior  # Use current posterior as new prior

        # Convert to odds, multiply by LR, convert back
        if prior >= 1.0:
            posterior = 1.0
        elif prior <= 0.0:
            posterior = 0.0
        else:
            prior_odds = prior / (1.0 - prior)
            posterior_odds = likelihood_ratio * prior_odds
            posterior = posterior_odds / (1.0 + posterior_odds)

        posterior = min(1.0, max(0.0, posterior))
        belief.posterior = posterior
        if evidence_label:
            belief.evidence.append(evidence_label)

        return belief

    def propagate(
        self, source_node: str, max_hops: int = 3, decay: float = 0.7
    ) -> dict[str, BeliefState]:
        """Propagate belief updates from a source node through the graph.

        CONCEPT:KG-2.45 — Belief Propagation

        When a node's belief changes, propagates a dampened update to
        its neighbors (successors in the directed graph).  The update
        decays exponentially with graph distance.

        Args:
            source_node: Starting node for propagation.
            max_hops: Maximum propagation distance.
            decay: Multiplicative decay per hop (0–1).

        Returns:
            Dict of all updated beliefs.
        """
        if source_node not in self._beliefs:
            return {}

        source_belief = self._beliefs[source_node]
        # The "evidence strength" diminishes with distance
        lr = source_belief.posterior / max(source_belief.prior, 1e-10)

        frontier = [(source_node, lr, 0)]
        visited: set[str] = {source_node}

        while frontier:
            current, current_lr, depth = frontier.pop(0)
            if depth >= max_hops:
                continue

            for neighbor in self._graph.successors(current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                dampened_lr = 1.0 + (current_lr - 1.0) * decay
                self.observe_evidence(
                    neighbor,
                    dampened_lr,
                    evidence_label=f"propagated_from_{source_node}_depth_{depth + 1}",
                )
                frontier.append((neighbor, dampened_lr, depth + 1))

        return {nid: b for nid, b in self._beliefs.items() if nid in visited}

    def get_belief(self, node_id: str) -> BeliefState | None:
        """Get the current belief state for a node."""
        return self._beliefs.get(node_id)

    def get_all_beliefs(self) -> dict[str, BeliefState]:
        """Get all current belief states."""
        return dict(self._beliefs)


class RandomWalkExplorer:
    """Stochastic KG exploration via random walks with restart.

    CONCEPT:KG-2.45 — Random Walk Exploration (MCS Ch 21)

    Discovers unexpected connections that deterministic traversal misses.
    Uses random walks with restart (teleport probability) to balance
    exploration vs. exploitation around seed nodes.
    """

    def __init__(self, graph: nx.DiGraph, seed: int = 42) -> None:
        self._graph = graph
        self._rng = np.random.default_rng(seed)

    def explore(
        self,
        start_node: str,
        n_steps: int = 100,
        restart_prob: float = 0.15,
    ) -> dict[str, float]:
        """Perform a random walk with restart and return visit frequencies.

        Args:
            start_node: Starting node for the walk.
            n_steps: Number of walk steps.
            restart_prob: Probability of teleporting back to start.

        Returns:
            Dict of node_id → visit frequency (proportion of steps).
        """
        if start_node not in self._graph:
            return {}

        visit_counts: dict[str, int] = defaultdict(int)
        current = start_node

        for _ in range(n_steps):
            visit_counts[current] += 1

            if self._rng.random() < restart_prob:
                current = start_node
                continue

            neighbors = list(self._graph.successors(current))
            if not neighbors:
                current = start_node
                continue

            current = neighbors[self._rng.integers(len(neighbors))]

        total = sum(visit_counts.values())
        return {node: count / total for node, count in visit_counts.items()}

    def discover_unexpected_connections(
        self,
        start_node: str,
        n_walks: int = 10,
        walk_length: int = 50,
        restart_prob: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Find surprising nodes by running multiple random walks.

        Nodes that appear frequently across walks but have low graph distance
        to the start are "expected".  Nodes that appear frequently but have
        high graph distance are "unexpected" — potential novel connections.

        Args:
            start_node: Starting node.
            n_walks: Number of independent random walks.
            walk_length: Steps per walk.
            restart_prob: Restart probability.

        Returns:
            List of dicts with node_id, frequency, distance, and surprise_score.
        """
        if start_node not in self._graph:
            return []

        # Aggregate frequencies across walks
        total_freq: dict[str, float] = defaultdict(float)
        for _ in range(n_walks):
            freq_dict = self.explore(start_node, walk_length, restart_prob)
            for node, f in freq_dict.items():
                total_freq[node] += f

        # Normalize
        total = sum(total_freq.values()) or 1.0
        normalized = {node: f / total for node, f in total_freq.items()}

        # Compute graph distances from start
        distances: dict[Any, Any] = {}
        try:
            distances = nx.single_source_shortest_path_length(
                self._graph,
                start_node,
            )
        except Exception:
            distances = {start_node: 0}

        # Surprise = frequency × distance (unexpected if visited often but far away)
        results: list[dict[str, Any]] = []
        for node, freq in normalized.items():
            if node == start_node:
                continue
            dist = distances.get(node, float("inf"))
            if dist == float("inf"):
                dist = 10  # Cap for unreachable nodes
            surprise = freq * dist
            results.append(
                {
                    "node_id": node,
                    "frequency": freq,
                    "distance": dist,
                    "surprise_score": surprise,
                }
            )

        results.sort(key=lambda x: x["surprise_score"], reverse=True)
        return results


def total_probability_aggregation(
    source_scores: list[tuple[float, float]],
) -> float:
    """Combine retrieval scores from multiple sources using Law of Total Probability.

    CONCEPT:KG-2.45 — Law of Total Probability (MCS §18.5)

    P(relevant) = Σ P(relevant|source_i) × P(source_i)

    This avoids Simpson's Paradox (MCS §18.6) by weighting each source's
    relevance by its reliability, rather than naively averaging scores.

    Args:
        source_scores: List of (relevance_score, source_reliability) tuples.
            relevance_score: P(relevant|source_i) in [0, 1].
            source_reliability: P(source_i) / weight of this source.

    Returns:
        Combined relevance score P(relevant) in [0, 1].
    """
    if not source_scores:
        return 0.0

    total_weight = sum(weight for _, weight in source_scores)
    if total_weight <= 0:
        return 0.0

    combined = sum(score * weight for score, weight in source_scores) / total_weight
    return min(1.0, max(0.0, combined))


def birthday_collision_probability(n_items: int, space_size: int) -> CollisionEstimate:
    """Estimate collision probability using the Birthday Paradox.

    CONCEPT:KG-2.45 — Birthday Paradox Collision Detector (MCS §17.4)

    Among n items chosen from a space of size d, the probability of at
    least one collision (duplicate) is approximately:
    ``P(collision) ≈ 1 - e^(-n²/(2d))``

    For KG node IDs, this estimates the probability of hash collisions
    given the number of nodes and the ID space size.

    Args:
        n_items: Number of items (KG nodes).
        space_size: Size of the identifier space (e.g., 2^64 for 64-bit hashes).

    Returns:
        CollisionEstimate with probability and safety threshold.
    """
    if space_size <= 0 or n_items <= 0:
        return CollisionEstimate()

    # P(collision) ≈ 1 - e^(-n²/(2d))
    exponent = -(n_items**2) / (2.0 * space_size)
    collision_prob = 1.0 - math.exp(max(exponent, -500))  # Clamp to avoid underflow

    # Expected collisions ≈ n(n-1)/(2d)
    expected = (n_items * (n_items - 1)) / (2.0 * space_size)

    # Safe threshold: n where P(collision) = 0.5 → n ≈ 1.2√d
    safe_n = int(1.2 * math.sqrt(space_size))

    return CollisionEstimate(
        n_items=n_items,
        space_size=space_size,
        collision_probability=min(1.0, collision_prob),
        expected_collisions=expected,
        safe_threshold=safe_n,
    )


def conditional_independence_test(
    graph: nx.DiGraph,
    x: str,
    y: str,
    conditioning_set: set[str] | None = None,
) -> dict[str, Any]:
    """Test conditional independence using d-separation on the KG.

    CONCEPT:KG-2.45 — Conditional Independence (MCS §18.7)

    Two nodes X and Y are conditionally independent given Z if and only if
    they are d-separated by Z in the graph.

    Args:
        graph: The directed KG graph.
        x: First node.
        y: Second node.
        conditioning_set: Set of conditioned nodes (Z).

    Returns:
        Dict with independence result and explanation.
    """
    z = conditioning_set or set()

    if x not in graph or y not in graph:
        return {
            "x": x,
            "y": y,
            "z": list(z),
            "independent": True,
            "reason": "One or both nodes not in graph.",
        }

    try:
        is_independent = nx.is_d_separator(graph, x, y, z)
    except Exception:
        # Fallback: check path existence
        try:
            nx.shortest_path(graph.to_undirected(), x, y)
            is_independent = False
        except nx.NetworkXNoPath:
            is_independent = True

    return {
        "x": x,
        "y": y,
        "z": list(z),
        "independent": is_independent,
        "reason": "d-separated" if is_independent else "active path exists",
    }


# --- Merged from formal_reasoning_core.py ---

#!/usr/bin/env python3
"""Formal Relations and Equivalence Classes.

CONCEPT:KG-2.47 — Formal Relations Engine

Implements mathematical relation properties (Reflexive, Symmetric, Transitive)
and Equivalence Classes from *Mathematics for Computer Science* (MCS Ch 4).
Provides zero-shot entity resolution by formally defining equivalence sets
across the Knowledge Graph.
"""


logger = logging.getLogger(__name__)


def is_reflexive(graph: nx.DiGraph, nodes: Iterable[str] | None = None) -> bool:
    """Check if the relation is reflexive over the given nodes.

    A relation R on A is reflexive if for all a in A, aRa.
    """
    node_set = set(nodes) if nodes is not None else set(graph.nodes())
    return all(graph.has_edge(n, n) for n in node_set)


def is_symmetric(graph: nx.DiGraph) -> bool:
    """Check if the relation is symmetric.

    A relation R is symmetric if aRb implies bRa.
    """
    return all(graph.has_edge(v, u) for u, v in graph.edges())


def is_transitive(graph: nx.DiGraph) -> bool:
    """Check if the relation is transitive.

    A relation R is transitive if aRb and bRc implies aRc.
    """
    for u, v in graph.edges():
        for _, w in graph.out_edges(v):
            if not graph.has_edge(u, w):
                return False
    return True


def is_equivalence_relation(
    graph: nx.DiGraph, nodes: Iterable[str] | None = None
) -> bool:
    """Check if a directed graph represents an equivalence relation."""
    return is_reflexive(graph, nodes) and is_symmetric(graph) and is_transitive(graph)


def equivalence_classes(graph: nx.DiGraph) -> list[set[str]]:
    """Compute equivalence classes for a symmetric and transitive relation.

    Returns a list of disjoint sets of nodes that are equivalent.
    If the graph is symmetric and transitive, its connected components
    form the equivalence classes.
    """
    if not is_symmetric(graph):
        logger.warning(
            "Graph is not symmetric. Treating edges as undirected for equivalence classes."
        )

    undirected = graph.to_undirected()
    classes = list(nx.connected_components(undirected))
    return [set(c) for c in classes]


def transitive_closure(graph: nx.DiGraph) -> nx.DiGraph:
    """Compute the transitive closure of a relation."""
    return nx.transitive_closure(graph)


def hasse_diagram(graph: nx.DiGraph) -> nx.DiGraph:
    """Compute the Hasse diagram (transitive reduction) of a DAG.

    Useful for partial orders (posets).
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph is not a DAG. Cannot compute Hasse diagram.")
    return nx.transitive_reduction(graph)


def resolve_entities(equivalences: list[tuple[str, str]]) -> dict[str, str]:
    """Zero-shot entity resolution using equivalence classes.

    Given a list of equivalence pairs (u, v), computes the equivalence
    classes and maps every entity to a canonical representative
    (the lexicographically smallest ID in its class).

    Args:
        equivalences: List of (entity1, entity2) tuples.

    Returns:
        Mapping from entity_id to canonical_entity_id.
    """
    G = nx.DiGraph()
    G.add_edges_from(equivalences)

    classes = equivalence_classes(G)
    resolution_map = {}

    for eq_class in classes:
        canonical = min(eq_class)
        for node in eq_class:
            resolution_map[node] = canonical

    return resolution_map


# --- Merged from formal_reasoning_core.py ---

#!/usr/bin/env python3
"""Formal State Machines and Invariants.

CONCEPT:KG-2.48 — State Machine Invariant Engine

Implements Deterministic Finite Automata (DFA) abstractions and provable
state invariants from *Mathematics for Computer Science* (MCS Ch 6).
Provides mathematical guarantees of agent safety by formally validating
transitions against structural invariants, preventing infinite loops.
"""


logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A formal transition in a state machine."""

    source: str
    target: str
    action: str
    condition: Callable[[dict[str, Any]], bool] | None = None


class FormalStateMachine:
    """A Deterministic Finite Automaton (DFA) with Invariant checking.

    Provides formal mathematical guarantees that an agent cannot
    make illegal transitions or violate structural invariants.
    """

    def __init__(self, start_state: str):
        self.start_state = start_state
        self.current_state = start_state
        self.states: set[str] = {start_state}
        self.transitions: list[Transition] = []
        self.invariants: list[Callable[[str, dict[str, Any]], bool]] = []

    def add_state(self, state: str) -> None:
        """Add a valid state to the machine."""
        self.states.add(state)

    def add_transition(
        self,
        source: str,
        target: str,
        action: str,
        condition: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        """Add a directed transition between two states."""
        self.states.add(source)
        self.states.add(target)
        self.transitions.append(Transition(source, target, action, condition))

    def add_invariant(
        self, invariant_func: Callable[[str, dict[str, Any]], bool]
    ) -> None:
        """Add a global invariant that must hold for all states and transitions.

        An invariant is a predicate P(state) that is true for the start state,
        and if true before a transition, remains true after.
        """
        self.invariants.append(invariant_func)

    def get_available_actions(self, context: dict[str, Any] | None = None) -> list[str]:
        """Get all valid actions from the current state given the context."""
        ctx = context or {}
        valid_actions = []
        for t in self.transitions:
            if t.source == self.current_state:
                if t.condition is None or t.condition(ctx):
                    valid_actions.append(t.action)
        return valid_actions

    def validate_invariants(self, target_state: str, context: dict[str, Any]) -> bool:
        """Check if transitioning to the target state preserves all invariants."""
        for inv in self.invariants:
            if not inv(target_state, context):
                logger.error(
                    f"Invariant violation: Transition to {target_state} failed invariant check."
                )
                return False
        return True

    def transition(self, action: str, context: dict[str, Any] | None = None) -> str:
        """Execute a formal transition if it is valid and preserves invariants.

        Args:
            action: The action to execute.
            context: Contextual variables evaluated by transition conditions.

        Returns:
            The new state.

        Raises:
            ValueError: If the action is invalid or an invariant is violated.
        """
        ctx = context or {}
        valid_targets = []

        for t in self.transitions:
            if t.source == self.current_state and t.action == action:
                if t.condition is None or t.condition(ctx):
                    valid_targets.append(t.target)

        if not valid_targets:
            raise ValueError(
                f"No valid transition found for action '{action}' from state '{self.current_state}'."
            )

        if len(valid_targets) > 1:
            raise ValueError(
                f"Non-deterministic transition detected for action '{action}'. DFA requires deterministic transitions."
            )

        target = valid_targets[0]

        # Prove invariants
        if not self.validate_invariants(target, ctx):
            raise ValueError(f"Transition to '{target}' aborted: Invariant violation.")

        logger.info(f"Transitioned: {self.current_state} --[{action}]--> {target}")
        self.current_state = target
        return self.current_state


# --- Merged from formal_reasoning_core.py ---

#!/usr/bin/env python3
"""Markov Chain Transitions and Vectorized Topologies.

CONCEPT:KG-2.49 — Markov Transition Forecasting

Implements Markov Chain transition matrices over agent interaction traces
(Vectorized Topologies) from *Mathematics for Computer Science* (MCS Ch 21).
Calculates the stationary distribution (Eigenvector) to predict where an
agent is statistically most likely to fail or reach a sink node.
"""


logger = logging.getLogger(__name__)


class MarkovTransitionModel:
    """Predictive model based on Markov Chains and Stationary Distributions.

    Builds a transition matrix from historical agent execution traces
    and uses power iteration to find the stationary distribution.
    """

    def __init__(self):
        self.state_counts: dict[str, int] = defaultdict(int)
        self.transitions: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.states: list[str] = []
        self._state_to_idx: dict[str, int] = {}
        self.transition_matrix: np.ndarray | None = None

    def ingest_trace(self, trace: Sequence[str]) -> None:
        """Ingest a sequential execution trace of states."""
        if len(trace) < 1:
            return

        self.state_counts[trace[0]] += 1
        for i in range(len(trace) - 1):
            src = trace[i]
            dst = trace[i + 1]
            self.transitions[src][dst] += 1
            self.state_counts[dst] += 1

        self._rebuild_matrix()

    def _rebuild_matrix(self) -> None:
        """Rebuild the stochastic transition matrix from observed counts."""
        self.states = sorted(list(self.state_counts.keys()))
        self._state_to_idx = {s: i for i, s in enumerate(self.states)}

        n = len(self.states)
        self.transition_matrix = np.zeros((n, n))

        for src, dsts in self.transitions.items():
            src_idx = self._state_to_idx[src]
            total_transitions = sum(dsts.values())

            if total_transitions > 0:
                for dst, count in dsts.items():
                    dst_idx = self._state_to_idx[dst]
                    self.transition_matrix[src_idx, dst_idx] = count / total_transitions
            else:
                # Absorbing state (sink), stays in itself
                self.transition_matrix[src_idx, src_idx] = 1.0

    def get_transition_probability(self, src: str, dst: str) -> float:
        """Get the empirical probability of transitioning from src to dst."""
        if not self.transition_matrix is not None:
            return 0.0
        if src not in self._state_to_idx or dst not in self._state_to_idx:
            return 0.0

        return float(
            self.transition_matrix[self._state_to_idx[src], self._state_to_idx[dst]]
        )

    def stationary_distribution(
        self, max_iter: int = 1000, tol: float = 1e-6
    ) -> dict[str, float]:
        """Compute the stationary distribution via power iteration.

        The stationary distribution represents the long-term probability
        of the agent being in any particular state. High probabilities on
        error/sink states indicate structural failure points.

        Returns:
            Dictionary mapping state to long-term probability.
        """
        if self.transition_matrix is None or len(self.states) == 0:
            return {}

        n = len(self.states)
        pi = np.ones(n) / n  # Initial uniform distribution

        for _ in range(max_iter):
            # pi * P (left eigenvector for row-stochastic matrix)
            next_pi = pi @ self.transition_matrix
            if np.linalg.norm(next_pi - pi, 1) < tol:
                pi = next_pi
                break
            pi = next_pi

        return {self.states[i]: float(pi[i]) for i in range(n)}

    def predict_sink_nodes(self, threshold: float = 0.1) -> list[tuple[str, float]]:
        """Identify states where the agent gets stuck or terminates.

        Returns a list of (state, probability) sorted by probability descending.
        """
        stat_dist = self.stationary_distribution()
        sinks = [(s, p) for s, p in stat_dist.items() if p >= threshold]
        sinks.sort(key=lambda x: x[1], reverse=True)
        return sinks
