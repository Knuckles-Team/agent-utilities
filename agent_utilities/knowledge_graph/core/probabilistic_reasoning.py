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

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

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
