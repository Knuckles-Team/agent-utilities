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

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import networkx as nx

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
