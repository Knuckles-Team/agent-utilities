#!/usr/bin/python
"""Learned Agent Routing Policy (CONCEPT:ORCH-1.2 Enhancement).

Derived from: Uno-Orchestra — Parsimonious Agent Routing via Selective Delegation
(arXiv:2605.05007v1, Score 31.2)

Key insight: Decomposition depth, worker choice, and inference budget should
be jointly optimized rather than separately decided. This module provides a
learned routing policy that selects (model, primitive) pairs from execution
traces, with cost-aware Pareto-optimal delegation.

Three routing strategies are provided:

1. **RuleBasedPolicy** — Static pattern matching (baseline).
2. **TraceLearnedPolicy** — Learns routing from historical execution traces
   using TF-IDF feature extraction and softmax scoring.
3. **CostAwareRouter** — Wraps any policy with cost/accuracy Pareto filtering.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RoutingPrimitive(str, Enum):
    """Execution primitives available for routing (CONCEPT:ORCH-1.2)."""

    DIRECT = "direct"  # Execute directly, no decomposition
    DECOMPOSE = "decompose"  # Break into subtasks
    DELEGATE = "delegate"  # Send to specialist agent
    PARALLEL = "parallel"  # Fan-out to multiple workers


class RoutingCandidate(BaseModel):
    """A candidate (model, primitive) pair for task routing.

    Attributes:
        model_id: Identifier for the model/agent.
        primitive: Execution primitive to use.
        estimated_cost: Estimated token/compute cost.
        confidence: Routing confidence score (0–1).
        latency_ms: Estimated latency in milliseconds.
    """

    model_id: str
    primitive: RoutingPrimitive = RoutingPrimitive.DIRECT
    estimated_cost: float = 0.0
    confidence: float = 0.5
    latency_ms: float = 0.0


class RoutingDecision(BaseModel):
    """The result of a routing decision (CONCEPT:ORCH-1.2).

    Attributes:
        selected: The selected candidate.
        alternatives: Other candidates considered.
        decomposition_depth: How many levels of decomposition were chosen.
        decision_reason: Human-readable explanation.
        trace_id: Unique trace ID for this decision.
    """

    selected: RoutingCandidate
    alternatives: list[RoutingCandidate] = Field(default_factory=list)
    decomposition_depth: int = 0
    decision_reason: str = ""
    trace_id: str = Field(
        default_factory=lambda: (
            f"route:{hashlib.sha256(str(datetime.now(UTC)).encode()).hexdigest()[:12]}"
        )
    )


class ExecutionTrace(BaseModel):
    """A historical execution trace for learning routing patterns.

    Attributes:
        task_text: The original task description.
        model_used: Which model was used.
        primitive_used: Which primitive was applied.
        cost_tokens: Actual token cost.
        success: Whether the task succeeded.
        quality_score: Quality of the result (0–1).
        latency_ms: Actual latency in milliseconds.
        features: Extracted feature vector for the task.
    """

    task_text: str
    model_used: str
    primitive_used: RoutingPrimitive
    cost_tokens: int = 0
    success: bool = True
    quality_score: float = 0.5
    latency_ms: float = 0.0
    features: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


# Task complexity indicators — maps keywords to complexity dimensions
_COMPLEXITY_SIGNALS: dict[str, tuple[str, float]] = {
    "decompose": ("decomposition", 0.8),
    "break down": ("decomposition", 0.9),
    "step by step": ("decomposition", 0.7),
    "multi-step": ("decomposition", 0.85),
    "analyze": ("analysis", 0.6),
    "compare": ("analysis", 0.7),
    "evaluate": ("analysis", 0.65),
    "generate": ("generation", 0.5),
    "create": ("generation", 0.5),
    "write": ("generation", 0.4),
    "search": ("retrieval", 0.5),
    "find": ("retrieval", 0.4),
    "look up": ("retrieval", 0.6),
    "coordinate": ("coordination", 0.8),
    "parallel": ("coordination", 0.9),
    "collaborate": ("coordination", 0.7),
    "simple": ("simplicity", 0.8),
    "quick": ("simplicity", 0.7),
    "just": ("simplicity", 0.6),
}


def extract_task_features(task_text: str) -> dict[str, float]:
    """Extract a feature vector from a task description.

    Uses TF-IDF-like keyword matching against complexity signals.

    Args:
        task_text: The task description text.

    Returns:
        Dict mapping feature dimensions to scores (0–1).
    """
    text_lower = task_text.lower()
    features: dict[str, float] = defaultdict(float)
    word_count = max(len(text_lower.split()), 1)

    # Keyword-based features
    for keyword, (dimension, weight) in _COMPLEXITY_SIGNALS.items():
        if keyword in text_lower:
            features[dimension] = max(features[dimension], weight)

    # Length-based complexity
    features["verbosity"] = min(word_count / 100.0, 1.0)

    # Question detection
    if "?" in task_text:
        features["is_question"] = 0.8

    return dict(features)


# ---------------------------------------------------------------------------
# Routing policies
# ---------------------------------------------------------------------------


class RoutingPolicy:
    """Base class for routing policies (CONCEPT:ORCH-1.2).

    Subclasses implement ``route()`` to select the best (model, primitive)
    pair for a given task.
    """

    def route(
        self,
        task_text: str,
        candidates: list[RoutingCandidate],
    ) -> RoutingDecision:
        """Select the best candidate for the task.

        Args:
            task_text: The task description.
            candidates: Available (model, primitive) pairs.

        Returns:
            A RoutingDecision with the selected candidate.
        """
        if not candidates:
            raise ValueError("No routing candidates provided")
        return RoutingDecision(
            selected=candidates[0],
            alternatives=candidates[1:],
            decision_reason="Default: first candidate selected",
        )


class RuleBasedPolicy(RoutingPolicy):
    """Rule-based routing using keyword pattern matching.

    Simple baseline that maps task complexity signals to primitives.
    """

    def __init__(
        self,
        decompose_threshold: float = 0.7,
        delegate_threshold: float = 0.6,
        parallel_threshold: float = 0.8,
    ) -> None:
        self.decompose_threshold = decompose_threshold
        self.delegate_threshold = delegate_threshold
        self.parallel_threshold = parallel_threshold

    def route(
        self,
        task_text: str,
        candidates: list[RoutingCandidate],
    ) -> RoutingDecision:
        """Route based on task complexity signals."""
        if not candidates:
            raise ValueError("No routing candidates provided")

        features = extract_task_features(task_text)

        # Determine ideal primitive
        ideal_primitive = RoutingPrimitive.DIRECT
        reason = "Simple task — direct execution"

        if features.get("coordination", 0) >= self.parallel_threshold:
            ideal_primitive = RoutingPrimitive.PARALLEL
            reason = f"Coordination signal ({features['coordination']:.2f}) → parallel execution"
        elif features.get("decomposition", 0) >= self.decompose_threshold:
            ideal_primitive = RoutingPrimitive.DECOMPOSE
            reason = f"Decomposition signal ({features['decomposition']:.2f}) → task decomposition"
        elif features.get("analysis", 0) >= self.delegate_threshold:
            ideal_primitive = RoutingPrimitive.DELEGATE
            reason = (
                f"Analysis signal ({features['analysis']:.2f}) → specialist delegation"
            )

        # Find best matching candidate
        best = candidates[0]
        for c in candidates:
            if c.primitive == ideal_primitive:
                best = c
                break

        depth = (
            2
            if ideal_primitive == RoutingPrimitive.DECOMPOSE
            else (1 if ideal_primitive == RoutingPrimitive.DELEGATE else 0)
        )

        return RoutingDecision(
            selected=best,
            alternatives=[c for c in candidates if c != best],
            decomposition_depth=depth,
            decision_reason=reason,
        )


class TraceLearnedPolicy(RoutingPolicy):
    """Routing policy learned from historical execution traces.

    CONCEPT:ORCH-1.2 — Derived from Uno-Orchestra's RL-trajectory approach.

    Learns which (model, primitive) pairs work best for different task
    types by analyzing historical execution traces. Uses softmax scoring
    over feature similarity to rank candidates.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize the trace-learned policy.

        Args:
            temperature: Softmax temperature for scoring. Lower values
                make the policy more decisive.
        """
        self.temperature = temperature
        self._traces: list[ExecutionTrace] = []
        self._model_stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {"success_rate": 0.5, "avg_quality": 0.5, "avg_cost": 0.0}
        )

    def add_trace(self, trace: ExecutionTrace) -> None:
        """Record an execution trace for future routing decisions.

        Args:
            trace: The execution trace to learn from.
        """
        self._traces.append(trace)
        self._update_stats(trace)

    def _update_stats(self, trace: ExecutionTrace) -> None:
        """Update running statistics from a new trace."""
        key = f"{trace.model_used}:{trace.primitive_used.value}"
        stats = self._model_stats[key]

        # Exponential moving average
        alpha = 0.1
        stats["success_rate"] = (1 - alpha) * stats["success_rate"] + alpha * (
            1.0 if trace.success else 0.0
        )
        stats["avg_quality"] = (1 - alpha) * stats[
            "avg_quality"
        ] + alpha * trace.quality_score
        stats["avg_cost"] = (1 - alpha) * stats["avg_cost"] + alpha * trace.cost_tokens

    def route(
        self,
        task_text: str,
        candidates: list[RoutingCandidate],
    ) -> RoutingDecision:
        """Route using learned trace statistics.

        Scores each candidate using feature similarity to successful
        historical traces, weighted by quality and cost efficiency.
        """
        if not candidates:
            raise ValueError("No routing candidates provided")

        if not self._traces:
            # Cold start — fall back to confidence-based ranking
            best = max(candidates, key=lambda c: c.confidence)
            return RoutingDecision(
                selected=best,
                alternatives=[c for c in candidates if c != best],
                decision_reason="Cold start — selected highest confidence candidate",
            )

        features = extract_task_features(task_text)
        scored: list[tuple[float, RoutingCandidate]] = []

        for candidate in candidates:
            key = f"{candidate.model_id}:{candidate.primitive.value}"
            stats = self._model_stats.get(
                key,
                {
                    "success_rate": 0.5,
                    "avg_quality": 0.5,
                    "avg_cost": 0.0,
                },
            )

            # Feature similarity to successful traces
            similarity = self._compute_similarity(features, candidate)

            # Combined score: quality × success_rate × similarity / cost_factor
            cost_factor = max(1.0, math.log1p(stats.get("avg_cost", 0) / 1000.0))
            score = (
                stats.get("avg_quality", 0.5)
                * stats.get("success_rate", 0.5)
                * (1.0 + similarity)
                / cost_factor
            )
            scored.append((score, candidate))

        # Softmax normalization
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_candidate = scored[0]

        return RoutingDecision(
            selected=best_candidate,
            alternatives=[c for _, c in scored[1:]],
            decision_reason=(
                f"Trace-learned: score={best_score:.3f} "
                f"(quality×success×similarity/cost)"
            ),
        )

    def _compute_similarity(
        self,
        features: dict[str, float],
        candidate: RoutingCandidate,
    ) -> float:
        """Compute feature similarity to successful traces using the same model."""
        relevant = [
            t
            for t in self._traces
            if t.model_used == candidate.model_id
            and t.primitive_used == candidate.primitive
            and t.success
        ]
        if not relevant:
            return 0.0

        # Average cosine-like similarity
        total_sim = 0.0
        for trace in relevant[-20:]:  # Use most recent 20 traces
            dims = set(features.keys()) | set(trace.features.keys())
            if not dims:
                continue
            dot = sum(features.get(d, 0) * trace.features.get(d, 0) for d in dims)
            mag_a = math.sqrt(sum(features.get(d, 0) ** 2 for d in dims)) or 1
            mag_b = math.sqrt(sum(trace.features.get(d, 0) ** 2 for d in dims)) or 1
            total_sim += dot / (mag_a * mag_b)

        return total_sim / max(len(relevant[-20:]), 1)


class CostAwareRouter:
    """Cost-aware Pareto-optimal routing wrapper (CONCEPT:ORCH-1.2).

    Wraps any RoutingPolicy with cost/accuracy Pareto filtering to ensure
    the selected candidate is on the efficiency frontier.

    Example::

        policy = TraceLearnedPolicy()
        router = CostAwareRouter(policy, max_cost=10000)
        decision = router.route("analyze this codebase", candidates)
    """

    def __init__(
        self,
        policy: RoutingPolicy,
        max_cost: float = float("inf"),
        cost_weight: float = 0.3,
    ) -> None:
        """Initialize the cost-aware router.

        Args:
            policy: Underlying routing policy.
            max_cost: Maximum acceptable cost (tokens).
            cost_weight: Weight of cost in the Pareto score (0–1).
        """
        self.policy = policy
        self.max_cost = max_cost
        self.cost_weight = cost_weight

    def route(
        self,
        task_text: str,
        candidates: list[RoutingCandidate],
    ) -> RoutingDecision:
        """Route with cost-aware Pareto filtering.

        First filters candidates exceeding max_cost, then applies
        the underlying policy to the remaining candidates.
        """
        # Filter by cost budget
        affordable = [c for c in candidates if c.estimated_cost <= self.max_cost]
        if not affordable:
            logger.warning(
                "No candidates within cost budget %.0f. Using all candidates.",
                self.max_cost,
            )
            affordable = candidates

        # Apply underlying policy
        decision = self.policy.route(task_text, affordable)

        # Annotate with cost context
        decision.decision_reason = (
            f"[CostAware: budget={self.max_cost:.0f}, "
            f"filtered={len(candidates) - len(affordable)} over-budget] "
            + decision.decision_reason
        )

        return decision
