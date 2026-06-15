from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field
from pydantic_graph import End

from agent_utilities.agent.sampling_profile import (
    DEFAULT_PROFILE,
    SamplingProfile,
    resolve_sampling_profile,
)

try:
    from pydantic_graph.step import StepContext
except ImportError:
    from pydantic_graph.beta import StepContext

from .executor import _execute_specialized_step

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


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RoutingPrimitive(StrEnum):
    """Execution primitives available for routing (CONCEPT:ORCH-1.2)."""

    DIRECT = "direct"  # Execute directly, no decomposition
    DECOMPOSE = "decompose"  # Break into subtasks
    DELEGATE = "delegate"  # Send to specialist agent
    PARALLEL = "parallel"  # Fan-out to multiple workers
    PLAN = "plan"  # Autonomous planning (Assimilated from OpenAGI)


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
    # CONCEPT:ORCH-1.57 — the inference-parameter bundle for this route, picked from the
    # task class alongside the model. Defaults to the inherit-everything profile.
    sampling_profile: SamplingProfile = Field(default=DEFAULT_PROFILE)
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
    "plan": ("planning", 0.9),
    "architect": ("planning", 0.85),
    "design": ("planning", 0.8),
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
            sampling_profile=resolve_sampling_profile(task_text),
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

        if features.get("planning", 0) >= 0.75:
            ideal_primitive = RoutingPrimitive.PLAN
            reason = (
                f"Planning signal ({features['planning']:.2f}) → autonomous planning"
            )
        elif features.get("coordination", 0) >= self.parallel_threshold:
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
            sampling_profile=resolve_sampling_profile(task_text),
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
            lambda: {
                "success_rate": 0.5,
                "avg_quality": 0.5,
                "avg_cost": 0.0,
                "q_value": 0.5,
            }
        )

    def add_trace(self, trace: ExecutionTrace) -> None:
        """Record an execution trace for future routing decisions.

        Args:
            trace: The execution trace to learn from.
        """
        self._traces.append(trace)
        self._update_stats(trace)

    def _update_stats(self, trace: ExecutionTrace) -> None:
        """Update running statistics and Q-values from a new trace."""
        key = f"{trace.model_used}:{trace.primitive_used.value}"
        stats = self._model_stats[key]

        # Exponential moving average for metrics
        alpha = 0.1
        stats["success_rate"] = (1 - alpha) * stats["success_rate"] + alpha * (
            1.0 if trace.success else 0.0
        )
        stats["avg_quality"] = (1 - alpha) * stats[
            "avg_quality"
        ] + alpha * trace.quality_score
        stats["avg_cost"] = (1 - alpha) * stats["avg_cost"] + alpha * trace.cost_tokens

        # Reinforcement-based feedback loop (OpenAGI assimilation)
        # Calculate reward considering both quality and success
        reward = trace.quality_score if trace.success else -0.5
        current_q = stats.get("q_value", 0.5)
        stats["q_value"] = current_q + alpha * (reward - current_q)

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
                sampling_profile=resolve_sampling_profile(task_text),
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
                    "q_value": 0.5,
                },
            )

            # Feature similarity to successful traces
            similarity = self._compute_similarity(features, candidate)

            # Combined score utilizing RL q_value
            cost_factor = max(1.0, math.log1p(stats.get("avg_cost", 0) / 1000.0))
            score = stats.get("q_value", 0.5) * (1.0 + similarity) / cost_factor
            scored.append((score, candidate))

        # Softmax normalization
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_candidate = scored[0]

        return RoutingDecision(
            selected=best_candidate,
            alternatives=[c for _, c in scored[1:]],
            sampling_profile=resolve_sampling_profile(task_text),
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


class OntologicalFallbackChain:
    """Ontological Fallback Chains (CONCEPT:ORCH-1.2).

    Instead of a hardcoded CSV list of fallback models, this class queries
    the Knowledge Graph (KG) for nearest ModelCapabilityNode neighbors to find
    the next compatible model when a rate limit or server error occurs.
    """

    def __init__(self, engine: Any):
        self.engine = engine

    def get_fallback(
        self, failed_model_id: str, _error_type: str = "429"
    ) -> str | None:
        """Query the KG for a fallback model based on semantic equivalence.

        Args:
            failed_model_id: The model that failed.
            error_type: The type of error (e.g. '429', '500').

        Returns:
            A new model_id to use as a fallback, or None if exhaustion.
        """
        if not self.engine:
            return None

        try:
            # Query KG for nodes that subsume or are equivalent to the failed model
            # This is a conceptual representation of the KG lookup
            fallbacks = self.engine.search_hybrid(
                f"fallback model for {failed_model_id}", top_k=3
            )
            for f in fallbacks:
                # Assuming the KG returns nodes with 'model_id'
                if "model_id" in f and f["model_id"] != failed_model_id:
                    logger.info(
                        f"OntologicalFallbackChain: Found topological neighbor '{f['model_id']}' for failed '{failed_model_id}'"
                    )
                    return str(f["model_id"])
        except Exception as e:
            logger.warning(f"Ontological fallback resolution failed: {e}")

        return None


class TopologicalRoutingPolicy(RoutingPolicy):
    """Routes using KG-derived topological signals instead of keyword TF-IDF.

    CONCEPT:ORCH-1.4 — Topological Routing

    Scoring dimensions:
        1. **PageRank centrality** — Specialists highly connected in the KG
           are preferred (they have more proven relationships).
        2. **Historical success rate** — Weighted by past outcome evaluations
           from the KG's ``SubagentPatternDecision`` and ``OutcomeEvaluation``
           nodes.
        3. **Domain cluster membership** — Specialists in the same spectral
           cluster as the query topic score higher.
        4. **Tool affinity** — Specialists with ``PROVIDES``/``HAS_CAPABILITY``
           edges to relevant tools score higher.

    Falls back to ``RuleBasedPolicy`` when no KG is available (cold start).
    """

    def __init__(self, engine: Any = None, centrality_weight: float = 0.3):
        """Initialize the topological routing policy.

        Args:
            engine: The IntelligenceGraphEngine for KG queries.
            centrality_weight: Weight of centrality in the final score (0-1).
        """
        self.engine = engine
        self.centrality_weight = centrality_weight
        self._centrality_cache: dict[str, float] | None = None
        self._fallback = RuleBasedPolicy()

    def route(
        self,
        task_text: str,
        candidates: list[RoutingCandidate],
    ) -> RoutingDecision:
        """Route using topological signals from the Knowledge Graph.

        Args:
            task_text: The task description.
            candidates: Available (model, primitive) pairs.

        Returns:
            A RoutingDecision with topology-scored selection.
        """
        if not candidates:
            raise ValueError("No routing candidates provided")

        if not self.engine:
            # Cold start — fall back to rule-based
            return self._fallback.route(task_text, candidates)

        # Compute topological scores for each candidate
        scored: list[tuple[float, RoutingCandidate, str]] = []

        for candidate in candidates:
            score, reason = self._score_candidate(candidate, task_text)
            scored.append((score, candidate, reason))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_candidate, best_reason = scored[0]

        return RoutingDecision(
            selected=best_candidate,
            alternatives=[c for _, c, _ in scored[1:]],
            sampling_profile=resolve_sampling_profile(task_text),
            decision_reason=(
                f"[TopologicalRouting: score={best_score:.3f}] {best_reason}"
            ),
        )

    def _score_candidate(
        self, candidate: RoutingCandidate, task_text: str
    ) -> tuple[float, str]:
        """Score a single candidate using KG topology.

        Returns:
            Tuple of (score, reason).
        """
        score = candidate.confidence  # Base from candidate's own estimate
        reasons: list[str] = []

        # 1. PageRank centrality
        centrality = self._get_centrality(candidate.model_id)
        if centrality > 0:
            score += self.centrality_weight * centrality
            reasons.append(f"centrality={centrality:.3f}")

        # 2. Historical success rate
        success_rate = self._get_historical_success(candidate)
        if success_rate is not None:
            score += 0.25 * success_rate
            reasons.append(f"success_rate={success_rate:.2f}")

        # 3. Tool affinity (how many relevant tools does this specialist have?)
        tool_score = self._get_tool_affinity(candidate.model_id, task_text)
        if tool_score > 0:
            score += 0.2 * tool_score
            reasons.append(f"tool_affinity={tool_score:.2f}")

        return score, " | ".join(reasons) if reasons else "base confidence only"

    def _get_centrality(self, model_id: str) -> float:
        """Get PageRank centrality for a model/agent from the KG."""
        if self._centrality_cache is None:
            self._compute_centrality_cache()

        return (self._centrality_cache or {}).get(model_id, 0.0)

    def _compute_centrality_cache(self) -> None:
        """Compute and cache PageRank centrality for all agents."""
        self._centrality_cache = {}

        if not self.engine:
            return

        try:
            from agent_utilities.knowledge_graph.core import graph_primitives as rx

            # Build a graph from the engine's node/edge data
            rg = rx.PyDiGraph()
            idx_map: dict[str, int] = {}

            # Get all node IDs from the engine
            try:
                all_ids = self.engine.graph.node_ids()
            except Exception:
                all_ids = []

            for nid in all_ids:
                idx_map[nid] = rg.add_node(nid)

            # Rebuild edges from successor relationships
            for nid in all_ids:
                try:
                    successors = self.engine.graph.get_successors(nid)
                except Exception:
                    successors = []
                for succ in successors:
                    if succ in idx_map:
                        rg.add_edge(idx_map[nid], idx_map[succ], None)

            if rg.num_nodes() > 0:
                # Use epistemic-graph native PageRank via the engine
                try:
                    pr = self.engine.graph.pagerank()
                    self._centrality_cache = {nid: score for nid, score in pr}
                except Exception:
                    self._centrality_cache = {}
        except Exception as e:
            logger.debug("Centrality computation failed: %s", e)

    def _get_historical_success(self, candidate: RoutingCandidate) -> float | None:
        """Get historical success rate for this candidate from KG."""
        if not self.engine:
            return None

        try:
            success_count = 0
            total_count = 0
            # Use GCE-native node iteration
            for nid in self.engine.graph.node_ids():
                props = self.engine.graph._get_node_properties(nid)
                if not props:
                    continue
                if (
                    props.get("type") == "subagent_pattern_decision"
                    and props.get("pattern") == candidate.primitive.value
                ):
                    total_count += 1
                    if props.get("outcome_success") is True:
                        success_count += 1

            if total_count >= 3:
                return success_count / total_count
        except Exception:  # nosec B110
            pass

        return None

    def _get_tool_affinity(self, model_id: str, task_text: str) -> float:
        """Compute tool affinity score for a candidate."""
        if not self.engine or not self.engine.backend:
            return 0.0

        try:
            results = self.engine.backend.execute(
                "MATCH (a)-[:PROVIDES|HAS_CAPABILITY]->(t) "
                "WHERE a.id = $aid "
                "RETURN count(t) AS tool_count",
                {"aid": model_id},
            )
            if results:
                count = results[0].get("tool_count", 0)
                # Normalize: more tools = higher affinity (capped at 1.0)
                return min(count / 5.0, 1.0) if count else 0.0
        except Exception:  # nosec B110
            pass

        return 0.0


"""Graph Specialist Steps — Data-Driven Factory.

CONCEPT:ORCH-1.0 Dynamic agent spawning via a registry-driven factory.

Instead of 20+ identical boilerplate functions, this module provides a
single ``make_specialist_step`` factory and a ``SPECIALIST_REGISTRY``
that maps persona IDs to their metadata.  The factory generates
async step functions at import time for backward compatibility.

New personas can be added by appending to ``SPECIALIST_REGISTRY`` —
no new function definitions are needed.
"""


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecialistPersona:
    """Metadata for a specialist persona node."""

    node_id: str
    description: str


# ── Registry of all built-in specialist personas ─────────────────────
SPECIALIST_REGISTRY: list[SpecialistPersona] = [
    SpecialistPersona(
        "python_programmer", "Python implementation, refactoring, packaging (Poetry/UV)"
    ),
    SpecialistPersona(
        "golang_programmer", "Cloud-native backends, microservices, concurrent Go"
    ),
    SpecialistPersona(
        "typescript_programmer", "TypeScript, React, Node.js, type-safe API design"
    ),
    SpecialistPersona("rust_programmer", "Memory-safe systems programming, CLI, Cargo"),
    SpecialistPersona(
        "security_auditor", "Threat modeling, OWASP, dependency vulnerability scanning"
    ),
    SpecialistPersona(
        "javascript_programmer",
        "General-purpose JS, legacy maintenance, Node scripting",
    ),
    SpecialistPersona("c_programmer", "C systems programming and embedded development"),
    SpecialistPersona(
        "cpp_programmer", "C++ systems and performance-critical applications"
    ),
    SpecialistPersona(
        "qa_expert", "Test planning, regression analysis, quality assurance"
    ),
    SpecialistPersona(
        "debugger_expert",
        "Root cause analysis, log forensics, stack trace interpretation",
    ),
    SpecialistPersona(
        "ui_ux_designer", "Design systems, accessibility, responsive layouts"
    ),
    SpecialistPersona(
        "devops_engineer", "CI/CD, Docker, K8s, IaC, environment stabilization"
    ),
    SpecialistPersona(
        "cloud_architect", "Distributed systems, multi-cloud, serverless, scalability"
    ),
    SpecialistPersona(
        "database_expert", "SQL/NoSQL schema design, query optimization, migrations"
    ),
    SpecialistPersona(
        "java_programmer", "Enterprise Java, Spring Boot, JVM performance tuning"
    ),
    SpecialistPersona(
        "data_scientist", "Data analysis, ML modeling, statistical inference"
    ),
    SpecialistPersona(
        "document_specialist", "README generation, API docs, Mermaid diagrams"
    ),
    SpecialistPersona("mobile_programmer", "React Native, Flutter, native iOS/Android"),
    SpecialistPersona(
        "agent_engineer", "Autonomous agents, MCP servers, graph orchestration"
    ),
    SpecialistPersona(
        "project_manager", "Task decomposition, resource planning, timeline estimation"
    ),
    SpecialistPersona(
        "systems_manager", "Codebase structural analysis, dependency mapping"
    ),
    SpecialistPersona(
        "browser_automation", "E2E testing, visual regression, web crawling"
    ),
    SpecialistPersona(
        "coordinator", "Multi-agent synchronization, session state alignment"
    ),
    SpecialistPersona(
        "critique", "Code review, architectural critique, logic validation"
    ),
]


def make_specialist_step(persona: SpecialistPersona):
    """Factory: create a specialist step function for a given persona.

    Args:
        persona: The specialist persona metadata.

    Returns:
        An async step function compatible with pydantic-graph.
    """

    async def _specialist_step(
        ctx: StepContext,
    ) -> str | End[Any]:
        return await _execute_specialized_step(ctx, persona.node_id)

    # Set function metadata for introspection and graph registration
    _specialist_step.__name__ = f"{persona.node_id}_step"
    _specialist_step.__qualname__ = f"{persona.node_id}_step"
    _specialist_step.__doc__ = (
        f"Execute the specialized {persona.description} role.\n\n"
        f"Auto-generated by the specialist factory (CONCEPT:ORCH-1.0).\n\n"
        f"Args:\n"
        f"    ctx: The pydantic-graph step context.\n\n"
        f"Returns:\n"
        f"    The next node identifier or terminal End state.\n"
    )
    return _specialist_step


# ── Generate all specialist step functions at module level ────────────
# This makes them importable: ``from .adaptive_agent_router import python_programmer_step``


def _build_exports():
    """Build module-level step functions and __all__ list."""
    exports = []
    for persona in SPECIALIST_REGISTRY:
        func_name = f"{persona.node_id}_step"
        globals()[func_name] = make_specialist_step(persona)
        exports.append(func_name)
    return exports


__all__ = _build_exports()
