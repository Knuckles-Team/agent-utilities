"""CONCEPT:AU-ORCH.execution.coordination-protocol-metadata — Coordination Protocol Layer.

Explicit coordination layer for multi-agent orchestration.

Research: 2605.03310v1 — Coordination as an Architectural Layer
for LLM-Based Multi-Agent Systems.

Key insight from the paper: *"Coordination in LLM-based multi-agent
systems should be treated as an explicit architectural layer"* rather
than being implicit in agent-to-agent communication. This module
implements that principle by providing pluggable coordination protocols
that sit between the ``AgentOrchestrationEngine`` and the actual
graph execution.

Architecture:
    - **CoordinationProtocol**: Declarative protocol definition
      (consensus, voting, delegation, handoff).
    - **CoordinationLayer**: Selects and applies protocols based on
      task type, agent count, and historical success rates.
    - **CoordinationTrace**: KG-persisted record of coordination events
      for observability and learning.

See docs/pillars/1_graph_orchestration.md §CONCEPT:AU-ORCH.execution.coordination-protocol-metadata
"""

from __future__ import annotations

import logging
import statistics
import time
import uuid
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


# ── Named aggregation registry (CONCEPT:AU-ORCH.execution.coordination-protocol-metadata) ──────────────────
# The coordination layer's named aggregation taxonomy (b1-02 F5). Scalar
# reductions live here; pairwise/uncertainty operators (Bradley–Terry,
# conservative-rating, contribution-weighted) are delegated to the unified
# selection registry (``harness.selection_operators``) so there is ONE place the
# whole stack picks/aggregates candidates (STRATEGY synergy #2).


class AggregationOperator(StrEnum):
    """Named scalar aggregation operators for coordinated outputs."""

    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    MIN = "min"
    LOG_POOL = (
        "log_pool"  # logarithmic opinion pool (geometric mean), for probabilities
    )


def aggregate_scores(
    values: list[float], operator: AggregationOperator | str = AggregationOperator.MEAN
) -> float:
    """Aggregate a list of scores by a named operator (CONCEPT:AU-ORCH.execution.coordination-protocol-metadata).

    ``log_pool`` is the geometric mean (clamped to >0), the principled pool for
    combining independent probabilities; the rest are the obvious reductions.
    Returns 0.0 for an empty input.
    """
    if not values:
        return 0.0
    op = AggregationOperator(operator)
    if op is AggregationOperator.MEAN:
        return sum(values) / len(values)
    if op is AggregationOperator.MEDIAN:
        return float(statistics.median(values))
    if op is AggregationOperator.MAX:
        return max(values)
    if op is AggregationOperator.MIN:
        return min(values)
    # LOG_POOL — geometric mean of strictly-positive values
    import math

    clamped = [max(v, 1e-9) for v in values]
    return math.exp(sum(math.log(v) for v in clamped) / len(clamped))


# ── Protocol Definitions ───────────────────────────────────────────


class ProtocolType(StrEnum):
    """Available coordination protocol types.

    CONCEPT:AU-ORCH.execution.coordination-protocol-metadata — Research: 2605.03310v1
    """

    CONSENSUS = "consensus"  # All agents must agree
    VOTING = "voting"  # Majority rules
    DELEGATION = "delegation"  # One agent takes authority
    HANDOFF = "handoff"  # Sequential pass with context
    BROADCAST = "broadcast"  # One-to-many, no convergence needed


class ConvergenceCriterion(StrEnum):
    """When to consider coordination converged."""

    MAJORITY = "majority"  # >50% agree
    UNANIMOUS = "unanimous"  # 100% agree
    THRESHOLD = "threshold"  # Configurable % agree
    FIRST_RESPONSE = "first"  # First valid response wins
    BEST_CONFIDENCE = "best"  # Highest confidence wins


class CoordinationProtocol(BaseModel):
    """Declarative coordination protocol definition.

    CONCEPT:AU-ORCH.execution.coordination-protocol-metadata — Research: 2605.03310v1

    Each protocol specifies how multiple agents should coordinate
    on a shared task: who speaks, when convergence is reached, and
    whether a moderator is needed.

    Attributes:
        protocol_id: Unique protocol identifier.
        protocol_type: The coordination strategy.
        name: Human-readable name.
        min_agents: Minimum agents required.
        max_rounds: Maximum coordination rounds before timeout.
        requires_moderator: Whether a moderator agent is needed.
        convergence: When to consider the coordination complete.
        convergence_threshold: Fraction for THRESHOLD convergence.
        timeout_seconds: Max wall-clock time for coordination.
    """

    protocol_id: str = Field(default_factory=lambda: f"proto:{uuid.uuid4().hex[:8]}")
    protocol_type: ProtocolType = ProtocolType.DELEGATION
    name: str = ""
    min_agents: int = 1
    max_rounds: int = 3
    requires_moderator: bool = False
    convergence: ConvergenceCriterion = ConvergenceCriterion.FIRST_RESPONSE
    convergence_threshold: float = 0.66
    timeout_seconds: float = 120.0


class CoordinationResult(BaseModel):
    """Outcome of a coordination round.

    CONCEPT:AU-ORCH.execution.coordination-protocol-metadata

    Attributes:
        protocol_id: Protocol that was applied.
        protocol_type: The coordination strategy used.
        agents_involved: List of agent IDs that participated.
        rounds_taken: Number of coordination rounds executed.
        converged: Whether the agents reached agreement.
        quality_score: Coordination quality (0.0–1.0).
        winning_output: The selected/merged output.
        duration_seconds: Wall-clock time for coordination.
        metadata: Additional coordination metadata.
    """

    protocol_id: str = ""
    protocol_type: str = ""
    agents_involved: list[str] = Field(default_factory=list)
    rounds_taken: int = 1
    converged: bool = True
    quality_score: float = 1.0
    winning_output: str = ""
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Built-in Protocol Registry ────────────────────────────────────

BUILTIN_PROTOCOLS: dict[str, CoordinationProtocol] = {
    "single_agent": CoordinationProtocol(
        protocol_id="proto:single",
        protocol_type=ProtocolType.DELEGATION,
        name="Single Agent Delegation",
        min_agents=1,
        max_rounds=1,
        convergence=ConvergenceCriterion.FIRST_RESPONSE,
    ),
    "pair_consensus": CoordinationProtocol(
        protocol_id="proto:pair",
        protocol_type=ProtocolType.CONSENSUS,
        name="Pair Consensus",
        min_agents=2,
        max_rounds=3,
        convergence=ConvergenceCriterion.UNANIMOUS,
    ),
    "team_voting": CoordinationProtocol(
        protocol_id="proto:vote",
        protocol_type=ProtocolType.VOTING,
        name="Team Majority Voting",
        min_agents=3,
        max_rounds=2,
        requires_moderator=True,
        convergence=ConvergenceCriterion.MAJORITY,
    ),
    "sequential_handoff": CoordinationProtocol(
        protocol_id="proto:handoff",
        protocol_type=ProtocolType.HANDOFF,
        name="Sequential Handoff",
        min_agents=2,
        max_rounds=1,
        convergence=ConvergenceCriterion.FIRST_RESPONSE,
    ),
    "broadcast": CoordinationProtocol(
        protocol_id="proto:broadcast",
        protocol_type=ProtocolType.BROADCAST,
        name="Broadcast (no convergence)",
        min_agents=2,
        max_rounds=1,
        convergence=ConvergenceCriterion.FIRST_RESPONSE,
    ),
}


# ── Coordination Layer ─────────────────────────────────────────────


class CoordinationLayer:
    """Explicit coordination layer for multi-agent tasks.

    CONCEPT:AU-ORCH.execution.coordination-protocol-metadata — Research: 2605.03310v1

    Sits between ``AgentOrchestrationEngine.synthesize_team()`` and
    graph execution. Selects the optimal coordination protocol based
    on task type, agent count, and historical success rates from the KG.

    Args:
        engine: Optional KG engine for historical success lookups and trace persistence.
        custom_protocols: Optional additional protocols to register.
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        custom_protocols: dict[str, CoordinationProtocol] | None = None,
    ) -> None:
        self.engine = engine
        self.protocols: dict[str, CoordinationProtocol] = dict(BUILTIN_PROTOCOLS)
        if custom_protocols:
            self.protocols.update(custom_protocols)

        # Historical success rates (populated from KG on init)
        self._success_history: dict[str, list[float]] = {}

    # ── Protocol Selection ─────────────────────────────────────────

    def aggregate(
        self,
        values: list[float],
        operator: AggregationOperator | str = AggregationOperator.MEAN,
    ) -> float:
        """Aggregate coordinated scalar outputs by a named operator (CONCEPT:AU-ORCH.execution.coordination-protocol-metadata)."""
        return aggregate_scores(values, operator)

    def rank(
        self,
        items: list[str],
        comparisons: list[tuple[str, str]],
        *,
        method: str = "bradley_terry",
    ) -> list[tuple[str, float]]:
        """Rank candidates from pairwise judgments via the unified selection registry.

        Delegates to :mod:`agent_utilities.harness.selection_operators` so the
        coordination layer and the evolution/variant paths share ONE selection
        implementation (CONCEPT:AU-ORCH.optimization.selection-on-unseen-data; STRATEGY synergy #2).
        """
        from ..harness.selection_operators import rank_from_comparisons

        return rank_from_comparisons(items, comparisons, method=method)

    def select_protocol(
        self,
        agent_count: int,
        task_type: str = "general",
        execution_mode: str = "sequential",
    ) -> CoordinationProtocol:
        """Select the optimal coordination protocol for a task.

        CONCEPT:AU-ORCH.execution.coordination-protocol-metadata — Research: 2605.03310v1

        Selection logic:
        1. If single agent → delegation (no coordination needed).
        2. If execution_mode is parallel → voting (concurrent + merge).
        3. If execution_mode is sequential → handoff (pass context).
        4. If 2 agents → consensus (pair agreement).
        5. If 3+ agents → voting with moderator.

        Falls back to historical success rates from KG when available.

        Args:
            agent_count: Number of agents in the team.
            task_type: Task classification (general, research, code, etc.).
            execution_mode: Team execution mode (sequential, parallel, mixed).

        Returns:
            The selected coordination protocol.
        """
        # Single agent — no coordination needed
        if agent_count <= 1:
            return self.protocols["single_agent"]

        # Check KG for historical best protocol for this task type
        best_from_history = self._lookup_best_protocol(task_type)
        if best_from_history:
            return best_from_history

        # Heuristic selection based on agent count and mode
        if execution_mode == "parallel":
            if agent_count >= 3:
                return self.protocols["team_voting"]
            return self.protocols["pair_consensus"]

        if execution_mode == "sequential":
            return self.protocols["sequential_handoff"]

        # Mixed mode — default to voting for 3+, consensus for 2
        if agent_count >= 3:
            return self.protocols["team_voting"]
        return self.protocols["pair_consensus"]

    def _lookup_best_protocol(self, task_type: str) -> CoordinationProtocol | None:
        """Query KG for the historically best-performing protocol.

        Returns None if no history is available.
        """
        if not self.engine or not self.engine.backend:
            return None

        try:
            results = self.engine.backend.execute(
                "MATCH (ct:CoordinationTrace) "
                "WHERE ct.task_type = $task_type AND ct.converged = true "
                "RETURN ct.protocol_type AS protocol, "
                "avg(ct.quality_score) AS avg_quality, "
                "count(*) AS uses "
                "ORDER BY avg_quality DESC LIMIT 1",
                {"task_type": task_type},
            )
            for r in results:
                proto_type = r.get("protocol")
                uses = r.get("uses", 0)
                if proto_type and uses >= 3:  # Need at least 3 data points
                    # Find matching protocol
                    for p in self.protocols.values():
                        if p.protocol_type.value == proto_type:
                            logger.info(
                                "[ORCH-1.5] KG-selected protocol '%s' for task '%s' "
                                "(avg_quality=%.2f, uses=%d)",
                                p.name,
                                task_type,
                                r.get("avg_quality", 0),
                                uses,
                            )
                            return p
        except Exception as e:
            logger.debug("CoordinationLayer: KG lookup failed: %s", e)

        return None

    # ── Protocol Application ───────────────────────────────────────

    def apply_protocol(
        self,
        protocol: CoordinationProtocol,
        agent_ids: list[str],
        task: str = "",
        task_type: str = "general",
    ) -> CoordinationResult:
        """Apply a coordination protocol and return the result.

        CONCEPT:AU-ORCH.execution.coordination-protocol-metadata — Research: 2605.03310v1

        This is the synchronous coordination step that occurs *before*
        graph execution begins. It determines how agents will coordinate
        during execution (the actual coordination happens in the graph
        nodes themselves — this sets up the protocol).

        Args:
            protocol: The selected coordination protocol.
            agent_ids: List of participating agent IDs.
            task: The task description.
            task_type: Task classification for KG tracing.

        Returns:
            A CoordinationResult describing the setup outcome.
        """
        start = time.monotonic()

        result = CoordinationResult(
            protocol_id=protocol.protocol_id,
            protocol_type=protocol.protocol_type.value,
            agents_involved=agent_ids,
            rounds_taken=1,
            converged=True,
            quality_score=1.0,
            metadata={
                "task_type": task_type,
                "task_preview": task[:200],
                "convergence_criterion": protocol.convergence.value,
            },
        )

        # Protocol-specific setup
        if protocol.protocol_type == ProtocolType.DELEGATION:
            # Single agent takes authority — no coordination overhead
            result.quality_score = 1.0
            result.metadata["delegated_to"] = agent_ids[0] if agent_ids else ""

        elif protocol.protocol_type == ProtocolType.HANDOFF:
            # Sequential pass — quality depends on chain length
            chain_penalty = max(0.5, 1.0 - (len(agent_ids) - 2) * 0.1)
            result.quality_score = chain_penalty
            result.metadata["handoff_order"] = agent_ids

        elif protocol.protocol_type == ProtocolType.CONSENSUS:
            # Pair consensus — quality depends on agent compatibility
            result.quality_score = 0.9
            result.metadata["requires_agreement"] = True

        elif protocol.protocol_type == ProtocolType.VOTING:
            # Majority voting — quality depends on voter count
            result.quality_score = min(1.0, 0.7 + len(agent_ids) * 0.05)
            result.metadata["voter_count"] = len(agent_ids)
            result.metadata["requires_moderator"] = protocol.requires_moderator

        elif protocol.protocol_type == ProtocolType.BROADCAST:
            result.quality_score = 0.8

        result.duration_seconds = time.monotonic() - start

        logger.info(
            "[ORCH-1.5] Applied coordination protocol '%s' (%s) "
            "for %d agents (quality=%.2f)",
            protocol.name,
            protocol.protocol_type.value,
            len(agent_ids),
            result.quality_score,
        )

        return result

    # ── KG Trace Persistence ───────────────────────────────────────

    def log_coordination_trace(
        self,
        result: CoordinationResult,
    ) -> str | None:
        """Persist a coordination trace to the Knowledge Graph.

        CONCEPT:AU-ORCH.execution.coordination-protocol-metadata — Research: 2605.03310v1

        Creates a ``CoordinationTrace`` node in the KG for observability,
        learning, and historical protocol selection.

        Args:
            result: The coordination outcome to persist.

        Returns:
            The trace node ID, or None if KG unavailable.
        """
        if not self.engine or not self.engine.backend:
            return None

        trace_id = f"coord_trace:{uuid.uuid4().hex[:8]}"
        try:
            props = {
                "id": trace_id,
                "type": "CoordinationTrace",
                "name": f"Coordination: {result.protocol_type}",
                "protocol_id": result.protocol_id,
                "protocol_type": result.protocol_type,
                "agent_count": len(result.agents_involved),
                "agents_involved": ",".join(result.agents_involved),
                "rounds_taken": result.rounds_taken,
                "converged": result.converged,
                "quality_score": result.quality_score,
                "duration_seconds": round(result.duration_seconds, 4),
                "task_type": result.metadata.get("task_type", ""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "importance_score": result.quality_score,
            }
            self.engine.graph.add_node(trace_id, **props)

            # Link trace to participating agents
            for agent_id in result.agents_involved:
                if agent_id in self.engine.graph:
                    self.engine.graph.add_edge(
                        trace_id, agent_id, type="COORDINATED_WITH"
                    )

            logger.debug("[ORCH-1.5] Persisted coordination trace %s to KG", trace_id)
            return trace_id

        except Exception as e:
            logger.debug("CoordinationLayer: trace persistence failed: %s", e)
            return None
