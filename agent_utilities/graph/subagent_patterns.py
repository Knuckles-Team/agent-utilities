#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:ORCH-1.6 — Subagent Lifecycle Patterns.

Formalizes the four-tier subagent interaction taxonomy identified by
Phil Schmid (2026) as first-class graph orchestration primitives:

    1. **INLINE_TOOL** — Single specialist via direct tool call
    2. **FAN_OUT** — Parallel dispatch with result aggregation
    3. **AGENT_POOL** — Persistent pool with messaging
    4. **TEAMS** — Cross-agent A2A collaboration

Each pattern maps to existing agent-utilities infrastructure:
    - INLINE_TOOL → ``executor.py`` single specialist path
    - FAN_OUT → ``SwarmPresetEngine`` parallel dispatch
    - AGENT_POOL → ``Council`` with advisory messaging
    - TEAMS → ``A2AClient`` with ``send_message``

The ``SubagentPatternRouter`` selects the optimal pattern based on
task complexity, parallelizability, and collaboration requirements.
Pattern selection decisions are recorded as ``RoutingDecisionNode``
entries in the Knowledge Graph for harness learning.

See docs/overview.md §CONCEPT:ORCH-1.6.
"""


import logging
import time
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class SubagentPattern(StrEnum):
    """Four-tier subagent interaction patterns (Schmid, 2026).

    Ordered by increasing coordination complexity and resource cost.
    """

    INLINE_TOOL = "inline_tool"
    """Single specialist executes a tool and returns. Cheapest pattern.
    Use for atomic, well-scoped tasks with clear input/output."""

    FAN_OUT = "fan_out"
    """Multiple adaptive_agent_router execute in parallel with result aggregation.
    Use for embarrassingly parallel tasks (multi-source research)."""

    AGENT_POOL = "agent_pool"
    """Persistent pool of adaptive_agent_router with inter-agent messaging.
    Use when adaptive_agent_router need to share intermediate findings."""

    TEAMS = "teams"
    """Full cross-agent collaboration via A2A protocol.
    Use for complex multi-step tasks requiring negotiation."""


class PatternComplexity(IntEnum):
    """Task complexity tiers for pattern selection."""

    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    EXPERT = 5


class SubagentPatternDecision(BaseModel):
    """Records a pattern selection decision for KG persistence.

    Stored as a node in the Knowledge Graph to enable the harness
    to learn which patterns succeed for which task types.
    """

    pattern: SubagentPattern
    task_complexity: PatternComplexity
    parallelizable: bool = False
    needs_collaboration: bool = False
    specialist_count: int = 1
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    reasoning: str = ""
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    # Outcome tracking (populated post-execution)
    outcome_success: bool | None = None
    outcome_duration_ms: float | None = None


class SubagentPatternRouter:
    """Selects the optimal subagent interaction pattern for a task.

    CONCEPT:ORCH-1.6 — Subagent Lifecycle Patterns

    Decision logic:
        - INLINE_TOOL: complexity ≤ SIMPLE, not parallelizable
        - FAN_OUT: parallelizable, no collaboration needed
        - AGENT_POOL: needs collaboration, complexity ≤ COMPLEX
        - TEAMS: complexity = EXPERT or cross-agent A2A required

    The router integrates with the Knowledge Graph to:
        1. Record decisions as ``SubagentPatternDecision`` nodes
        2. Learn from historical outcomes via MemoryRetriever
        3. Adjust thresholds based on team success rates (AHE-3.3)

    Args:
        engine: Optional KG engine for historical decision tracking.
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        self.engine = engine

    def select_pattern(
        self,
        task_complexity: int | PatternComplexity = PatternComplexity.MODERATE,
        parallelizable: bool = False,
        needs_collaboration: bool = False,
        specialist_count: int = 1,
        has_a2a_peers: bool = False,
    ) -> SubagentPatternDecision:
        """Select the optimal subagent pattern for the given task parameters.

        Args:
            task_complexity: Estimated task complexity (1-5 scale).
            parallelizable: Whether sub-tasks can run independently.
            needs_collaboration: Whether adaptive_agent_router need to exchange messages.
            specialist_count: Number of adaptive_agent_router the planner wants to invoke.
            has_a2a_peers: Whether remote A2A agents are available.

        Returns:
            A ``SubagentPatternDecision`` with the selected pattern and reasoning.
        """
        complexity = PatternComplexity(min(task_complexity, 5))

        # Decision tree (ordered by cost — prefer cheaper patterns)
        if complexity <= PatternComplexity.SIMPLE and specialist_count <= 1:
            pattern = SubagentPattern.INLINE_TOOL
            reasoning = (
                f"Low complexity ({complexity.name}) with single specialist — "
                f"inline tool execution is sufficient."
            )
            confidence = 0.9

        elif parallelizable and not needs_collaboration:
            pattern = SubagentPattern.FAN_OUT
            reasoning = (
                f"Task is parallelizable with {specialist_count} adaptive_agent_router, "
                f"no inter-agent messaging needed — fan-out with aggregation."
            )
            confidence = 0.85

        elif needs_collaboration and complexity <= PatternComplexity.COMPLEX:
            pattern = SubagentPattern.AGENT_POOL
            reasoning = (
                f"Collaboration required at complexity {complexity.name} — "
                f"persistent agent pool with shared messaging."
            )
            confidence = 0.75

        elif complexity >= PatternComplexity.EXPERT or has_a2a_peers:
            pattern = SubagentPattern.TEAMS
            reasoning = (
                "Expert-level complexity or A2A peers available — "
                "full team collaboration via A2A protocol."
            )
            confidence = 0.7

        elif specialist_count > 1 and parallelizable:
            pattern = SubagentPattern.FAN_OUT
            reasoning = f"Multiple adaptive_agent_router ({specialist_count}) with parallel execution."
            confidence = 0.8

        else:
            # Default to inline for anything that doesn't fit above
            pattern = SubagentPattern.INLINE_TOOL
            reasoning = "Default fallback to inline tool execution."
            confidence = 0.6

        # Check historical performance if KG is available
        if self.engine is not None:
            confidence = self._adjust_from_history(pattern, confidence)

        decision = SubagentPatternDecision(
            pattern=pattern,
            task_complexity=complexity,
            parallelizable=parallelizable,
            needs_collaboration=needs_collaboration,
            specialist_count=specialist_count,
            confidence=confidence,
            reasoning=reasoning,
        )

        logger.info(
            "[CONCEPT:ORCH-1.6] Pattern selected: %s (confidence=%.2f, reason=%s)",
            pattern.value,
            confidence,
            reasoning[:80],
        )

        # Persist decision to KG
        if self.engine is not None:
            self._persist_decision(decision)

        return decision

    def estimate_specialist_count(self, query: str) -> int:
        """Estimate specialist count using KG topology.

        CONCEPT:ORCH-1.15 — KG-Driven Specialist Estimation

        Instead of the caller guessing specialist_count, query the KG
        for agents/tools topologically proximate to the task.
        """
        if self.engine is None:
            return 1

        count = 1
        try:
            if self.engine.backend:
                # Use backend for O(1) lookup instead of O(N) scan
                results = self.engine.backend.execute(
                    "MATCH (a:Agent)-[:PROVIDES|HAS_CAPABILITY]->() "
                    "RETURN count(DISTINCT a) AS agent_count",
                    {},
                )
                if results:
                    count = max(1, min(results[0].get("agent_count", 1), 10))
            else:
                # Fallback: count agent nodes in NX
                for _, data in self.engine.graph.nodes(data=True):
                    if data.get("type") == "agent":
                        count += 1
                count = min(count, 10)
        except Exception:  # nosec B110
            pass

        return count

    def _adjust_from_history(
        self, pattern: SubagentPattern, base_confidence: float
    ) -> float:
        """Adjust confidence based on historical pattern success rates.

        Integrates with CONCEPT:AHE-3.3 (TeamConfig) and
        CONCEPT:KG-2.1 (MemoryRetriever) for learned pattern preferences.
        """
        if self.engine is None:
            return base_confidence

        try:
            # Prefer backend (Tier 1) for O(1) lookups
            if self.engine.backend:
                try:
                    results = self.engine.backend.execute(
                        "MATCH (d:SubagentPatternDecision) "
                        "WHERE d.pattern = $pattern AND d.outcome_success IS NOT NULL "
                        "RETURN d.outcome_success AS success",
                        {"pattern": pattern.value},
                    )
                    if results and len(results) >= 3:
                        total_count = len(results)
                        success_count = sum(
                            1 for r in results if r.get("success") is True
                        )
                        historical_rate = success_count / total_count
                        adjusted = 0.7 * base_confidence + 0.3 * historical_rate
                        logger.debug(
                            "[CONCEPT:ORCH-1.6] Adjusted confidence for %s: %.2f → %.2f "
                            "(historical: %d/%d = %.2f, source=backend)",
                            pattern.value,
                            base_confidence,
                            adjusted,
                            success_count,
                            total_count,
                            historical_rate,
                        )
                        return adjusted
                except Exception:  # nosec B110
                    pass  # Fall through to NX fallback

            # Fallback: O(N) NX graph scan
            success_count = 0
            total_count = 0
            for nid, data in self.engine.graph.nodes(data=True):
                if (
                    data.get("type") == "subagent_pattern_decision"
                    and data.get("pattern") == pattern.value
                ):
                    total_count += 1
                    if data.get("outcome_success") is True:
                        success_count += 1

            if total_count >= 3:  # Need minimum sample size
                historical_rate = success_count / total_count
                # Blend: 70% base + 30% historical
                adjusted = 0.7 * base_confidence + 0.3 * historical_rate
                logger.debug(
                    "[CONCEPT:ORCH-1.6] Adjusted confidence for %s: %.2f → %.2f "
                    "(historical: %d/%d = %.2f)",
                    pattern.value,
                    base_confidence,
                    adjusted,
                    success_count,
                    total_count,
                    historical_rate,
                )
                return adjusted
        except Exception as e:
            logger.debug("Historical pattern lookup failed: %s", e)

        return base_confidence

    def _persist_decision(self, decision: SubagentPatternDecision) -> None:
        """Store a pattern decision in the Knowledge Graph.

        Tiered write path: backend (Tier 1) when available, NX fallback.
        """
        if self.engine is None:
            return

        try:
            import uuid

            node_id = f"spd:{uuid.uuid4().hex[:8]}"
            node_data = {
                "id": node_id,
                "type": "subagent_pattern_decision",
                "pattern": decision.pattern.value,
                "task_complexity": decision.task_complexity.value,
                "parallelizable": decision.parallelizable,
                "needs_collaboration": decision.needs_collaboration,
                "specialist_count": decision.specialist_count,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "timestamp": decision.timestamp,
                "outcome_success": None,
                "importance_score": 0.3,
            }

            # Tier 1: Backend is source of truth
            if hasattr(self.engine, "backend") and self.engine.backend:
                self.engine._upsert_node("SubagentPatternDecision", node_id, node_data)
            else:
                # Tier 2 fallback: NX only
                self.engine.graph.add_node(node_id, **node_data)
        except Exception as e:
            logger.debug("Failed to persist pattern decision: %s", e)

    def record_outcome(
        self,
        decision: SubagentPatternDecision,
        success: bool,
        duration_ms: float = 0.0,
    ) -> None:
        """Record the outcome of a pattern decision for learning.

        Args:
            decision: The original decision to update.
            success: Whether the execution succeeded.
            duration_ms: Wall-clock execution time.
        """
        decision.outcome_success = success
        decision.outcome_duration_ms = duration_ms

        if self.engine is None:
            return

        # Find and update the persisted node
        try:
            for nid, data in self.engine.graph.nodes(data=True):
                if (
                    data.get("type") == "subagent_pattern_decision"
                    and data.get("timestamp") == decision.timestamp
                    and data.get("pattern") == decision.pattern.value
                ):
                    self.engine.graph.nodes[nid]["outcome_success"] = success
                    self.engine.graph.nodes[nid]["outcome_duration_ms"] = duration_ms
                    logger.info(
                        "[CONCEPT:ORCH-1.6] Pattern outcome recorded: %s → %s (%.0fms)",
                        decision.pattern.value,
                        "SUCCESS" if success else "FAILURE",
                        duration_ms,
                    )
                    break
        except Exception as e:
            logger.debug("Failed to record pattern outcome: %s", e)


def get_infrastructure_mapping() -> dict[SubagentPattern, dict[str, Any]]:
    """Map each pattern to its agent-utilities infrastructure component.

    Returns a dict mapping pattern → implementation details, including
    the module path, class name, and required capabilities.
    """
    return {
        SubagentPattern.INLINE_TOOL: {
            "module": "agent_utilities.graph.executor",
            "class": "_execute_dynamic_mcp_agent",
            "description": "Single specialist via direct MCP execution",
            "requires": [],
        },
        SubagentPattern.FAN_OUT: {
            "module": "agent_utilities.graph.dynamic_graph_orchestrator",
            "class": "DynamicSubgraphOrchestrator",
            "description": "Parallel dispatch with DynamicSubgraphOrchestrator",
            "requires": ["dynamic_graph_orchestrator"],
        },
        SubagentPattern.AGENT_POOL: {
            "module": "agent_utilities.graph.dynamic_graph_orchestrator",
            "class": "DynamicSubgraphOrchestrator",
            "description": "Persistent advisory pool mapped to dynamic subgraphs",
            "requires": ["dynamic_graph_orchestrator"],
        },
        SubagentPattern.TEAMS: {
            "module": "agent_utilities.knowledge_graph.engine_query",
            "class": "A2AClient (via find_a2a_peers)",
            "description": "Cross-agent A2A collaboration",
            "requires": ["a2a_protocol"],
        },
    }
