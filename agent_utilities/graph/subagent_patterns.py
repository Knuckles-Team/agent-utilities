#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-ORCH.execution.active-subagent-lifecycle — Subagent Lifecycle Patterns.

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

The ``SubagentPatternRouter`` maps a task onto one of these executor
families. The topology itself is EG's decision (SWARM-TOPOLOGY-DECIDE-DESIGN,
EH-048): when a topology asker is installed the family is the projection of
EG's certified plan; the cost-ordered tree below is only the deterministic
fallback. Nothing here persists outcomes or learns from them (invariant T5):
outcomes are credited by independent evaluation to the committed plan's
whole slate in EG.

See docs/overview.md §CONCEPT:AU-ORCH.execution.active-subagent-lifecycle
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


class SubagentMode(StrEnum):
    """Execution mode for subagents (CONCEPT:AU-ORCH.execution.subagent-pattern-variant).

    Controls whether a subagent can modify the codebase or is
    restricted to read-only exploration.  Read-only subagents map
    a subsystem and write findings to a structured file, then the
    parent agent edits with the full picture.
    """

    READ_WRITE = "read_write"
    """Full access — subagent can read and modify files."""

    READ_ONLY = "read_only"
    """Exploration only — subagent can read files, run queries, and
    search but cannot write, delete, or run modifying commands.
    Findings are written to a structured output file."""


class PatternComplexity(IntEnum):
    """Task complexity tiers for pattern selection."""

    TRIVIAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4
    EXPERT = 5


def _decided_pattern(
    pattern: SubagentPattern,
    reasoning: str,
    complexity: int,
    flags: tuple[bool, bool, bool],
    specialist_count: int,
) -> tuple[SubagentPattern, str]:
    """The tree's pattern, unless EG decides another (EH-048)."""
    from agent_utilities.decide.consumers.topology import TaskShape, decided_topology

    parallelizable, needs_collaboration, has_a2a_peers = flags
    shape = TaskShape(
        int(complexity),
        parallelizable,
        needs_collaboration,
        specialist_count,
        has_a2a_peers,
    )
    chosen, why = decided_topology(pattern.value, reasoning, shape)
    return SubagentPattern(chosen), why


class SubagentPatternDecision(BaseModel):
    """One pattern selection: the family, the facts it read and why."""

    pattern: SubagentPattern
    mode: SubagentMode = SubagentMode.READ_WRITE
    task_complexity: PatternComplexity
    parallelizable: bool = False
    needs_collaboration: bool = False
    specialist_count: int = 1
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    reasoning: str = ""
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


class SubagentPatternRouter:
    """Selects the optimal subagent interaction pattern for a task.

    CONCEPT:AU-ORCH.execution.active-subagent-lifecycle — Subagent Lifecycle Patterns

    Decision logic:
        - INLINE_TOOL: complexity ≤ SIMPLE, not parallelizable
        - FAN_OUT: parallelizable, no collaboration needed
        - AGENT_POOL: needs collaboration, complexity ≤ COMPLEX
        - TEAMS: complexity = EXPERT or cross-agent A2A required

    The tree is the fallback; EG's topology plan decides when an asker is
    installed (:func:`agent_utilities.decide.consumers.topology.install_topology`).

    Args:
        engine: Optional KG engine for the specialist-count estimate.
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
        read_only: bool = False,
    ) -> SubagentPatternDecision:
        """Select the optimal subagent pattern for the given task parameters.

        Args:
            task_complexity: Estimated task complexity (1-5 scale).
            parallelizable: Whether sub-tasks can run independently.
            needs_collaboration: Whether adaptive_agent_router need to exchange messages.
            specialist_count: Number of adaptive_agent_router the planner wants to invoke.
            has_a2a_peers: Whether remote A2A agents are available.
            read_only: If True, subagent is restricted to exploration-only
                (read files, search, query) with no write/delete/modify access.
                Findings are written to a structured output file for the
                parent agent to act on.

        Returns:
            A ``SubagentPatternDecision`` with the selected pattern and reasoning.
        """
        complexity = PatternComplexity(min(task_complexity, 5))
        mode = SubagentMode.READ_ONLY if read_only else SubagentMode.READ_WRITE

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

        # EH-048: EG Decide over the topology family; the tree above is the fallback.
        pattern, reasoning = _decided_pattern(
            pattern,
            reasoning,
            complexity,
            (parallelizable, needs_collaboration, has_a2a_peers),
            specialist_count,
        )

        # Append read-only note to reasoning
        if read_only:
            reasoning += " [READ_ONLY: exploration-only, no file modifications]"

        decision = SubagentPatternDecision(
            pattern=pattern,
            mode=mode,
            task_complexity=complexity,
            parallelizable=parallelizable,
            needs_collaboration=needs_collaboration,
            specialist_count=specialist_count,
            confidence=confidence,
            reasoning=reasoning,
        )

        logger.info(
            "[CONCEPT:AU-ORCH.execution.active-subagent-lifecycle] Pattern selected: %s (confidence=%.2f, reason=%s)",
            pattern.value,
            confidence,
            reasoning[:80],
        )

        return decision

    def estimate_specialist_count(self, query: str) -> int:
        """Estimate specialist count using KG topology.

        CONCEPT:AU-ORCH.routing.kg-specialist-estimation — KG-Driven Specialist Estimation

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
                    if data.get("node_type") == "agent":
                        count += 1
                count = min(count, 10)
        except Exception:  # nosec B110
            pass

        return count


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
            "module": "agent_utilities.orchestration.graph_orchestrator",
            "class": "AgentOrchestrationEngine",
            "description": "Parallel dispatch with AgentOrchestrationEngine",
            "requires": ["graph_orchestrator"],
        },
        SubagentPattern.AGENT_POOL: {
            "module": "agent_utilities.orchestration.graph_orchestrator",
            "class": "AgentOrchestrationEngine",
            "description": "Persistent advisory pool mapped to dynamic subgraphs",
            "requires": ["graph_orchestrator"],
        },
        SubagentPattern.TEAMS: {
            "module": "agent_utilities.knowledge_graph.engine_query",
            "class": "A2AClient (via find_a2a_peers)",
            "description": "Cross-agent A2A collaboration",
            "requires": ["a2a_protocol"],
        },
    }
