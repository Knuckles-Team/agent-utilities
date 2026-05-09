#!/usr/bin/python
"""KG-Driven Team Composer (CONCEPT:ORCH-1.15).

Assembles specialist teams from Knowledge Graph topology instead of
static ``discover_agents()`` registration.  The KG becomes the single
source of truth for *who* participates in a task, *what tools* they get,
*which model* they use, and *how* they collaborate.

Composition flow:
    1. Embed the task query via the engine's hybrid search.
    2. Search for a proven ``TeamConfigNode`` (AHE-3.3 reuse).
    3. If match found with high enough success_rate: reuse directly.
    4. If no match: compose from scratch using KG topology:
       - Walk ``PROVIDES`` / ``HAS_CAPABILITY`` edges for tool affinity.
       - Walk ``SIMILAR_TO`` edges for semantic relevance.
       - Check ``MCPServer`` health (circuit breaker state).
       - Match domain ontology alignment (OWL classes).
    5. Materialize the ``TopologyTemplateNode`` into a ``TeamComposition``.
    6. On success: promote to ``TeamConfigNode`` for future reuse.

This replaces the ad-hoc ``if deps.knowledge_engine:`` pattern in
``routing.py`` with a single call: ``composer.compose_team(query, deps)``.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

from ..models.knowledge_graph import (
    RegistryNodeType,
    TeamComposition,
)
from .dynamic_subgraph import DynamicSubgraphOrchestrator

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class KGTeamComposer:
    """Assembles specialist teams from KG topology.

    CONCEPT:ORCH-1.15 — KG-Driven Team Composition

    This is the primary entry point for KG-driven orchestration.
    Instead of static agent registration, the KG dynamically determines
    which specialists participate, what tools they receive, and how they
    collaborate.

    Args:
        engine: The IntelligenceGraphEngine for KG queries.
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        self.engine = engine
        self._topologies_seeded = False

    def compose_team(
        self,
        query: str,
        domain: str = "general",
        complexity: int = 3,
        available_tools: list[str] | None = None,
        available_agents: list[str] | None = None,
        delegated_authority: str | None = None,
    ) -> TeamComposition:
        """Compose the optimal specialist team for a task dynamically.

        Flow:
            1. Try reusing a proven TeamConfigNode (AHE-3.3)
            2. If no proven team: dynamically synthesize subgraph using KG primitives
            3. Return a fully specified TeamComposition
        """
        team_id = f"team:{uuid.uuid4().hex[:12]}"

        # Step 1: Try to reuse a proven team configuration
        proven = self._try_reuse_proven_team(query)
        if proven is not None:
            proven.team_id = team_id
            logger.info(
                "[CONCEPT:ORCH-1.15] Reusing proven team config '%s' "
                "(success_rate=%.2f, usage=%d)",
                proven.team_config_id,
                proven.confidence,
                0,
            )
            return proven

        # Step 2: Dynamically synthesize the topology using ORCH-1.19
        orchestrator = DynamicSubgraphOrchestrator(engine=self.engine)
        composition = orchestrator.synthesize_team(
            query=query,
            domain=domain,
            complexity=complexity,
            available_tools=available_tools,
            available_agents=available_agents,
            delegated_authority=delegated_authority,
        )

        return composition

    def promote_to_team_config(
        self,
        composition: TeamComposition,
        success: bool,
        quality_score: float = 0.5,
    ) -> str | None:
        """Promote a successful composition to a reusable TeamConfigNode.

        Called after execution completes successfully. This is how the
        KG learns which team compositions work — the evolutionary
        feedback loop.

        Args:
            composition: The team composition to promote.
            success: Whether the execution succeeded.
            quality_score: Quality of the result (0-1).

        Returns:
            The TeamConfigNode ID if promoted, else None.
        """
        if not success or quality_score < 0.6:
            return None
        if not self.engine:
            return None

        # Build a TeamConfigNode from the composition
        specialist_ids = [
            s.get("agent_id", s.get("role", "")) for s in composition.specialists
        ]
        tool_assignments = {}
        for s in composition.specialists:
            role = s.get("role", "")
            tools = s.get("tools", [])
            if tools:
                tool_assignments[role] = tools

        node_id = f"tc:{uuid.uuid4().hex[:8]}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node_data = {
            "id": node_id,
            "name": f"Proven team: {composition.execution_mode} ({len(specialist_ids)} agents)",
            "type": RegistryNodeType.TEAM_CONFIG.value,
            "task_pattern": composition.reasoning[:200]
            if composition.reasoning
            else "",
            "specialist_ids": specialist_ids,
            "tool_assignments": tool_assignments,
            "success_rate": quality_score,
            "usage_count": 1,
            "origin": "local",
            "timestamp": timestamp,
        }

        try:
            self.engine._upsert_node("TeamConfig", node_id, node_data)
            logger.info(
                "[CONCEPT:ORCH-1.15] Promoted composition to TeamConfig '%s'",
                node_id,
            )
            return node_id
        except Exception as e:
            logger.warning("Failed to promote team composition: %s", e)
            return None

    # -----------------------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------------------

    def _try_reuse_proven_team(self, query: str) -> TeamComposition | None:
        """Search KG for a proven TeamConfigNode matching this query."""
        if not self.engine:
            return None

        try:
            # Try to find matching team configs via the registry mixin
            from ..knowledge_graph.core.engine_registry import RegistryMixin

            if isinstance(self.engine, RegistryMixin) and hasattr(
                self.engine, "find_matching_team_config"
            ):
                matching_teams = self.engine.find_matching_team_config(query)
                if matching_teams:
                    best = matching_teams[0]
                    # Only reuse if success rate is high enough
                    if best.get("success_rate", 0) >= 0.7:
                        specialists = []
                        for sid in best.get("specialist_ids", []):
                            tool_map = best.get("tool_assignments", {})
                            specialists.append(
                                {
                                    "role": sid,
                                    "agent_id": sid,
                                    "tools": tool_map.get(sid, []),
                                    "model_id": "",
                                    "system_prompt": "",
                                }
                            )

                        return TeamComposition(
                            team_id="",
                            source="reused",
                            team_config_id=best.get("id", ""),
                            specialists=specialists,
                            execution_mode="sequential",
                            confidence=best.get("success_rate", 0.7),
                            reasoning=f"Reused proven team '{best.get('name', '')}' "
                            f"(success_rate={best.get('success_rate', 0):.2f})",
                        )
        except Exception as e:
            logger.debug("Team config reuse lookup failed: %s", e)

        return None

    def seed_default_topologies(self) -> int:
        """No-op. Default topologies are deprecated in favor of dynamic synthesis."""
        return 0
