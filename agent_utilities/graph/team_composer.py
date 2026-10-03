#!/usr/bin/python
from __future__ import annotations

"""KG-Driven Team Composer (CONCEPT:AU-ORCH.dispatch.kg-team-composition).

Assembles specialist teams from Knowledge Graph topology instead of
static ``discover_agents()`` registration.  The KG becomes the single
source of truth for *who* participates in a task, *what tools* they get,
*which model* they use, and *how* they collaborate.

Composition synthesizes the team from KG topology (tool affinity, semantic
relevance, server health, domain alignment) through
``AgentOrchestrationEngine.synthesize_team``. There is no success-rate reuse
and no promotion of outcomes (AU-CONTROL-R014, R015): reusing a composition
is L3 promotion under policy, and learning which topology works is EG's
calibrated, independently labelled, slate-credited statistical rung -- never
a self-reported threshold.

This replaces the ad-hoc ``if deps.knowledge_engine:`` pattern in
``routing.py`` with a single call: ``composer.compose_team(query, deps)``.
"""


import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine
    from ..models.knowledge_graph import TeamComposition

logger = logging.getLogger(__name__)


class KGTeamComposer:
    """Assembles specialist teams from KG topology.

    CONCEPT:AU-ORCH.dispatch.kg-team-composition — KG-Driven Team Composition

    This is the primary entry point for KG-driven orchestration.
    Instead of static agent registration, the KG dynamically determines
    which adaptive_agent_router participate, what tools they receive, and how they
    collaborate.

    Args:
        engine: The IntelligenceGraphEngine for KG queries.
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None):
        self.engine = engine

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

        Synthesizes the subgraph from KG primitives and returns a fully
        specified ``TeamComposition``.
        """
        # Dynamically synthesize the topology (ORCH-1.19).
        from ..orchestration.engine import AgentOrchestrationEngine

        orchestrator = AgentOrchestrationEngine(engine=self.engine)
        composition = orchestrator.synthesize_team(
            query=query,
            domain=domain,
            complexity=complexity,
            available_tools=available_tools,
            available_agents=available_agents,
            delegated_authority=delegated_authority,
        )

        return composition
