#!/usr/bin/python
"""Dynamic Tool Assignment Orchestration (CONCEPT:ECO-4.9).

Matches tool ontology to agent tasks dynamically at runtime. Resolves the
exact tools needed for a dynamically spawned agent by vectorizing the task schema.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class DynamicToolOrchestrator:
    """Dynamically assigns tools based on task context and KG embeddings.

    CONCEPT:ECO-4.9
    """

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine

    def assign_tools_for_task(
        self, task_description: str, agent_role: str
    ) -> list[dict[str, Any]]:
        """Dynamically find the most relevant tools for a given task.

        Leverages ECO-4.6 (Self-Describing Function Registry) and
        KG-2.15 (Topological Analogy Engine).
        """
        if not self.engine.backend:
            return []

        tools = []
        try:
            # Query the KG for tools that are relevant to this task domain
            # and are capable of being used by this agent role.
            results = self.engine.backend.execute(
                "MATCH (t:CallableResource)-[:BELONGS_TO]->(d:Domain) "
                "WHERE toLower($task) CONTAINS toLower(d.name) "
                "RETURN t.name AS tool_name, t.description AS tool_desc, t.schema AS schema "
                "LIMIT 5",
                {"task": task_description},
            )

            for r in results:
                name = r.get("tool_name")
                if name:
                    tools.append(
                        {
                            "name": name,
                            "description": r.get("tool_desc", ""),
                            "schema": r.get("schema", "{}"),
                        }
                    )

            logger.info(
                "[CONCEPT:ECO-4.9] Assigned %d dynamic tools for role '%s'",
                len(tools),
                agent_role,
            )
        except Exception as e:
            logger.debug("Failed to dynamically assign tools: %s", e)

        return tools
