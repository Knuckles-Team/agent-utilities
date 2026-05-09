#!/usr/bin/python
"""KG-Driven Pydantic Graph Engine (CONCEPT:ORCH-1.20).

Shifts control of pydantic-graph workflows from Python logic to Knowledge Graph state transitions.
Every step polls the KG for the next optimal execution node instead of relying on statically compiled transitions.
"""

import logging
from typing import Any

from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class KGDrivenExecutionEngine:
    """Orchestrates dynamic pydantic-graph execution using the KG.

    CONCEPT:ORCH-1.20 — KG-Driven Pydantic Graph Engine
    """

    def __init__(self, engine: IntelligenceGraphEngine):
        self.engine = engine

    def determine_next_node(self, current_node_id: str, context: dict[str, Any]) -> str:
        """Poll the KG to determine the optimal next execution node based on current context.

        Args:
            current_node_id: The identifier of the node that just completed.
            context: The execution context containing findings, errors, and task state.

        Returns:
            The identifier of the next node to execute.
        """
        # If no backend is available, fallback to a simple sequential heuristic or end.
        if not self.engine.backend:
            logger.warning("No backend available. Falling back to default END node.")
            return "END"

        # Query the KG for potential transitions linked to the current state node
        query = """
        MATCH (current:ExecutionStateNode {node_id: $current_id})-[r:TRANSITION_TO]->(next:ExecutionStateNode)
        RETURN next.node_id as next_id, r.condition as condition, r.priority as priority
        ORDER BY r.priority DESC
        """

        results = self.engine.backend.execute(query, {"current_id": current_node_id})

        for row in results:
            condition = row.get("condition")
            # Evaluate the condition dynamically against the context.
            # In a production system, this would be a safe eval or pattern match.
            # For this KG-native engine, we check if the condition text matches context markers.
            if self._evaluate_condition(condition, context):
                next_id = row.get("next_id")
                if next_id:
                    logger.debug(
                        f"[ORCH-1.20] KG routing from {current_node_id} -> {next_id}"
                    )
                    return str(next_id)

        # Fallback to the dynamic orchestrator if no predefined static route matches
        logger.debug("[ORCH-1.20] No static route matched. Using dynamic synthesis.")
        task = context.get("task", "")
        if task:
            # Re-synthesize team or fetch next best from orchestrator capabilities
            # For now, default to END if we reach the edge of the graph.
            pass

        return "END"

    def _evaluate_condition(
        self, condition: str | None, context: dict[str, Any]
    ) -> bool:
        """Evaluate a KG string condition against the current execution context."""
        if not condition or condition.lower() == "always":
            return True
        if condition.lower() == "on_error" and context.get("error"):
            return True
        if condition.lower() == "on_success" and not context.get("error"):
            return True

        # More complex ontological evaluations could be plugged in here
        return False
