#!/usr/bin/python
"""Graph Integration Layer.

This module provides integration points between the graph orchestration (HSM/BT)
and the Knowledge Graph (LadybugDB), including automatic outcome recording
and self-improvement triggers.
"""

import logging
import time
from typing import Any

from ..knowledge_graph.engine import IntelligenceGraphEngine
from ..models.knowledge_graph import OutcomeEvaluationNode, RegistryNodeType
from .hsm import register_on_exit_hook

logger = logging.getLogger(__name__)


async def record_specialist_outcome_hook(
    deps: Any,
    state: Any,
    agent_name: str,
    success: bool,
    server_name: str,
    duration: float,
) -> None:
    """Hook to record the outcome of a specialist execution in the Knowledge Graph.

    This creates a basic OutcomeEvaluationNode for every specialist execution,
    providing the raw data for downstream self-improvement cycles.
    """
    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return

    import uuid

    outcome_id = f"outcome:{uuid.uuid4().hex[:8]}"

    # Calculate a basic reward: 1.0 for success, 0.0 for failure
    reward = 1.0 if success else 0.0

    # If it succeeded but took too long, maybe penalize?
    # (Just an example of potential reward shaping)
    if success and duration > 30.0:
        reward = 0.8

    node = OutcomeEvaluationNode(
        id=outcome_id,
        type=RegistryNodeType.OUTCOME_EVALUATION,
        name=f"Outcome: {agent_name}",
        reward=reward,
        success_criteria_met=["execution_completed"] if success else [],
        feedback_text=f"Agent '{agent_name}' {'succeeded' if success else 'failed'} in {duration:.2f}s.",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    try:
        # Add node to graph
        engine.graph.add_node(node.id, **node.model_dump())

        # Link to agent if we can find it in the graph
        # Note: agent_name here is usually the tag or name
        agent_matches = [
            n for n, d in engine.graph.nodes(data=True) if d.get("name") == agent_name
        ]
        if agent_matches:
            engine.graph.add_edge(agent_matches[0], node.id, type="PRODUCED_OUTCOME")

        # Link to current episode if available in state or deps
        episode_id = getattr(state, "session_id", getattr(deps, "request_id", None))
        if episode_id and episode_id in engine.graph:
            engine.graph.add_edge(episode_id, node.id, type="PRODUCED_OUTCOME")

        logger.debug(f"Recorded outcome '{outcome_id}' for agent '{agent_name}'")
    except Exception as e:
        logger.warning(f"Failed to record outcome to graph: {e}")


def initialize_integration():
    """Initialize the graph integration hooks."""
    register_on_exit_hook(record_specialist_outcome_hook)
    logger.info("Graph integration hooks initialized.")
