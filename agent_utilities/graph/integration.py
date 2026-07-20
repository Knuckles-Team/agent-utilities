#!/usr/bin/python
"""Graph Integration Layer.

This module provides integration points between the graph orchestration (HSM/BT)
and the Knowledge Graph (LadybugDB), including automatic outcome recording
and self-improvement triggers.
"""

import logging
from typing import Any

from agent_utilities.observability.trace_ontology import (
    OUTCOME_NODE_LABEL,
    TRACE_PRODUCED_OUTCOME_EDGE,
    next_event_sequence,
    outcome_id,
    outcome_properties,
    trace_id,
)

from ..knowledge_graph.core.engine import IntelligenceGraphEngine
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

    # Calculate a basic reward: 1.0 for success, 0.0 for failure
    reward = 1.0 if success else 0.0

    # If it succeeded but took too long, maybe penalize?
    # (Just an example of potential reward shaping)
    if success and duration > 30.0:
        reward = 0.8

    run_id = str(
        getattr(state, "run_id", "")
        or getattr(state, "session_id", "")
        or getattr(deps, "request_id", "")
    )
    if not run_id:
        return
    import time

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tid = trace_id(run_id)
    oid = outcome_id(run_id)
    props = outcome_properties(
        run_id=run_id,
        status="completed" if success else "failed",
        timestamp=ts,
        event_sequence=next_event_sequence(),
        feedback="execution_completed" if success else "execution_failed",
        reward=reward,
    )
    props["duration_ms"] = max(0.0, float(duration) * 1000.0)

    try:
        engine.add_node(oid, OUTCOME_NODE_LABEL, properties=props)
        engine.link_nodes(tid, oid, TRACE_PRODUCED_OUTCOME_EDGE)
        logger.debug("Recorded normalized execution outcome")
    except Exception as e:
        logger.warning(f"Failed to record outcome to graph: {e}")


def initialize_integration():
    """Initialize the graph integration hooks."""
    register_on_exit_hook(record_specialist_outcome_hook)
    logger.info("Graph integration hooks initialized.")
