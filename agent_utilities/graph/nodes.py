# agent_utilities/graph/nodes.py
from __future__ import annotations

import logging
from typing import cast

from pydantic_graph import End

from .client import get_graph_client
from .models import Policy, ProcessFlow
from .state import GraphState

try:
    from pydantic_graph.step import StepContext
except ImportError:  # pragma: no cover - older pydantic-graph
    from pydantic_graph.beta import StepContext  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


async def load_and_execute_process_flow(ctx: StepContext) -> str | End:
    """Dynamically load a ProcessFlow from the KG and stage it for the dispatcher.

    Builder step (CONCEPT:AU-ORCH.planning.recursion-nesting-depth). Reads the target flow id from ``state.current_flow_id``,
    loads the flow + its applicable policies into state, and dispatches to the ``dispatcher`` node.
    Returns the next node id (or ``End``), matching the other ``g.step`` functions — previously this
    was a ``BaseNode`` subclass registered via ``g.step``, which the builder API does not accept.
    """
    state = cast(GraphState, ctx.state)
    client = await get_graph_client()

    flow_id = state.current_flow_id
    if not flow_id:
        logger.warning("No flow_id provided for load_and_execute_process_flow")
        return "dispatcher"

    # 1. Retrieve the relevant flow + steps
    query = """
    MATCH (f:ProcessFlow {flow_id: $flow_id})
    MATCH (f)-[:HAS_START]->(start:ProcessStep)
    MATCH path = (f)-[*]->(step:ProcessStep)
    RETURN f, start, collect(DISTINCT step) as steps
    """
    results = client.execute(query, {"flow_id": flow_id})

    if not results:
        logger.error(f"ProcessFlow {flow_id} not found in Knowledge Graph")
        return "dispatcher"

    flow_data = results[0]
    flow = ProcessFlow.model_validate(flow_data["f"])

    # 2. Store in state for the Dispatcher
    state.current_process_flow = flow

    # 3. Load applicable Policies for this flow/context
    policy_query = """
    MATCH (p:Policy)-[:APPLIES_TO]->(f:ProcessFlow {flow_id: $flow_id})
    RETURN p
    """
    policies = client.execute(policy_query, {"flow_id": flow_id})
    state.active_policies = [Policy.model_validate(p["p"]) for p in policies]

    logger.info(
        f"Loaded ProcessFlow '{flow.name}' with {len(flow_data['steps'])} steps "
        f"and {len(state.active_policies)} policies."
    )

    # 4. Transition to the next logical step
    return "dispatcher"


async def find_best_matching_process_flow_via_kg(goal: str) -> ProcessFlow | None:
    """Search for a ProcessFlow that matches the current goal semantically."""
    client = await get_graph_client()
    query = """
    MATCH (f:ProcessFlow)
    WHERE f.goal CONTAINS $goal OR f.name CONTAINS $goal
    RETURN f LIMIT 1
    """
    results = client.execute(query, {"goal": goal})
    if results:
        return ProcessFlow.model_validate(results[0]["f"])
    return None
