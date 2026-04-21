# agent_utilities/graph/nodes.py
import logging

from pydantic import BaseModel
from pydantic_graph import BaseNode, End, GraphRunContext

from .client import get_graph_client
from .models import Policy, ProcessFlow
from .state import GraphDeps, GraphState

logger = logging.getLogger(__name__)


class LoadAndExecuteProcessFlow(BaseModel, BaseNode[GraphState, GraphDeps, str | End]):
    """Dynamically loads a ProcessFlow from KG and turns it into executable pydantic-graph nodes.
    Called natively by the Planner as part of SDD. Policies can be applied as pre/post checks.
    """

    flow_id: str | None = None

    async def run(self, ctx: GraphRunContext[GraphState, GraphDeps]) -> str | End:
        state: GraphState = ctx.state
        client = await get_graph_client()

        flow_id = self.flow_id or state.current_flow_id
        if not flow_id:
            logger.warning("No flow_id provided for LoadAndExecuteProcessFlow")
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

        # 2. Store in state for Dispatcher
        state.current_process_flow = flow

        # 3. Load applicable Policies for this flow/context
        # (This can be used by the Dispatcher or Verifier later)
        policy_query = """
        MATCH (p:Policy)-[:APPLIES_TO]->(f:ProcessFlow {flow_id: $flow_id})
        RETURN p
        """
        policies = client.execute(policy_query, {"flow_id": flow_id})
        state.active_policies = [Policy.model_validate(p["p"]) for p in policies]

        logger.info(
            f"Loaded ProcessFlow '{flow.name}' with {len(flow_data['steps'])} steps and {len(state.active_policies)} policies."
        )

        # 4. Transition to the next logical step (e.g., Dispatcher)
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
