import asyncio
from unittest.mock import MagicMock

from pydantic_graph.beta import StepContext

from agent_utilities.graph.state import GraphDeps, GraphState


async def run():
    state = GraphState(query="Calculate optimal Almgren-Chriss trajectory")
    deps = GraphDeps(tag_prompts={}, tag_env_vars={}, mcp_toolsets=[])
    deps.knowledge_engine = MagicMock()
    StepContext(state=state, deps=deps, inputs=())

    # We want to see what use_rlm is. Let's patch router_step to print it.
    pass


if __name__ == "__main__":
    asyncio.run(run())
