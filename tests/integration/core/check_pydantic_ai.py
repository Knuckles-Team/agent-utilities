"""CONCEPT:AU-ORCH.execution.inject-signal-board-observations"""

import asyncio

from pydantic_ai.models.test import TestModel

from agent_utilities.core.contextual_model import create_context_agent


async def test():
    try:
        # We don't actually run it, just check the result type hints if possible
        # Or run with a mock
        agent = create_context_agent(TestModel())
        res = await agent.run("hello")
        print(f"Result type: {type(res)}")
        print(f"Result attrs: {dir(res)}")
        if hasattr(res, "data"):
            print(f"Data: {res.data}")
    except Exception as e:
        print(f"Error: {e}")


asyncio.run(test())
