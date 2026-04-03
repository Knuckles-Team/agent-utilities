from pydantic_ai import Agent
import asyncio


async def test():
    agent = Agent("google-gla:gemini-1.5-flash")  # Dummy model
    try:
        # We don't actually run it, just check the result type hints if possible
        # Or run with a mock
        from pydantic_ai.models.test import TestModel

        agent = Agent(TestModel())
        res = await agent.run("hello")
        print(f"Result type: {type(res)}")
        print(f"Result attrs: {dir(res)}")
        if hasattr(res, "data"):
            print(f"Data: {res.data}")
    except Exception as e:
        print(f"Error: {e}")


asyncio.run(test())
