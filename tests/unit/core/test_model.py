import asyncio
import os

from agent_utilities.core.model_factory import create_model


async def test():
    print(f"Env: {os.environ.get('LLM_BASE_URL')}")
    try:
        model = create_model()
        print(f"Model: {model}")
        # result = await model.request("Hello") # This might be too much
    except Exception as e:
        print(f"Error: {e}")


asyncio.run(test())
