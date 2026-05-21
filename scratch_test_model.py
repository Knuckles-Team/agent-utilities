import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.DEBUG)

from dotenv import load_dotenv

load_dotenv()

from pydantic_ai import Agent

from agent_utilities.core.model_factory import create_model


async def main():
    print("Creating model...")
    model = create_model(
        model_id="qwen/qwen3.5-9b",
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        provider=os.environ.get("LLM_PROVIDER"),
    )
    print(f"Model created: {model}")

    agent = Agent(model=model, system_prompt="You are a helpful assistant.")

    print("Running agent...")
    try:
        res = await agent.run("Hello, who are you? Respond with one short sentence.")
        print(f"Result: {res.output}")
    except Exception as e:
        print(f"Error running agent: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
