import asyncio
import os
import sys


try:
    from agent_utilities.agent_utilities import create_agent
    from pydantic_ai import RunContext
except ImportError:
    print("agent_utilities is not installed or available.")
    sys.exit(1)

async def main():

    print("Creating parent agent...")
    parent_agent = create_agent(
        name="TestParent",
        system_prompt="You are a parent agent. You MUST use the spawn_agent tool to ask a sub-agent to pick a random number between a provided min and max.",
    )



    os.environ["DYNAMIC_TOOLS"] = "True"
    os.environ["GITLAB_URL"] = "https://gitlab.com"
    os.environ["GITLAB_PERSONAL_ACCESS_TOKEN"] = "dummy"



    print("Running parent agent and asking it to spawn a gitlab sub-agent...")
    prompt = "Please spawn a sub-agent using the 'gitlab' template. Just ask it 'What is your name and what do you do?' and return its exact response."

    result = await parent_agent.run(prompt)
    print("="*40)
    print("Final Result from Parent Agent:")
    print("="*40)
    data = getattr(result, 'data', None)
    if data is None:
        data = getattr(result, 'result', result)
    print(data)

if __name__ == "__main__":
    asyncio.run(main())
