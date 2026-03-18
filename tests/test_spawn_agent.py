import asyncio
import os
import sys

# Pre-requisite: we need agent_utilities
try:
    from agent_utilities.agent_utilities import create_agent
    from pydantic_ai import RunContext
except ImportError:
    print("agent_utilities is not installed or available.")
    sys.exit(1)

async def main():
    # 1. Create a parent agent
    print("Creating parent agent...")
    parent_agent = create_agent(
        name="TestParent",
        system_prompt="You are a parent agent. You MUST use the spawn_agent tool to ask a sub-agent to pick a random number between a provided min and max.",
    )

    # 2. Add the dynamic tools to the parent agent
    # Assuming dynamic tools is enabled by default or we can force it
    os.environ["DYNAMIC_TOOLS"] = "True"
    os.environ["GITLAB_URL"] = "https://gitlab.com"
    os.environ["GITLAB_PERSONAL_ACCESS_TOKEN"] = "dummy"
    # Note: create_agent automatically calls register_agent_tools inside its factory, no need to call it twice.

    # 3. Test running the parent agent, telling it to spawn a sub agent
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
