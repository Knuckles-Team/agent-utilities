#!/usr/bin/env python3
"""
Basic Agent Example

Demonstrates the simplest way to create and use an agent with agent-utilities.
"""

import asyncio

from agent_utilities import create_agent


async def main():
    """Create and run a basic agent."""

    # Create a simple agent with default workspace tools
    agent, _ = create_agent(name="BasicAgent")

    # Run the agent with a simple query
    result = await agent.run(
        "Hello! Can you help me understand this codebase?", message_history=[]
    )

    print("Agent Response:")
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
