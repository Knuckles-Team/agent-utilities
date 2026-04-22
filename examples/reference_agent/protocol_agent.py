#!/usr/bin/env python3
"""
Protocol Agent Example

Demonstrates using protocol adapters (ACP, A2A, AG-UI) for agent communication.
"""

import asyncio

from agent_utilities import create_agent


async def main():
    """Create and run an agent with protocol support."""

    # Create an agent with ACP (Agent Communication Protocol) support
    agent, _ = create_agent(name="ProtocolAgent", skill_types=["universal"])

    # Run the agent with a query
    result = await agent.run(
        "Help me understand the ACP protocol and how it enables agent coordination.",
        message_history=[],
    )

    print("Protocol Agent Response:")
    print(result.content)

    # The protocol adapters provide:
    # 1. ACP: Agent Communication Protocol for standardized sessions and planning
    # 2. A2A: Agent-to-Agent communication for peer-to-peer messaging
    # 3. AG-UI: Legacy streaming interface for native Pydantic AI clients
    # 4. SSE: Server-Sent Events for real-time streaming
    # 5. Human-in-the-loop: Tool approval and elicitation support


if __name__ == "__main__":
    asyncio.run(main())
