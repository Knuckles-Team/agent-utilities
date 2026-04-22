#!/usr/bin/env python3
"""
Graph Agent Example

Demonstrates the Graph Orchestration with Router → Planner → Dispatcher pipeline.
This is the recommended approach for complex, multi-step tasks.
"""

import asyncio

from agent_utilities import create_agent


async def main():
    """Create and run a graph agent with universal skills."""

    # Create a Graph Agent with Universal Skills
    # This automatically discovers domain specialists from registries
    agent, _ = create_agent(name="GraphAgent", skill_types=["universal", "graphs"])

    # Run the agent with a complex query that requires orchestration
    result = await agent.run(
        "Analyze this codebase and identify opportunities for improvement. "
        "Focus on code quality, performance, and architecture.",
        message_history=[],
    )

    print("Graph Agent Response:")
    print(result.content)

    # The graph orchestrator will:
    # 1. Route the query to the appropriate specialist
    # 2. Plan the approach
    # 3. Dispatch to domain specialists in parallel
    # 4. Verify the results
    # 5. Synthesize the final response


if __name__ == "__main__":
    asyncio.run(main())
