#!/usr/bin/env python3
"""
Memory Agent Example

Demonstrates using memory primitives and the Knowledge Base for long-term memory.
"""

import asyncio

from agent_utilities import create_agent


async def main():
    """Create and run an agent with memory primitives."""

    # Create an agent with knowledge base enabled
    agent, _ = create_agent(name="MemoryAgent", skill_types=["universal"])

    # Run the agent with a query that requires memory
    result = await agent.run(
        "Save this important decision: We should use Pydantic for all data validation. "
        "Then retrieve it later.",
        message_history=[],
    )

    print("Memory Agent Response:")
    print(result.content)

    # The memory primitives provide:
    # 1. Short-term memory: Current session context
    # 2. Long-term memory: Persistent knowledge graph storage
    # 3. Shared team memory: Collaborative memory across agents
    # 4. Knowledge Base: LLM-maintained wiki system
    # 5. MAGMA orthogonal views: Semantic, Temporal, Causal, Entity
    # 6. Agent Lightning: Self-improvement loops


if __name__ == "__main__":
    asyncio.run(main())
