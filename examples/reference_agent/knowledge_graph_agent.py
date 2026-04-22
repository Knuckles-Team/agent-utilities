#!/usr/bin/env python3
"""
Knowledge Graph Agent Example

Demonstrates using the Knowledge Graph for context-aware reasoning.
The Knowledge Graph provides a 12-phase unified intelligence pipeline.
"""

import asyncio

from agent_utilities import create_agent


async def main():
    """Create and run an agent with knowledge graph integration."""

    # Create an agent with knowledge graph enabled
    agent, _ = create_agent(
        name="KnowledgeGraphAgent",
        skill_types=["universal", "graphs"],
    )

    # Run the agent with a query that requires codebase understanding
    result = await agent.run(
        "What are the main components of this codebase? "
        "How do they interact with each other?",
        message_history=[],
    )

    print("Knowledge Graph Agent Response:")
    print(result.content)

    # The knowledge graph will:
    # 1. Scan the filesystem and identify source code files
    # 2. Parse code using tree-sitter to extract symbols
    # 3. Resolve import dependencies
    # 4. Build a call graph
    # 5. Cluster nodes into modules using Louvain algorithm
    # 6. Run PageRank to identify critical components
    # 7. Generate semantic embeddings
    # 8. Sync to persistent graph storage (LadybugDB/Neo4j)
    # 9. Provide hybrid search capabilities
    # 10. Enable code impact analysis


if __name__ == "__main__":
    asyncio.run(main())
