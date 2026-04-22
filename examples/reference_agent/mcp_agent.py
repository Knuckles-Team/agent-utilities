#!/usr/bin/env python3
"""
MCP Agent Example

Demonstrates MCP (Model Context Protocol) tool integration.
This allows the agent to use external tools from MCP servers.
"""

import asyncio
from pathlib import Path

from agent_utilities import create_agent


async def main():
    """Create and run an agent with MCP tools."""

    # Check if mcp_config.json exists
    config_path = Path("mcp_config.json")
    if not config_path.exists():
        print("Warning: mcp_config.json not found.")
        print("Create an mcp_config.json file with your MCP server configurations.")
        print("Example:")
        print("""
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"]
    }
  }
}
        """)
        return

    # Create an agent that uses MCP tools
    # The agent will automatically discover and load tools from mcp_config.json
    agent, _ = create_agent(
        name="MCPAgent",
        skill_types=["universal"],
        mcp_config="mcp_config.json",  # Path to your MCP configuration
    )

    # Run the agent with a query that requires external tools
    result = await agent.run(
        "Search for recent Python documentation and summarize the key changes.",
        message_history=[],
    )

    print("MCP Agent Response:")
    print(result.content)

    # The agent will:
    # 1. Discover available MCP tools from mcp_config.json
    # 2. Partition tools into focused specialist agents (~10-20 tools each)
    # 3. Register specialists as graph nodes
    # 4. Route queries to the appropriate specialist
    # 5. Execute tool calls with proper error handling and retries


if __name__ == "__main__":
    asyncio.run(main())
