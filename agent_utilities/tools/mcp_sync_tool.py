#!/usr/bin/python
"""MCP Synchronization Tool Module.

This module provides a tool for manually triggering a synchronization
of the dynamic MCP specialist agent registry from the workspace configuration.
"""

import logging
from typing import Any

from pydantic_ai import RunContext

from ..mcp_agent_manager import sync_mcp_agents

logger = logging.getLogger(__name__)


async def trigger_mcp_sync(ctx: RunContext[Any], force_reprompt: bool = False) -> str:
    """Synchronize the MCP specialist agent registry with the current configuration.

    Updates the NODE_AGENTS.md file based on the available MCP servers
    defined in mcp_config.json.

    Args:
        ctx: The agent run context.
        force_reprompt: Whether to force an LLM-based regeneration of
                        specialist role definitions.

    Returns:
        A status message indicating success or failure.

    """
    try:
        await sync_mcp_agents(force_reprompt=force_reprompt)
        return "✅ MCP agent registry synchronized successfully."
    except Exception as e:
        logger.error(f"Manual MCP sync failed: {e}")
        return f"❌ MCP sync failed: {str(e)}"
