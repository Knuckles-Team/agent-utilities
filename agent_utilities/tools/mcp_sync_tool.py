import logging
from typing import Any
from pydantic_ai import RunContext
from ..mcp_agent_manager import sync_mcp_agents

logger = logging.getLogger(__name__)


async def trigger_mcp_sync(ctx: RunContext[Any], force_reprompt: bool = False) -> str:
    """
    Manually triggers a synchronization of the MCP agent registry (MCP_AGENTS.md).
    Use this if you update the mcp_config.json or want to regenerate specialized agents.
    """
    try:
        await sync_mcp_agents(force_reprompt=force_reprompt)
        return "✅ MCP agent registry synchronized successfully."
    except Exception as e:
        logger.error(f"Manual MCP sync failed: {e}")
        return f"❌ MCP sync failed: {str(e)}"
