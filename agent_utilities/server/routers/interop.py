import json
import logging
from contextlib import suppress
from pathlib import Path

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Interoperability"])


@router.get("/mcp/config", summary="Get MCP Configuration")
async def get_mcp_config():
    """Returns the current mcp_config.json contents."""
    from agent_utilities.core.workspace import CORE_FILES as _cf
    from agent_utilities.core.workspace import get_workspace_path

    mcp_config_path = get_workspace_path(_cf.get("MCP_CONFIG", "mcp_config.json"))
    if not mcp_config_path.exists():
        # Fallback to local agent_data/mcp_config.json if not in workspace
        mcp_config_path = (
            Path(__file__).parent.parent.parent / "agent_data" / "mcp_config.json"
        )

    if mcp_config_path.exists():
        try:
            return json.loads(mcp_config_path.read_text(encoding="utf-8"))
        except Exception:
            return {"mcpServers": {}}
    return {"mcpServers": {}}


@router.get("/mcp/tools", summary="List Available MCP Tools")
async def list_mcp_tools(request: Request):
    """Returns a list of all tools from all connected MCP servers."""
    tools = []
    agent_instance = getattr(request.app.state, "agent_instance", None)
    if agent_instance and hasattr(agent_instance, "toolsets"):
        for ts in agent_instance.toolsets:
            # Skip the SkillsToolset which is handled separately via A2A if needed
            if type(ts).__name__ == "SkillsToolset":
                continue

            # For MCPServer toolsets, we can extract tool info
            if hasattr(ts, "get_tools"):
                with suppress(Exception):
                    # Some toolsets might be async or require a context
                    ts_tools = ts.get_tools()
                    for t in ts_tools:
                        tools.append(
                            {
                                "name": getattr(t, "name", str(t)),
                                "description": getattr(t, "description", ""),
                                "tag": getattr(
                                    ts, "name", "mcp"
                                ),  # Use toolset name as tag
                            }
                        )
    return tools


@router.post("/mcp/reload", summary="Hot-reload MCP servers and rebuild graph")
async def reload_mcp_config(request: Request):
    """Re-sync MCP agents from config and rebuild graph without restarting."""
    try:
        from agent_utilities.core.workspace import resolve_mcp_config_path
        from agent_utilities.mcp.agent_manager import sync_mcp_agents

        from ...graph_orchestration import load_node_agents_registry

        mcp_config = getattr(request.app.state, "mcp_config", "mcp_config.json")
        _mcp_cfg_path = resolve_mcp_config_path(mcp_config or "mcp_config.json")
        if _mcp_cfg_path:
            await sync_mcp_agents(config_path=_mcp_cfg_path)
        registry = load_node_agents_registry()
        return {
            "status": "reloaded",
            "agents": len(registry.agents),
            "tools": len(registry.tools),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
