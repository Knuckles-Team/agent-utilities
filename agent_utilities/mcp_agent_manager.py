import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional

from .workspace import (
    get_workspace_path,
    CORE_FILES,
    load_workspace_file,
    write_workspace_file,
    parse_node_registry,
    serialize_node_registry,
)
from .models import MCPAgent, MCPAgentRegistryModel, MCPToolInfo
from .mcp_utilities import load_mcp_config

logger = logging.getLogger(__name__)


def should_sync(config_path: Path, agents_path: Path) -> bool:
    """
    Checks if MCP agents synchronization is required based on existence,
    content, and modification time of the config file vs the registry.
    """
    if not config_path.exists():
        return False

    if not agents_path.exists() or agents_path.stat().st_size == 0:
        return True

    # Check if the registry is actually empty (only headers)
    try:
        content = agents_path.read_text(encoding="utf-8")
        registry = parse_node_registry(content)
        if not registry.agents:
            return True
    except Exception:
        return True

    # Check if config is newer than registry
    config_mtime = config_path.stat().st_mtime
    agents_mtime = agents_path.stat().st_mtime

    if config_mtime > agents_mtime + 2.0:
        return True

    return False


async def _extract_single_server_metadata(
    server,
    mcp_servers_config: Dict,
    timeout: int = 300,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[MCPToolInfo]:
    """Helper to extract metadata from a single MCP server with timeout and fallback."""
    if semaphore:
        async with semaphore:
            return await _extract_single_server_metadata_inner(
                server, mcp_servers_config, timeout
            )
    return await _extract_single_server_metadata_inner(
        server, mcp_servers_config, timeout
    )


async def _extract_single_server_metadata_inner(
    server, mcp_servers_config: Dict, timeout: int = 300
) -> List[MCPToolInfo]:
    """Internal helper to extract metadata from a single MCP server with timeout and fallback."""
    server_name = getattr(server, "name", getattr(server, "_id", "unknown"))
    all_tools = []
    try:
        # Attempt Dynamic Extraction with timeout
        async with asyncio.timeout(timeout):
            async with server as session:
                result = await session.list_tools()
                # Handle both ListToolsResult object and raw List[Tool] (SDK version variance)
                tools_list = (
                    getattr(result, "tools", result)
                    if not isinstance(result, list)
                    else result
                )

                for tool in tools_list:
                    tags = []
                    if hasattr(tool, "annotations") and tool.annotations:
                        ann = tool.annotations
                        if isinstance(ann, dict):
                            tags_data = ann.get("tags") or ann.get("tag")
                        else:
                            tags_data = getattr(ann, "tags", None) or getattr(
                                ann, "tag", None
                            )
                        if isinstance(tags_data, (list, set, tuple)):
                            tags = [str(t) for t in tags_data]
                        elif isinstance(tags_data, str):
                            tags = [tags_data]

                    # Support FastMCP meta tags if annotations is missing (common in some protocol versions)
                    if (
                        not tags
                        and hasattr(tool, "meta")
                        and isinstance(tool.meta, dict)
                    ):
                        fastmcp_meta = tool.meta.get("fastmcp", {})
                        tags_data = fastmcp_meta.get("tags")
                        if isinstance(tags_data, (list, set, tuple)):
                            tags = [str(t) for t in tags_data]
                        elif isinstance(tags_data, str):
                            tags = [tags_data]

                    # Primary tag for grouping (partitioning)
                    tag = tags[0] if tags else None

                    if not tag:
                        # Heuristic: split by underscore or hyphen
                        import re

                        parts = re.split(r"[-_]", tool.name)
                        if len(parts) > 1:
                            # Skip common generic verbs to find a more descriptive domain tag
                            generic_verbs = {
                                "get",
                                "list",
                                "create",
                                "delete",
                                "update",
                                "remove",
                                "add",
                                "set",
                                "start",
                                "stop",
                                "restart",
                                "post",
                                "patch",
                                "put",
                            }
                            if parts[0].lower() in generic_verbs and len(parts) > 1:
                                # Use the next descriptive part (e.g., get_containers -> containers)
                                tag = parts[1].lower()
                            else:
                                # Use first part as domain tag
                                tag = parts[0].lower()

                    all_tools.append(
                        MCPToolInfo(
                            name=tool.name,
                            description=getattr(tool, "description", "") or "",
                            tag=tag or "general",
                            mcp_server=server_name,
                            all_tags=tags,
                        )
                    )
    except Exception as e:
        error_msg = str(e)
        # Handle ExceptionGroup (Python 3.11+) and report the first sub-exception
        if hasattr(e, "exceptions") and e.exceptions:
            error_msg = f"{type(e).__name__}({str(e.exceptions[0])})"

        logger.warning(
            f"Dynamic extraction failed for {server_name}, falling back to static hints: {error_msg}"
        )
        # Fallback: Parse env-based hints from the raw config
        server_cfg = mcp_servers_config.get(server_name, {})
        env = server_cfg.get("env", {})

        hints_found = False
        for key, val in env.items():
            if key.endswith("TOOL"):
                tag = key.lower().replace("tool", "")
                hints_found = True
                # Create a mock tool to represent this capability
                all_tools.append(
                    MCPToolInfo(
                        name=f"{server_name}_{tag}_toolset",
                        description=f"Static hint toolset for {tag} based on config env.",
                        tag=tag,
                        mcp_server=server_name,
                    )
                )

        if not hints_found:
            # Absolute fallback: one 'general' tag for the whole server
            all_tools.append(
                MCPToolInfo(
                    name=f"{server_name}_general_tools",
                    description=f"General tools for {server_name} (offline extraction).",
                    tag=server_name.replace("-mcp", "").replace("-agent", ""),
                    mcp_server=server_name,
                )
            )

    return all_tools


async def extract_tool_metadata(
    config_path: Path, timeout: int = 300
) -> List[MCPToolInfo]:
    """Connect to MCP servers in parallel and extract tool metadata with static fallback."""
    if not config_path.exists():
        logger.warning(f"MCP config not found at {config_path}")
        return []

    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
            mcp_servers_config = config_data.get("mcpServers", {})
    except Exception as e:
        logger.error(f"Failed to read raw config for static fallback: {e}")
        mcp_servers_config = {}

    servers = load_mcp_config(config_path)

    # Sequential loading to avoid AnyIO cross-task cancel scope errors
    # (FastMCP uses anyio.create_task_group internally which conflicts with asyncio.gather)
    all_tools = []
    for server in servers:
        server_tools = await _extract_single_server_metadata_inner(
            server, mcp_servers_config, timeout=timeout
        )
        all_tools.extend(server_tools)

    return all_tools


async def partition_tools(tools: List[MCPToolInfo]) -> Dict[str, List[MCPToolInfo]]:
    """Group tools into logical agent partitions by tag or LLM classification."""
    partitions = {}

    # Primary partitioning by TAG (Multi-tag support)
    untracked_tools = []
    for tool in tools:
        tags = tool.all_tags if tool.all_tags else ([tool.tag] if tool.tag else [])
        if tags:
            for tag in tags:
                if tag not in partitions:
                    partitions[tag] = []
                partitions[tag].append(tool)
        else:
            untracked_tools.append(tool)

    # Secondary partitioning via LLM for untracked tools
    if untracked_tools:
        # For untracked tools, we group by MCP server name as a safe fallback
        for tool in untracked_tools:
            tag = f"{tool.mcp_server}_general"
            if tag not in partitions:
                partitions[tag] = []
            partitions[tag].append(tool)

    return partitions


async def generate_system_prompt(
    agent_name: str, tools: List[MCPToolInfo], tag: str, server_name: str
) -> str:
    """Generate a specialized system prompt deterministically based on server and tag."""
    clean_server = (
        server_name.replace("-mcp", "").replace("-agent", "").replace("_", " ").title()
    )
    clean_tag = tag.replace("_", " ").title()

    # Improve naming: ensure the server name is part of the specialist identity if not already present
    if clean_server.lower() in clean_tag.lower():
        specialist_name = clean_tag
    else:
        specialist_name = f"{clean_server} {clean_tag}"

    return (
        f"You are a {specialist_name} specialist. "
        f"Help users manage and interact with {clean_tag} functionality using the available tools."
    )


async def sync_mcp_agents(
    force_reprompt: bool = False, config_path: Optional[Path] = None
):
    """Main synchronization engine for MCP agents and the NODE_AGENTS.md registry."""
    if not config_path:
        config_path = get_workspace_path(CORE_FILES["MCP_CONFIG"])

    # 1. Extract Tool Metadata
    tools_inventory = await extract_tool_metadata(config_path)
    if not tools_inventory:
        logger.info("No tools found to sync.")
        return

    # 2. Partition Tools
    partitions = await partition_tools(tools_inventory)

    # 3. Load Current Registry
    content = load_workspace_file(CORE_FILES["NODE_AGENTS"])
    current_registry = parse_node_registry(content)

    new_agents = []

    # Mapping of existing custom agents to preserve them
    custom_agents = {a.name: a for a in current_registry.agents if a.is_custom}
    existing_generated_agents = {
        a.name: a for a in current_registry.agents if not a.is_custom
    }

    # 4. Process Partitions
    for tag, group in partitions.items():
        server_name = group[0].mcp_server
        clean_server = (
            server_name.replace("-mcp", "")
            .replace("-agent", "")
            .replace("_", " ")
            .title()
        )

        # Handle partitioned suffixes (e.g. group_1, group_2) to keep names unique but descriptive
        clean_tag = tag.replace("_", " ").title()

        if clean_server.lower() in clean_tag.lower():
            agent_name = f"{clean_tag} Specialist"
        else:
            agent_name = f"{clean_server} {clean_tag} Specialist"

        # Check if we already have a custom version
        if agent_name in custom_agents:
            # Preserve custom agent BUT update its tools/tag metadata if needed?
            # User wants to be able to modify, so we should keep their custom version
            new_agents.append(custom_agents[agent_name])
            continue

        # Check if we should regenerate or create new
        if agent_name in existing_generated_agents and not force_reprompt:
            existing = existing_generated_agents[agent_name]
            # Update tools just in case
            existing.tools = [t.name for t in group]
            existing.tag = tag
            new_agents.append(existing)
        else:
            # Create NEW agent
            logger.info(
                f"Generating deterministic system prompt for NEW agent: {agent_name}"
            )
            description = f"Expert specialist for {tag} domain tasks."
            system_prompt = await generate_system_prompt(
                agent_name, group, tag, server_name
            )

            new_agents.append(
                MCPAgent(
                    name=agent_name,
                    description=description,
                    system_prompt=system_prompt,
                    tools=[t.name for t in group],
                    mcp_server=group[0].mcp_server,
                    tag=tag,
                    is_custom=False,
                )
            )

    # 5. Update Registry Model
    updated_registry = MCPAgentRegistryModel(agents=new_agents, tools=tools_inventory)

    # 6. Write back to disk
    new_content = serialize_node_registry(updated_registry)
    write_workspace_file(CORE_FILES["NODE_AGENTS"], new_content)
    logger.info(
        f"✅ Synced {len(new_agents)} agents and {len(tools_inventory)} tools to NODE_AGENTS.md"
    )


if __name__ == "__main__":
    # Setup basic logging when run directly
    logging.basicConfig(level=logging.INFO)
    asyncio.run(sync_mcp_agents())
