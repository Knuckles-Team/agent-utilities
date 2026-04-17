#!/usr/bin/python
# coding: utf-8
"""MCP Agent Manager Module.

This module manages the lifecycle of agents derived from MCP servers. It
handles the extraction of tool metadata from running servers, partitioning
tools into logical domain specialists, deterministic tool relevance scoring,
and synchronizing these specialists with the NODE_AGENTS.md registry.
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Any

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
from .tool_guard import is_sensitive_tool

logger = logging.getLogger(__name__)


def should_sync(config_path: Path, agents_path: Path) -> bool:
    """Determine if a synchronization of MCP agents is required.

    Compares the modification times of the MCP config and the agent registry
    to decide if updates are necessary.

    Args:
        config_path: Path to the mcp_config.json file.
        agents_path: Path to the NODE_AGENTS.md registry file.

    Returns:
        True if synchronization is needed, False otherwise.

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
    server: Any,
    mcp_servers_config: Dict,
    timeout: int = 300,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[MCPToolInfo]:
    """Wrapper for metadata extraction with optional concurrency control.

    Args:
        server: The MCP server instance.
        mcp_servers_config: Raw configuration dictionary for fallback hints.
        timeout: Execution timeout in seconds.
        semaphore: Optional semaphore to limit parallel connections.

    Returns:
        A list of extracted MCPToolInfo objects.

    """
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
    """Internal logic for connecting to an MCP server and listing tools.

    Uses a robust fallback mechanism:
    1. Dynamic extraction via session.list_tools().
    2. Environmental metadata hints from the config.
    3. Heuristic tagging based on tool name prefixes.

    Args:
        server: The MCP server instance.
        mcp_servers_config: Raw configuration dictionary.
        timeout: Timeout in seconds.

    Returns:
        A list of extracted MCPToolInfo objects.

    """
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
                            requires_approval=is_sensitive_tool(tool.name),
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
    """Load MCP servers and extract tool metadata in parallel.

    Args:
        config_path: Path to the mcp_config.json.
        timeout: Per-server connection timeout.

    Returns:
        A unified list of all discovered MCPToolInfo objects.

    """
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


def compute_agent_metadata_score(description: str, skills: List[str]) -> int:
    """Compute a deterministic relevance score for an agent based on metadata.

    Args:
        description: The agent's specialization description.
        skills: List of skills/capabilities.

    Returns:
        An integer score between 0 and 100.

    """
    score = 0
    # Description quality (0-50)
    dlen = len(description or "")
    if dlen > 150:
        score += 50
    elif dlen > 80:
        score += 40
    elif dlen > 40:
        score += 20
    elif dlen > 0:
        score += 5

    # Skills quality (0-50)
    slen = len(skills or [])
    if slen > 10:
        score += 50
    elif slen > 5:
        score += 40
    elif slen > 2:
        score += 20
    elif slen > 0:
        score += 10

    return min(100, score)


def compute_tool_relevance_score(tool: MCPToolInfo) -> int:
    """Compute a deterministic relevance score for a tool (0-100).

    The score reflects how well-described, well-tagged, and specific the
    tool is.  Higher scores indicate tools that are more likely to be
    correctly routed and effectively used by specialist agents.

    Scoring breakdown (deterministic, no LLM):

    * **Description quality (0-30)**: Length and keyword richness of the
      description field.
    * **Tag confidence (0-30)**: Annotation-derived tags score higher
      than heuristic-derived tags.
    * **Name specificity (0-20)**: Longer, multi-segment names that avoid
      generic verbs are more specific.
    * **Multi-tag coverage (0-20)**: Tools with multiple tags can serve
      more specialist domains.

    Args:
        tool: The tool metadata to score.

    Returns:
        An integer score between 0 and 100 inclusive.

    """
    score = 0

    # --- Description quality (0-30) ---
    desc = tool.description or ""
    desc_len = len(desc)
    if desc_len > 100:
        score += 30
    elif desc_len > 50:
        score += 20
    elif desc_len > 15:
        score += 10
    elif desc_len > 0:
        score += 5

    # --- Tag confidence (0-30) ---
    if tool.all_tags:
        # Multiple explicit tags = highest confidence (from annotations)
        if len(tool.all_tags) >= 2:
            score += 30
        else:
            tag_val = tool.all_tags[0]
            # Heuristic-derived tags are usually single lowercase words
            if "_" in tag_val or len(tag_val) > 6:
                score += 25
            else:
                score += 15
    elif tool.tag:
        score += 10
    # No tag at all = 0

    # --- Name specificity (0-20) ---
    name = tool.name or ""
    generic_verbs = {"get", "list", "create", "update", "delete", "set", "run"}
    segments = [s for s in name.replace("-", "_").split("_") if s]
    meaningful = [s for s in segments if s.lower() not in generic_verbs and len(s) > 2]
    if len(meaningful) >= 3:
        score += 20
    elif len(meaningful) >= 2:
        score += 15
    elif len(meaningful) >= 1:
        score += 10
    elif segments:
        score += 5

    # --- Multi-tag coverage (0-20) ---
    tag_count = len(tool.all_tags)
    if tag_count >= 3:
        score += 20
    elif tag_count == 2:
        score += 15
    elif tag_count == 1:
        score += 10

    return min(score, 100)


def score_tools(tools: List[MCPToolInfo]) -> List[MCPToolInfo]:
    """Apply deterministic relevance scoring to all tools in-place.

    Args:
        tools: The list of tool metadata to score.

    Returns:
        The same list with ``relevance_score`` populated on each item.

    """
    for tool in tools:
        tool.relevance_score = compute_tool_relevance_score(tool)
    return tools


async def partition_tools(tools: List[MCPToolInfo]) -> Dict[str, List[MCPToolInfo]]:
    """Group tools into logical domain partitions using tags.

    Args:
        tools: List of all extracted tool information.

    Returns:
        A dictionary mapping tag names to their specific tool lists.

    """
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
    """Generate a deterministic system prompt for a partitioned agent.

    Args:
        agent_name: Preferred name of the specialist agent.
        tools: The tools assigned to this specialist.
        tag: The partition tag.
        server_name: The source MCP server name.

    Returns:
        A system prompt string describing the specialist's role.

    """
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
    """Orchestrate the full synchronization of MCP servers with the agent registry.

    Performs metadata extraction, tool partitioning, and registry updates
    in NODE_AGENTS.md.

    Args:
        force_reprompt: Whether to regenerate system prompts for existing agents.
        config_path: Optional path override for mcp_config.json.

    """
    if not config_path:
        config_path = get_workspace_path(CORE_FILES["MCP_CONFIG"])

    # 1. Extract Tool Metadata
    tools_inventory = await extract_tool_metadata(config_path)
    if not tools_inventory:
        logger.info("No tools found to sync.")
        return

    # 1b. Score all tools deterministically
    score_tools(tools_inventory)
    avg_score = (
        sum(t.relevance_score for t in tools_inventory) // len(tools_inventory)
        if tools_inventory
        else 0
    )
    logger.info(
        f"Tool scoring complete: {len(tools_inventory)} tools, "
        f"avg relevance {avg_score}/100"
    )

    # 2. Partition Tools
    partitions = await partition_tools(tools_inventory)

    # 3. Load Current Registry
    from .agent_registry_builder import rebuild_node_agents_md

    # Trigger full rebuild first to ensure we have latest prompts/A2A
    await rebuild_node_agents_md()

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
            existing.mcp_tools = tag
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
                    agent_type="mcp",
                    endpoint_url="stdio" if "command" in group[0].mcp_server else "-",
                    description=description,
                    system_prompt=system_prompt,
                    tools=[t.name for t in group],
                    mcp_server=group[0].mcp_server,
                    mcp_tools=tag,
                    is_custom=False,
                    tool_count=len(group),
                    avg_relevance_score=(
                        sum(t.relevance_score for t in group) // len(group)
                        if group
                        else 0
                    ),
                )
            )

    # 5. Update Registry Model (Merge back prompt agents from before)
    # We should preserve agents from the current_registry that are NOT in existing_generated_agents or custom_agents of this run
    prompt_agents = [a for a in current_registry.agents if a.agent_type == "prompt"]
    a2a_agents = [a for a in current_registry.agents if a.agent_type == "a2a"]

    updated_registry = MCPAgentRegistryModel(
        agents=prompt_agents + a2a_agents + new_agents, tools=tools_inventory
    )

    # 6. Write back to disk
    new_content = serialize_node_registry(updated_registry)
    write_workspace_file(CORE_FILES["NODE_AGENTS"], new_content)
    logger.info(
        f"✅ Synced {len(new_agents)} MCP specialists and updated unified registry."
    )


if __name__ == "__main__":
    # Setup basic logging when run directly
    logging.basicConfig(level=logging.INFO)
    asyncio.run(sync_mcp_agents())
