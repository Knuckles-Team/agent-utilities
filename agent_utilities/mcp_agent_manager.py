#!/usr/bin/python
"""MCP Agent Manager Module.

This module manages the lifecycle of agents derived from MCP servers. It
handles the extraction of tool metadata from running servers, partitioning
tools into logical domain specialists, deterministic tool relevance scoring,
and synchronizing these specialists directly with the Knowledge Graph.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .mcp_utilities import load_mcp_config
from .models import MCPToolInfo
from .tool_guard import is_sensitive_tool
from .workspace import (
    CORE_FILES,
    get_workspace_path,
)

logger = logging.getLogger(__name__)


def should_sync(config_path: Path) -> bool:
    """Determine if a synchronization of MCP agents is required.

    Compares the modification time of the MCP config with the last sync
    recorded in the Knowledge Graph.

    Args:
        config_path: Path to the mcp_config.json file.

    Returns:
        True if synchronization is needed, False otherwise.

    """
    if not config_path.exists():
        return False

    from .knowledge_graph.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return True

    # Check if we have any tools and when they were last synced
    try:
        res = engine.backend.execute(
            "MATCH (t:Tool) RETURN max(t.last_sync) as last_sync"
        )
        last_sync = res[0].get("last_sync") if res else 0
        if not last_sync:
            return True

        config_mtime = config_path.stat().st_mtime
        if config_mtime > last_sync + 2.0:
            return True
    except Exception:
        return True

    return False


async def _extract_single_server_metadata(
    server: Any,
    mcp_servers_config: dict,
    timeout: int = 300,
    semaphore: asyncio.Semaphore | None = None,
) -> list[MCPToolInfo]:
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
    server, mcp_servers_config: dict, timeout: int = 300
) -> list[MCPToolInfo]:
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
) -> list[MCPToolInfo]:
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
        with open(config_path) as f:
            config_data = json.load(f)
            mcp_servers_config = config_data.get("mcpServers", {})
    except Exception as e:
        logger.error(f"Failed to read raw config for static fallback: {e}")
        mcp_servers_config = {}

    servers = load_mcp_config(config_path)

    # Parallel extraction using anyio to handle FastMCP/anyio internals correctly
    import anyio

    all_tools: list[MCPToolInfo] = []
    semaphore = anyio.Semaphore(5)  # Limit to 5 concurrent connections

    async def _safe_extract(server, results):
        async with semaphore:
            server_tools = await _extract_single_server_metadata_inner(
                server, mcp_servers_config, timeout=timeout
            )
            results.extend(server_tools)

    async with anyio.create_task_group() as tg:
        for server in servers:
            tg.start_soon(_safe_extract, server, all_tools)

    return all_tools


def compute_agent_metadata_score(description: str, skills: list[str]) -> int:
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


def score_tools(tools: list[MCPToolInfo]) -> list[MCPToolInfo]:
    """Apply deterministic relevance scoring to all tools in-place.

    Args:
        tools: The list of tool metadata to score.

    Returns:
        The same list with ``relevance_score`` populated on each item.

    """
    for tool in tools:
        tool.relevance_score = compute_tool_relevance_score(tool)
    return tools


async def partition_tools(tools: list[MCPToolInfo]) -> dict[str, list[MCPToolInfo]]:
    """Group tools into logical domain partitions using tags.

    Args:
        tools: List of all extracted tool information.

    Returns:
        A dictionary mapping tag names to their specific tool lists.

    """
    partitions: dict[str, list[MCPToolInfo]] = {}

    # Primary partitioning by TAG (Multi-tag support)
    for tool in tools:
        # Sanitize server name for better agent identity (e.g. repository-manager -> repository)
        server_tag = (
            tool.mcp_server.lower()
            .replace("-mcp", "")
            .replace("_mcp", "")
            .replace("-manager", "")
            .replace("-agent", "")
            .replace("-server", "")
        )

        tags = tool.all_tags if tool.all_tags else ([tool.tag] if tool.tag else [])

        # If no descriptive tags, fall back to server-specific general bucket
        if not tags or tags == ["general"]:
            all_partition_tags = {f"{tool.mcp_server}_general"}
        else:
            all_partition_tags = set(tags)
            # Also include the specialized server tag for cross-domain discovery
            all_partition_tags.add(server_tag)

        for tag in all_partition_tags:
            if tag not in partitions:
                partitions[tag] = []
            partitions[tag].append(tool)

    return partitions


async def generate_system_prompt(
    agent_name: str, tools: list[MCPToolInfo], tag: str, server_name: str
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
    force_reprompt: bool = False, config_path: Path | None = None
):
    """Orchestrate the full synchronization of MCP servers with the Knowledge Graph.

    Performs metadata extraction, tool scoring, and stores them natively as
    ToolNode entities in the Cypher backend.
    """
    if not config_path:
        config_path = get_workspace_path(CORE_FILES["MCP_CONFIG"])

    from filelock import FileLock, Timeout

    lock_path = config_path.with_suffix(".sync.lock")
    lock = FileLock(str(lock_path), timeout=0)

    try:
        with lock.acquire(timeout=0):
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

            # 2. Ingest into Knowledge Graph
            import networkx as nx

            from .knowledge_graph.engine import IntelligenceGraphEngine
            from .workspace import get_agent_workspace

            logger.info(
                f"Starting Knowledge Graph ingestion for {len(tools_inventory)} tools"
            )
            try:
                engine = IntelligenceGraphEngine.get_active()
                if not engine:
                    ws_path = get_agent_workspace()
                    db_path = str(ws_path / "knowledge_graph.db")
                    logger.info(
                        f"No active engine, creating new IntelligenceGraphEngine with db_path: {db_path}"
                    )
                    engine = IntelligenceGraphEngine(
                        graph=nx.MultiDiGraph(), db_path=db_path
                    )

                backend = engine.backend
                if backend is None:
                    logger.error(
                        "Graph backend is not available. Cannot sync tools to graph."
                    )
                    return

                # 2a. Sync Prompts from registry builder
                from .agent_registry_builder import ingest_prompts_to_graph

                await ingest_prompts_to_graph()

                # 2b. Upsert Tool Nodes
                import time

                sync_ts = int(time.time())

                for tool in tools_inventory:
                    query = "MERGE (t:Tool {id: $id}) SET t.name = $name, t.description = $description, t.mcp_server = $mcp_server, t.relevance_score = $score, t.tags = $tags, t.requires_approval = $requires_approval, t.last_sync = $sync_ts"
                    props = {
                        "id": f"tool:{tool.name}",
                        "name": tool.name,
                        "description": tool.description,
                        "mcp_server": tool.mcp_server,
                        "score": tool.relevance_score,
                        "tags": tool.all_tags or [tool.tag] if tool.tag else [],
                        "requires_approval": tool.requires_approval,
                        "sync_ts": sync_ts,
                    }
                    backend.execute(query, props)

                    # Link Tool to Server node
                    query_link = "MERGE (s:Server {id: $server_id}) SET s.name = $server_name WITH s MATCH (t:Tool {id: $tool_id}) MERGE (s)-[:PROVIDES]->(t)"
                    backend.execute(
                        query_link,
                        {
                            "server_id": f"server:{tool.mcp_server}",
                            "server_name": tool.mcp_server,
                            "tool_id": f"tool:{tool.name}",
                        },
                    )

                # 2c. Partition tools and create specialist agents
                partitions = await partition_tools(tools_inventory)
                logger.info(
                    f"Partitioned {len(tools_inventory)} tools into {len(partitions)} specialist domains: {list(partitions.keys())}"
                )

                for tag, partition_tools_list in partitions.items():
                    # Heuristic: use the server of the first tool as the source
                    source_server = (
                        partition_tools_list[0].mcp_server
                        if partition_tools_list
                        else "unknown"
                    )

                    system_prompt = await generate_system_prompt(
                        tag, partition_tools_list, tag, source_server
                    )

                    agent_id = f"agent:{tag}"
                    agent_name = tag
                    agent_desc = f"Specialist for {tag} (from {source_server})"

                    logger.info(
                        f"Upserting agent {agent_id} with {len(partition_tools_list)} tools"
                    )

                    query_agent = """
                    MERGE (a:Agent {id: $id})
                    SET a.name = $name,
                        a.description = $description,
                        a.agent_type = 'mcp',
                        a.system_prompt = $system_prompt,
                        a.tool_count = $tool_count,
                        a.mcp_server = $mcp_server,
                        a.last_sync = $sync_ts
                    """
                    backend.execute(
                        query_agent,
                        {
                            "id": agent_id,
                            "name": agent_name,
                            "description": agent_desc,
                            "system_prompt": system_prompt,
                            "tool_count": len(partition_tools_list),
                            "mcp_server": source_server,
                            "sync_ts": sync_ts,
                        },
                    )

                    # Link Agent to its Tools
                    for t in partition_tools_list:
                        query_link_tool = "MATCH (a:Agent {id: $agent_id}), (t:Tool {id: $tool_id}) MERGE (a)-[:PROVIDES]->(t)"
                        backend.execute(
                            query_link_tool,
                            {"agent_id": agent_id, "tool_id": f"tool:{t.name}"},
                        )

                # Cleanup old tools and agents that were not in this sync
                try:
                    query_clean_tools = "MATCH (t:Tool) WHERE t.last_sync < $sync_ts OR t.last_sync IS NULL DETACH DELETE t"
                    backend.execute(query_clean_tools, {"sync_ts": sync_ts})

                    query_clean_agents = "MATCH (a:Agent) WHERE a.last_sync < $sync_ts OR a.last_sync IS NULL DETACH DELETE a"
                    backend.execute(query_clean_agents, {"sync_ts": sync_ts})
                except Exception as e:
                    logger.debug(f"Cleanup skipped (likely empty DB): {e}")

                logger.info(
                    f"✅ Synced {len(tools_inventory)} MCP tools and {len(partitions)} specialist agents directly to the Knowledge Graph."
                )
            except Exception as e:
                logger.exception(f"Failed to sync MCP agents to Knowledge Graph: {e}")
    except Timeout:
        logger.info("Another process is currently syncing MCP agents. Skipping...")


if __name__ == "__main__":
    # Setup basic logging when run directly
    logging.basicConfig(level=logging.INFO)
    asyncio.run(sync_mcp_agents())
