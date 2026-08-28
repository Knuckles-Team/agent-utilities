#!/usr/bin/python
"""MCP Agent Manager Module.

This module manages the lifecycle of agents derived from MCP servers. It
handles the extraction of tool metadata from running servers, partitioning
tools into logical domain adaptive_agent_router, deterministic tool relevance scoring,
and synchronizing these adaptive_agent_router directly with the Knowledge Graph.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from agent_utilities.core.config import load_mcp_servers_from_config
from agent_utilities.core.workspace import (
    CORE_FILES,
    get_workspace_path,
)
from agent_utilities.models import MCPToolInfo
from agent_utilities.security.tool_guard import is_sensitive_tool

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

    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    except ImportError:
        # A broken/incomplete environment can still fail this guarded import. The
        # supported package contract always includes the mandatory full engine and its
        # numeric component, so this is runtime fault containment, not an install shape.
        return False

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        return True

    # Check if we have any tools and when they were last synced
    try:
        res = engine.query_cypher(
            "MATCH (t:Tool) RETURN t.id AS id, t.last_sync AS last_sync "
            "ORDER BY t.last_sync DESC LIMIT 1"
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
    timeout: int = 300,
    semaphore: asyncio.Semaphore | None = None,
) -> list[MCPToolInfo]:
    """Wrapper for metadata extraction with optional concurrency control.

    Args:
        server: The MCP server instance.
        timeout: Execution timeout in seconds.
        semaphore: Optional semaphore to limit parallel connections.

    Returns:
        A list of extracted MCPToolInfo objects.

    """
    if semaphore:
        async with semaphore:
            return await _extract_single_server_metadata_inner(server, timeout)
    return await _extract_single_server_metadata_inner(server, timeout)


def _tags_from_annotations(tool: Any) -> list[str]:
    """Tags carried in ``tool.annotations`` (dict- or attribute-shaped)."""
    if not (hasattr(tool, "annotations") and tool.annotations):
        return []
    ann = tool.annotations
    if isinstance(ann, dict):
        tags_data = ann.get("tags") or ann.get("tag")
    else:
        tags_data = getattr(ann, "tags", None) or getattr(ann, "tag", None)
    if isinstance(tags_data, list | set | tuple):
        return [str(t) for t in tags_data]
    if isinstance(tags_data, str):
        return [tags_data]
    return []


def _tags_from_fastmcp_meta(tool: Any) -> list[str]:
    """Tags carried in FastMCP's ``tool.meta["fastmcp"]["tags"]`` (protocol fallback)."""
    if not (hasattr(tool, "meta") and isinstance(tool.meta, dict)):
        return []
    fastmcp_meta = tool.meta.get("fastmcp", {})
    tags_data = fastmcp_meta.get("tags")
    if isinstance(tags_data, list | set | tuple):
        return [str(t) for t in tags_data]
    if isinstance(tags_data, str):
        return [tags_data]
    return []


def _tool_tags(tool: Any) -> list[str]:
    """A tool's tags: annotation-derived, falling back to FastMCP meta tags."""
    return _tags_from_annotations(tool) or _tags_from_fastmcp_meta(tool)


def _infer_tag_from_name(name: str) -> str | None:
    """Heuristic domain tag from a tool name: split on ``-``/``_``, skip a leading
    generic verb (e.g. ``get_containers`` -> ``containers``)."""
    import re

    parts = re.split(r"[-_]", name)
    if len(parts) <= 1:
        return None
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
    if parts[0].lower() in generic_verbs:
        return parts[1].lower()
    return parts[0].lower()


def _tool_domain_tag(tool: Any, tags: list[str]) -> str:
    """The primary partitioning tag: an explicit tag, else the name heuristic, else ``general``."""
    tag = tags[0] if tags else None
    if not tag:
        tag = _infer_tag_from_name(tool.name)
    return tag or "general"


def _tool_info_from_tool(tool: Any, server_name: str) -> MCPToolInfo:
    """Build one :class:`MCPToolInfo` from a raw MCP SDK tool object."""
    tags = _tool_tags(tool)
    return MCPToolInfo(
        name=tool.name,
        description=getattr(tool, "description", "") or "",
        tag=_tool_domain_tag(tool, tags),
        mcp_server=server_name,
        all_tags=tags,
        requires_approval=is_sensitive_tool(tool.name),
    )


async def _extract_single_server_metadata_inner(
    server: Any, timeout: int = 300
) -> list[MCPToolInfo]:
    """Internal logic for connecting to an MCP server and listing tools.

    Uses dynamic extraction via ``session.list_tools()`` and heuristic tagging
    based only on returned tool metadata. Connection failure is fail-closed:
    environment keys/config values are never converted into synthetic tools.

    Args:
        server: The MCP server instance.
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

                for tool in tools_list or []:
                    all_tools.append(_tool_info_from_tool(tool, server_name))
    except Exception as exc:
        logger.warning(
            "Dynamic extraction failed; using static hints (exception_type=%s)",
            type(exc).__name__,
        )
        # Do not fabricate capabilities from environment/config metadata. A
        # later successful discovery will populate the inventory accurately.

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
        logger.warning("MCP configuration was not found")
        return []

    servers = load_mcp_servers_from_config(config_path)
    server_count = len(servers)
    logger.info(
        f"Extracting tool metadata from {server_count} MCP servers in parallel..."
    )

    # Parallel extraction using anyio to handle FastMCP/anyio internals correctly
    import time as _time

    import anyio

    all_tools: list[MCPToolInfo] = []
    completed = {"count": 0}
    # Scale concurrency with server count: min 2, max 4
    concurrency = min(4, max(2, server_count // 8))
    semaphore = anyio.Semaphore(concurrency)

    async def _safe_extract(server, results):
        server_name = getattr(
            server, "name", getattr(server, "_id", getattr(server, "id", "unknown"))
        )
        t0 = _time.monotonic()
        async with semaphore:
            server_tools = await _extract_single_server_metadata_inner(
                server, timeout=timeout
            )
            results.extend(server_tools)
            completed["count"] += 1
            elapsed = _time.monotonic() - t0
            logger.info(
                f"  [{completed['count']}/{server_count}] {server_name}: "
                f"{len(server_tools)} tools ({elapsed:.1f}s)"
            )

    async with anyio.create_task_group() as tg:
        for server in servers:
            tg.start_soon(_safe_extract, server, all_tools)

    logger.info(
        f"Parallel extraction complete: {len(all_tools)} tools from {server_count} servers"
    )
    return all_tools


def _agent_description_quality_score(description: str) -> int:
    """Description quality (0-50)."""
    dlen = len(description or "")
    if dlen > 150:
        return 50
    if dlen > 80:
        return 40
    if dlen > 40:
        return 20
    if dlen > 0:
        return 5
    return 0


def _agent_skills_quality_score(skills: list[str]) -> int:
    """Skills quality (0-50)."""
    slen = len(skills or [])
    if slen > 10:
        return 50
    if slen > 5:
        return 40
    if slen > 2:
        return 20
    if slen > 0:
        return 10
    return 0


def compute_agent_metadata_score(description: str, skills: list[str]) -> int:
    """Compute a deterministic relevance score for an agent based on metadata.

    Args:
        description: The agent's specialization description.
        skills: List of skills/capabilities.

    Returns:
        An integer score between 0 and 100.

    """
    score = _agent_description_quality_score(description) + _agent_skills_quality_score(
        skills
    )
    return min(100, score)


def _tool_description_quality_score(tool: MCPToolInfo) -> int:
    """Description quality (0-30)."""
    desc_len = len(tool.description or "")
    if desc_len > 100:
        return 30
    if desc_len > 50:
        return 20
    if desc_len > 15:
        return 10
    if desc_len > 0:
        return 5
    return 0


def _tool_tag_confidence_score(tool: MCPToolInfo) -> int:
    """Tag confidence (0-30): explicit annotation tags score higher than heuristic ones."""
    if tool.all_tags:
        # Multiple explicit tags = highest confidence (from annotations)
        if len(tool.all_tags) >= 2:
            return 30
        tag_val = tool.all_tags[0]
        # Heuristic-derived tags are usually single lowercase words
        if "_" in tag_val or len(tag_val) > 6:
            return 25
        return 15
    if tool.tag:
        return 10
    return 0  # No tag at all


def _meaningful_name_segments(tool_name: str) -> tuple[list[str], list[str]]:
    """A tool name's ``-``/``_``-split segments, and the non-generic-verb ones."""
    generic_verbs = {"get", "list", "create", "update", "delete", "set", "run"}
    segments = [s for s in tool_name.replace("-", "_").split("_") if s]
    meaningful = [s for s in segments if s.lower() not in generic_verbs and len(s) > 2]
    return segments, meaningful


def _tool_name_specificity_score(tool: MCPToolInfo) -> int:
    """Name specificity (0-20): longer, multi-segment names beat generic verbs."""
    segments, meaningful = _meaningful_name_segments(tool.name or "")
    if len(meaningful) >= 3:
        return 20
    if len(meaningful) >= 2:
        return 15
    if len(meaningful) >= 1:
        return 10
    if segments:
        return 5
    return 0


def _tool_multi_tag_coverage_score(tool: MCPToolInfo) -> int:
    """Multi-tag coverage (0-20): more tags can serve more specialist domains."""
    tag_count = len(tool.all_tags)
    if tag_count >= 3:
        return 20
    if tag_count == 2:
        return 15
    if tag_count == 1:
        return 10
    return 0


def compute_tool_relevance_score(tool: MCPToolInfo) -> int:
    """Compute a deterministic relevance score for a tool (0-100).

    The score reflects how well-described, well-tagged, and specific the
    tool is.  Higher scores indicate tools that are more likely to be
    correctly routed and effectively used by specialist agents.

    Scoring breakdown (deterministic, no LLM) -- see the four
    ``_tool_*_score`` helpers for each component's own docstring.

    Args:
        tool: The tool metadata to score.

    Returns:
        An integer score between 0 and 100 inclusive.

    """
    score = (
        _tool_description_quality_score(tool)
        + _tool_tag_confidence_score(tool)
        + _tool_name_specificity_score(tool)
        + _tool_multi_tag_coverage_score(tool)
    )
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


async def _extract_and_score_tools(config_path: Path) -> list[MCPToolInfo] | None:
    """Extract + deterministically score tool metadata.

    Extracted from :func:`sync_mcp_agents`. ``None`` means there is nothing to
    sync (already logged).
    """
    tools_inventory = await extract_tool_metadata(config_path)
    if not tools_inventory:
        logger.info("No tools found to sync.")
        return None

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
    return tools_inventory


def _resolve_sync_engine() -> tuple[Any, bool]:
    """The active engine, or a freshly created local one. Returns ``(engine, is_local)``."""
    from agent_utilities.core.workspace import get_agent_workspace
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if engine:
        return engine, False
    ws_path = get_agent_workspace()
    db_path = str(ws_path / "knowledge_graph.db")
    return IntelligenceGraphEngine.get_or_create(db_path=db_path), True


def _upsert_tool_nodes(
    engine: Any, tools_inventory: list[MCPToolInfo], sync_ts: int
) -> None:
    """Batch-upsert Server/Tool nodes + ``PROVIDES`` edges. Extracted from
    :func:`sync_mcp_agents`."""
    logger.info(f"Batching {len(tools_inventory)} tool upserts to Knowledge Graph...")
    # Pre-create server nodes in bulk (deduplicated)
    seen_servers: set[str] = set()
    for tool in tools_inventory:
        if tool.mcp_server not in seen_servers:
            seen_servers.add(tool.mcp_server)
            engine.add_node(
                f"server:{tool.mcp_server}",
                "Server",
                {"name": tool.mcp_server, "last_sync": sync_ts},
            )

    for tool in tools_inventory:
        props = {
            "name": tool.name,
            "description": tool.description,
            "mcp_server": tool.mcp_server,
            "relevance_score": tool.relevance_score,
            "tags": tool.all_tags or [tool.tag] if tool.tag else [],
            "requires_approval": tool.requires_approval,
            "last_sync": sync_ts,
        }
        engine.add_node(f"tool:{tool.name}", "Tool", props)

        # Link Tool to Server node
        engine.link_nodes(
            f"server:{tool.mcp_server}",
            f"tool:{tool.name}",
            "PROVIDES",
        )

    logger.info(
        f"✅ Synced {len(tools_inventory)} MCP tools directly to the Knowledge Graph."
    )


async def _ingest_tools_to_graph(tools_inventory: list[MCPToolInfo]) -> None:
    """Ingest scored tools into the Knowledge Graph. Extracted from :func:`sync_mcp_agents`."""
    logger.info(f"Starting Knowledge Graph ingestion for {len(tools_inventory)} tools")
    is_local_engine = False
    backend = None
    try:
        engine, is_local_engine = _resolve_sync_engine()

        backend = engine.backend
        if backend is None:
            logger.error("Graph backend is not available. Cannot sync tools to graph.")
            if is_local_engine:
                pass  # no backend to close
            return

        # Sync Prompts from registry builder
        from agent_utilities.agent.registry_builder import ingest_prompts_to_graph

        await ingest_prompts_to_graph()

        import time

        sync_ts = int(time.time())
        _upsert_tool_nodes(engine, tools_inventory, sync_ts)

        # CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Invalidate hot cache after sync
        from agent_utilities.core.config import invalidate_registry_cache

        invalidate_registry_cache()

    except Exception as exc:
        logger.error(
            "Failed to sync MCP agents to Knowledge Graph (exception_type=%s)",
            type(exc).__name__,
        )
    finally:
        if is_local_engine and backend:
            backend.close()


async def sync_mcp_agents(
    force_reprompt: bool = False, config_path: Path | None = None
):
    """Orchestrate the full synchronization of MCP servers with the Knowledge Graph.

    Performs metadata extraction, tool scoring, and stores them natively as
    ToolNode entities in the Cypher backend.
    """
    if not config_path:
        config_path = get_workspace_path(CORE_FILES["MCP_CONFIG"])

    config_path = Path(config_path)

    if not (force_reprompt or should_sync(config_path)):
        return

    tools_inventory = await _extract_and_score_tools(config_path)
    if tools_inventory is None:
        return

    await _ingest_tools_to_graph(tools_inventory)


if __name__ == "__main__":
    # Setup basic logging when run directly
    logging.basicConfig(level=logging.INFO)
    asyncio.run(sync_mcp_agents())
