#!/usr/bin/python
from __future__ import annotations

"""Graph Configuration Helpers Module.

CONCEPT:ORCH-1.2 — Hot Cache Layer & Registry Optimization

This module provides utility functions for loading and saving graph-related
configurations, managing dynamic agent registries, emitting sideband events
for the WebUI, and resolving specialized prompts from the package resources.

The ``_RegistryCache`` class (CONCEPT:ORCH-1.2) provides zero-cost typed lookups for
registry data.  The cache is populated on first access and invalidated by
explicit signals from ``sync_mcp_agents()``, pipeline completion,
``promote_coalition_to_template()``, and ``MemoryRetriever.update_after_session()``.
"""


import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from agent_utilities.core.config import config
from agent_utilities.core.workspace import (
    CORE_FILES,
    get_workspace_path,
)

from ..base_utilities import to_integer
from ..models import MCPAgent, MCPAgentRegistryModel, MCPConfigModel, MCPToolInfo

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_TIMEOUT = to_integer(config.graph_timeout or "1200000")


# ---------------------------------------------------------------------------
# CONCEPT:ORCH-1.2 — Session-Scoped Registry Cache
# ---------------------------------------------------------------------------


class _RegistryCache:
    """Session-scoped cache for KG registry data.

    CONCEPT:ORCH-1.2 — Hot Cache Layer

    Populated on first access, invalidated by explicit event signals.
    No TTL — pure event-driven invalidation from four callsites:

    1. ``agent_manager.sync_mcp_agents()`` (MCP reload)
    2. Pipeline completion (``PipelineRunner.run()``)
    3. ``promote_coalition_to_template()`` (TeamConfig creation)
    4. ``MemoryRetriever.update_after_session()`` (proficiency update)
    """

    _registry: MCPAgentRegistryModel | None = None
    _prompts: dict[str, str] = {}
    _tool_agent_map: dict[str, list[str]] = {}

    @classmethod
    def invalidate(cls) -> None:
        """Clear all cached data.  Called by event-driven signals."""
        cls._registry = None
        cls._prompts.clear()
        cls._tool_agent_map.clear()
        logger.info("[CACHE] Registry cache invalidated (CONCEPT:ORCH-1.2).")

    @classmethod
    def get_registry(cls) -> MCPAgentRegistryModel:
        """Return the cached registry, populating on first access."""
        if cls._registry is None:
            cls._registry = _fetch_registry_from_kg()
            logger.info(
                "[CACHE] Registry cache populated: %d agents, %d tools.",
                len(cls._registry.agents),
                len(cls._registry.tools),
            )
        return cls._registry


def invalidate_registry_cache() -> None:
    """Public API to invalidate the hot cache.

    CONCEPT:ORCH-1.2 — Hot Cache Layer

    Call this after any operation that changes the registry state:
    MCP reload, pipeline ingestion, TeamConfig promotion, or
    Self-Model update.
    """
    _RegistryCache.invalidate()


def _fetch_registry_from_kg() -> MCPAgentRegistryModel:
    """Fetch the full registry from the Knowledge Graph (uncached).

    This is the expensive operation that ``_RegistryCache`` wraps.
    Delegates to focused sub-functions for each data source.
    """
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        import networkx as nx

        from agent_utilities.core.workspace import get_agent_workspace

        ws = get_agent_workspace()
        db_path = str(ws / "knowledge_graph.db")
        engine = IntelligenceGraphEngine(graph=nx.MultiDiGraph(), db_path=db_path)

    if not engine or not engine.backend:
        return MCPAgentRegistryModel()

    agents: list[MCPAgent] = []
    agents.extend(_fetch_prompt_agents(engine))
    agents.extend(_fetch_specialist_agents(engine))

    tools = _fetch_tools(engine)
    agents.extend(_synthesize_partition_agents(tools, {a.name for a in agents}))

    return MCPAgentRegistryModel(agents=agents, tools=tools)


def _fetch_prompt_agents(engine: Any) -> list[MCPAgent]:
    """Fetch Prompt-based agents from the KG."""
    agents: list[MCPAgent] = []
    try:
        prompt_rows = engine.backend.execute(
            "MATCH (p:Prompt) RETURN p.name AS name, p.description AS descriptionription, p.capabilities AS capabilities, p.system_prompt AS system_prompt, p.json_blueprint AS json_blueprint"
        )
        for row in prompt_rows:
            blueprint = row.get("json_blueprint")
            if isinstance(blueprint, str):
                try:
                    blueprint = json.loads(blueprint)
                except Exception:
                    try:
                        import ast

                        blueprint = ast.literal_eval(blueprint)
                    except Exception:
                        logger.debug(
                            f"Failed to parse json_blueprint as JSON or literal: {blueprint[:100]}..."
                        )

            if blueprint and not isinstance(blueprint, dict):
                logger.debug(
                    f"json_blueprint for {row.get('name')} is not a dict, type={type(blueprint)}"
                )

            parsed_blueprint: dict[str, Any] | None = (
                blueprint if isinstance(blueprint, dict) else None
            )
            agents.append(
                MCPAgent(
                    name=row.get("name", ""),
                    description=row.get("description", ""),
                    agent_type="specialist",
                    capabilities=row.get("capabilities", []),
                    system_prompt=row.get("system_prompt", ""),
                    json_blueprint=parsed_blueprint,
                )
            )
    except Exception as e:
        logger.debug(f"Failed to fetch Prompt nodes: {e}")
    return agents


def _fetch_specialist_agents(engine: Any) -> list[MCPAgent]:
    """Fetch Agent-type specialist nodes from the KG."""
    agents: list[MCPAgent] = []
    try:
        agent_rows = engine.backend.execute(
            "MATCH (a:Agent) RETURN a.name AS name, a.description AS descriptionription, a.agent_type AS agent_type, a.system_prompt AS system_prompt, a.tool_count AS tool_count, a.mcp_server AS mcp_server"
        )
        for row in agent_rows:
            # CONCEPT:ORCH-1.2: Normalize legacy prompt/mcp to unified specialist
            _raw_type = row.get("agent_type", "specialist")
            _agent_type = _raw_type if _raw_type == "a2a" else "specialist"
            agents.append(
                MCPAgent(
                    name=row.get("name", "unknown"),
                    description=row.get("description", ""),
                    agent_type=_agent_type,
                    system_prompt=row.get("system_prompt", ""),
                    tool_count=row.get("tool_count", 0),
                    mcp_server=row.get("mcp_server"),
                )
            )
    except Exception as e:
        logger.debug(f"Failed to fetch specialist agents from KG: {e}")
    return agents


def _fetch_tools(engine: Any) -> list[MCPToolInfo]:
    """Fetch Tool nodes from the KG."""
    tools: list[MCPToolInfo] = []
    try:
        tool_rows = engine.backend.execute(
            "MATCH (t:Tool) RETURN t.name, t.description, t.mcp_server, t.relevance_score, t.tags, t.requires_approval"
        )
        for row in tool_rows:
            tools.append(
                MCPToolInfo(
                    name=row.get("t.name", ""),
                    description=row.get("t.description", ""),
                    mcp_server=row.get("t.mcp_server", "unknown"),
                    relevance_score=row.get("t.relevance_score", 0),
                    all_tags=row.get("t.tags", []),
                    requires_approval=row.get("t.requires_approval", False),
                )
            )
    except Exception as e:
        logger.debug(f"Failed to fetch Tool nodes: {e}")
    return tools


def _synthesize_partition_agents(
    tools: list[MCPToolInfo],
    existing_agent_names: set[str],
) -> list[MCPAgent]:
    """Synthesize partition-based agents from tool tags.

    CONCEPT:ORCH-1.2 — Re-derive Server Agents from Tools (Dynamic Partitioning at read-time)
    """
    partitions: dict[str, list[MCPToolInfo]] = {}
    for t in tools:
        tags = t.all_tags if t.all_tags else ([t.tag] if t.tag else [])
        server_tag = (
            t.mcp_server.lower()
            .replace("-mcp", "")
            .replace("_mcp", "")
            .replace("-manager", "")
            .replace("-agent", "")
            .replace("-server", "")
        )
        if not tags or tags == ["general"]:
            all_partition_tags = {f"{t.mcp_server}_general"}
        else:
            all_partition_tags = set(tags)
            all_partition_tags.add(server_tag)

        for tag in all_partition_tags:
            if tag not in partitions:
                partitions[tag] = []
            partitions[tag].append(t)

    agents: list[MCPAgent] = []
    for tag, partition_tools in partitions.items():
        if tag in existing_agent_names:
            continue

        mcp_servers = list(set(t.mcp_server for t in partition_tools))
        primary_server = mcp_servers[0] if mcp_servers else "unknown"

        agents.append(
            MCPAgent(
                name=tag,
                description=f"Dynamically synthesized agent for {tag} capabilities.",
                agent_type="specialist",
                system_prompt=f"You are the {tag} specialist.",
                tool_count=len(partition_tools),
                mcp_server=primary_server,
                tools=[t.name for t in partition_tools],
                capabilities=list(
                    set(
                        c_tag
                        for t in partition_tools
                        for c_tag in (
                            t.all_tags if t.all_tags else ([t.tag] if t.tag else [])
                        )
                    )
                ),
            )
        )

    return agents


def get_discovery_registry() -> MCPAgentRegistryModel:
    """Load the unified agent discovery registry (cached).

    CONCEPT:ORCH-1.2 — Hot Cache Layer

    Returns the registry from the in-memory cache.  On first call,
    populates the cache from the Knowledge Graph.  Subsequent calls
    are O(1) until ``invalidate_registry_cache()`` is called.

    Returns:
        The populated MCPAgentRegistryModel.
    """
    return _RegistryCache.get_registry()


def get_relevant_specialists(
    query: str,
    engine: Any | None = None,
    top_n: int = 7,
) -> list[MCPAgent]:
    """Return the top-N adaptive_agent_router most relevant to a query.

    CONCEPT:ORCH-1.2 — Hot Cache Layer

    Uses KG discovery results (hybrid search + tool matching) to filter
    the full specialist list down to the most relevant agents for a
    given query.  Falls back to the full list if KG discovery returns
    nothing or the engine is unavailable.

    Args:
        query: The user query to match against.
        engine: Optional ``IntelligenceGraphEngine`` for hybrid search.
        top_n: Maximum number of adaptive_agent_router to return.

    Returns:
        A list of the most relevant ``MCPAgent`` objects.
    """
    registry = get_discovery_registry()
    all_agents = registry.agents

    if not all_agents:
        return []

    if not engine or not query:
        return all_agents[:top_n]

    # Use hybrid search to find relevant nodes
    try:
        results = engine.search_hybrid(query, top_k=top_n * 3)
        matched_names: set[str] = set()
        for r in results:
            name = r.get("name", "")
            if name:
                matched_names.add(name.lower())
            # Also check the node type for agent/prompt matches
            node_type = str(r.get("type", "")).lower()
            if node_type in ("agent", "prompt"):
                matched_names.add(name.lower())

        # Score agents by whether they appear in search results
        relevant = [a for a in all_agents if a.name.lower() in matched_names]

        if relevant:
            return relevant[:top_n]
    except Exception as e:
        logger.debug(f"Hybrid search for adaptive_agent_router failed: {e}")

    # Fallback: return all agents (capped)
    return all_agents[:top_n]


def load_node_agents_registry() -> MCPAgentRegistryModel:
    """Legacy alias for get_discovery_registry."""
    return get_discovery_registry()


def load_mcp_config() -> MCPConfigModel:
    """Retrieve the global MCP server configuration from the workspace.

    Loads the mcp_config.json file which contains the definitions of
    external MCP servers (e.g., Docker, GitHub) and their connection
    parameters.

    Returns:
        An MCPConfigModel object containing server definitions and settings.

    """
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return MCPConfigModel.model_validate(data)
        except Exception:
            return MCPConfigModel()
    return MCPConfigModel()


def save_mcp_config(config: MCPConfigModel):
    """Persist the MCP configuration model back to the workspace file.

    Args:
        config: The MCPConfigModel to be saved.

    """
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")


def emit_graph_event(eq: asyncio.Queue[Any] | None, event_type: str, **kwargs):
    """Emit a standardized graph event for real-time UI visualization.

    Formats the event data as a sideband part compatible with the
    Agentic UI streaming protocol, allowing the frontend to visualize
    graph progression and tool activity.  Also emits a structured log
    line so the full execution trace is visible in server-side logs
    without requiring the UI.

    Args:
        eq: The asynchronous event queue to publish to.
        event_type: A string identifier for the event category.
        **kwargs: Additional metadata to include in the event payload.

    """
    ts = time.time()
    trace_kwargs = {k: v for k, v in kwargs.items() if k != "timestamp"}
    _log_graph_trace(event_type, ts, **trace_kwargs)

    if not eq:
        return

    try:
        eq.put_nowait(
            {
                "type": "data-graph-event",
                "data": {
                    "event": event_type,
                    "timestamp": ts,
                    **kwargs,
                },
            }
        )
    except Exception as e:
        logger.warning(f"Failed to emit graph event '{event_type}': {e}")


# ---------------------------------------------------------------------------
# Structured graph trace logging
# ---------------------------------------------------------------------------

_graph_trace_logger = logging.getLogger("agent_utilities.graph.trace")

_PHASE_MAP: dict[str, str] = {
    # ── Lifecycle ──────────────────────────────────────────────────────
    "graph_start": "LIFECYCLE",
    "graph_complete": "LIFECYCLE",
    "node_start": "LIFECYCLE",
    "node_complete": "LIFECYCLE",
    # ── Safety & Policy ───────────────────────────────────────────────
    "safety_warning": "SAFETY",
    # ── Routing & Planning ────────────────────────────────────────────
    "routing_started": "ROUTING",
    "routing_completed": "ROUTING",
    "plan_created": "PLANNING",
    "replanning_started": "REPLANNING",
    "replanning_completed": "REPLANNING",
    # ── Dispatch ──────────────────────────────────────────────────────
    "step_dispatched": "DISPATCH",
    "batch_dispatched": "DISPATCH",
    # ── Context Enrichment ────────────────────────────────────────────
    "context_gap_detected": "ENRICHMENT",
    # ── Specialist Execution ──────────────────────────────────────────
    "specialist_enter": "EXECUTION",
    "specialist_exit": "EXECUTION",
    "specialist_fallback": "FALLBACK",
    "expert_metadata": "EXECUTION",
    "expert_thinking": "EXECUTION",
    "expert_warning": "EXECUTION",
    "expert_text": "EXECUTION",
    "expert_complete": "EXECUTION",
    "tools_bound": "EXECUTION",
    "subagent_started": "EXECUTION",
    "subagent_completed": "EXECUTION",
    "subagent_thought": "EXECUTION",
    # ── Tool Calls ────────────────────────────────────────────────────
    "expert_tool_call": "TOOL_CALL",
    "subagent_tool_call": "TOOL_CALL",
    "tool_result": "TOOL_RESULT",
    # ── Parallel / Orthogonal Regions ─────────────────────────────────
    "orthogonal_regions_start": "PARALLEL",
    "orthogonal_regions_complete": "PARALLEL",
    "region_start": "PARALLEL",
    "region_complete": "PARALLEL",
    # ── Verification & Synthesis ──────────────────────────────────────
    "verification_result": "VERIFICATION",
    "agent_node_delta": "SYNTHESIS",
    "synthesis_fallback": "SYNTHESIS",
    # ── Human-in-the-Loop ─────────────────────────────────────────────
    "approval_required": "APPROVAL",
    "approval_resolved": "APPROVAL",
    "elicitation": "APPROVAL",
    # ── Recovery & Termination ────────────────────────────────────────
    "error_recovery_replan": "RECOVERY",
    "error_recovery_terminal": "RECOVERY",
    "graph_force_terminated": "TERMINATION",
    # ── Council Deliberation ──────────────────────────────────────────
    "council_started": "COUNCIL",
    "council_stage": "COUNCIL",
    "council_advisor_complete": "COUNCIL",
    "council_reviewer_complete": "COUNCIL",
    "council_completed": "COUNCIL",
    # ── KG-Driven Graph Materialization (CONCEPT:ORCH-1.4) ─────────────
    "kg_query_start": "KG_BRIDGE",
    "kg_query_complete": "KG_BRIDGE",
    "kg_template_resolved": "KG_BRIDGE",
    "kg_prompt_injected": "KG_BRIDGE",
    "kg_topology_materialized": "KG_BRIDGE",
}


def _log_graph_trace(event_type: str, timestamp: float, **kwargs):
    """Emit a structured log line for a graph event."""
    phase = _PHASE_MAP.get(event_type, "GRAPH")
    detail_parts: list[str] = []

    for key in ("agent", "expert", "node_id", "domain", "server"):
        if key in kwargs:
            detail_parts.append(f"{key}={kwargs[key]}")
    for key in ("count", "score", "batch_size", "attempt", "duration_ms"):
        if key in kwargs:
            detail_parts.append(f"{key}={kwargs[key]}")
    if "tool_name" in kwargs:
        detail_parts.append(f"tool={kwargs['tool_name']}")
    if "success" in kwargs:
        detail_parts.append(f"ok={kwargs['success']}")
    if "message" in kwargs and event_type in ("expert_warning", "safety_warning"):
        detail_parts.append(f"msg={kwargs['message'][:120]}")

    detail = " ".join(detail_parts) if detail_parts else ""
    _graph_trace_logger.info(f"[{phase}] {event_type} {detail}".rstrip())


def _render_prompt_payload(data: dict[str, Any]) -> str:
    """Render a prompt blueprint dict to the string the LLM should see.

    Prefers the modern JSON blueprint schema (with a ``content`` key) and
    falls back to :class:`StructuredPrompt` for legacy ``task``/``input``
    payloads. The returned string is always valid JSON so callers can
    forward it directly to ``system_prompt=`` kwargs.
    """
    content = data.get("content")
    if isinstance(content, str) and content.strip():
        return json.dumps(data, indent=2)

    try:
        from agent_utilities.prompting.structured import StructuredPrompt

        return StructuredPrompt.model_validate(data).render()
    except Exception as e:
        logger.debug(f"StructuredPrompt validation failed: {e}")
        return json.dumps(data, indent=2)


def load_specialized_prompts(prompt_name: str) -> str:
    """Load a specialized agent persona prompt from the registry defined path.

    The loader checks, in order:

    1. A matching agent in the Knowledge Graph registry with a
       ``json_blueprint`` payload.
    2. An agent whose ``prompt_file`` points at a local ``*.json`` file.
    3. A fallback ``agent_utilities/prompts/<prompt_name>.json`` file.

    Args:
        prompt_name: The slugified name/tag of the expert (e.g. ``router``).

    Returns:
        The specialized system prompt serialized as a JSON string.

    """
    registry = get_discovery_registry()
    agent = next((a for a in registry.agents if a.name == prompt_name), None)

    if agent:
        if agent.json_blueprint:
            return _render_prompt_payload(dict(agent.json_blueprint))

        if agent.prompt_file:
            # Check if it's a JSON file
            prompt_path = (Path(__file__).parent.parent / agent.prompt_file).resolve()
            if prompt_path.suffix == ".json" and prompt_path.exists():
                data = json.loads(prompt_path.read_text(encoding="utf-8"))
                return _render_prompt_payload(data)

    # Unified JSON loading from prompts/
    json_path = (
        Path(__file__).parent.parent / "prompts" / f"{prompt_name}.json"
    ).resolve()
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return _render_prompt_payload(data)
        except Exception as e:
            logger.warning(
                f"Failed to load structured prompt JSON for '{prompt_name}': {e}"
            )

    logger.warning(
        f"Specialized prompt for '{prompt_name}' not found in registry "
        "or prompts/*.json."
    )
    return f"You are a helpful assistant specialized in {prompt_name}."
