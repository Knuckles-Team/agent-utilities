#!/usr/bin/python
# coding: utf-8
"""Graph Configuration Helpers Module.

This module provides utility functions for loading and saving graph-related
configurations, managing dynamic agent registries, emitting sideband events
for the WebUI, and resolving specialized prompts from the package resources.
"""

from __future__ import annotations

import os
import json
import logging
import time
import asyncio
import re
from pathlib import Path
from typing import Optional

from ..workspace import (
    get_workspace_path,
    CORE_FILES,
)
from ..base_utilities import to_integer
from ..models import MCPConfigModel, MCPAgentRegistryModel

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_TIMEOUT = to_integer(os.environ.get("GRAPH_TIMEOUT", "1200000"))


def get_discovery_registry() -> MCPAgentRegistryModel:
    """Load the unified agent discovery registry from the Knowledge Graph.

    Returns:
        The populated MCPAgentRegistryModel.
    """
    from ..knowledge_graph.engine import IntelligenceGraphEngine
    from ..models import MCPAgent, MCPToolInfo

    engine = IntelligenceGraphEngine.get_active()
    if not engine or not engine.backend:
        # Fallback to local prompt discovery if graph not active
        return MCPAgentRegistryModel()

    agents = []
    # 1. Fetch Prompt Agents
    try:
        prompt_rows = engine.backend.execute(
            "MATCH (p:Prompt) RETURN p.name, p.desc, p.capabilities, p.content"
        )
        for row in prompt_rows:
            agents.append(
                MCPAgent(
                    name=row.get("p.name", ""),
                    description=row.get("p.desc", ""),
                    agent_type="prompt",
                    capabilities=row.get("p.capabilities", []),
                    system_prompt=row.get("p.content", ""),
                )
            )
    except Exception as e:
        logger.debug(f"Failed to fetch Prompt nodes: {e}")

    # 2. Fetch Tools
    tools = []
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

    return MCPAgentRegistryModel(agents=agents, tools=tools)


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


def emit_graph_event(eq: Optional[asyncio.Queue], event_type: str, **kwargs):
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


def load_specialized_prompts(prompt_name: str) -> str:
    """Load a specialized agent persona prompt from the registry defined path.

    Args:
        prompt_name: The slugified name/tag of the expert (e.g., 'researcher').

    Returns:
        The raw markdown content of the specialized system prompt.

    """
    registry = get_discovery_registry()
    agent = next((a for a in registry.agents if a.name == prompt_name), None)

    if agent and agent.prompt_file:
        prompt_path = (Path(__file__).parent.parent / agent.prompt_file).resolve()
        if prompt_path.exists():
            content = prompt_path.read_text(encoding="utf-8")
            # Remove frontmatter if present
            content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
            return content

    # Fallback to local file check if registry is missing
    prompt_path = (
        Path(__file__).parent.parent / "prompts" / f"{prompt_name}.md"
    ).resolve()
    if prompt_path.exists():
        content = prompt_path.read_text(encoding="utf-8")
        content = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.DOTALL)
        return content

    logger.warning(f"Specialized prompt for '{prompt_name}' not found.")
    return f"You are a helpful assistant specialized in {prompt_name}."
