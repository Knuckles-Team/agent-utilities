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
from typing import Optional

from ..workspace import get_workspace_path, CORE_FILES, load_workspace_file
from ..base_utilities import to_integer
from ..models import MCPConfigModel, MCPAgentRegistryModel

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_TIMEOUT = to_integer(os.environ.get("GRAPH_TIMEOUT", "1200000"))


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


def load_node_agents_registry() -> MCPAgentRegistryModel:
    """Parse and load the specialized expert registry from NODE_AGENTS.md.

    This registry maps specialist tags (e.g., 'python_programmer') to
    their corresponding MCP servers or local agent packages, enabling
    dynamic tool discovery for the graph orchestrator.

    Returns:
        A model containing the list of registered MCP specialists.

    """
    from ..workspace import parse_node_registry

    content = load_workspace_file(CORE_FILES["NODE_AGENTS"])
    if not content:
        return MCPAgentRegistryModel()
    try:
        return parse_node_registry(content)
    except Exception as e:
        logger.error(f"Failed to parse MCP Agent registry: {e}")
        return MCPAgentRegistryModel()


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
    graph progression and tool activity. Also emites a structured log
    line so the full execution trace is visible in server-side logs
    without requiring a UI

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


_graph_trace_logger = logging.getLogger("agent_utilities.graph.trace")

_PHASE_MAP: dict[str, str] = {
    "routing_started": "ROUTING",
    "routing_completed": "ROUTING",
    "plan_created": "PLANNING",
    "replanning_started": "REPLANNING",
    "replanning_completed": "REPLANNING",
    "step_dispatched": "DISPATCH",
    "batch_dispatched": "DISPATCH",
    "specialist_enter": "EXECUTION",
    "specialist_exit": "EXECUTION",
    "specialist_fallback": "FALLBACK",
    "expert-metadata": "EXECUTION",
    "expert-thinking": "EXECUTION",
    "tools-bound": "EXECUTION",
    "expert-warning": "EXECUTION",
    "expert_tool_call": "TOOL_CALL",
    "expert_text": "EXECUTION",
    "tool-result": "TOOL_RESULT",
    "subagent_completed": "EXECUTION",
    "verification_result": "VERIFICATION",
    "agent-node-delta": "SYNTHESIS",
    "synthesis_fallback": "SYNTHESIS",
    "graph_force_terminated": "TERMINATION",
    "safety_warning": "SAFETY",
    "approval_required": "APPROVAL",
    "orthogonal_regions_start": "PARALLEL",
    "orthogonal_regions_complete": "PARALLEL",
    "graph-start": "LIFECYCLE",
    "graph-complete": "LIFECYCLE",
    "node_start": "EXECUTION",
    "node_complete": "EXECUTION",
    "error_recovery_replan": "RECOVERY",
    "error_recovery_terminal": "RECOVERY",
    "context_gap_detected": "ENRICHMENT",
}


def _log_graph_trace(event_type: str, timestamp: float, **kwargs):
    """Emit a structured log line for a graph event.
    Provides server-side traceability of every graph transition, enabling
    post-hoc analysis without the UI. Each line includes the phase label,
    event type, and key metadata extracted from kwargs.

    Args:
        event_type: The graph event identifier.
        timestamp: Unix epoch timestamp of the event.
        **kwargs: Event-specific metadata.
    """
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
    if "message" in kwargs and event_type in ("expert-warning", "safety_warning"):
        detail_parts.append(f"msg={kwargs['message'][:120]}")

    detail = " ".join(detail_parts) if detail_parts else ""
    _graph_trace_logger.info(f"[{phase}] {event_type}: {detail}".rstrip())


def load_specialized_prompts(name: str) -> str:
    """Load a specialized role-based prompt from the package resources.

    Attempts to locate the markdown prompt file within the internal
    prompts/ directory, with a fallback to local filesystem for development.

    Args:
        name: The basename of the prompt file (e.g., 'researcher').

    Returns:
        The content of the prompt file as a string.

    """
    import importlib.resources as pkg_resources
    from .. import prompts

    try:
        # Pydantic 3.11+ / Python 3.9+ standard way
        with pkg_resources.files(prompts).joinpath(f"{name}.md").open("r") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Could not load specialized prompt '{name}': {e}")
        # Build-time fallback for development if needed
        local_path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.md")
        if os.path.exists(local_path):
            with open(local_path, "r") as f:
                return f.read()
        return ""


# Mapping of Graph Nodes to Universal Skills and Skill Graphs
# This ensures each specialist has the highest-fidelity tools/docs for their domain.
NODE_SKILL_MAP = {
    "researcher": [
        "web-search",
        "web-fetch",
        "web-crawler",
        "agent-browser",
        "systems-manager",
        "browser-tools",
        "web-artifacts",
        "web-design-guidelines",
        "workspace-analyst",
    ],
    "architect": [
        "c4-architecture",
        "product-management",
        "product-strategy",
        "user-research",
        "brainstorming",
        "mermaid-diagrams",
    ],
    "planner": [
        "project-planning",
        "product-management",
        "brainstorming",
        "internal-comms",
    ],
    "verifier": ["qa-planning", "tdd-methodology"],
    "python_programmer": [
        "agent-builder",
        "tdd-methodology",
        "mcp-builder",
        "developer-utilities",
        "jupyter-notebook",
        "api-wrapper-builder",
        "python-docs",
        "pydantic-ai-docs",
        "pydantic-docs",
        "fastmcp-docs",
        "fastapi-docs",
        "agent-package-builder",
        "django-docs",
    ],
    "typescript_programmer": [
        "react-development",
        "web-artifacts",
        "tdd-methodology",
        "canvas-design",
        "nodejs-docs",
        "react-docs",
        "nextjs-docs",
        "shadcn-docs",
        "nestjs-docs",
        "reactrouter-docs",
        "redux-docs",
        "tanstack-docs",
        "vitejs-docs",
        "vercel-docs",
        "svelte-docs",
        "vuejs-docs",
        "remix-docs",
    ],
    "rust_programmer": ["rust-docs"],
    "golang_programmer": ["go-docs"],
    "java_programmer": [
        "java-docs",
        "laravel-docs",
    ],
    "devops_engineer": [
        "cloudflare-deploy",
        "repository-maintenance",
        "c4-architecture",
        "docker-docs",
        "terraform-docs",
        "temporal-docs",
        "minio-docs",
        "aws-docs",
        "azure-docs",
        "gcp-docs",
    ],
    "database_expert": [
        "database-tools",
        "postgres-docs",
        "mongodb-docs",
        "redis-docs",
        "mariadb-docs",
        "mssql-docs",
        "neo4j-docs",
        "couchbase-docs",
        "falkordb-docs",
        "chromadb-docs",
        "qdrant-docs",
    ],
    "security_auditor": ["security-tools", "linux-docs"],
    "qa_expert": [
        "qa-planning",
        "tdd-methodology",
        "testing-library-docs",
        "developer-utilities",
        "self-improver",
    ],
    "ui_ux_designer": [
        "theme-factory",
        "canvas-design",
        "brand-guidelines",
        "algorithmic-art",
        "web-artifacts",
        "website-builder",
        "website-cloner",
        "web-design-guidelines",
        "shadcn-docs",
        "tailwind-docs",
        "framer-docs",
        "radix-ui-docs",
        "material-ui-docs",
        "chakra-ui-docs",
    ],
    "data_scientist": [
        "jupyter-notebook",
        "numpy-docs",
        "pandas-docs",
        "matplotlib-docs",
        "scikit-learn-docs",
        "scipy-docs",
        "pytorch-docs",
        "tesnorflow-docs",
        "huggingface-docs",
        "langchain-docs",
    ],
    "document_specialist": [
        "document-tools",
        "document-converter",
        "marp-presentations",
        "creative-media",
    ],
    "mobile_programmer": [
        "react-native-skills",
        "react-docs",
    ],
    "agent_engineer": [
        "agent-builder",
        "agent-package-builder",
        "agent-spawner",
        "agent-workflows",
        "mcp-builder",
        "mcp-client",
        "skill-builder",
        "skill-graph-builder",
        "skill-installer",
        "agents-md-generator",
        "self-improver",
        "pydantic-ai-docs",
        "fastmcp-docs",
    ],
    "project_manager": [
        "jira-tools",
        "github-tools",
        "google-workspace",
        "repository-maintenance",
        "product-management",
        "session-handoff",
        "internal-comms",
    ],
    "systems_manager": [
        "systems-manager",
        "system-tools",
        "linux-docs",
        "home-assistant-docs",
        "uptime-kuma-docs",
        "owncast-docs",
        "postiz-docs",
    ],
    "c_programmer": [
        "developer-utilities",
        "c-docs",
    ],
    "cpp_programmer": [
        "developer-utilities",
        "cpp-docs",
    ],
    "javascript_programmer": [
        "web-artifacts",
        "canvas-design",
        "nodejs-docs",
        "react-docs",
        "developer-utilities",
    ],
    "cloud_architect": [
        "c4-architecture",
        "aws-docs",
        "azure-docs",
        "gcp-docs",
        "developer-utilities",
    ],
    "debugger_expert": [
        "developer-utilities",
        "agent-builder",
    ],
    "critique": [
        "qa-planning",
        "tdd-methodology",
        "self-improver",
    ],
    "browser_automation": [
        "browser-tools",
        "agent-browser",
        "web-artifacts",
        "web-design-guidelines",
        "web-crawler",
    ],
    "coordinator": [
        "project-planning",
        "agent-workflows",
        "session-handoff",
        "internal-comms",
    ],
}
