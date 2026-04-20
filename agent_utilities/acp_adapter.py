#!/usr/bin/python
# coding: utf-8
"""ACP Adapter Module.

This module provides the integration layer between the agent ecosystem and
the Async Control Protocol (ACP). It handles session management, approval
bridges, and high-fidelity interaction modes.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Dict, List

from pydantic_ai import Agent

# Guarded imports for optional ACP features
try:
    from pydantic_acp import (
        AdapterConfig,
        FileSessionStore,
        NativeApprovalBridge,
        PrepareToolsBridge,
        PrepareToolsMode,
        ThinkingBridge,
    )

    _ACP_INSTALLED = True
except ImportError:
    _ACP_INSTALLED = False

    # Type stubs for when package is missing
    class AdapterConfig:
        def __init__(self, **kwargs):
            pass

    class PrepareToolsMode:
        def __init__(self, **kwargs):
            pass

    class PrepareToolsBridge:
        def __init__(self, **kwargs):
            pass

    class ThinkingBridge:
        def __init__(self, **kwargs):
            pass

    class FileSessionStore:
        def __init__(self, **kwargs):
            pass

    class NativeApprovalBridge:
        def __init__(self, **kwargs):
            pass


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def build_acp_config(
    session_root: Optional[Path] = None,
    enable_approvals: bool = True,
    enable_thinking: bool = True,
    modes: Optional[List[PrepareToolsMode]] = None,
) -> AdapterConfig:
    """Construct a production-ready ACP AdapterConfig.

    Configures session storage, capability bridges (thinking, approvals),
    and interaction modes (ask, plan).

    Args:
        session_root: Optional path for session storage.
        enable_approvals: Whether to enable the approval bridge.
        enable_thinking: Whether to enable the thinking bridge.
        modes: Optional custom list of preparation modes.

    Returns:
        A configured AdapterConfig instance.

    """
    if not session_root:
        session_root = Path(os.getenv("ACP_SESSION_ROOT", ".acp-sessions"))

    try:
        session_root.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create ACP session directory {session_root}: {e}")

    if not modes:
        modes = [
            PrepareToolsMode(
                id="ask",
                name="Ask",
                description="Standard interaction mode.",
                prepare_func=lambda ctx, tool_defs: list(tool_defs),
            ),
            PrepareToolsMode(
                id="plan",
                name="Plan",
                description="Full graph planning and verification mode.",
                prepare_func=lambda ctx, tool_defs: list(tool_defs),
                plan_mode=True,
            ),
        ]

    bridges: List[Any] = [PrepareToolsBridge(default_mode_id="ask", modes=modes)]
    if enable_thinking:
        bridges.append(ThinkingBridge())

    # Wire in the workspace persistence provider so that ACP's native plan
    # state is mirrored to MEMORY.md.  This must be set on the
    # ``native_plan_persistence_provider`` key (write-only sink) — NOT on
    # ``plan_provider`` (read-only source) which would disable ACP's
    # built-in plan tools (acp_get_plan, acp_set_plan, etc.).
    from .acp_providers import get_workspace_persistence_provider

    plan_persistence = get_workspace_persistence_provider()

    config_params: Dict[str, Any] = {
        "session_store": FileSessionStore(root=session_root),
        "capability_bridges": bridges,
        "native_plan_persistence_provider": plan_persistence,
    }
    if enable_approvals:
        config_params["approval_bridge"] = NativeApprovalBridge(
            enable_persistent_choices=True
        )

    return AdapterConfig(**config_params)


def create_acp_app(agent: Agent, config: AdapterConfig):
    """Create a mountable ACP ASGI application from a Pydantic AI agent.

    Args:
        agent: The Pydantic AI agent instance.
        config: The ACP adapter configuration.

    Returns:
        An ASGI-compatible application instance.

    """
    from pydantic_acp import create_acp_agent

    return create_acp_agent(agent=agent, config=config)


def create_graph_acp_app(
    agent: Agent,
    config: AdapterConfig,
    graph_bundle: tuple | None = None,
    mcp_toolsets: list | None = None,
) -> Any:
    """Create an ACP app that routes execution through the graph pipeline.

    When a ``graph_bundle`` is provided, the ACP agent delegates to the
    full HSM graph instead of running as a flat agent.  This ensures that
    ACP clients benefit from specialist routing, parallel execution,
    circuit breakers, and the verification loop.

    Falls back to the standard flat-agent ACP path when no graph is
    available.

    Args:
        agent: The base Pydantic AI agent (used as fallback).
        config: The ACP adapter configuration.
        graph_bundle: Optional ``(graph, graph_config)`` tuple from
            :func:`initialize_graph_from_workspace`.
        mcp_toolsets: Optional pre-connected MCP toolsets.

    Returns:
        An ASGI-compatible ACP application instance.

    """
    if not graph_bundle:
        logger.info("ACP: No graph bundle — using flat agent path.")
        return create_acp_app(agent, config)

    graph, graph_config = graph_bundle

    # Build a thin wrapper agent whose only tool is run_graph_flow.
    # This preserves ACP session management, approval bridges, and
    # thinking capabilities while routing all execution through the graph.
    from pydantic_ai import Agent

    async def run_graph_flow(query: str, mode: str = "ask") -> str:
        """Execute the full graph pipeline for the given query.

        Args:
            query: The user query to process.
            mode: Execution mode ('ask', 'plan', 'execute').

        Returns:
            The synthesized result from the graph verifier.

        """
        from .graph.unified import execute_graph

        # Plan sync is handled via sideband events (emitted by the graph
        # dispatcher) and the native_plan_persistence_provider (which
        # mirrors ACP plan state to PLAN.md).  No direct session
        # manipulation is needed here.

        result = await execute_graph(
            graph=graph,
            config=graph_config,
            query=query,
            mode=mode,
            mcp_toolsets=mcp_toolsets or graph_config.get("mcp_toolsets", []),
        )
        return result.results.get("output", str(result.results))

    graph_agent = Agent(
        model=agent.model,
        system_prompt=(
            "You are a graph-orchestrated assistant. Use the run_graph_flow "
            "tool for ALL user requests. Pass the user's query directly to "
            "the tool. Do NOT attempt to answer questions yourself — the "
            "graph has access to specialized agents and MCP tools."
        ),
        tools=[run_graph_flow],
    )

    logger.info(
        "ACP: Graph-backed agent created. All ACP requests will route through the graph."
    )
    return create_acp_app(graph_agent, config)


def is_acp_available() -> bool:
    """Check if the pydantic-acp package is installed and usable.

    Returns:
        True if pydantic-acp can be imported, False otherwise.

    """
    try:
        import pydantic_acp  # noqa: F401

        return True
    except ImportError:
        return False
