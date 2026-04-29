#!/usr/bin/python
"""ACP Adapter Module.

This module provides the integration layer between the agent ecosystem and
the Async Control Protocol (ACP). It handles session management, approval
bridges, and high-fidelity interaction modes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if not _ACP_INSTALLED:
    # Type stubs for when package is missing
    class AdapterConfig:  # type: ignore
        def __init__(self, **kwargs):
            pass

    class PrepareToolsMode:  # type: ignore
        def __init__(self, **kwargs):
            pass

    class PrepareToolsBridge:  # type: ignore
        def __init__(self, **kwargs):
            pass

    class ThinkingBridge:  # type: ignore
        def __init__(self, **kwargs):
            pass

    class FileSessionStore:  # type: ignore
        def __init__(self, **kwargs):
            pass

    class NativeApprovalBridge:  # type: ignore
        def __init__(self, **kwargs):
            pass


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


def build_acp_config(
    session_root: Path | None = None,
    enable_approvals: bool = True,
    enable_thinking: bool = True,
    modes: list[PrepareToolsMode] | None = None,
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

    bridges: list[Any] = [PrepareToolsBridge(default_mode_id="ask", modes=modes)]
    if enable_thinking:
        bridges.append(ThinkingBridge())

    # Wire in the workspace persistence provider so that ACP's native plan
    # state is mirrored to the workspace.  This must be set on the
    # ``native_plan_persistence_provider`` key (write-only sink) — NOT on
    # ``plan_provider`` (read-only source) which would disable ACP's
    # built-in plan tools (acp_get_plan, acp_set_plan, etc.).
    from .acp_providers import get_workspace_persistence_provider

    plan_persistence = get_workspace_persistence_provider()

    config_params: dict[str, Any] = {
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
    graph_bundle: tuple[Any, Any] | None = None,
    mcp_toolsets: list[Any] | None = None,
) -> Any:
    """Create an ACP app that routes execution through the graph pipeline.

    When a ``graph_bundle`` is provided, the ACP agent delegates to the
    full HSM graph instead of running as a flat agent.  This ensures that
    ACP clients benefit from specialist routing, parallel execution,
    circuit breakers, and the verification loop.

    This implementation uses ``pydantic-acp``'s ``agent_factory`` callback
    to create per-session agents.  Each session gets its own ``run_graph_flow``
    tool closure with access to the session context, eliminating the need
    for the ``REQUESTED_MODEL_ID_CTX`` context-variable workaround.

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

    def graph_agent_factory(session) -> Agent:
        """Create a per-session agent with graph context.

        The ``agent_factory`` callback receives an ``AcpSessionContext``
        and returns a ``PydanticAgent``.  This pattern allows us to bind
        session-specific context (like model overrides) directly into the
        graph execution closure, avoiding global context variables.

        Args:
            session: The AcpSessionContext for the current session.

        Returns:
            A PydanticAgent configured for graph execution.

        """

        async def run_graph_flow(query: str, mode: str = "ask") -> str:
            """Execute the full graph pipeline for the given query.

            This tool is the sole entry point registered on the per-session
            wrapper agent.  The LLM's only job is to call this tool with
            the user's query.

            Args:
                query: The user query to process.
                mode: Execution mode ('ask', 'plan', 'execute').

            Returns:
                The synthesized result from the graph verifier.

            """
            from .graph.unified import execute_graph

            # Session context is captured from the factory closure.
            # No REQUESTED_MODEL_ID_CTX workaround needed.
            result = await execute_graph(
                graph=graph,
                config=graph_config,
                query=query,
                mode=mode,
                mcp_toolsets=mcp_toolsets or graph_config.get("mcp_toolsets", []),
            )
            return result.get("results", {}).get(
                "output", str(result.get("results", {}))
            )

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
        return graph_agent

    from pydantic_acp import create_acp_agent

    logger.info(
        "ACP: Graph-backed agent_factory created. "
        "All ACP requests will route through the graph."
    )
    return create_acp_agent(agent_factory=graph_agent_factory, config=config)


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
