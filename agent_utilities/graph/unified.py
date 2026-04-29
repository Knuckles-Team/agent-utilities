#!/usr/bin/python
"""Unified Execution Layer Module.

This module provides protocol-agnostic entry points for graph execution,
simplifying the interface for various adapters (ACP, AG-UI, SSE). It wraps
the core runner logic to provide a consistent execution contract.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from .runner import run_graph, run_graph_iter, run_graph_stream

logger = logging.getLogger(__name__)


class GraphEventHandler:
    """Base class for graph event handlers."""

    async def handle_event(self, event: dict[str, Any]):
        """Callback to handle a graph event part."""
        pass


class SSEEventHandler(GraphEventHandler):
    """Event handler specialized for SSE streaming."""

    pass


class ACPEventHandler(GraphEventHandler):
    """Event handler specialized for the Agent Communication Protocol."""

    pass


async def execute_graph(
    graph,
    config: dict,
    query: str,
    run_id: str | None = None,
    mode: str = "ask",
    mcp_toolsets: list[Any] | None = None,
    plan_sync=None,
    requested_model_id: str | None = None,
    **kwargs,
) -> dict:
    """Unified entry point for synchronous graph execution.

    Args:
        graph: The Pydantic Graph instance to execute.
        config: Execution configuration dictionary.
        query: User input query string.
        run_id: Optional unique identifier for the execution run.
        mode: Operational mode ('ask', 'plan', 'research').
        mcp_toolsets: Optional list of pre-initialized MCP toolsets.
        plan_sync: Optional async callback for bridging plan state to ACP.
        requested_model_id: Optional per-turn model id override from the
            ``x-agent-model-id`` request header. See :func:`run_graph`.
        **kwargs: Additional parameters passed to the runner.

    Returns:
        A dictionary containing the final graph execution results.

    """
    return await run_graph(
        graph=graph,
        config=config,
        query=query,
        run_id=run_id,
        mode=mode,
        mcp_toolsets=mcp_toolsets,
        plan_sync=plan_sync,
        requested_model_id=requested_model_id,
        **kwargs,
    )


async def execute_graph_stream(
    graph,
    config: dict,
    query: str,
    run_id: str | None = None,
    mode: str = "ask",
    mcp_toolsets: list[Any] | None = None,
    handler: GraphEventHandler | None = None,
    requested_model_id: str | None = None,
    **kwargs,
) -> AsyncGenerator[dict[str, Any], None]:
    """Unified entry point for asynchronous streaming graph execution.

    Args:
        graph: The Pydantic Graph instance to execute.
        config: Execution configuration dictionary.
        query: User input query string.
        run_id: Optional unique identifier for the execution run.
        mode: Operational mode ('ask', 'plan', 'research').
        mcp_toolsets: Optional list of pre-initialized MCP toolsets.
        handler: Optional event handler for sideband processing.
        requested_model_id: Optional per-turn model id override from the
            ``x-agent-model-id`` request header. See :func:`run_graph`.
        **kwargs: Additional parameters passed to the runner.

    Yields:
        A stream of graph event dictionaries (SSE compatible).

    """
    async for event in run_graph_stream(
        graph=graph,
        config=config,
        query=query,
        run_id=run_id,
        mode=mode,
        mcp_toolsets=mcp_toolsets,
        requested_model_id=requested_model_id,
        **kwargs,
    ):
        if handler:
            await handler.handle_event(event)
        yield event


async def execute_graph_iter(
    graph,
    config: dict,
    query: str,
    run_id: str | None = None,
    mode: str = "ask",
    mcp_toolsets: list[Any] | None = None,
    handler: GraphEventHandler | None = None,
    requested_model_id: str | None = None,
    elicitation_callback=None,
    **kwargs,
) -> AsyncGenerator[dict[str, Any], None]:
    """Unified entry point for step-by-step graph execution using ``iter()``.

    This function wraps :func:`run_graph_iter` to provide a protocol-agnostic
    interface for iterative graph execution.  It yields per-step events that
    protocol adapters (AG-UI, ACP, SSE) can translate into their native
    wire formats.

    Unlike :func:`execute_graph_stream` (which runs the graph in a background
    task and streams SSE events), this function drives the graph step-by-step
    in the caller's task, enabling pause/resume, elicitation, and per-node
    observability.

    Args:
        graph: The Pydantic Graph instance to execute.
        config: Execution configuration dictionary.
        query: User input query string.
        run_id: Optional unique identifier for the execution run.
        mode: Operational mode ('ask', 'plan', 'research').
        mcp_toolsets: Optional list of pre-initialized MCP toolsets.
        handler: Optional event handler for sideband processing.
        requested_model_id: Optional per-turn model id override.
        elicitation_callback: Optional async callable for human-in-the-loop
            pauses.  See :func:`run_graph_iter` for details.
        **kwargs: Additional parameters passed to the runner.

    Yields:
        A stream of graph event dictionaries with ``type`` keys such as
        ``"node_transition"``, ``"graph_complete"``, ``"error"``, etc.

    """
    async for event in run_graph_iter(
        graph=graph,
        config=config,
        query=query,
        run_id=run_id,
        mode=mode,
        mcp_toolsets=mcp_toolsets,
        requested_model_id=requested_model_id,
        elicitation_callback=elicitation_callback,
        **kwargs,
    ):
        if handler:
            await handler.handle_event(event)
        yield event
