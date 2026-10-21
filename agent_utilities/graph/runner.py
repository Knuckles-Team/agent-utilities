"""Protocol-neutral entry points of the run loop.

Adapters (A2A, SSE, the agent client protocol) call these with the
``(graph, config)`` bundle from :mod:`agent_utilities.graph.builder` and one
query. Each returns or yields the events of :func:`executor.iterate_run`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from . import executor
from .builder import RunGraph
from .state import RunRequest


class GraphEventHandler:
    """Side-band receiver of each event; the default does nothing."""

    async def handle_event(self, event: dict[str, Any]) -> None:
        return None


def _request_and_config(
    graph: Any,
    config: Mapping[str, Any] | None,
    query: str,
    run_id: str | None,
    mode: str,
    mcp_toolsets: list[Any] | None,
    requested_model_id: str | None,
    agent_name: str,
    metadata: Mapping[str, Any] | None,
) -> tuple[RunRequest, dict[str, Any]]:
    cfg = dict(config or {})
    if isinstance(graph, RunGraph):
        cfg.setdefault("name", graph.name)
        cfg["specialists"] = {
            **dict(graph.specialists),
            **(cfg.get("specialists") or {}),
        }
        cfg.setdefault("toolsets", list(graph.toolsets))
    if mcp_toolsets:
        cfg["toolsets"] = [*(cfg.get("toolsets") or ()), *mcp_toolsets]
    fields: dict[str, Any] = {
        "task": query,
        "mode": mode,
        "agent_name": agent_name,
        "model_id": requested_model_id,
    }
    if run_id:
        fields["run_id"] = run_id
    if metadata:
        fields["metadata"] = dict(metadata)
    return RunRequest(**fields), cfg


async def execute_graph_iter(
    graph: Any,
    config: Mapping[str, Any] | None,
    query: str,
    run_id: str | None = None,
    mode: str = "ask",
    mcp_toolsets: list[Any] | None = None,
    handler: GraphEventHandler | None = None,
    requested_model_id: str | None = None,
    agent_name: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield each event of one run in the caller's task.

    ``metadata`` passes through to the run request: the host decision
    provider sees it; the run loop itself never reads it.
    """
    request, cfg = _request_and_config(
        graph,
        config,
        query,
        run_id,
        mode,
        mcp_toolsets,
        requested_model_id,
        agent_name,
        metadata,
    )
    async for event in executor.iterate_run(request, cfg):
        if handler is not None:
            await handler.handle_event(event)
        yield event


execute_graph_stream = execute_graph_iter


async def execute_graph(
    graph: Any,
    config: Mapping[str, Any] | None,
    query: str,
    run_id: str | None = None,
    mode: str = "ask",
    mcp_toolsets: list[Any] | None = None,
    requested_model_id: str | None = None,
    agent_name: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one query to its end; return the terminal result as a dict."""
    final: dict[str, Any] = {}
    async for event in execute_graph_iter(
        graph,
        config,
        query,
        run_id=run_id,
        mode=mode,
        mcp_toolsets=mcp_toolsets,
        requested_model_id=requested_model_id,
        agent_name=agent_name,
        metadata=metadata,
    ):
        final = event
    return dict(final["result"])


run_graph = execute_graph

__all__ = [
    "GraphEventHandler",
    "execute_graph",
    "execute_graph_iter",
    "execute_graph_stream",
    "run_graph",
]
