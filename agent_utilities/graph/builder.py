"""Build the run graph bundle a connector or host passes to the run loop.

The bundle is ``(graph, config)``: ``graph`` is a :class:`RunGraph` naming the
specialists and the toolsets bound at build time; ``config`` is the plain
mapping the fallback plan reads. The builder binds what the caller names and
nothing else: it does not route, rank or discover. Routing settings the caller
passes (``router_model``, ``min_confidence``) are kept in ``config["routing"]``
for the host decision provider; the harness never reads them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_utilities import host_ports

#: Workspace file whose text becomes the default instructions.
WORKSPACE_PROMPT_FILE = "AGENTS.md"


@dataclass(frozen=True)
class RunGraph:
    """The bound shape of one agent: its name, specialists and toolsets."""

    name: str
    specialists: Mapping[str, str] = field(default_factory=dict)
    toolsets: tuple[Any, ...] = ()


def _toolsets(mcp_url: str | None, mcp_config: str | None) -> tuple[Any, ...]:
    bound: list[Any] = []
    if mcp_url:
        bound.append(host_ports.build_url_toolset(mcp_url))
    if mcp_config:
        bound.extend(host_ports.load_config_toolsets(mcp_config))
    return tuple(bound)


def _bundle(
    graph: RunGraph,
    *,
    agent_model: str | None,
    system_prompt: str,
    routing: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> tuple[RunGraph, dict[str, Any]]:
    config: dict[str, Any] = {
        "name": graph.name,
        "agent_model": agent_model,
        "system_prompt": system_prompt,
        "specialists": dict(graph.specialists),
        "toolsets": list(graph.toolsets),
        "routing": dict(routing),
        **extra,
    }
    return graph, config


def create_graph_agent(
    *,
    tag_prompts: Mapping[str, str] | None = None,
    tag_env_vars: Mapping[str, str] | None = None,
    mcp_url: str | None = None,
    mcp_config: str | None = None,
    name: str = "GraphAgent",
    router_model: str | None = None,
    agent_model: str | None = None,
    min_confidence: float = 0.0,
    system_prompt: str = "",
) -> tuple[RunGraph, dict[str, Any]]:
    """Bind one agent from specialist prompts and the caller's tool endpoints.

    Tool endpoints bind through the host remote toolset provider and fail
    closed without one.
    """
    graph = RunGraph(
        name=name,
        specialists=dict(tag_prompts or {}),
        toolsets=_toolsets(mcp_url, mcp_config),
    )
    return _bundle(
        graph,
        agent_model=agent_model,
        system_prompt=system_prompt,
        routing={"router_model": router_model, "min_confidence": min_confidence},
        extra={"tag_env_vars": dict(tag_env_vars or {})},
    )


def initialize_graph_from_workspace(
    *,
    mcp_config: str | None = None,
    router_model: str | None = None,
    agent_model: str | None = None,
    workspace: str | Path | None = None,
    name: str = "GraphAgent",
) -> tuple[RunGraph, dict[str, Any]]:
    """Bind one agent from a workspace directory.

    The workspace ``AGENTS.md`` (when present) becomes the instructions.
    """
    prompt = ""
    if workspace is not None:
        path = Path(workspace) / WORKSPACE_PROMPT_FILE
        if path.is_file():
            prompt = path.read_text(encoding="utf-8")
    graph = RunGraph(name=name, toolsets=_toolsets(None, mcp_config))
    return _bundle(
        graph,
        agent_model=agent_model,
        system_prompt=prompt,
        routing={"router_model": router_model},
        extra={"workspace": str(workspace) if workspace is not None else None},
    )


__all__ = [
    "RunGraph",
    "WORKSPACE_PROMPT_FILE",
    "create_graph_agent",
    "initialize_graph_from_workspace",
]
