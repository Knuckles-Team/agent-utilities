#!/usr/bin/python
from __future__ import annotations

"""Make the mcp-multiplexer fleet a native ``pydantic_ai.capabilities.ToolSearch`` provider.

CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog — Track 2 of the pydantic-ai
native-adoption program (``reports/program/pydantic-ai-native-adoption.md``).

We already do dynamic tool discovery by hand: ``find_tools`` / ``list_catalog`` /
``load_tools`` / ``unload_tools`` (``agent_utilities/mcp/multiplexer.py``) front
~123 tools across ~66 servers for any MCP CLIENT (Claude Code, another agent).
Pydantic AI ships the SAME idea as a first-class capability —
``pydantic_ai.capabilities.ToolSearch`` — for a pydantic-ai AGENT built directly
against our own fleet. Rather than build a second discovery mechanism for that
case, this module:

* :class:`FleetToolset` — a real ``AbstractToolset`` over
  ``MCPMultiplexer.probe_catalog``/``call_proxied_tool`` (the SAME catalog +
  child-execution boundary the MCP-facing ``find_tools``/``load_tools`` surface
  uses), with every tool registered ``defer_loading=True`` so the framework's own
  deferred-disclosure machinery — not a hand-rolled visibility flag — decides
  what the model sees.
* :func:`fleet_relevance_search` — a ``ToolSearchFunc`` that reuses
  ``MCPMultiplexer._relevance`` (the exact token-overlap backbone ``find_tools``
  ranks with) instead of pydantic-ai's own bundled keyword algorithm, so a
  pydantic-ai agent's native tool search and the MCP-facing ``find_tools`` agree
  on which tools match a query.

We do NOT fork ``ToolSearch`` to add governance/KG scoring here — this wraps it
(a ``strategy=`` callable is the native extension point) and keeps the KG/
provenance/authority layer as agent-utilities' own value-add, per the program's
"do not fork a native to add governance" rule.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic_ai.toolsets import AbstractToolset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pydantic_ai import RunContext
    from pydantic_ai.capabilities import ToolSearch
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.toolsets import ToolsetTool

    from agent_utilities.mcp.multiplexer import MCPMultiplexer

logger = logging.getLogger(__name__)

__all__ = ["FleetToolset", "fleet_relevance_search", "fleet_tool_search_capability"]


def fleet_relevance_search(
    ctx: RunContext[Any],
    queries: Sequence[str],
    tools: Sequence[ToolDefinition],
) -> list[str]:
    """``ToolSearchFunc`` backed by the multiplexer's own relevance backbone.

    Matches the exact contract of pydantic-ai's built-in ``keywords_search_fn``
    (sync, no truncation — ``ToolSearch.max_results`` bounds the result) but scores
    with :meth:`MCPMultiplexer._relevance`, the SAME deterministic token-overlap
    algorithm ``find_tools``/``discover_tools`` rank fleet tools with, so a native
    ``ToolSearch`` call and an MCP client's ``find_tools`` call agree on ranking
    for identical queries against identical tool text.
    """
    from agent_utilities.mcp.multiplexer import MCPMultiplexer

    del (
        ctx
    )  # unused — relevance here is query/text-only, matching the built-in contract
    joined_query = " ".join(queries)
    scored: list[tuple[float, str]] = []
    for tool_def in tools:
        text = f"{tool_def.name} {tool_def.description or ''}"
        score = MCPMultiplexer._relevance(joined_query, text)
        if score > 0:
            scored.append((score, tool_def.name))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in scored]


class FleetToolset(AbstractToolset[Any]):
    """Exposes the mcp-multiplexer's fleet catalog as a native pydantic-ai toolset.

    Every fleet tool is registered ``defer_loading=True`` — hidden from the model
    until ``ToolSearch`` (or a caller's own ``load_capability``-equivalent) reveals
    it — so a pydantic-ai agent built with ``toolsets=[FleetToolset(mux)]`` plus
    ``capabilities=[ToolSearch(strategy=fleet_relevance_search)]`` reaches the
    SAME ~123 tools / ~66 servers ``find_tools``/``load_tools`` reach for an MCP
    client, through the framework's own discovery surface instead of a second one.

    Calling a discovered tool forwards through
    ``MCPMultiplexer.resolve_and_mount`` + ``call_proxied_tool`` — the identical
    mount-then-call path ``load_tools``/``_make_forwarder`` use for the MCP-facing
    surface. One execution boundary, two front doors.
    """

    def __init__(self, multiplexer: MCPMultiplexer, *, id: str = "mcp-fleet") -> None:
        self._mux = multiplexer
        self._toolset_id = id

    @property
    def id(self) -> str | None:
        return self._toolset_id

    async def for_run(self, ctx: Any) -> FleetToolset:
        return self

    async def __aenter__(self) -> FleetToolset:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        return None

    async def get_instructions(self, ctx: Any) -> str | None:
        return None

    async def get_tools(self, ctx: Any) -> dict[str, ToolsetTool[Any]]:
        from pydantic_ai.mcp import TOOL_SCHEMA_VALIDATOR
        from pydantic_ai.tools import ToolDefinition
        from pydantic_ai.toolsets import ToolsetTool

        from agent_utilities.core.config import config as agent_config
        from agent_utilities.mcp.multiplexer import clean_tool_name

        probe = await self._mux.probe_catalog(
            budget=agent_config.mcp_dynamic_discovery_timeout
        )
        tools: dict[str, ToolsetTool[Any]] = {}
        for server, info in probe.items():
            if info.get("error"):
                continue
            prefix = self._mux.server_prefix(server)
            for entry in info.get("tools", []) or []:
                name = entry.get("name")
                if not name or not self._mux._tool_enabled(server, name):
                    continue
                prefixed = clean_tool_name(prefix, server, name)
                tool_def = ToolDefinition(
                    name=prefixed,
                    description=entry.get("description", "") or "",
                    parameters_json_schema=(
                        entry.get("inputSchema") or {"type": "object", "properties": {}}
                    ),
                    defer_loading=True,
                    metadata={"mcp_server": server, "mcp_tool": name},
                )
                tools[prefixed] = ToolsetTool[Any](
                    toolset=self,
                    tool_def=tool_def,
                    max_retries=1,
                    args_validator=TOOL_SCHEMA_VALIDATOR,
                )
        return tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: Any, tool: ToolsetTool[Any]
    ) -> Any:
        del ctx, tool
        _mounted_servers, _to_expose, failed = await self._mux.resolve_and_mount(
            tools=[name]
        )
        if name in failed:
            raise RuntimeError(f"fleet tool unavailable: {failed[name]}")
        result = await self._mux.call_proxied_tool(name, tool_args)
        if bool(getattr(result, "is_error", False)) or bool(
            getattr(result, "isError", False)
        ):
            raise RuntimeError("delegated_child_tool_failed")
        return _fleet_call_result(result)


def _fleet_call_result(result: Any) -> Any:
    """Decode one child ``CallToolResult`` into a plain tool-return value.

    Prefers structured content (mirrors ``multiplexer._child_result_payload``'s
    precedence) and falls back to joined text content, returned as-is — unlike
    ``_child_result_payload`` this does not require the text to be JSON, since a
    pydantic-ai tool return is not constrained to that shape.
    """
    structured = getattr(result, "structuredContent", None)
    if structured in (None, {}):
        structured = getattr(result, "structured_content", None)
    if structured not in (None, {}):
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured
    texts = [
        str(getattr(item, "text", ""))
        for item in (getattr(result, "content", None) or [])
        if getattr(item, "text", "")
    ]
    return "\n".join(texts)


def fleet_tool_search_capability(*, max_results: int = 10, **kwargs: Any) -> ToolSearch:
    """Build the ``ToolSearch`` capability wired to :func:`fleet_relevance_search`.

    Pair with ``toolsets=[FleetToolset(mux)]`` on the ``Agent`` construction so the
    fleet's deferred tools are both present (for search to rank) and discoverable
    (via this capability) — see module docstring.
    """
    from pydantic_ai.capabilities import ToolSearch

    return ToolSearch(
        strategy=fleet_relevance_search, max_results=max_results, **kwargs
    )
