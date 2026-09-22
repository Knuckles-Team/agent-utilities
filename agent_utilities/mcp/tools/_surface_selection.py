"""Shared registration helpers for conditionally served legacy tools."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def conditional_tool_decorator(
    mcp: Any,
    *,
    name: str,
    excluded_names: frozenset[str],
    include_excluded: bool,
    **options: Any,
) -> Any:
    """Return a real MCP decorator only when this tool is selected to be served."""
    if name in excluded_names and not include_excluded:
        return lambda handler: handler
    return mcp.tool(name=name, **options)


def register_conditional_tools(
    tool_registry: dict[str, Any],
    entries: Iterable[tuple[str, Any, str | None]],
    *,
    excluded_names: frozenset[str],
    include_excluded: bool,
    route_registry: dict[str, str] | None = None,
) -> None:
    """Register selected tools and optional REST routes in their authoritative maps."""
    for name, handler, route in entries:
        if name in excluded_names and not include_excluded:
            continue
        tool_registry[name] = handler
        if route_registry is not None and route is not None:
            route_registry[name] = route
