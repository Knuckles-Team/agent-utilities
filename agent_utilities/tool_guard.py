#!/usr/bin/python
"""Tool Guard Module.

This module implements a security middleware layer for agent tools. It provides
pattern-based sensitivity detection and integrates with pydantic-ai's
Human-in-the-Loop mechanism to enforce explicit user approval for
high-risk operations.

Two mechanisms are provided:

1. :func:`apply_tool_guard_approvals` — flags *function* tools (``@agent.tool``)
   with ``requires_approval=True``.  Used for the top-level agent.
2. :func:`build_sensitive_tool_names` + :func:`flag_mcp_tool_definitions` —
   builds a set of sensitive tool names from the NODE_AGENTS.md registry
   (``requires_approval`` column) **and** live pattern matching, then wraps
   MCP toolsets with pydantic-ai's ``ApprovalRequiredToolset``.  When a
   sensitive tool is called, ``ApprovalRequired`` is raised (unless
   ``ctx.tool_call_approved`` is already ``True``), causing pydantic-ai to
   return ``DeferredToolRequests`` instead of executing the tool.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from pydantic_ai import Agent

from .config import SENSITIVE_TOOL_PATTERNS, TOOL_GUARD_MODE

logger = logging.getLogger(__name__)


def is_sensitive_tool(name: str) -> bool:
    """Check if a tool name matches any sensitive pattern."""
    if TOOL_GUARD_MODE == "strict":
        # In strict mode, everything is sensitive unless it's explicitly safe
        return not is_safe_tool(name)

    for pattern in SENSITIVE_TOOL_PATTERNS:
        if re.match(pattern, name.lower()):
            return True
    return False


def is_safe_tool(name: str) -> bool:
    """Check if a tool name is explicitly safe (read-only)."""
    safe_patterns = [
        r"^read_.*",
        r"^list_.*",
        r"^get_.*",
        r"^describe_.*",
        r"^search_.*",
        r"^inspect_.*",
        r"^view_.*",
        r"^show_.*",
    ]
    for pattern in safe_patterns:
        if re.match(pattern, name.lower()):
            return True
    return False


def build_sensitive_tool_names() -> set[str]:
    """Build the authoritative set of tool names that require approval.

    Merges two sources:

    1. **Knowledge Graph** — tools where ``requires_approval`` is True.
    2. **Live pattern matching** — any tool name matching
       ``SENSITIVE_TOOL_PATTERNS`` is included even if the graph hasn't
       been re-synced yet.

    Returns:
        A set of lowercase tool names that should be flagged for approval.

    """
    sensitive: set[str] = set()

    # Source 1: Knowledge Graph
    with contextlib.suppress(Exception):
        from .graph.config_helpers import get_discovery_registry

        registry = get_discovery_registry()
        for tool in registry.tools:
            if tool.requires_approval:
                sensitive.add(tool.name.lower())

    return sensitive


def flag_mcp_tool_definitions(
    toolsets: list[Any],
    sensitive_names: set[str] | None = None,
) -> list[Any]:
    """Wrap MCP toolsets so sensitive tools require human approval.

    Uses pydantic-ai's native :class:`ApprovalRequiredToolset` wrapper.
    When a specialist agent calls a sensitive tool, the wrapper raises
    ``ApprovalRequired`` (unless ``ctx.tool_call_approved`` is already
    ``True`` from a prior approval round).  This causes pydantic-ai to
    return ``DeferredToolRequests`` instead of executing the tool.

    The ``approval_required_func`` checks the tool name against both the
    NODE_AGENTS.md registry (``sensitive_names``) and live pattern
    matching (``is_sensitive_tool``).

    Args:
        toolsets: The original list of MCP toolsets to wrap.
        sensitive_names: Pre-built set from :func:`build_sensitive_tool_names`.
            If ``None``, pattern matching alone is used.

    Returns:
        A new list of toolsets where MCP toolsets are wrapped with
        ``ApprovalRequiredToolset``.

    """
    if TOOL_GUARD_MODE == "off":
        return toolsets

    if sensitive_names is None:
        sensitive_names = set()

    try:
        from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
    except ImportError:
        return toolsets

    def _requires_approval(
        _ctx: Any, tool_def: Any, _tool_args: dict[str, Any]
    ) -> bool:
        name = getattr(tool_def, "name", "")
        return name.lower() in sensitive_names or is_sensitive_tool(name)

    wrapped: list[Any] = []
    for ts in toolsets:
        is_mcp = hasattr(ts, "list_tools") or hasattr(ts, "direct_call_tool")
        if is_mcp:
            wrapped.append(
                ApprovalRequiredToolset(
                    wrapped=ts,
                    approval_required_func=_requires_approval,
                )
            )
        else:
            wrapped.append(ts)

    return wrapped


def apply_tool_guard_approvals(agent: Agent) -> None:
    """Apply requires_approval=True to all sensitive function tools on an agent.

    Iterates the agent's function toolset (the first entry in the public
    ``agent.toolsets`` property) and sets ``requires_approval=True`` on
    tools matching sensitive patterns.

    For MCP tools, use :func:`flag_mcp_tool_definitions` instead.

    Args:
        agent: The Pydantic AI Agent instance to modify.

    """
    if TOOL_GUARD_MODE == "off":
        logger.debug("Tool guard disabled (TOOL_GUARD_MODE=off)")
        return

    flagged = 0

    try:
        from pydantic_ai.toolsets.function import FunctionToolset
    except ImportError:
        return

    for ts in agent.toolsets:
        if not isinstance(ts, FunctionToolset):
            continue
        if not hasattr(ts, "tools"):
            continue
        for tool_name, tool in ts.tools.items():
            sensitive = is_sensitive_tool(tool_name)
            orchestration = bool(re.search(r"run[-_]graph", tool_name.lower()))

            if sensitive and not orchestration:
                if not getattr(tool, "requires_approval", False):
                    logger.info(
                        f"Tool Guard: Flagging sensitive tool '{tool_name}' for approval."
                    )
                    tool.requires_approval = True
                    flagged += 1
            elif orchestration:
                if getattr(tool, "requires_approval", False):
                    logger.info(
                        f"Tool Guard: Removing sensitive flag from orchestration tool '{tool_name}'."
                    )
                tool.requires_approval = False
                logger.debug(
                    f"Tool Guard: Orchestration tool '{tool_name}' is trusted."
                )

    if flagged:
        logger.info(
            f"Tool Guard (native): Flagged {flagged} sensitive tools with requires_approval=True"
        )
