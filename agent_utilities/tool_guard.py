#!/usr/bin/python

from __future__ import annotations

import re
import logging


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


from pydantic_ai import Agent


from .config import SENSITIVE_TOOL_PATTERNS, TOOL_GUARD_MODE

logger = logging.getLogger(__name__)


def is_sensitive_tool(name: str) -> bool:
    """Check if a tool name matches any sensitive pattern."""
    for pattern in SENSITIVE_TOOL_PATTERNS:
        if re.match(pattern, name.lower()):
            logger.info(
                f"Tool Guard: Tool '{name}' matched sensitive pattern '{pattern}'"
            )
            return True
    return False


def apply_tool_guard_approvals(agent: "Agent") -> None:
    """Apply requires_approval=True to all sensitive tools on an agent.

    Uses pydantic-ai's native Human-in-the-Loop mechanism.
    When TOOL_GUARD_MODE is 'native', tools matching SENSITIVE_TOOL_PATTERNS
    will require frontend approval before execution.
    The AG-UI / Vercel AI SDK frontend renders an ApprovalCard,
    and the user's response flows back via DeferredToolResults.

    Args:
        agent: The Pydantic AI Agent instance to modify.
    """
    if TOOL_GUARD_MODE == "off":
        logger.debug("Tool guard disabled (TOOL_GUARD_MODE=off)")
        return

    flagged = 0

    if hasattr(agent, "_function_toolset") and hasattr(
        agent._function_toolset, "tools"
    ):
        for tool_name, tool in agent._function_toolset.tools.items():
            sensitive = is_sensitive_tool(tool_name)
            # Robust orchestration check: any tool with run_graph or run-graph is trusted
            orchestration = bool(re.search(r"run[-_]graph", tool_name.lower()))

            if sensitive and not orchestration:
                if not getattr(tool, "requires_approval", False):
                    logger.info(
                        f"Tool Guard: Flagging sensitive tool '{tool_name}' for approval."
                    )
                    tool.requires_approval = True
                    flagged += 1
            elif orchestration:
                # Orchestration tools are never sensitive, even if they match patterns
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
