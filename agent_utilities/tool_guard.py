#!/usr/bin/python

from __future__ import annotations

import re
import logging
import asyncio


from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    pass


from pydantic_ai import Agent



from .config import *
from .workspace import *


from .models import PeriodicTask

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()




logger = logging.getLogger(__name__)


def is_sensitive_tool(name: str) -> bool:
    """Check if a tool name matches any sensitive pattern."""
    return any(re.match(pattern, name.lower()) for pattern in SENSITIVE_TOOL_PATTERNS)


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
            if (
                is_sensitive_tool(tool_name)
                and tool_name != "run_graph_flow"
                and not getattr(tool, "requires_approval", False)
            ):
                tool.requires_approval = True
                flagged += 1

    if flagged:
        logger.info(
            f"Tool Guard (native): Flagged {flagged} sensitive tools with requires_approval=True"
        )
