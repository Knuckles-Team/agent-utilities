#!/usr/bin/python
"""Agentic Manual Testing Pattern.

This module enables agents to perform interactive verification using shell commands,
curl, or direct code execution.
"""

import logging

from ..models import AgentDeps
from .subagents import dispatch_subagent

logger = logging.getLogger(__name__)


async def run_agentic_manual_test(
    verification_goal: str,
    deps: AgentDeps,
    context: str = "",
) -> str:
    """Execute an agentic manual test loop.

    Args:
        verification_goal: What needs to be verified (e.g., 'Check if /health returns 200').
        deps: Agent dependencies.
        context: Additional code or configuration context.

    Returns:
        The result of the manual testing process.
    """
    logger.info(f"Running agentic manual test: {verification_goal}")

    result = await dispatch_subagent(
        goal=f"Perform a manual test to verify the following: {verification_goal}\n"
        f"Context: {context}\n"
        "You have access to shell tools. Use curl, python, or other available tools to verify behavior.",
        deps=deps,
        name="Manual-Test-Agent",
        skill_types=["universal"],
        system_prompt_suffix="You are an expert at runtime verification and edge-case testing.",
    )

    return result
