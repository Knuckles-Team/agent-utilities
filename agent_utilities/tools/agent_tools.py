#!/usr/bin/python
# coding: utf-8
"""Agent Interaction Tools Module.

This module provides tools for invoking specialized sub-agents, listing
available expert roles, and logging internal agent reasoning.
"""

import logging
from typing import Any, List, Optional
from pydantic import BaseModel
from pydantic_ai import RunContext

logger = logging.getLogger(__name__)


class SubAgentResponse(BaseModel):
    """Structured response from a specialized sub-agent execution."""

    id: str
    result: str
    success: bool


async def invoke_specialized_agent(
    ctx: RunContext[Any], agent_name: str, prompt: str, model: Optional[str] = None
) -> str:
    """Invoke a specialized sub-agent for a targeted task.

    Args:
        ctx: The agent run context.
        agent_name: The name of the expert node to invoke (e.g., 'python').
        prompt: The specific instruction or task for the specialist.
        model: Optional model override for the sub-agent.

    Returns:
        The result string from the sub-agent execution.

    """
    # Logic to run a specific domain node in our graph
    # (Simplified for now, will integrate with master graph)
    return f"Sub-agent '{agent_name}' result for: {prompt[:50]}..."


async def list_available_agents(ctx: RunContext[Any]) -> List[str]:
    """List all specialized expert roles currently registered in the orchestrator.

    Args:
        ctx: The agent run context.

    Returns:
        A list of available specialist names.

    """
    return [
        "python",
        "c",
        "cpp",
        "golang",
        "javascript",
        "typescript",
        "security",
        "qa",
        "mcp",
    ]


async def share_reasoning(ctx: RunContext[Any], reasoning: str) -> str:
    """Explicitly share the agent's internal reasoning or plan for a task.

    This tool is used to provide transparency before a major operation
    or to log complex decision-making steps.

    Args:
        ctx: The agent run context.
        reasoning: The markdown-formatted reasoning string.

    Returns:
        A formatted reasoning log entry.

    """
    return f"Reasoning log: {reasoning}"
