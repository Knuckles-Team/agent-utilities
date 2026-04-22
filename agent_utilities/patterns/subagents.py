#!/usr/bin/python
"""Subagent Orchestration Pattern.

This module provides utilities for spawning and managing specialized subagents,
enabling context isolation and parallelism.
"""

import logging
from typing import Any

from ..agent_factory import create_agent
from ..models import AgentDeps

logger = logging.getLogger(__name__)


async def dispatch_subagent(
    goal: str,
    deps: AgentDeps,
    name: str = "SubAgent",
    skill_types: list[str] | None = None,
    system_prompt_suffix: str = "",
    model_id: str | None = None,
    tool_tags: list[str] | None = None,
    output_type: Any | None = None,
) -> Any:
    """Dispatch a task to a fresh subagent instance.

    Args:
        goal: The specific task or goal for the subagent.
        deps: The parent agent's dependencies.
        name: Name for the subagent.
        skill_types: List of skills to load (e.g., ['universal', 'tdd-methodology']).
        system_prompt_suffix: Additional instructions for the subagent.
        model_id: Optional model override.
        tool_tags: Optional tags to filter tools.
        output_type: Optional structured output type.

    Returns:
        The result of the subagent execution.
    """
    logger.info(f"Dispatching subagent '{name}' for goal: {goal[:100]}...")

    # Initialize subagent with requested skills
    sub_agent, _ = create_agent(
        name=name,
        model_id=model_id or deps.model_id,
        skill_types=skill_types,
        tool_tags=tool_tags,
        output_type=output_type,
        system_prompt=f"You are a specialized subagent named {name}.\n{system_prompt_suffix}",
        enable_universal_tools=False,  # Keep subagents focused
    )

    try:
        result = await sub_agent.run(goal, deps=deps)
        return result.output
    except Exception as e:
        logger.error(f"Subagent '{name}' failed: {e}")
        raise
