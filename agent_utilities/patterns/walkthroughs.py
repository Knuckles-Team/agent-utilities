#!/usr/bin/python
"""Codebase Walkthrough Generation Pattern.

This module provides tools to generate linear walkthroughs of a codebase or
feature implementation using subagents.
"""

import logging
from typing import Any

from .subagents import dispatch_subagent

logger = logging.getLogger(__name__)


async def generate_linear_walkthrough(
    path_or_query: str,
    deps: Any,
) -> str:
    """Generate a linear walkthrough for a given path or KG query.

    Args:
        path_or_query: Directory path or Knowledge Graph query string.
        deps: Agent dependencies.

    Returns:
        A markdown-formatted walkthrough.
    """
    logger.info(f"Generating linear walkthrough for: {path_or_query}")

    # We use a subagent to explore the path and extract key logic flows.
    goal = (
        f"Create a linear, step-by-step walkthrough of the code located at '{path_or_query}'.\n"
        f"Focus on the main entry points, key data structures, and the primary flow of logic.\n"
        f"Format as a high-quality markdown document (walkthrough.md style)."
    )

    walkthrough = await dispatch_subagent(
        goal=goal,
        deps=deps,
        name="Walkthrough-Agent",
        skill_types=["universal", "walkthroughs"],
        system_prompt_suffix="You are an expert at technical documentation and code explanation.",
    )

    # Hoard in KG if available
    if deps.knowledge_engine:
        deps.knowledge_engine.store_pattern_template(
            name=f"Walkthrough: {path_or_query}",
            pattern_type="walkthrough",
            content=walkthrough,
            tags=["documentation", "walkthrough", path_or_query],
        )

    return walkthrough
