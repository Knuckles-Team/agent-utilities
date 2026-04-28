#!/usr/bin/python
"""Interactive Explanation Generation Pattern.

This module provides tools to generate self-contained HTML/JS artifacts
that explain complex concepts or logic flows interactively.
"""

import logging
from typing import Any

from .subagents import dispatch_subagent

logger = logging.getLogger(__name__)


async def generate_interactive_explanation(
    explanation_goal: str,
    content_to_explain: str,
    deps: Any,
) -> str:
    """Generate an interactive HTML/JS artifact for a given goal.

    Args:
        explanation_goal: The specific thing to explain (e.g. "How the dispatcher works").
        content_to_explain: The raw content/code to base the explanation on.
        deps: Agent dependencies.

    Returns:
        The HTML content of the interactive explanation.
    """
    logger.info(f"Generating interactive explanation for: {explanation_goal}")

    # We use a subagent to generate the HTML/JS.
    # Simon's preference: Vanilla JS, no React.
    goal = (
        f"Generate a self-contained, high-quality interactive HTML/JS explanation for: {explanation_goal}.\n"
        f"Use the following content as context: {content_to_explain}\n"
        f"Requirements:\n"
        f"1. Vanilla HTML/CSS/JS (no external frameworks like React/Vue).\n"
        f"2. Use CSS for modern aesthetics (vibrant colors, glassmorphism).\n"
        f"3. Make it interactive (e.g. clickable steps, hover effects, simple animations).\n"
        f"4. The output must be valid HTML that can be saved and opened in a browser."
    )

    html_artifact = await dispatch_subagent(
        goal=goal,
        deps=deps,
        name="Explanation-Agent",
        skill_types=["universal", "interactive-explain"],
        system_prompt_suffix="You are an expert at creating interactive web-based educational content.",
    )

    return html_artifact
