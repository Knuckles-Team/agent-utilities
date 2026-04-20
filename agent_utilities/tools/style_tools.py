#!/usr/bin/python
# coding: utf-8
"""Output style tools with knowledge base integration.

Allows agents to set and discover response styles (concise, formal, etc.)
stored as KB Articles in the knowledge graph.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import RunContext

logger = logging.getLogger(__name__)

BUILTIN_STYLES = {
    "concise": "Keep your responses extremely brief and direct. No filler, no pleasantries. Use bullet points where possible.",
    "explanatory": "Provide detailed, step-by-step explanations. Define technical terms. Use analogies to clarify complex points.",
    "formal": "Maintain a strictly professional, academic tone. Use passive voice where appropriate. No slang or contractions.",
    "conversational": "Be friendly and approachable. Use contractions. Use 'I' and 'you'. Keep it light but helpful.",
}


async def set_output_style(ctx: RunContext[Any], style_name: str) -> str:
    """Set the agent's response style for the current session.

    The style can be a built-in keyword or a style name stored in the KB.
    """
    style_content = BUILTIN_STYLES.get(style_name.lower())

    if not style_content:
        # Check Knowledge Base for the style
        engine = getattr(ctx.deps, "graph_engine", None)
        if engine:
            # Query KB for an article named style_name in kb:output-styles
            # MATCH (a:Article {name: $name})-[:BELONGS_TO_KB]->(kb:KnowledgeBase {topic: 'output-styles'})
            # For simplicity, we search by name in the graph
            for node_id, node_data in engine.graph.nodes(data=True):
                if (
                    node_data.get("type") == "article"
                    and node_data.get("name") == style_name
                ):
                    style_content = node_data.get("content")
                    break

    if not style_content:
        return f"Style '{style_name}' not found. Available built-ins: {', '.join(BUILTIN_STYLES.keys())}"

    # Inject into context for subsequent prompt building or just return to agent
    # In agent-utilities, we can use a dynamic instruction
    ctx.deps.metadata["output_style"] = style_content
    return f"Output style set to '{style_name}'. Instruction: {style_content[:100]}..."


async def list_output_styles(ctx: RunContext[Any]) -> str:
    """List all available output styles including those in the KB."""
    styles = list(BUILTIN_STYLES.keys())

    engine = getattr(ctx.deps, "graph_engine", None)
    if engine:
        for node_id, node_data in engine.graph.nodes(data=True):
            if node_data.get("type") == "article" and "style" in node_data.get(
                "tags", []
            ):
                styles.append(node_data.get("name"))

    return f"Available styles: {', '.join(sorted(set(styles)))}"
