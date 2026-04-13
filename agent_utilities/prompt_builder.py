#!/usr/bin/python
# coding: utf-8
"""Prompt Builder Module.

This module provides utilities for constructing and resolving agent system
prompts. It handles dynamic extraction of content from markdown files,
resolving workspace file references (using the '@' prefix), and aggregating
identity, user, and memory files into a unified prompt context.
"""

from __future__ import annotations

import os
import re
import logging


from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass
from pathlib import Path


from universal_skills.skill_utilities import (
    resolve_mcp_reference,
)


from .workspace import (
    CORE_FILES,
    load_workspace_file,
    parse_identity,
)

logger = logging.getLogger(__name__)


def extract_section_from_md(content: str, header: str) -> Optional[str]:
    """Extract content under a specific markdown header.

    Matches headers following the pattern '## Header Name' or '### Header Name'
    and captures all content until the next header of equal or higher level.

    Args:
        content: The raw markdown content string.
        header: The exact header text to search for (case-insensitive).

    Returns:
        The extracted section content as a string, or None if the header is not found.

    """
    escaped_header = re.escape(header)

    pattern = rf"^\s*#+\s*{escaped_header}\s*\n(.*?)(?=\n#|\Z)"
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def get_system_prompt_from_reference(agent_name: str) -> Optional[str]:
    """Retrieve the system prompt for an agent from its markdown reference file.

    Scans the filesystem for matching agent configuration files (e.g.
    [agent_name]-identity.md, [agent_name].md) and extracts the 'System Prompt'
    section.

    Args:
        agent_name: The slugified name of the agent to resolve.

    Returns:
        The extracted system prompt string, or None if no reference is found.

    """
    identity_query = f"{agent_name}-identity.md"
    md_path = resolve_mcp_reference(identity_query)

    if md_path and os.path.exists(md_path):
        return Path(md_path).read_text(encoding="utf-8")

    queries = [
        f"{agent_name}.md",
        f"{agent_name}-mcp.md",
        f"{agent_name}-agent.md",
        f"{agent_name}-api.md",
    ]

    md_path = None
    for query in queries:
        md_path = resolve_mcp_reference(query)
        if md_path:
            break

    if md_path and os.path.exists(md_path):
        content = Path(md_path).read_text(encoding="utf-8")

        return extract_section_from_md(content, "System Prompt")

    return None


def build_system_prompt_from_workspace(fallback_prompt: str = "") -> str:
    """Aggregate core workspace files into a unified system prompt.

    Combines IDENTITY.md, USER.md, and MEMORY.md content with an optional
    fallback string. The order of aggregation is fixed to ensure proper LLM
    instruction hierarchy.

    Args:
        fallback_prompt: An optional string to append if core files are insufficient.

    Returns:
        The final combined system prompt string.

    """
    parts = []
    included_files = []

    logger.debug(
        f"Building system prompt from workspace. Fallback provided: {bool(fallback_prompt)}"
    )
    for key in ["IDENTITY", "USER", "MEMORY"]:
        filename = CORE_FILES[key]
        logger.debug(f"Checking for {key} file: {filename}")
        content = load_workspace_file(filename)
        if content.strip():
            logger.debug(
                f"Including {filename} in system prompt (Snippet: {content[:50]}...)"
            )
            parts.append(f"---\n# {filename}\n{content}\n---")
            included_files.append(filename)
        else:
            logger.debug(f"File {filename} is empty or missing content.")

    if fallback_prompt:
        parts.append(fallback_prompt)
        included_files.append("fallback_prompt")

    prompt = "\n\n".join(parts).strip()
    logger.debug(f"Built System Prompt from files: {', '.join(included_files)}")
    return prompt


def resolve_prompt(prompt_str: str) -> str:
    """Resolve a prompt string, optionally loading from a file reference.

    If the string starts with '@', it is treated as a filename reference
    within the agent's workspace. Otherwise, the string is returned unchanged.

    Args:
        prompt_str: The prompt string to resolve.

    Returns:
        The resolved prompt content.

    """
    prompt_str = prompt_str.strip()
    if prompt_str.startswith("@"):
        filename = prompt_str[1:].strip()
        content = load_workspace_file(filename)
        if content and content.strip():
            return content.strip()
        logger.warning(
            f"Prompt file '{filename}' is empty or missing, using raw: {prompt_str}"
        )
    return prompt_str


def extract_agent_metadata(content: str) -> Dict[str, str]:
    """Extract structured agent metadata from IDENTITY.md content.

    Args:
        content: Raw markdown text from an identity file.

    Returns:
        A dictionary containing agent parameters like 'name', 'description',
        'emoji', 'vibe', and the extracted 'system_prompt'.

    """
    model = parse_identity(content)
    return {
        "name": model.name,
        "description": model.role,
        "emoji": model.emoji,
        "vibe": model.vibe,
        "content": model.system_prompt,
    }


def load_identity(tag: Optional[str] = None) -> Dict[str, str]:
    """Load the primary IDENTITY.md file and return agent metadata.

    Args:
        tag: Optional tag filter (not currently used in base implementation).

    Returns:
        A dictionary of agent metadata. Defaults to generic values if
        IDENTITY.md is missing.

    """
    content = load_workspace_file("IDENTITY.md")
    if not content:
        return {"name": "Agent", "description": "AI Agent", "content": ""}

    return extract_agent_metadata(content)
