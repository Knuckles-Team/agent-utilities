#!/usr/bin/python
"""Prompt Builder Module.

This module provides utilities for constructing and resolving agent system
prompts. It handles dynamic extraction of content from markdown files,
resolving workspace file references (using the '@' prefix), and aggregating
the main_agent.md configuration and Knowledge Graph context into a unified
prompt context for the agent.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass
from pathlib import Path

from universal_skills.skill_utilities import (
    resolve_mcp_reference,
)

from .workspace import (
    load_workspace_file,
)

logger = logging.getLogger(__name__)


def extract_section_from_md(content: str, header: str) -> str | None:
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


def get_system_prompt_from_reference(agent_name: str) -> str | None:
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

    Combines main_agent.md content with an optional
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

    # Try to load main_agent.md from workspace first
    content = load_workspace_file("main_agent.md")
    if content:
        parts.append(f"---\n# main_agent.md\n{content}\n---")
        included_files.append("main_agent.md")
    else:
        # Fallback to package resources
        try:
            from importlib.resources import files

            prompts_dir = files("agent_utilities") / "prompts"
            main_agent_path = prompts_dir / "main_agent.md"
            if main_agent_path.is_file():
                content = main_agent_path.read_text(encoding="utf-8")
                if content.strip():
                    parts.append(f"---\n# main_agent.md\n{content}\n---")
                    included_files.append("main_agent.md (pkg)")
        except Exception as e:
            logger.warning(f"Could not load main_agent.md from package: {e}")

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


def extract_agent_metadata(content: str) -> dict[str, Any]:
    """Extract metadata (name, description, emoji, vibe, etc.) from prompt content.

    Supports both YAML frontmatter (modern) and the legacy star-based format.
    """
    meta = {
        "name": "Agent",
        "description": "AI Agent",
        "emoji": "🤖",
        "content": content,
        "vibe": "Neutral",
    }

    # 1. Try YAML frontmatter
    import yaml

    fm_match = re.search(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if fm_match:
        try:
            fm_data = yaml.safe_load(fm_match.group(1))
            if isinstance(fm_data, dict):
                # Standardize keys
                if "role" in fm_data and "description" not in fm_data:
                    fm_data["description"] = fm_data.pop("role")

                meta.update(fm_data)
                meta["content"] = content[fm_match.end() :].strip()
                return meta
        except Exception:
            pass

    # 2. Try legacy star-based format
    # Example: * **Name:** TestBot
    name_match = re.search(r"\* \*\*Name:\*\* (.*)", content)
    if name_match:
        meta["name"] = name_match.group(1).strip()

    role_match = re.search(r"\* \*\*(Role|Description):\*\* (.*)", content)
    if role_match:
        meta["description"] = role_match.group(2).strip()

    emoji_match = re.search(r"\* \*\*Emoji:\*\* (.*)", content)
    if emoji_match:
        meta["emoji"] = emoji_match.group(1).strip()

    vibe_match = re.search(r"\* \*\*Vibe:\*\* (.*)", content)
    if vibe_match:
        meta["vibe"] = vibe_match.group(1).strip()

    # System Prompt section
    sp_match = re.search(
        r"### System Prompt\s*\n(.*?)(?=\n#|\Z)", content, re.DOTALL | re.MULTILINE
    )
    if sp_match:
        meta["content"] = sp_match.group(1).strip()

    return meta


def load_identity(tag: str | None = None) -> dict[str, str]:
    """Load the primary main_agent.md file and return agent metadata.

    Args:
        tag: Optional tag filter (not currently used in base implementation).

    Returns:
        A dictionary of agent metadata. Defaults to generic values if
        main_agent.md is missing.

    """
    try:
        from importlib.resources import files

        prompts_dir = files("agent_utilities") / "prompts"
        main_agent_path = prompts_dir / "main_agent.md"
        if main_agent_path.is_file():
            content = main_agent_path.read_text(encoding="utf-8")
            return extract_agent_metadata(content)
    except Exception as e:
        logger.warning(f"Could not load main_agent.md identity: {e}")

    return {"name": "Agent", "description": "AI Agent", "content": ""}
