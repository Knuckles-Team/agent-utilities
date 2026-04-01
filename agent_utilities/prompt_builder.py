#!/usr/bin/python

from __future__ import annotations

import os
import re
import logging
import asyncio


from typing import Dict, List, Optional, TYPE_CHECKING

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


from .models import PeriodicTask

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()


logger = logging.getLogger(__name__)


def extract_section_from_md(content: str, header: str) -> Optional[str]:
    """
    Extracts content under a specific markdown header (e.g., 'System Prompt').
    Matches headers like '## System Prompt' or '### System Prompt'.
    Returns None if the header is not found.
    """

    escaped_header = re.escape(header)

    pattern = rf"^\s*#+\s*{escaped_header}\s*\n(.*?)(?=\n#|\Z)"
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def get_system_prompt_from_reference(agent_template: str) -> Optional[str]:
    """
    Retrieves the system prompt for a template from its markdown reference.
    """

    identity_query = f"{agent_template}-identity.md"
    md_path = resolve_mcp_reference(identity_query)

    if md_path and os.path.exists(md_path):

        return Path(md_path).read_text(encoding="utf-8")

    queries = [
        f"{agent_template}.md",
        f"{agent_template}-mcp.md",
        f"{agent_template}-agent.md",
        f"{agent_template}-api.md",
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
    """
    Combine core files into a rich system prompt.
    Order matters — IDENTITY → USER → AGENTS → MEMORY → custom fallback
    """
    parts = []
    included_files = []

    logger.debug(
        f"Building system prompt from workspace. Fallback provided: {bool(fallback_prompt)}"
    )
    for key in ["IDENTITY", "USER", "AGENTS", "MEMORY"]:
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
    """Resolve a prompt string.

    If it starts with '@', load content from the referenced workspace file.
    Otherwise return the string as-is.
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
    """Extracts basic agent metadata from IDENTITY.md or returns defaults."""
    model = parse_identity(content)
    return {
        "name": model.name,
        "description": model.role,
        "emoji": model.emoji,
        "vibe": model.vibe,
        "content": model.system_prompt,
    }


def load_identity(tag: Optional[str] = None) -> Dict[str, str]:
    """
    Load IDENTITY.md and return metadata for the agent.
    """
    content = load_workspace_file("IDENTITY.md")
    if not content:
        return {"name": "Agent", "description": "AI Agent", "content": ""}

    return extract_agent_metadata(content)
