#!/usr/bin/python
"""Prompt Builder Module.

This module provides utilities for constructing and resolving agent system
prompts. It handles loading structured JSON prompt blueprints (with a
``content`` key) from both the workspace and the package ``prompts/``
directory, resolving workspace file references (using the ``@`` prefix),
and aggregating the ``main_agent.json`` configuration and Knowledge Graph
context into a unified prompt context for the agent.

Prompts are JSON-only; markdown fallbacks (YAML-frontmatter and the legacy
star-based format) have been removed. Companion files such as ``AGENTS.md``
and ``MEMORY.md`` are still read as plain markdown — they are not prompt
blueprints, they are contextual memory surfaces.
"""

from __future__ import annotations

import json
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


def _extract_prompt_content(raw: str) -> str:
    """Return the body of a JSON prompt blueprint.

    Args:
        raw: The raw file contents. Must be a JSON object containing a
            ``content`` (preferred) or ``input`` key.

    Returns:
        The value of the ``content`` key, falling back to ``input``.

    Raises:
        ValueError: If ``raw`` is empty, not valid JSON, not a JSON object,
            or the object lacks both ``content`` and ``input`` keys.
    """
    if not raw or not raw.strip():
        raise ValueError("Prompt payload is empty")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Prompt payload is not valid JSON; expected a blueprint object "
            "with a 'content' key"
        ) from e
    if not isinstance(data, dict):
        raise ValueError("Prompt JSON must decode to an object")
    for key in ("content", "input"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(
        "Prompt JSON object is missing a 'content' or 'input' string value"
    )


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


def _load_main_agent_content() -> str:
    """Return the ``content`` body of ``main_agent.json``.

    Checks the workspace first, then the packaged default. Returns an empty
    string when neither source yields a usable blueprint; malformed JSON is
    logged as a warning rather than raised so agent startup is resilient.
    """
    raw = load_workspace_file("main_agent.json")
    if raw:
        try:
            return _extract_prompt_content(raw)
        except ValueError as e:
            logger.warning(
                "Invalid main_agent.json in workspace (%s); trying package default",
                e,
            )

    try:
        from importlib.resources import files

        prompts_dir = files("agent_utilities") / "prompts"
        main_agent_path = prompts_dir / "main_agent.json"
        if main_agent_path.is_file():
            raw = main_agent_path.read_text(encoding="utf-8")
            try:
                return _extract_prompt_content(raw)
            except ValueError as e:
                logger.warning(
                    "Invalid packaged main_agent.json (%s); using empty prompt",
                    e,
                )
    except Exception as e:
        logger.warning(f"Could not load main_agent.json from package: {e}")

    return ""


def build_system_prompt_from_workspace(fallback_prompt: str = "") -> str:
    """Aggregate core workspace files into a unified system prompt.

    Combines the ``content`` body of ``main_agent.json`` with
    ``AGENTS.md`` / ``MEMORY.md`` (if present) and an optional
    ``fallback_prompt`` string. ``main_agent.json`` is strictly JSON; only
    its ``content`` key is used as the prompt body.

    Args:
        fallback_prompt: An optional string to append if core files are
            insufficient.

    Returns:
        The final combined system prompt string.
    """
    parts: list[str] = []
    included_files: list[str] = []

    logger.debug(
        "Building system prompt from workspace. "
        f"Fallback provided: {bool(fallback_prompt)}"
    )

    main_content = _load_main_agent_content()
    if main_content.strip():
        parts.append(f"---\n# main_agent.json\n{main_content}\n---")
        included_files.append("main_agent.json")

    # AGENTS.md (Claude Code parity) — markdown is the source-of-truth here
    agents_content = load_workspace_file("AGENTS.md")
    if agents_content:
        parts.append(
            f"---\n# AGENTS.md (Project Rules & Memory)\n{agents_content}\n---"
        )
        included_files.append("AGENTS.md")

    # MEMORY.md (Auto Memory) — markdown surface, not a prompt blueprint
    memory_content = load_workspace_file("MEMORY.md")
    if memory_content:
        parts.append(f"---\n# MEMORY.md (Learned Context)\n{memory_content}\n---")
        included_files.append("MEMORY.md")
    else:
        memory_content = load_workspace_file(".agents/memory/MEMORY.md")
        if memory_content:
            parts.append(f"---\n# MEMORY.md\n{memory_content}\n---")
            included_files.append(".agents/memory/MEMORY.md")

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
    """Extract metadata (name, description, emoji, vibe, etc.) from a JSON blueprint.

    Only the modern JSON blueprint schema (a dict with ``name``,
    ``description``, ``content`` keys) is supported. Legacy YAML-frontmatter
    and star-based markdown formats have been removed.

    If ``content`` cannot be parsed as a JSON object, a generic default is
    returned and a warning is logged so agent startup remains resilient.
    """
    meta: dict[str, Any] = {
        "name": "Agent",
        "description": "AI Agent",
        "emoji": "🤖",
        "content": content,
        "vibe": "Neutral",
    }

    stripped = content.lstrip() if content else ""
    if not stripped.startswith("{"):
        logger.warning(
            "extract_agent_metadata: content is not a JSON blueprint; "
            "returning generic default metadata"
        )
        return meta

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        logger.warning(
            f"extract_agent_metadata: failed to parse JSON blueprint ({e}); "
            "returning generic default metadata"
        )
        return meta

    if not isinstance(data, dict):
        logger.warning(
            "extract_agent_metadata: JSON blueprint is not an object; "
            "returning generic default metadata"
        )
        return meta

    if "role" in data and "description" not in data:
        data["description"] = data.pop("role")
    meta.update(data)
    body = data.get("content") or data.get("input") or ""
    if isinstance(body, str):
        meta["content"] = body
    return meta


def load_identity(tag: str | None = None) -> dict[str, str]:
    """Load the primary ``main_agent.json`` file and return agent metadata.

    Args:
        tag: Optional tag filter (not currently used in base implementation).

    Returns:
        A dictionary of agent metadata. Defaults to generic values if
        ``main_agent.json`` is missing.

    """
    try:
        from importlib.resources import files

        prompts_dir = files("agent_utilities") / "prompts"
        main_agent_path = prompts_dir / "main_agent.json"
        if main_agent_path.is_file():
            content = main_agent_path.read_text(encoding="utf-8")
            return extract_agent_metadata(content)
    except Exception as e:
        logger.warning(f"Could not load main_agent.json identity: {e}")

    return {"name": "Agent", "description": "AI Agent", "content": ""}
