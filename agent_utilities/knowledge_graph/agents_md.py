#!/usr/bin/python
"""Project-Aware Context — AGENTS.md Loader.

CONCEPT:KG-2.14 — Project-Aware Context

This module provides dedicated entry points for loading ``AGENTS.md``
project rules and injecting them into the system prompt.  It consolidates
the AGENTS.md-related logic that previously existed only in
:mod:`agent_utilities.prompting.builder` and
:mod:`agent_utilities.tools.memory_tools` into a standalone module
registered in the conceptual registry.

Usage::

    from agent_utilities.knowledge_graph.agents_md import (
        load_agents_md,
        inject_project_context,
    )

    # Load all AGENTS.md content from a workspace
    content = load_agents_md("/path/to/workspace")

    # Inject project rules into a system prompt
    enriched = inject_project_context(system_prompt, "/path/to/workspace")

The file discovery order mirrors the existing ``memory_tools.py`` logic:

1. ``<workspace>/AGENTS.md``
2. ``<workspace>/.agents/AGENTS.md``
3. ``<workspace>/AGENTS.local.md`` (merged, not overriding)

See Also:
    - :func:`agent_utilities.prompting.builder.build_system_prompt_from_workspace`
    - :func:`agent_utilities.tools.memory_tools.read_agents_md`
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical file discovery order
_AGENTS_FILENAMES: list[str] = [
    "AGENTS.md",
    ".agents/AGENTS.md",
    "AGENTS.local.md",
]


def load_agents_md(workspace_path: str | Path) -> str | None:
    """Load AGENTS.md content from the workspace root.

    CONCEPT:KG-2.14 — Project-Aware Context

    Searches for AGENTS.md variants in the workspace directory using the
    canonical discovery order.  If multiple files exist, their contents
    are concatenated with section separators.

    Args:
        workspace_path: Absolute path to the project workspace root.

    Returns:
        The combined AGENTS.md content, or ``None`` if no files are found.

    Example::

        content = load_agents_md("/home/user/project")
        if content:
            print(f"Loaded {len(content)} chars of project rules")
    """
    root = Path(workspace_path)
    if not root.is_dir():
        logger.debug("Workspace path does not exist: %s", root)
        return None

    parts: list[str] = []
    for filename in _AGENTS_FILENAMES:
        path = root / filename
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    parts.append(f"--- Content from {filename} ---\n{text}")
                    logger.debug("Loaded project rules from %s", path)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", path, exc)

    if not parts:
        return None

    return "\n\n".join(parts)


def find_agents_md(start_path: str | Path) -> Path | None:
    """Walk up the directory tree to find the nearest AGENTS.md.

    CONCEPT:KG-2.14 — Project-Aware Context

    Searches upward from ``start_path`` until the filesystem root,
    returning the first directory that contains an ``AGENTS.md`` file.

    Args:
        start_path: Starting directory or file path.

    Returns:
        The ``Path`` to the found ``AGENTS.md``, or ``None``.

    Example::

        path = find_agents_md("/home/user/project/src/module")
        # Returns: PosixPath('/home/user/project/AGENTS.md')
    """
    current = Path(start_path).resolve()
    if current.is_file():
        current = current.parent

    visited: set[Path] = set()
    while current not in visited:
        visited.add(current)
        candidate = current / "AGENTS.md"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def inject_project_context(
    system_prompt: str,
    workspace_path: str | Path,
    *,
    include_memory: bool = True,
) -> str:
    """Inject AGENTS.md project rules into a system prompt.

    CONCEPT:KG-2.14 — Project-Aware Context

    Appends the loaded project rules as a clearly delimited section
    within the system prompt.  Optionally also injects ``MEMORY.md``
    content for full workspace context parity.

    Args:
        system_prompt: The base system prompt string.
        workspace_path: Absolute path to the project workspace root.
        include_memory: If True, also include MEMORY.md content.

    Returns:
        The enriched system prompt with project rules injected.

    Example::

        prompt = inject_project_context(
            "You are a helpful coding assistant.",
            "/home/user/project",
        )
    """
    parts: list[str] = [system_prompt]

    agents_content = load_agents_md(workspace_path)
    if agents_content:
        parts.append(
            f"\n\n---\n# AGENTS.md (Project Rules & Memory)\n{agents_content}\n---"
        )

    if include_memory:
        root = Path(workspace_path)
        memory_paths = [
            root / "MEMORY.md",
            root / ".agents" / "memory" / "MEMORY.md",
        ]
        for mp in memory_paths:
            if mp.is_file():
                try:
                    memory_content = mp.read_text(encoding="utf-8").strip()
                    if memory_content:
                        parts.append(
                            f"\n\n---\n# MEMORY.md (Learned Context)\n{memory_content}\n---"
                        )
                        break
                except Exception as exc:
                    logger.warning("Failed to read %s: %s", mp, exc)

    return "\n".join(parts)


def extract_project_metadata(workspace_path: str | Path) -> dict[str, Any]:
    """Extract structured metadata from AGENTS.md sections.

    CONCEPT:KG-2.14 — Project-Aware Context

    Parses the AGENTS.md content to extract key sections as structured
    metadata.  Recognized sections: Build Commands, Test Commands, Style
    Guidelines, Useful Commands.

    Args:
        workspace_path: Absolute path to the project workspace root.

    Returns:
        Dictionary of extracted metadata. Empty dict if AGENTS.md is
        not found.

    Example::

        meta = extract_project_metadata("/home/user/project")
        print(meta.get("test_commands", []))
    """
    import re

    content = load_agents_md(workspace_path)
    if not content:
        return {}

    metadata: dict[str, Any] = {"raw_content": content}
    sections: dict[str, str] = {}

    # Extract markdown sections
    current_header = ""
    current_lines: list[str] = []
    for line in content.split("\n"):
        header_match = re.match(r"^#{1,3}\s+(.+)$", line)
        if header_match:
            if current_header and current_lines:
                sections[current_header.lower().strip()] = "\n".join(
                    current_lines
                ).strip()
            current_header = header_match.group(1)
            current_lines = []
        else:
            current_lines.append(line)

    if current_header and current_lines:
        sections[current_header.lower().strip()] = "\n".join(current_lines).strip()

    # Map known sections to structured fields
    section_mapping = {
        "build commands": "build_commands",
        "test commands": "test_commands",
        "style guidelines": "style_guidelines",
        "useful commands": "useful_commands",
        "project": "project_name",
    }
    for header_key, meta_key in section_mapping.items():
        if header_key in sections:
            metadata[meta_key] = sections[header_key]

    metadata["sections"] = sections
    return metadata


__all__ = [
    "extract_project_metadata",
    "find_agents_md",
    "inject_project_context",
    "load_agents_md",
]
