#!/usr/bin/python
from __future__ import annotations

"""Tool Filtering Module.

CONCEPT:AU-ECO.messaging.native-backend-abstraction

This module provides utility functions for filtering skills and MCP tools based
on tags, categories, and frontmatter metadata. It supports dynamic discovery
from directory structures and robust extraction of metadata from various
tool definition objects.
"""


import logging
import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore

logger = logging.getLogger(__name__)


def extract_skill_tags(skill_path: str) -> list[str]:
    """Extract tags from the frontmatter of a skill's SKILL.md file.

    Args:
        skill_path: Path to the skill directory.

    Returns:
        A list of tag strings extracted from the frontmatter.

    """
    skill_file = Path(skill_path) / "SKILL.md"
    if not skill_file.exists():
        return []

    try:
        with open(skill_file) as f:
            content = f.read()
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = parts[1]
                    data = yaml.safe_load(frontmatter)
                    tags = data.get("tags", [])
                    if isinstance(tags, str):
                        return [tags]
                    if isinstance(tags, list):
                        return [str(t) for t in tags]
    except Exception as e:
        logger.debug("Error reading configured skill tags (%s)", type(e).__name__)

    return []


def skill_in_tag(skill_path: str, tag: str) -> bool:
    """Check if a skill belongs to a specific tag.

    Args:
        skill_path: Path to the skill directory.
        tag: The tag string to check for.

    Returns:
        True if the tag is present in the skill's metadata, False otherwise.

    """
    tool_tags = extract_skill_tags(skill_path)
    return tag in tool_tags


def filter_skills_by_tag(skills: list[str], tag: str) -> list[str]:
    """Filter a list of skill paths based on a given tag.

    Args:
        skills: A list of skill directory paths.
        tag: The tag string to filter by.

    Returns:
        A list of skill paths that contain the specified tag.

    """
    return [s for s in skills if skill_in_tag(s, tag)]


def get_skill_directories_by_tag(base_dir: str, tag: str) -> list[str]:
    """Find all skill directories within a base directory that match a tag.

    Args:
        base_dir: The parent directory to scan.
        tag: The tag string to match.

    Returns:
        A list of absolute or relative paths to matching skill directories.

    """
    skill_dirs: list[str] = []
    base_path = Path(base_dir)

    if not base_path.exists() or not base_path.is_dir():
        return skill_dirs

    for item in base_path.iterdir():
        if item.is_dir() and skill_in_tag(str(item), tag):
            skill_dirs.append(str(item))

    return skill_dirs


def skill_matches_tags(skill_dir: str, tags: list[str]) -> bool:
    """Check if a skill directory matches any of the provided tags.

    Reads frontmatter 'tags' and 'categories' to perform a multi-vector match.

    Args:
        skill_dir: Path to the skill directory.
        tags: List of tag strings to check against.

    Returns:
        True if any tag matches, False otherwise.

    """
    skill_md = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(skill_md):
        return False

    try:
        with open(skill_md) as f:
            content = f.read()

        import re

        import yaml

        fm_match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL | re.MULTILINE)
        if not fm_match:
            return False

        data = yaml.safe_load(fm_match.group(1)) or {}
        tool_tags = data.get("tags", [])
        if isinstance(tool_tags, str):
            tool_tags = [tool_tags]

        skill_categories = data.get("categories", [])
        if isinstance(skill_categories, str):
            skill_categories = [skill_categories]

        all_skill_metadata = set(
            [t.lower() for t in tool_tags] + [c.lower() for c in skill_categories]
        )

        all_skill_metadata.add(os.path.basename(skill_dir).lower())

        return any(tag.lower() in all_skill_metadata for tag in tags)
    except Exception as e:  # noqa: BLE001 — fails closed (excludes the skill from this tag-filtered subset rather than including it), the safe direction for a filter; the skill itself is untouched, only its visibility under this particular tag query is affected
        logger.debug(f"Error checking tags for skill {skill_dir}: {e}")
        return False


def extract_tool_tags(tool_def: Any) -> list[str]:
    """Extract tags from multiple potential locations in a tool definition.

    Reads the protocol-neutral ``tags`` the graph-os host normalizes into a
    tool's ``meta``/``metadata`` (the host owns any server-vendor metadata
    namespace), plus a direct ``tags`` attribute.

    Args:
        tool_def: The tool definition object (from a host-provided toolset).

    Returns:
        A list of tag strings associated with the tool.

    """
    tags_list = []

    meta = getattr(tool_def, "meta", None)
    if isinstance(meta, dict):
        tags_list = meta.get("tags", [])
        if tags_list:
            return tags_list

    metadata = getattr(tool_def, "metadata", None)
    if isinstance(metadata, dict):
        tags_list = metadata.get("tags", [])
        if tags_list:
            return tags_list

        meta_nested = metadata.get("meta") or {}
        tags_list = meta_nested.get("tags", [])
        if tags_list:
            return tags_list

    tags_list = getattr(tool_def, "tags", [])
    if isinstance(tags_list, list | set | tuple) and tags_list:
        return list(tags_list)

    return []


def tool_in_tag(tool_def: Any, tag: str) -> bool:
    """Check if a tool definition belongs to a specific tag.

    Args:
        tool_def: The tool definition object.
        tag: The tag string to check for.

    Returns:
        True if the tool is tagged with the specified value, False otherwise.

    """
    tool_tags = extract_tool_tags(tool_def)
    if tag in tool_tags:
        return True
    else:
        return False


def filter_tools_by_tag(tools: Any, tags: str | list[str]) -> Any:
    """Filter a list of tools or a ToolSet by one or more tags.

    If multiple tags are provided, a tool is included if it matches any
    of the specified tags (OR logic).

    Args:
        tools: A list of tools or a filterable ToolSet instance.
        tags: A single tag string or a list of tag strings.

    Returns:
        The filtered collection of tools.

    """
    if isinstance(tags, str):
        tag_list = [tags]
    else:
        tag_list = tags

    if hasattr(tools, "filtered"):
        return tools.filtered(
            lambda ctx, tool_def: any(
                tag.lower() in [t.lower() for t in extract_tool_tags(tool_def)]
                for tag in tag_list
            )
        )
    elif isinstance(tools, list):
        return [t for t in tools if any(tool_in_tag(t, tag) for tag in tag_list)]
    return tools
