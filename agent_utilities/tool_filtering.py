#!/usr/bin/python

from __future__ import annotations

import os
import re
import logging
import asyncio
import yaml
from typing import Any, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from fasta2a import Skill
from pathlib import Path


from .models import PeriodicTask

tasks: List[PeriodicTask] = []
lock = asyncio.Lock()


logger = logging.getLogger(__name__)


def _parse_skill_from_md(skill_file: Path, skill_id: str) -> Optional[Skill]:
    from fasta2a import Skill
    import yaml

    try:
        with open(skill_file, "r") as f:
            content = f.read()

            fm_match = re.search(
                r"^---\s*\n(.*?)\n---", content, re.DOTALL | re.MULTILINE
            )
            if fm_match:
                frontmatter = fm_match.group(1)
                data = yaml.safe_load(frontmatter)

                skill_name = data.get("name", skill_id)
                skill_desc = data.get("description", f"Access to {skill_name} tools")

                skill_version = str(
                    data.get(
                        "version", data.get("metadata", {}).get("version", "0.1.0")
                    )
                )

                tool_tags = data.get("tags", [skill_id])
                if not isinstance(tool_tags, list):
                    tool_tags = [str(tool_tags)]

                return Skill(
                    id=skill_id,
                    name=skill_name,
                    description=skill_desc,
                    version=skill_version,
                    tags=tool_tags,
                    input_modes=["text"],
                    output_modes=["text"],
                )
    except Exception as e:
        logger.debug(f"Error parsing skill from {skill_file}: {e}")
    return None


def load_skills_from_directory(directory: str) -> List[Skill]:

    skills = []
    base_path = Path(directory)

    if not base_path.exists():
        logger.debug(f"Skills directory not found: {directory}")
        return skills

    skill_file = base_path / "SKILL.md"
    if skill_file.exists():
        skill = _parse_skill_from_md(skill_file, base_path.name)
        if skill:
            skills.append(skill)
            return skills

    if base_path.is_dir():
        for item in base_path.iterdir():
            if item.is_dir():
                sub_skill_file = item / "SKILL.md"
                if sub_skill_file.exists():
                    skill = _parse_skill_from_md(sub_skill_file, item.name)
                    if skill:
                        skills.append(skill)
    return skills


def extract_skill_tags(skill_path: str) -> List[str]:
    """
    Extracts tags from the frontmatter of a skill's SKILL.md.
    """
    skill_file = Path(skill_path) / "SKILL.md"
    if not skill_file.exists():
        return []

    try:
        with open(skill_file, "r") as f:
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
        logger.debug(f"Error reading tags from {skill_file}: {e}")

    return []


def skill_in_tag(skill_path: str, tag: str) -> bool:
    """
    Checks if a skill belongs to a specific tag.
    """
    tool_tags = extract_skill_tags(skill_path)
    return tag in tool_tags


def filter_skills_by_tag(skills: List[str], tag: str) -> List[str]:
    """
    Filters a list of skill paths for a given tag.
    """
    return [s for s in skills if skill_in_tag(s, tag)]


def get_skill_directories_by_tag(base_dir: str, tag: str) -> List[str]:
    """
    Finds all skill directories within base_dir that have the specified tag in their SKILL.md.
    """
    skill_dirs = []
    base_path = Path(base_dir)

    if not base_path.exists() or not base_path.is_dir():
        return skill_dirs

    for item in base_path.iterdir():
        if item.is_dir() and skill_in_tag(str(item), tag):
            skill_dirs.append(str(item))

    return skill_dirs


def skill_matches_tags(skill_dir: str, tags: List[str]) -> bool:
    """
    Checks if a skill directory matches any of the given tags.
    Reads SKILL.md frontmatter for 'tags' and 'categories'.
    """
    skill_md = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(skill_md):
        return False

    try:
        with open(skill_md, "r") as f:
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
    except Exception as e:
        logger.debug(f"Error checking tags for skill {skill_dir}: {e}")
        return False


def extract_tool_tags(tool_def: Any) -> List[str]:
    """
    Extracts tags from a tool definition object.

    Found structure in debug:
    tool_def.name (str)
    tool_def.meta (dict) -> {'fastmcp': {'tags': ['tag']}}

    This function checks multiple paths to be robust:
    1. tool_def.meta['fastmcp']['tags']
    2. tool_def.meta['tags']
    3. tool_def.metadata['tags'] (legacy/alternative wrapper)
    4. tool_def.metadata.get('meta')... (nested path)
    """
    tags_list = []

    meta = getattr(tool_def, "meta", None)
    if isinstance(meta, dict):
        fastmcp = meta.get("fastmcp") or meta.get("_fastmcp") or {}
        tags_list = fastmcp.get("tags", [])
        if tags_list:
            return tags_list

        tags_list = meta.get("tags", [])
        if tags_list:
            return tags_list

    metadata = getattr(tool_def, "metadata", None)
    if isinstance(metadata, dict):
        tags_list = metadata.get("tags", [])
        if tags_list:
            return tags_list

        meta_nested = metadata.get("meta") or {}
        fastmcp = meta_nested.get("fastmcp") or meta_nested.get("_fastmcp") or {}
        tags_list = fastmcp.get("tags", [])
        if tags_list:
            return tags_list

        tags_list = meta_nested.get("tags", [])
        if tags_list:
            return tags_list

    tags_list = getattr(tool_def, "tags", [])
    if isinstance(tags_list, (list, set, tuple)) and tags_list:
        return list(tags_list)

    return []


def tool_in_tag(tool_def: Any, tag: str) -> bool:
    """
    Checks if a tool belongs to a specific tag.
    """
    tool_tags = extract_tool_tags(tool_def)
    if tag in tool_tags:
        return True
    else:
        return False


def filter_tools_by_tag(tools: Any, tags: Union[str, List[str]]) -> Any:
    """
    Filters a list of tools, or a ToolSet like MCPServer for a given tag(s).
    If multiple tags are provided, it returns tools that match ANY of the tags.
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
