#!/usr/bin/python
# coding: utf-8
"""Workspace Management Tools Module.

This module provides tools for interacting with the agentic workspace,
including reading/writing core metadata files, managing dynamic skills,
and auditing the filesystem.
"""

import os
import json
import logging
from typing import Any, Union
from pydantic_ai import RunContext
from ..workspace import (
    CORE_FILES,
    parse_identity,
    parse_user_info,
    parse_memory,
    parse_cron_registry,
    parse_a2a_registry,
    parse_node_registry,
    parse_cron_log,
    read_md_file,
    append_to_md_file,
    create_new_skill,
    delete_skill_from_disk,
    write_skill_md,
    read_skill_md,
)
from ..models import (
    IdentityModel,
    UserModel,
    MemoryModel,
    CronRegistryModel,
    A2ARegistryModel,
    CronLogModel,
    MCPConfigModel,
)

logger = logging.getLogger(__name__)


async def read_workspace_file(ctx: RunContext[Any], filename: str) -> Union[
    IdentityModel,
    UserModel,
    MemoryModel,
    CronRegistryModel,
    A2ARegistryModel,
    CronLogModel,
    MCPConfigModel,
    str,
]:
    """Read and parse the content of a file within the workspace.

    Core configuration files (e.g., IDENTITY.md, CRON.md) are automatically
    parsed into their respective structured models. Other files are returned
    as raw strings.

    Args:
        ctx: The agent run context.
        filename: The relative path or identifier of the file to read.

    Returns:
        A structured model for core files, or the raw string content for others.

    """
    content = read_md_file(filename)
    if "File not found" in content:
        # Try loading as raw workspace file (maybe JSON) from the project root
        project_root = os.getcwd()
        full_path = os.path.join(project_root, filename)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

    # Match against core files to return structured models
    if filename == CORE_FILES["IDENTITY"]:
        return parse_identity(content)
    if filename == CORE_FILES["USER"]:
        return parse_user_info(content)
    if filename == CORE_FILES["MEMORY"]:
        return parse_memory(content)
    if filename == CORE_FILES["CRON"]:
        return parse_cron_registry(content)
    if filename == CORE_FILES["A2A_AGENTS"]:
        return parse_a2a_registry(content)
    if filename == CORE_FILES["NODE_AGENTS"]:
        return parse_node_registry(content)
    if filename == CORE_FILES["CRON_LOG"]:
        return parse_cron_log(content)
    if filename == CORE_FILES["MCP_CONFIG"]:
        try:
            return MCPConfigModel.model_validate(json.loads(content))
        except Exception:
            return content

    return content


async def append_note_to_file(ctx: RunContext[Any], filename: str, text: str) -> str:
    """Append a markdown entry or note to an existing workspace file.

    Args:
        ctx: The agent run context.
        filename: The target markdown file.
        text: The content to append.

    Returns:
        A confirmation message indicating success.

    """
    append_to_md_file(filename, text)
    return f"Appended to {filename}"


async def create_skill(
    ctx: RunContext[Any],
    name: str,
    description: str,
    when_to_use: str = "",
    how_to_use: str = "",
) -> str:
    """Initialize a new dynamic skill with its corresponding SKILL.md definition.

    New skills are automatically loaded into the agent's toolset in
    subsequent sessions.

    Args:
        ctx: The agent run context.
        name: The unique slug for the new skill.
        description: A brief summary of the skill's purpose.
        when_to_use: Guidance on appropriate trigger conditions.
        how_to_use: Implementation details and examples.

    Returns:
        A confirmation message indicating success.

    """
    return create_new_skill(name, description, when_to_use, how_to_use)


async def delete_skill(ctx: RunContext[Any], name: str) -> str:
    """Permanently remove a dynamic skill folder from the local workspace.

    Args:
        ctx: The agent run context.
        name: The slug of the skill to delete.

    Returns:
        A confirmation message indicating success.

    """
    return delete_skill_from_disk(name)


async def edit_skill(ctx: RunContext[Any], name: str, new_content: str) -> str:
    """Update the logic or documentation of an existing workspace skill.

    Args:
        ctx: The agent run context.
        name: The slug of the skill to update.
        new_content: The full replacement text for the SKILL.md file.

    Returns:
        A confirmation message indicating success.

    """
    return write_skill_md(name, new_content)


async def get_skill_content(ctx: RunContext[Any], name: str) -> str:
    """Retrieve the raw markdown definition of a workspace skill.

    Args:
        ctx: The agent run context.
        name: The slug of the skill to read.

    Returns:
        The content of the SKILL.md file.

    """
    return read_skill_md(name)


# New: List Workspace Files (Code Puppy Port)
async def list_files(
    ctx: RunContext[Any], path: str = ".", recursive: bool = False
) -> str:
    """Audit and list files within the workspace or a sub-directory.

    Args:
        ctx: The agent run context.
        path: The target directory to scan.
        recursive: Whether to perform a deep scan of sub-directories.

    Returns:
        A newline-separated list of relative file paths.

    """
    try:
        files = []
        for root, dirs, filenames in os.walk(path):
            for filename in filenames:
                files.append(os.path.join(root, filename))
            if not recursive:
                break
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {e}"


# Tool grouping for registration
workspace_tools = [
    read_workspace_file,
    append_note_to_file,
    create_skill,
    delete_skill,
    edit_skill,
    get_skill_content,
    list_files,
]
