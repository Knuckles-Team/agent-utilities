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
    parse_mcp_registry,
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
    """Read content of any .md or .json file in workspace.
    Returns a structured object for core files (IDENTITY.md, CRON.md, etc.)
    or a string for other files.
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
    if filename == CORE_FILES["MCP_AGENTS"]:
        return parse_mcp_registry(content)
    if filename == CORE_FILES["CRON_LOG"]:
        return parse_cron_log(content)
    if filename == CORE_FILES["MCP_CONFIG"]:
        try:
            return MCPConfigModel.model_validate(json.loads(content))
        except Exception:
            return content

    return content


async def append_note_to_file(ctx: RunContext[Any], filename: str, text: str) -> str:
    """Append a short note or section to a workspace .md file."""
    append_to_md_file(filename, text)
    return f"Appended to {filename}"


async def create_skill(
    ctx: RunContext[Any],
    name: str,
    description: str,
    when_to_use: str = "",
    how_to_use: str = "",
) -> str:
    """Create a brand-new skill folder + SKILL.md that will be auto-loaded on next run."""
    return create_new_skill(name, description, when_to_use, how_to_use)


async def delete_skill(ctx: RunContext[Any], name: str) -> str:
    """Delete a skill folder from the workspace. Only works for workspace skills."""
    return delete_skill_from_disk(name)


async def edit_skill(ctx: RunContext[Any], name: str, new_content: str) -> str:
    """
    Overwrite the SKILL.md of an existing workspace skill.
    Use this to refine a skill's logic, description, or examples.
    """
    return write_skill_md(name, new_content)


async def get_skill_content(ctx: RunContext[Any], name: str) -> str:
    """Read the current SKILL.md of a workspace skill to prepare for editing."""
    return read_skill_md(name)


# New: List Workspace Files (Code Puppy Port)
async def list_files(
    ctx: RunContext[Any], path: str = ".", recursive: bool = False
) -> str:
    """List files in the workspace directory."""
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
