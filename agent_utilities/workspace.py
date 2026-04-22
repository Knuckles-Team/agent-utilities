#!/usr/bin/python
"""Workspace Management Module.

This module provides the core logic for discovering, initializing, and
managing the agent's filesystem workspace. it includes templates for
standard markdown files (IDENTITY, etc.), parsing and
serialization logic for these files, and robust path discovery for
skills and MCP configurations.
"""

import logging
import os
import re
import shutil
from datetime import datetime
from importlib.resources import as_file, files
from pathlib import Path

from .base_utilities import load_env_vars, retrieve_package_name

logger = logging.getLogger(__name__)


WORKSPACE_DIR: str | None = None


def md_table_escape(text: str) -> str:
    """Escape markdown table delimiters and handle newlines for table safety.

    This function ensures that literal pipes are escaped and newline characters
    (both actual newlines and literal '\n' sequences) are converted to <br/>
    to maintain table integrity.

    Args:
        text: The raw string to escape.

    Returns:
        An escaped string suitable for a markdown table cell.

    """
    if not text:
        return ""
    # Standardize newlines and literal \n sequences to <br/>
    escaped = text.replace("\n", "<br/>").replace("\\n", "<br/>")
    # Escape pipe character which breaks markdown tables
    escaped = escaped.replace("|", "\\|")
    return escaped.strip()


def smart_truncate(text: str | None, limit: int) -> str:
    """Truncate text to a limit while respecting word boundaries.

    Args:
        text: The string to truncate.
        limit: The maximum character count.

    Returns:
        The truncated string with an ellipsis if applicable, cut at the last space
        before the limit to avoid partial words.

    """
    if not text or len(text) <= limit:
        return text or "-"

    truncated = text[:limit]
    # Respect word boundaries (split at last space)
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]

    return f"{truncated}..."


CORE_FILES = {
    "MAIN_AGENT": "main_agent.json",
    "MCP_CONFIG": "mcp_config.json",
}

TEMPLATES = {
    "MAIN_AGENT": """{
  "task": "main-agent",
  "input": "# main-agent\\n\\nYou are the primary orchestrator for this workspace. Your goal is to help the user manage their projects and coordinate specialized agents.\\n\\n### Core Principles\\n* Be concise and efficient.\\n* Use the knowledge graph to discover tools and experts.\\n* Verify your work before concluding.\\n\\nYour personality:\\n* **Emoji:** 🤖\\n* **Vibe:** Professional, efficient, helpful",
  "type": "prompt",
  "description": "The primary orchestrator agent for this workspace.",
  "tools": [
    "workspace-manager",
    "agent-workflows"
  ],
  "topic": "General Expertise",
  "tone": "technical and precise",
  "style": "professional assistant",
  "goal": "Coordinate specialized agents and manage the workspace."
}
""",
    "MCP_CONFIG": """{
    "mcpServers": {}
}
""",
}


def get_skills_path() -> list[str]:
    """Discover the filesystem paths to agent skills.

    Scans the local package and parent directories for 'skills' folders
    using importlib.resources and manual path construction.

    Returns:
        A list of absolute paths to discovered skills directories.

    """
    try:
        package_name = retrieve_package_name()
        if not package_name:
            # Fallback to current directory skills if no package found
            local_skills = Path.cwd() / "skills"
            if local_skills.exists():
                return [str(local_skills)]
            return []

        # Check for skills in standard subdirectories
        for sub in ["agent_data/skills", "agent/skills", "skills"]:
            try:
                from importlib.resources import files

                skills_dir = os.path.join(str(files(package_name)), sub)
                if os.path.exists(skills_dir):
                    logger.debug(f"Found skills at {skills_dir}")
                    return [str(skills_dir)]
            except (ImportError, ValueError, TypeError) as e:
                logger.debug(f"Subdirectory lookup fail for {sub}: {e}")
                continue

        # Fallback to manual path construction if resources.files fails
        try:
            import importlib.util

            spec = importlib.util.find_spec(package_name)
            if spec and spec.origin:
                pkg_root = Path(spec.origin).parent
                for sub in ["agent_data/skills", "agent/skills", "skills"]:
                    skills_path = pkg_root / sub
                    if skills_path.exists():
                        return [str(skills_path)]
        except Exception as e:
            logger.debug(f"Manual path fallback fail: {e}")

    except Exception as e:
        logger.debug(f"Error accessing skills path: {e}")
    return []


def get_mcp_config_path() -> str | None:
    """Retrieve the absolute path to the local MCP configuration file.

    Returns:
        The path to mcp_config.json if found, otherwise None.

    """
    try:
        package_name = retrieve_package_name()
        for sub in ["agent_data", "agent"]:
            mcp_config_file = os.path.join(
                str(files(package_name)), sub, "mcp_config.json"
            )
            if os.path.isfile(mcp_config_file):
                with as_file(Path(mcp_config_file)) as path:
                    return str(path)
        return None
    except Exception as e:
        logger.debug(f"Error accessing mcp_config path: {e}")
        return None


def get_agent_workspace() -> Path:
    """Discover the root workspace directory for the agent.

    Uses a tiered discovery strategy:
    1. Explicit global WORKSPACE_DIR override.
    2. AGENT_WORKSPACE environment variable.
    3. Local package subdirectories (agent_data, agent).
    4. CWD-relative search.
    5. Fallback to package-native agent_data.

    Returns:
        The absolute Path to the resolved workspace directory.

    """
    global WORKSPACE_DIR
    if WORKSPACE_DIR:
        p = Path(WORKSPACE_DIR).resolve()
        logger.debug(f"get_agent_workspace: Tier 1 SUCCESS (Override): {p}")
        return p

    env_workspace = os.getenv("AGENT_WORKSPACE")
    if env_workspace:
        p = Path(env_workspace).resolve()
        logger.debug(f"get_agent_workspace: Tier 2 Checking: {p}")
        if p.exists():
            logger.debug(f"get_agent_workspace: Tier 2 SUCCESS: {p}")
            WORKSPACE_DIR = str(p)
        return p

    pkg = retrieve_package_name()
    if pkg:
        try:
            # 1. Check local package directory in CWD
            pkg_local = Path.cwd() / pkg
            if pkg_local.is_dir():
                p = pkg_local.resolve()
                WORKSPACE_DIR = str(p)
                return p

            # 2. Check for legacy agent_data/agent folders (temporary compatibility)
            for sub in ["agent_data", "agent"]:
                candidate = Path.cwd() / pkg / sub
                if candidate.is_dir():
                    return candidate.resolve()

            # 3. Use importlib to find package origin
            import importlib.util
            spec = importlib.util.find_spec(pkg)
            if spec and spec.origin:
                origin_path = Path(spec.origin).resolve()
                # Package root is parent of __init__.py
                p = origin_path.parent
                WORKSPACE_DIR = str(p)
                return p
        except (OSError, ValueError):
            pass

    # 4. Fallback to CWD or parent-based discovery
    local_mcp = Path.cwd() / "mcp_config.json"
    if local_mcp.exists():
        return Path.cwd().resolve()

    return Path.cwd().resolve()


def validate_workspace_path(path: Path) -> Path:
    """Ensure a path is within the agent workspace to prevent traversal.

    Args:
        path: The path to validate.

    Returns:
        The resolved absolute Path.

    Raises:
        ValueError: If the path resolves outside the workspace directory.

    """
    ws = get_agent_workspace()
    resolved = path.resolve()
    try:
        resolved.relative_to(ws.resolve())
    except ValueError:
        logger.warning(f"Path traversal blocked: {path} (resolves to {resolved})")
        raise ValueError(f"Access denied: path is outside the workspace: {path}")
    return resolved


def get_workspace_path(filename: str) -> Path:
    """Construct an absolute path to a specific file within the agent workspace.

    Args:
        filename: The relative filename or sub-path within the workspace.

    Returns:
        The absolute Path to the file.

    """
    ws = get_agent_workspace()
    path = ws / filename
    # We don't call validate_workspace_path here because the file might not exist yet
    # but we can check if it WOULD resolve inside the workspace
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        # Resolve it and check
        resolved = path.resolve()
        try:
            resolved.relative_to(ws.resolve())
        except ValueError:
            raise ValueError(f"Invalid workspace filename: {filename}")
    return path


def resolve_mcp_config_path(mcp_config: str | None) -> Path | None:
    """Resolve the absolute path for an MCP configuration identifier.

    Checks absolute paths, workspace-relative paths, local package data,
    and CWD fallbacks to find a valid mcp_config.json.

    Args:
        mcp_config: The filename, relative path, or absolute path for the config.

    Returns:
        The absolute Path to the config if found, otherwise None.

    """
    from .base_utilities import retrieve_package_name

    if not mcp_config:
        return get_workspace_path(CORE_FILES["MCP_CONFIG"])

    path = Path(mcp_config)
    if path.is_absolute() and path.exists():
        return path

    # Check Workspace
    ws_config = get_workspace_path(mcp_config)
    if ws_config.exists():
        return ws_config

    # Check Local Package
    pkg = retrieve_package_name()
    if pkg and pkg != "agent_utilities":
        # Strategy A: importlib.resources (standard)
        try:
            from importlib.resources import files

            for sub in ["", "agent_data", "agent"]:
                p_bin = Path(str(files(pkg).joinpath(sub).joinpath(mcp_config)))
                if p_bin.exists():
                    return p_bin
        except (ImportError, ValueError, TypeError):
            pass

        # Strategy B: importlib.util.find_spec (robust fallback for development installs)
        try:
            import importlib.util

            spec = importlib.util.find_spec(pkg)
            if spec and spec.origin:
                pkg_root = Path(spec.origin).parent
                for sub in ["", "agent_data", "agent"]:
                    p_spec = pkg_root / sub / mcp_config
                    if p_spec.exists():
                        return p_spec
        except Exception:
            pass

        # Strategy C: CWD fallback (legacy)
        candidates = [
            Path.cwd() / pkg / "agent_data" / mcp_config,
            Path.cwd() / pkg / mcp_config,
            Path.cwd() / "agent_data" / mcp_config,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    # Check CWD directly
    local_config = Path.cwd() / mcp_config
    if local_config.exists():
        return local_config

    return None


def initialize_workspace(overwrite: bool = False):
    """Scaffold a fresh agent workspace with standard files and directories.
    Creates IDENTITY.md, USER.md, etc., using predefined
    templates if they do not already exist.

    Args:
        overwrite: Whether to overwrite existing files. Defaults to False.

    """
    load_env_vars()
    for key, fname in CORE_FILES.items():
        path = get_workspace_path(fname)
        if not path.exists() or overwrite:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            content = TEMPLATES.get(key, "# " + fname + "\n\n(empty)")
            if "{now}" in content:
                content = content.format(now=now_str)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content.strip() + "\n", encoding="utf-8")
            logger.debug(f"Initialized {path}")

    discovered = get_agent_workspace()
    internal_dirs = [
        str(Path(__file__).parent / "agent_data"),
        str(Path(__file__).parent / "agent"),
    ]
    try:
        with as_file(files("agent_utilities") / "agent_data") as p:
            internal_dirs.append(str(p.resolve()))
    except (OSError, ValueError):
        pass

    if str(discovered.resolve()) not in internal_dirs:
        global WORKSPACE_DIR
        WORKSPACE_DIR = str(discovered)
        logger.debug(f"Workspace cached: {WORKSPACE_DIR}")


def load_workspace_file(filename: str, default: str = "") -> str:
    """Read the full text content of a file from the agent's workspace.

    Args:
        filename: The filename or sub-path within the workspace.
        default: Fallback text to return if the file is missing.

    Returns:
        The file content string, or the default value.

    """
    path = get_workspace_path(filename)
    if path.exists():
        validate_workspace_path(path)
        return path.read_text(encoding="utf-8").strip()
    return default


def load_all_core_files() -> dict[str, str]:
    """Read all standard agent configuration and memory files into a map.

    Returns:
        A dictionary mapping core file identifiers to their raw text content.

    """
    return {k: load_workspace_file(v) for k, v in CORE_FILES.items()}


def write_workspace_file(filename: str, content: str):
    """Write or overwrite a file in the agent's workspace.

    Args:
        filename: The target filename.
        content: The text content to write.

    """
    path = get_workspace_path(filename)
    # Ensure the target directory is within the workspace
    if path.exists():
        validate_workspace_path(path)
    else:
        # Check parent
        validate_workspace_path(path.parent)

    path.write_text(content, encoding="utf-8")


def list_workspace_files() -> list[str]:
    """List all files present in the current agent workspace.

    Returns:
        A list of filenames.

    """
    workspace = get_agent_workspace()
    if not workspace.exists():
        return []
    return [f.name for f in workspace.iterdir() if f.is_file()]


def get_agent_icon_path() -> str | None:
    """Retrieve the absolute filesystem path to the agent's icon image.

    Returns:
        The absolute path string if the icon exists, otherwise None.

    """
    icon_path = get_workspace_path(CORE_FILES["ICON"])
    if icon_path.exists():
        return str(icon_path)
    return None


def read_md_file(filename: str) -> str:
    """Read a specific markdown file from the agent's workspace.

    Args:
        filename: The filename or sub-path (must end in .md).

    Returns:
        The text content of the file, or an error message if the file
        is missing or not a markdown file.

    """
    path = get_workspace_path(filename)
    if path.exists() and path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8")
    return f"File not found or not markdown: {filename}"


def write_md_file(filename: str, content: str):
    """Write or overwrite a markdown file in the workspace.

    Automatically handles parent directory creation.

    Args:
        filename: The target filename (must end in .md).
        content: The text content to write.

    Raises:
        ValueError: If the filename does not end in '.md'.

    """
    if not filename.lower().endswith(".md"):
        raise ValueError("Only .md files allowed")
    path = get_workspace_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def append_to_md_file(filename: str, text: str):
    """Append a block of text to an existing markdown file in the workspace.

    Args:
        filename: The target filename (must end in .md).
        text: The content to append.

    Raises:
        ValueError: If the filename does not end in '.md'.

    """
    if not filename.lower().endswith(".md"):
        raise ValueError("Only .md files allowed")
    path = get_workspace_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n" + text + "\n")


NEW_SKILL_TEMPLATE = """---
name: {name}
description: {description}
version: '0.1.0'
tags: {tags}
---
# {name}

{description}

### When to use
{when_to_use}

### How to use
{how_to_use}
"""


def create_new_skill(
    name: str,
    description: str,
    when_to_use: str = "",
    how_to_use: str = "",
    tags: str = "custom",
) -> str:
    """Scaffold a new agent skill within the workspace 'skills/' directory.

    Args:
        name: Human-readable name of the skill.
        description: A brief summary of what the skill does.
        when_to_use: Guidance on when the LLM should use this skill.
        how_to_use: Guidance on how to parameterize the skill calls.
        tags: Comma-separated or single tag string.

    Returns:
        A success message with the path to the newly created skill.

    """
    safe_name = re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "-"))
    skills_dir = get_agent_workspace() / "skills"
    skill_dir = skills_dir / safe_name
    skill_dir.mkdir(parents=True, exist_ok=True)

    content = NEW_SKILL_TEMPLATE.format(
        name=safe_name,
        description=description,
        when_to_use=when_to_use or "When the user needs this capability.",
        how_to_use=how_to_use or "Call the skill with appropriate parameters.",
        tags=tags,
    )
    (skill_dir / "SKILL.md").write_text(content.strip() + "\n", encoding="utf-8")
    return f"✅ Created new skill '{safe_name}' at {skill_dir}"


def delete_skill_from_disk(name: str) -> str:
    """Remove a skill directory and all its contents from the workspace.

    Args:
        name: The name or slug of the skill to delete.

    Returns:
        A status message indicating success or failure.

    """
    safe_name = re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "-"))
    skills_dir = get_agent_workspace() / "skills"
    skill_dir = skills_dir / safe_name

    if not skill_dir.exists():
        return f"❌ Skill '{safe_name}' not found at {skill_dir}"

    try:
        shutil.rmtree(skill_dir)
        return f"✅ Deleted skill '{safe_name}' and all its contents."
    except Exception as e:
        return f"❌ Error deleting skill '{safe_name}': {e}"


def read_skill_md(name: str) -> str:
    """Read the SKILL.md definition for a specific workspace skill.

    Args:
        name: The name or slug of the skill.

    Returns:
        The content of SKILL.md, or an error message if missing.

    """
    safe_name = re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "-"))
    skills_dir = get_agent_workspace() / "skills"
    skill_file = skills_dir / safe_name / "SKILL.md"

    if skill_file.exists():
        return skill_file.read_text(encoding="utf-8")
    return f"❌ SKILL.md not found for skill '{safe_name}'"


def write_skill_md(name: str, content: str) -> str:
    """Overwrite the SKILL.md definition for a specific workspace skill.

    Args:
        name: The name or slug of the skill.
        content: The new markdown content.

    Returns:
        A success message or an error message.

    """
    safe_name = re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "-"))
    skills_dir = get_agent_workspace() / "skills"
    skill_dir = skills_dir / safe_name
    skill_file = skill_dir / "SKILL.md"

    if not skill_dir.exists():
        return f"❌ Skill '{safe_name}' folder does not exist."

    try:
        skill_file.write_text(content.strip() + "\n", encoding="utf-8")
        return f"✅ Updated SKILL.md for skill '{safe_name}'."
    except Exception as e:
        return f"❌ Error writing SKILL.md for skill '{safe_name}': {e}"
