import os
import logging
import re
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from importlib.resources import files, as_file

from .base_utilities import retrieve_package_name, load_env_vars

logger = logging.getLogger(__name__)

                                         
WORKSPACE_DIR: Optional[str] = None

CORE_FILES = {
    "IDENTITY": "IDENTITY.md",
    "USER": "USER.md",
    "AGENTS": "A2A_AGENTS.md",
    "MEMORY": "MEMORY.md",
    "CRON": "CRON.md",
    "CRON_LOG": "CRON_LOG.md",
    "CHATS": "chats",
    "MCP_CONFIG": "mcp_config.json",
    "HEARTBEAT": "HEARTBEAT.md",
    "ICON": "icon.png",
}

TEMPLATES = {
    "IDENTITY": """# IDENTITY.md - Who I Am, Core Personality, & Boundaries

## [default]
 * **Name:** AI Agent
 * **Role:** A versatile AI agent capable of research, task delegation, and workspace management.
 * **Emoji:** 🤖
 * **Vibe:** Professional, efficient, helpful

 ### System Prompt
 You are a highly capable AI Agent.
 You have access to various tools and MCP servers to assist the user.
 Your responsibilities:
 1. Analyze the user's request.
 2. Use available tools and skills to gather information or perform actions.
 Synthesize findings into clear, well-structured responses.
""",
    "USER": """# USER.md - About the Human

* **Name:** User
* **Emoji:** 👤
""",
    "AGENTS": """# A2A_AGENTS.md - Known A2A Peer Agents

This file is the local registry of other A2A agents this agent can discover and call.

## Registered A2A Peers

| Name | Endpoint URL | Description | Capabilities | Auth | Notes / Last Connected |
|------|--------------|-------------|--------------|------|------------------------|
""",
    "MEMORY": """# MEMORY.md - Long-term Memory

This file stores important decisions, user preferences, and historical outcomes.

## Log of Important Events
- [{now}] Workspace initialized.
""",
    "CRON": """# CRON.md - Persistent Scheduled Tasks

## Active Tasks

| ID | Name | Interval (min) | Prompt | Last run | Next approx |
|----|------|----------------|--------|----------|-------------|
| log-cleanup | Log Cleanup | 720 | __internal:cleanup_cron_log | — | — |
""",
    "CRON_LOG": """# CRON_LOG.md - Scheduled Task History

| Timestamp | Task ID | Status | Message |
|-----------|---------|--------|---------|
""",
    "HEARTBEAT": """# Heartbeat — Periodic Self-Check

You are running a self-diagnostic heartbeat.
Please verify that:
1. Your core skills and MCP tools are responsive.
2. The user's recent instructions are still being followed.
3. Your long-term memory (MEMORY.md) is updated if necessary.

No specific user input is required unless you detect an issue.
""",
    "MCP_CONFIG": """{
    "mcpServers": {}
}
""",
}

def get_skills_path() -> Optional[str]:
    try:
        package_name = retrieve_package_name()
        for sub in ["agent_data/skills", "agent/skills", "skills"]:
            skills_dir = os.path.join(files(package_name), sub)
            if os.path.isdir(skills_dir):
                with as_file(Path(skills_dir)) as path:
                    return str(path)
        return None
    except Exception as e:
        logger.debug(f"Error accessing skills path: {e}")
        return None

def get_mcp_config_path() -> Optional[str]:
    try:
        package_name = retrieve_package_name()
        for sub in ["agent_data", "agent"]:
            mcp_config_file = os.path.join(files(package_name), sub, "mcp_config.json")
            if os.path.isfile(mcp_config_file):
                with as_file(Path(mcp_config_file)) as path:
                    return str(path)
        return None
    except Exception as e:
        logger.debug(f"Error accessing mcp_config path: {e}")
        return None

def get_agent_workspace() -> Path:
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
    if pkg and pkg != "agent_utilities":
        try:
            pkg_local_data = Path.cwd() / pkg / "agent_data"
            pkg_local_agent = Path.cwd() / pkg / "agent"
            for candidate in [pkg_local_data, pkg_local_agent]:
                if candidate.is_dir():
                    p = candidate.resolve()
                    WORKSPACE_DIR = str(p)
                    return p

            for sub in ["agent_data", "agent"]:
                try:
                    pkg_resource_dir = files(pkg) / sub
                    if pkg_resource_dir.is_dir():
                        with as_file(pkg_resource_dir) as path:
                            p = path.resolve()
                            WORKSPACE_DIR = str(p)
                            return p
                except Exception:
                    pass

            import importlib.util
            spec = importlib.util.find_spec(pkg)
            if spec and spec.origin:
                origin_path = Path(spec.origin).resolve()
                candidates = [
                    origin_path.parent / "agent_data",
                    origin_path.parent / "agent",
                    origin_path.parent.parent / "agent_data",
                    origin_path.parent.parent / "agent",
                ]
                for candidate in candidates:
                    if candidate.is_dir():
                        return candidate.resolve()
        except Exception:
            pass

    for sub in ["agent_data", "agent"]:
        local_dir = Path.cwd() / sub
        if local_dir.is_dir():
            p = local_dir.resolve()
            WORKSPACE_DIR = str(p)
            return p

    try:
        for entry in Path.cwd().iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                for sub in ["agent_data", "agent"]:
                    candidate = entry / sub
                    if candidate.is_dir():
                        p = candidate.resolve()
                        WORKSPACE_DIR = str(p)
                        return p
    except Exception:
        pass

    for sub in ["agent_data", "agent"]:
        native_path = Path(__file__).parent / sub
        if native_path.is_dir():
            return native_path.resolve()

    for sub in ["agent_data", "agent"]:
        workspace_dir = files("agent_utilities") / sub
        if workspace_dir.is_dir():
            with as_file(workspace_dir) as path:
                return path.resolve()

    return Path(__file__).parent / "agent_data"

def get_workspace_path(filename: str) -> Path:
    ws = get_agent_workspace()
    path = ws / filename
    return path

def initialize_workspace(overwrite: bool = False):
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
    except Exception:
        pass

    if str(discovered.resolve()) not in internal_dirs:
        global WORKSPACE_DIR
        WORKSPACE_DIR = str(discovered)
        logger.debug(f"Workspace cached: {WORKSPACE_DIR}")

def load_workspace_file(filename: str, default: str = "") -> str:
    path = get_workspace_path(filename)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default

def load_all_core_files() -> Dict[str, str]:
    return {k: load_workspace_file(v) for k, v in CORE_FILES.items()}

def write_workspace_file(filename: str, content: str):
    path = get_workspace_path(filename)
    path.write_text(content, encoding="utf-8")

def list_workspace_files() -> List[str]:
    workspace = get_agent_workspace()
    if not workspace.exists():
        return []
    return [f.name for f in workspace.iterdir() if f.is_file()]

def get_agent_icon_path() -> Optional[str]:
    icon_path = get_workspace_path(CORE_FILES["ICON"])
    if icon_path.exists():
        return str(icon_path)
    return None

def read_md_file(filename: str) -> str:
    """Read any md file in workspace."""
    path = get_workspace_path(filename)
    if path.exists() and path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8")
    return f"File not found or not markdown: {filename}"

def write_md_file(filename: str, content: str):
    """Overwrite markdown file."""
    if not filename.lower().endswith(".md"):
        raise ValueError("Only .md files allowed")
    path = get_workspace_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def append_to_md_file(filename: str, text: str):
    """Append text to a markdown file."""
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
    """Helper to scaffold a new skill."""
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
    """Delete a skill folder from workspace."""
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
    """Read the SKILL.md content of a workspace skill."""
    safe_name = re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "-"))
    skills_dir = get_agent_workspace() / "skills"
    skill_file = skills_dir / safe_name / "SKILL.md"

    if skill_file.exists():
        return skill_file.read_text(encoding="utf-8")
    return f"❌ SKILL.md not found for skill '{safe_name}'"


def write_skill_md(name: str, content: str) -> str:
    """Overwrite the SKILL.md content of a workspace skill."""
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
