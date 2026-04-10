import os
import logging
import re
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
from importlib.resources import files, as_file

from .base_utilities import retrieve_package_name, load_env_vars
from .models import (
    IdentityModel,
    UserModel,
    A2ARegistryModel,
    A2APeerModel,
    MemoryModel,
    MemoryEntryModel,
    CronRegistryModel,
    CronTaskModel,
    CronLogModel,
    CronLogEntryModel,
    MCPAgentRegistryModel,
    MCPAgent,
    MCPToolInfo,
)

logger = logging.getLogger(__name__)


WORKSPACE_DIR: Optional[str] = None


def md_table_escape(text: str) -> str:
    """Escape markdown table delimiters and handle newlines."""
    if not text:
        return ""
    # Replace pipe character which breaks markdown tables
    escaped = text.replace("|", "\\|")
    # Replace newlines with <br/> to keep the cell on one row
    escaped = escaped.replace("\n", "<br/>")
    return escaped.strip()


CORE_FILES = {
    "IDENTITY": "IDENTITY.md",
    "USER": "USER.md",
    "A2A_AGENTS": "A2A_AGENTS.md",
    "MEMORY": "MEMORY.md",
    "CRON": "CRON.md",
    "CRON_LOG": "CRON_LOG.md",
    "CHATS": "chats",
    "MCP_CONFIG": "mcp_config.json",
    "NODE_AGENTS": "NODE_AGENTS.md",
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
    "A2A_AGENTS": """# A2A_AGENTS.md - Known A2A Peer Agents

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
    "NODE_AGENTS": """# NODE_AGENTS.md - Dynamic Agent Registry

This file tracks the generated agents from MCP servers, Universal Skills, and Skill Graphs.

## Agent Mapping Table

| Name | Description | System Prompt | Tools | Tag / ID | Source MCP / Skill |
|------|-------------|---------------|-------|----------|--------------------|

## Tool Inventory Table

| Tool Name | Description | Tag | Source |
|-----------|-------------|-----|--------|
""",
}


def get_skills_path() -> List[str]:
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
                    skills_dir = pkg_root / sub
                    if skills_dir.exists():
                        return [str(skills_dir)]
        except Exception as e:
            logger.debug(f"Manual path fallback fail: {e}")

    except Exception as e:
        logger.debug(f"Error accessing skills path: {e}")
    return []


def get_mcp_config_path() -> Optional[str]:
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
                    pkg_resource_dir = files(pkg).joinpath(sub)
                    if pkg_resource_dir.is_dir():
                        with as_file(pkg_resource_dir) as path:
                            p = path.resolve()
                            WORKSPACE_DIR = str(p)
                            return p
                except (OSError, ValueError):
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
        except (OSError, ValueError):
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
    except (OSError, ValueError):
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


def resolve_mcp_config_path(mcp_config: str) -> Optional[Path]:
    """
    Resolves the absolute path for an MCP config string.
    Checks:
    1. Absolute path
    2. Relative to workspace
    3. Relative to local package's agent_data
    4. Relative to CWD
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

            for sub in ["agent_data", "agent", ""]:
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
                for sub in ["agent_data", "agent", ""]:
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


def parse_identity(content: str) -> IdentityModel:
    """Parse IDENTITY.md into IdentityModel."""
    model = IdentityModel()
    name_match = re.search(r"\*\s+\*\*Name:\*\*\s*(.*)", content)
    if name_match:
        model.name = name_match.group(1).strip()
    role_match = re.search(r"\*\s+\*\*Role:\*\*\s*(.*)", content)
    if role_match:
        model.role = role_match.group(1).strip()
    emoji_match = re.search(r"\*\s+\*\*Emoji:\*\*\s*(.*)", content)
    if emoji_match:
        model.emoji = emoji_match.group(1).strip()
    vibe_match = re.search(r"\*\s+\*\*Vibe:\*\*\s*(.*)", content)
    if vibe_match:
        model.vibe = vibe_match.group(1).strip()

    prompt_split = re.split(r"### System Prompt", content, flags=re.IGNORECASE)
    if len(prompt_split) > 1:
        model.system_prompt = prompt_split[1].strip()
    return model


def serialize_identity(model: IdentityModel) -> str:
    """Serialize IdentityModel back to IDENTITY.md format."""
    return f"""# IDENTITY.md - Who I Am, Core Personality, & Boundaries

## [default]
 * **Name:** {model.name}
 * **Role:** {model.role}
 * **Emoji:** {model.emoji}
 * **Vibe:** {model.vibe}

 ### System Prompt
 {model.system_prompt}
"""


def parse_user_info(content: str) -> UserModel:
    """Parse USER.md into UserModel."""
    model = UserModel()
    name_match = re.search(r"\*\s+\*\*Name:\*\*\s*(.*)", content)
    if name_match:
        model.name = name_match.group(1).strip()
    emoji_match = re.search(r"\*\s+\*\*Emoji:\*\*\s*(.*)", content)
    if emoji_match:
        model.emoji = emoji_match.group(1).strip()
    return model


def serialize_user_info(model: UserModel) -> str:
    """Serialize UserModel back to USER.md format."""
    return f"""# USER.md - About the Human

* **Name:** {model.name}
* **Emoji:** {model.emoji}
"""


def parse_a2a_registry(content: str) -> A2ARegistryModel:
    """Parse A2A_AGENTS.md table into A2ARegistryModel."""
    peers = []
    lines = content.splitlines()
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| Name") or stripped.startswith("| ID"):
            in_table = True
            continue
        if (
            in_table
            and stripped.startswith("|")
            and "|" in stripped
            and not (
                stripped.startswith("|---")
                or stripped.startswith("| ID")
                or stripped.startswith("| Name")
            )
        ):
            parts = [p.strip() for p in stripped.strip("| ").split("|")]
            if len(parts) >= 2:
                peers.append(
                    A2APeerModel(
                        name=parts[0],
                        url=parts[1],
                        description=parts[2] if len(parts) > 2 else "",
                        capabilities=parts[3] if len(parts) > 3 else "",
                        auth=parts[4] if len(parts) > 4 else "none",
                        notes=parts[5] if len(parts) > 5 else "",
                    )
                )
    return A2ARegistryModel(peers=peers)


def serialize_a2a_registry(model: A2ARegistryModel) -> str:
    """Serialize A2ARegistryModel back to A2A_AGENTS.md format."""
    lines = [
        "# A2A_AGENTS.md - Known A2A Peer Agents",
        "",
        "This file is the local registry of other A2A agents this agent can discover and call.",
        "",
        "## Registered A2A Peers",
        "",
        "| Name | Endpoint URL | Description | Capabilities | Auth | Notes / Last Connected |",
        "|------|--------------|-------------|--------------|------|------------------------|",
    ]
    for p in model.peers:
        lines.append(
            f"| {p.name} | {p.url} | {p.description or ''} | {p.capabilities or ''} | {p.auth or 'none'} | {p.notes or ''} |"
        )
    return "\n".join(lines).strip() + "\n"


def parse_memory(content: str) -> MemoryModel:
    """Parse MEMORY.md into MemoryModel."""
    entries = []
    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- [") or stripped.startswith("* ["):
            match = re.match(r"[-*]\s+\[(.*?)\]\s*(.*)", stripped)
            if match:
                entries.append(
                    MemoryEntryModel(timestamp=match.group(1), text=match.group(2))
                )
    return MemoryModel(entries=entries)


def serialize_memory(model: MemoryModel) -> str:
    """Serialize MemoryModel back to MEMORY.md format."""
    lines = [
        "# MEMORY.md - Long-term Memory",
        "",
        "This file stores important decisions, user preferences, and historical outcomes.",
        "",
        "## Log of Important Events",
    ]
    for e in model.entries:
        lines.append(f"- [{e.timestamp}] {e.text}")
    return "\n".join(lines).strip() + "\n"


def parse_cron_registry(content: str) -> CronRegistryModel:
    """Parse CRON.md table into CronRegistryModel."""
    tasks = []
    lines = content.splitlines()
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| ID") or stripped.startswith("| ID |"):
            in_table = True
            continue
        if in_table and stripped.startswith("|") and not stripped.startswith("|---"):
            parts = [p.strip() for p in stripped.strip("| ").split("|")]
            if len(parts) >= 4:
                tasks.append(
                    CronTaskModel(
                        id=parts[0],
                        name=parts[1],
                        interval_minutes=int(parts[2]) if parts[2].isdigit() else 0,
                        prompt=parts[3],
                        last_run=parts[4] if len(parts) > 4 else "—",
                        next_approx=parts[5] if len(parts) > 5 else "—",
                    )
                )
    return CronRegistryModel(tasks=tasks)


def serialize_cron_registry(model: CronRegistryModel) -> str:
    """Serialize CronRegistryModel back to CRON.md format."""
    lines = [
        "# CRON.md - Persistent Scheduled Tasks",
        "",
        "## Active Tasks",
        "",
        "| ID | Name | Interval (min) | Prompt | Last run | Next approx |",
        "|----|------|----------------|--------|----------|-------------|",
    ]
    for t in model.tasks:
        lines.append(
            f"| {t.id} | {t.name} | {t.interval_minutes} | {t.prompt} | {t.last_run} | {t.next_approx} |"
        )
    return "\n".join(lines).strip() + "\n"


def parse_cron_log(content: str) -> CronLogModel:
    """Parse CRON_LOG.md into CronLogModel."""
    entries = []
    import re

    parts = re.split(r"(?=^### \[)", content, flags=re.MULTILINE)
    for part in parts:
        if not part.strip() or not part.startswith("### ["):
            continue
        header_match = re.search(
            r"^### \[(.*?)\] (.*?) \(`(.*?)`\)(?: \| \[View Chat\]\((.*?)\))?", part
        )
        if header_match:
            ts = header_match.group(1)
            name = header_match.group(2)
            tid = header_match.group(3)
            cid = (
                header_match.group(4).lstrip("/")
                if header_match.lastindex is not None
                and header_match.lastindex >= 4
                and header_match.group(4)
                else None
            )

            body = part.split("\n\n", 1)[1] if "\n\n" in part else ""
            msg = body.split("\n---")[0].strip()

            entries.append(
                CronLogEntryModel(
                    timestamp=ts,
                    task_id=tid,
                    task_name=name,
                    message=msg,
                    chat_id=cid,
                )
            )
    return CronLogModel(entries=entries)


def serialize_cron_log(model: CronLogModel) -> str:
    """Serialize CronLogModel back to CRON_LOG.md format."""
    lines = ["# CRON_LOG.md - Scheduled Task History", ""]
    for e in model.entries:
        chat_info = f" | [View Chat](/{e.chat_id})" if e.chat_id else ""
        lines.append(f"### [{e.timestamp}] {e.task_name} (`{e.task_id}`){chat_info}")
        lines.append("")
        lines.append(e.message)
        lines.append("")
        lines.append("---")
    return "\n".join(lines).strip() + "\n"


def parse_node_registry(content: str) -> MCPAgentRegistryModel:
    """Parse NODE_AGENTS.md tables into MCPAgentRegistryModel."""
    agents = []
    tools = []
    lines = content.splitlines()

    current_table = None  # 1 for Agents, 2 for Tools

    for line in lines:
        stripped = line.strip()
        if "## Agent Mapping Table" in stripped:
            current_table = 1
            continue
        elif "## Tool Inventory Table" in stripped:
            current_table = 2
            continue

        if (
            stripped.startswith("|")
            and not stripped.startswith("|---")
            and not stripped.startswith("| Name")
            and not stripped.startswith("| Tool Name")
        ):
            parts = [p.strip() for p in stripped.strip("| ").split("|")]
            if current_table == 1 and len(parts) >= 6:
                agents.append(
                    MCPAgent(
                        name=parts[0],
                        description=parts[1],
                        system_prompt=parts[2],
                        tools=[t.strip() for t in parts[3].split(",") if t.strip()],
                        tag=parts[4] if parts[4] else None,
                        mcp_server=parts[5],
                        is_custom=False,
                    )
                )
            elif current_table == 2 and len(parts) >= 4:
                tools.append(
                    MCPToolInfo(
                        name=parts[0],
                        description=parts[1],
                        tag=parts[2] if parts[2] else None,
                        mcp_server=parts[3],
                    )
                )

    return MCPAgentRegistryModel(agents=agents, tools=tools)


def serialize_node_registry(model: MCPAgentRegistryModel) -> str:
    """Serialize MCPAgentRegistryModel back to NODE_AGENTS.md format."""
    lines = [
        "# NODE_AGENTS.md - Dynamic Agent Registry",
        "",
        "This file tracks the generated agents from MCP servers, Universal Skills, and Skill Graphs.",
        "",
        "## Agent Mapping Table",
        "",
        "| Name | Description | System Prompt | Tools | Tag / ID | Source MCP / Skill |",
        "|------|-------------|---------------|-------|----------|--------------------|",
    ]
    for a in model.agents:
        tools_str = ", ".join(a.tools)
        lines.append(
            f"| {md_table_escape(a.name)} | {md_table_escape(a.description)} | {md_table_escape(a.system_prompt)} | {md_table_escape(tools_str)} | {md_table_escape(a.tag or '')} | {md_table_escape(a.mcp_server)} |"
        )

    lines.extend(
        [
            "",
            "## Tool Inventory Table",
            "",
            "| Tool Name | Description | Tag | Source |",
            "|-----------|-------------|-----|--------|",
        ]
    )
    for t in model.tools:
        # Show all tags in the Tag column
        tags_display = ", ".join(t.all_tags) if t.all_tags else (t.tag or "")
        lines.append(
            f"| {md_table_escape(t.name)} | {md_table_escape(t.description)} | {md_table_escape(tags_display)} | {md_table_escape(t.mcp_server)} |"
        )

    return "\n".join(lines).strip() + "\n"
