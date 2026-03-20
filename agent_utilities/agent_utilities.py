#!/usr/bin/python
# coding: utf-8
from __future__ import annotations

import os
import re
import shutil
import json
import logging
import asyncio
import yaml
import httpx
import argparse
import base64
import contextvars

# import uvicorn  # Optional
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from fasta2a import Skill
    from fastapi import FastAPI
from pathlib import Path
from contextlib import asynccontextmanager
from importlib.resources import files, as_file

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response, StreamingResponse
from pydantic import ValidationError

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.mcp import (
    load_mcp_servers,
    MCPServerStreamableHTTP,
    MCPServerSSE,
)

from universal_skills.skill_utilities import resolve_mcp_reference


from .base_utilities import (
    to_boolean,
    to_integer,
    to_float,
    to_list,
    to_dict,
    retrieve_package_name,
    GET_DEFAULT_SSL_VERIFY,
    load_env_vars,
)

# from .tools import register_agent_tools  # Breaks circular import

from .models import PeriodicTask

# Global state for periodic tasks
tasks: List[PeriodicTask] = []
lock = asyncio.Lock()


elicitation_queue_var: contextvars.ContextVar[Optional[asyncio.Queue]] = (
    contextvars.ContextVar("elicitation_queue", default=None)
)


class ElicitationManager:
    """Manages pending elicitation requests and their resolutions."""

    def __init__(self):
        self.pending_requests: Dict[str, asyncio.Future] = {}

    async def wait_for_resolution(self, request_id: str) -> Any:
        future = asyncio.get_running_loop().create_future()
        self.pending_requests[request_id] = future
        try:
            return await future
        finally:
            self.pending_requests.pop(request_id, None)

    async def resolve_request(self, request_id: str, result: Any) -> bool:
        future = self.pending_requests.get(request_id)
        if future and not future.done():
            future.set_result(result)
            return True
        return False


elicitation_manager = ElicitationManager()


async def global_elicitation_callback(
    context: Optional["RequestContext"] = None, params: Any = None
) -> Any:
    """
    Standardized elicitation callback for MCP servers.
    Sends the request to the UI via elicitation_queue_var and waits for resolution.
    """
    queue = elicitation_queue_var.get()

    # If not in contextvars, try to find it via context.session
    if not queue and context and hasattr(context, "session"):
        # This is a bit of a hack to find the MCPServer from the session
        # In pydantic-ai, the MCPServer instance usually has the callback set
        # We'll try to get it from the callback's closure if we can,
        # but a better way is to check if we can find it in the current task.
        pass

    if not queue:
        logger.warning("No elicitation queue found in context. Blocking request.")
        if mcp_types and params is not None:
            return mcp_types.ErrorData(
                code=mcp_types.INVALID_REQUEST, message="No elicitation queue"
            )
        return {"status": "error", "message": "No elicitation queue"}

    # Handle both direct dict calls and MCP protocol params
    if params is not None:
        if hasattr(params, "model_dump"):
            request_data = params.model_dump()
        else:
            request_data = params
    elif isinstance(context, dict):
        request_data = context
    else:
        logger.error(
            f"Invalid elicitation callback arguments: context={type(context)}, params={type(params)}"
        )
        return {"status": "error", "message": "Invalid arguments"}

    request_id = request_data.get("id")
    if not request_id:
        import uuid

        request_id = str(uuid.uuid4())
        request_data["id"] = request_id

    logger.info(f"Triggering elicitation: {request_id}")
    await queue.put({"type": "elicitation", **request_data})

    # Wait for the UI to resolve the request via /api/elicit
    result = await elicitation_manager.wait_for_resolution(request_id)

    # If called by MCP session, return ElicitResult
    if mcp_types and params is not None:
        try:
            return mcp_types.ElicitResult(**result)
        except Exception as e:
            logger.error(f"Error creating ElicitResult: {e}")
            return result

    return result


SENSITIVE_TOOL_PATTERNS = [
    # Destructive Ops
    r".*delete.*",
    r".*remove.*",
    r".*rm_.*",
    r".*rmdir.*",
    r".*drop.*",
    r".*truncate.*",
    r".*prune.*",
    # Process/System Interrupts
    r".*kill.*",
    r".*terminate.*",
    r".*reboot.*",
    r".*shutdown.*",
    # Deployment & Versioning
    r".*install.*",
    r".*uninstall.*",
    r".*redeploy.*",
    r".*bump.*",
    # Resource Creation
    r".*create.*",
    r".*add.*",
    r".*post.*",
    r".*put.*",
    r".*insert.*",
    r".*upload.*",
    r".*ingest.*",
    r".*write.*",
    # State Modifications
    r".*update.*",
    r".*patch.*",
    r".*set.*",
    r".*reset.*",
    r".*clear.*",
    r".*revert.*",
    r".*replace.*",
    r".*rename.*",
    r".*move.*",
    r".*rotate.*",
    # Lifecycle Management
    r".*start.*",
    r".*stop.*",
    r".*restart.*",
    r".*pause.*",
    r".*unpause.*",
    # Command/Script Execution
    r".*execute.*",
    r".*shell.*",
    r".*run_.*",
    r".*git_.*",
    # Access & Features
    r".*enable.*",
    r".*disable.*",
    r".*activate.*",
    r".*approve.*",
    # Direct Data Protocols
    r".*graphql.*",
    r".*mutation.*",
]


async def global_tool_guard(ctx, call_tool, name, tool_args):
    """
    Global tool guard that intercepts sensitive tool calls and requires elicitation.
    Standardized to handle elicitation through the client-side MCP lifecycle.
    """
    import re

    try:
        # Check if guard is disabled
        if to_boolean(os.getenv("DISABLE_TOOL_GUARD", "False")):
            return await call_tool(name, tool_args)

        # Check if tool name matches any sensitive pattern
        is_sensitive = any(
            re.match(pattern, name.lower()) for pattern in SENSITIVE_TOOL_PATTERNS
        )

        if is_sensitive:
            logger.info(f"Universal Tool Guard: Intercepted sensitive tool '{name}'")
            # In the new architecture, the MCP client handles this if the tool
            # is called through the MCP protocol. For native tools being guarded,
            # we might still need a way to trigger elicitation, but for now
            # we block/proceed based on the guard.
            #
            # If this is a native tool call that isn't wrapped in MCP,
            # server-side elicitation is NO LONGER SUPPORTED.
            pass

        # Proceed with the original tool call
        return await call_tool(name, tool_args)
    except Exception as e:
        logger.error(f"Universal Tool Guard error: {e}")
        return await call_tool(name, tool_args)


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

NEW_SKILL_TEMPLATE = """---
name: {skill_name}
description: {skill_description}
version: '0.1.0'
---
# {skill_name}

{skill_description}
"""

import contextvars

import warnings

# Suppress RequestsDependencyWarning due to chardet 6.x / requests 2.32.x mismatch
# We use a message-based filter to avoid importing from requests, which triggers the warning
warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")

try:
    from pydantic_ai.models.openai import OpenAIChatModel
    from openai import AsyncOpenAI
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:
    print("Unable to import OpenAI Chat Model / OpenAI Provider from agent-utilities")
    OpenAIModel = None
    AsyncOpenAI = None
    OpenAIProvider = None

try:
    import logfire

    HAS_LOGFIRE = True
except ImportError:
    HAS_LOGFIRE = False

try:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.mistral import GoogleProvider
except ImportError:
    GoogleModel = None
    GoogleProvider = None

try:
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider
except ImportError:
    HuggingFaceModel = None
    HuggingFaceProvider = None

try:
    from pydantic_ai.models.groq import GroqModel
    from groq import AsyncGroq
    from pydantic_ai.providers.groq import GroqProvider
except ImportError:
    GroqModel = None
    AsyncGroq = None
    GroqProvider = None

try:
    from mcp import types as mcp_types
    from mcp.shared.context import RequestContext
    from mcp.client.session import ClientSession
except ImportError:
    mcp_types = None
    RequestContext = None
    ClientSession = None
    AsyncGroq = None
    GroqProvider = None

try:
    from pydantic_ai.models.mistral import MistralModel
    from mistralai import Mistral
    from pydantic_ai.providers.mistral import MistralProvider
except ImportError:
    MistralModel = None
    Mistral = None
    MistralProvider = None

try:
    from pydantic_ai.models.anthropic import AnthropicModel
    from anthropic import AsyncAnthropic
    from pydantic_ai.providers.anthropic import AnthropicProvider
except ImportError:
    AnthropicModel = None
    AsyncAnthropic = None
    AnthropicProvider = None

logger = logging.getLogger(__name__)
__version__ = "0.2.31"

# Load environment variables early
load_env_vars()

# Global override for workspace directory
WORKSPACE_DIR: Optional[str] = None


def get_skills_path() -> Optional[str]:
    try:
        package_name = retrieve_package_name()
        # Check agent_data/skills first, then skills/
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
    """Find mcp_config.json in the calling package's resources."""
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


def _parse_skill_from_md(skill_file: Path, skill_id: str) -> Optional[Skill]:
    from fasta2a import Skill
    import yaml
    import re

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

                # Support version in top-level or metadata
                skill_version = str(
                    data.get(
                        "version", data.get("metadata", {}).get("version", "0.1.0")
                    )
                )

                # Extract tags from frontmatter if available
                skill_tags = data.get("tags", [skill_id])
                if not isinstance(skill_tags, list):
                    skill_tags = [str(skill_tags)]

                return Skill(
                    id=skill_id,
                    name=skill_name,
                    description=skill_desc,
                    version=skill_version,
                    tags=skill_tags,
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

    # 1. Check if the directory itself is a skill
    skill_file = base_path / "SKILL.md"
    if skill_file.exists():
        skill = _parse_skill_from_md(skill_file, base_path.name)
        if skill:
            skills.append(skill)
            return skills

    # 2. Check subdirectories
    if base_path.is_dir():
        for item in base_path.iterdir():
            if item.is_dir():
                sub_skill_file = item / "SKILL.md"
                if sub_skill_file.exists():
                    skill = _parse_skill_from_md(sub_skill_file, item.name)
                    if skill:
                        skills.append(skill)
    return skills


def extract_section_from_md(content: str, header: str) -> Optional[str]:
    """
    Extracts content under a specific markdown header (e.g., 'System Prompt').
    Matches headers like '## System Prompt' or '### System Prompt'.
    Returns None if the header is not found.
    """
    import re

    # Escape header for regex
    escaped_header = re.escape(header)
    # Match any level header (# to ######) followed by the header text
    pattern = rf"^\s*#+\s*{escaped_header}\s*\n(.*?)(?=\n#|\Z)"
    match = re.search(pattern, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def get_system_prompt_from_reference(agent_template: str) -> Optional[str]:
    """
    Retrieves the system prompt for a template from its markdown reference.
    """

    # Prioritize specialized identity file
    identity_query = f"{agent_template}-identity.md"
    md_path = resolve_mcp_reference(identity_query)

    if md_path and os.path.exists(md_path):
        # Read the entire file as the system prompt
        return Path(md_path).read_text(encoding="utf-8")

    # Fallback to searching and parsing standard reference files
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
        # Try 'System Prompt' section
        return extract_section_from_md(content, "System Prompt")

    return None


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
    skill_tags = extract_skill_tags(skill_path)
    return tag in skill_tags


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


def get_http_client(
    ssl_verify: bool = True, timeout: float = 300.0
) -> httpx.AsyncClient | None:
    if not ssl_verify:
        return httpx.AsyncClient(verify=False, timeout=timeout)
    return None


def get_agent_workspace() -> Path:
    """
    Dynamically discover the agent workspace path.
    Tiers:
    1. Global WORKSPACE_DIR override (from CLI)
    2. AGENT_WORKSPACE environment variable
    3. Calling package's /agent_data or /agent directory (via retrieve_package_name)
    4. Current directory's /agent_data or /agent folder
    5. Internal agent-utilities/agent_data resource (fallback)
    """
    global WORKSPACE_DIR
    # 1. Explicit global override
    if WORKSPACE_DIR:
        p = Path(WORKSPACE_DIR).resolve()
        logger.debug(f"get_agent_workspace: Tier 1 SUCCESS (Override): {p}")
        return p

    # 2. Environment variable
    env_workspace = os.getenv("AGENT_WORKSPACE")
    if env_workspace:
        p = Path(env_workspace).resolve()
        logger.debug(f"get_agent_workspace: Tier 2 Checking (Env AGENT_WORKSPACE): {p}")
        # Only cache if it exists
        if p.exists():
            logger.debug(f"get_agent_workspace: Tier 2 SUCCESS (Env exists): {p}")
            WORKSPACE_DIR = str(p)
        else:
            logger.warning(f"get_agent_workspace: Tier 2 path does NOT exist: {p}")
        return p

    # 3. Discovery via caller package
    pkg = retrieve_package_name()
    if pkg and pkg != "agent_utilities":
        try:
            # A. Check for local dev structure: Path.cwd() / pkg / "agent_data" or "agent"
            # This is critical for users running from source without -e install
            pkg_local_data = Path.cwd() / pkg / "agent_data"
            pkg_local_agent = Path.cwd() / pkg / "agent"

            logger.debug(
                f"get_agent_workspace: Tier 3A Checking Local Dev: {pkg_local_data}, {pkg_local_agent}"
            )
            for candidate in [pkg_local_data, pkg_local_agent]:
                if candidate.is_dir():
                    p = candidate.resolve()
                    logger.debug(
                        f"get_agent_workspace: Tier 3A SUCCESS (Local Package Dev {pkg}): {p}"
                    )
                    WORKSPACE_DIR = str(p)
                    return p

            # B. Try built-in resources (for installed packages)
            logger.debug(
                f"get_agent_workspace: Tier 3B Checking Package Resources: {pkg}"
            )
            for sub in ["agent_data", "agent"]:
                try:
                    pkg_resource_dir = files(pkg) / sub
                    if pkg_resource_dir.is_dir():
                        with as_file(pkg_resource_dir) as path:
                            p = path.resolve()
                            logger.debug(
                                f"get_agent_workspace: Tier 3B SUCCESS (Package Resource {pkg} - {sub}): {p}"
                            )
                            WORKSPACE_DIR = str(p)
                            return p
                except Exception as e:
                    logger.debug(
                        f"get_agent_workspace: Tier 3B check failed for {pkg}/{sub}: {e}"
                    )

            # C. Alternative: find spec and check parent
            import importlib.util

            spec = importlib.util.find_spec(pkg)
            if spec and spec.origin:
                origin_path = Path(spec.origin).resolve()
                # Check if it's package/sub/file.py or package/file.py
                candidates = [
                    origin_path.parent / "agent_data",
                    origin_path.parent / "agent",
                    origin_path.parent.parent / "agent_data",
                    origin_path.parent.parent / "agent",
                ]
                for candidate in candidates:
                    if candidate.is_dir():
                        logger.debug(
                            f"Discovery Tier 3C (Package Spec {pkg}): {candidate}"
                        )
                        return candidate.resolve()
        except (ImportError, ModuleNotFoundError, Exception) as e:
            logger.debug(f"Discovery Tier 3 error for {pkg}: {e}")

    # 4. Local fallback (Current directory / agent_data or agent)
    for sub in ["agent_data", "agent"]:
        local_dir = Path.cwd() / sub
        if local_dir.is_dir():
            p = local_dir.resolve()
            logger.debug(f"get_agent_workspace: Tier 4 SUCCESS ({sub}): {p}")
            WORKSPACE_DIR = str(p)
            return p

    # 4.5. Deep Local Search (Search subdirectories of CWD for agent_data or agent)
    # This helps when running from the root of a project with multiple packages
    logger.debug("get_agent_workspace: Tier 4.5 Checking Subdirectories of CWD...")
    try:
        for entry in Path.cwd().iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                for sub in ["agent_data", "agent"]:
                    candidate = entry / sub
                    if candidate.is_dir():
                        p = candidate.resolve()
                        logger.debug(
                            f"get_agent_workspace: Tier 4.5 SUCCESS (Found nested {sub} in {entry.name}): {p}"
                        )
                        WORKSPACE_DIR = str(p)
                        return p
    except Exception as e:
        logger.debug(f"Tier 4.5 check failed: {e}")

    # 5. Native fallback
    for sub in ["agent_data", "agent"]:
        native_path = Path(__file__).parent / sub
        if native_path.is_dir():
            p = native_path.resolve()
            logger.debug(f"Discovery Tier 5 (Native {sub}): {p}")
            return p

    # Discovery Tier 6: Fallback Resource
    for sub in ["agent_data", "agent"]:
        workspace_dir = files("agent_utilities") / sub
        if workspace_dir.is_dir():
            with as_file(workspace_dir) as path:
                p = path.resolve()
                logger.debug(f"Discovery Tier 6 (Fallback Resource {sub}): {p}")
                # CAUTION: Do NOT set WORKSPACE_DIR global here to avoid poisoning
                return p

    # Absolute last resort
    return Path(__file__).parent / "agent_data"


def get_workspace_path(filename: str) -> Path:
    """Return full path for a file in discovered workspace."""
    ws = get_agent_workspace()
    path = ws / filename
    logger.debug(f"get_workspace_path: Resolved {filename} -> {path}")
    return path


def initialize_workspace(overwrite: bool = False):
    """Create missing files with templates."""
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

    # Set global workspace dir after initialization to ensure discovery is cached.
    # We only cache if we found a non-fallback workspace.
    discovered = get_agent_workspace()
    # Check if discovered workspace is one of our internal fallbacks
    internal_dirs = [
        str(Path(__file__).parent / "agent_data"),
        str(Path(__file__).parent / "agent"),
    ]
    try:
        from importlib.resources import files, as_file

        with as_file(files("agent_utilities") / "agent_data") as p:
            internal_dirs.append(str(p.resolve()))
    except Exception:
        pass

    if str(discovered.resolve()) not in internal_dirs:
        global WORKSPACE_DIR
        WORKSPACE_DIR = str(discovered)
        logger.debug(f"Workspace cached: {WORKSPACE_DIR}")


def load_workspace_file(filename: str, default: str = "") -> str:
    """Read markdown file content. Returns default if missing."""
    path = get_workspace_path(filename)
    logger.debug(f"Final resolution for {filename}: {path}")
    if path.exists():
        logger.debug(f"Loading workspace file: {path}")
        return path.read_text(encoding="utf-8").strip()
    logger.warning(f"Workspace file not found: {path} (using default)")
    return default


def load_all_core_files() -> Dict[str, str]:
    """Load all core markdown files into a dict."""
    return {k: load_workspace_file(v) for k, v in CORE_FILES.items()}


def write_workspace_file(filename: str, content: str):
    """Write content to a file in the workspace."""
    path = get_workspace_path(filename)
    path.write_text(content, encoding="utf-8")
    logger.debug(f"Updated {filename}")


def list_workspace_files() -> List[str]:
    """List all files in the agent workspace."""
    workspace = get_agent_workspace()
    if not workspace.exists():
        return []
    return [f.name for f in workspace.iterdir() if f.is_file()]


def get_agent_icon_path() -> Optional[str]:
    """Get the absolute path to the agent icon if it exists."""
    icon_path = get_workspace_path(CORE_FILES["ICON"])
    if icon_path.exists():
        return str(icon_path)
    return None


def get_cron_tasks_from_md() -> List[Dict[str, Any]]:
    """Parse CRON.md and return active tasks."""
    content = load_workspace_file(CORE_FILES["CRON"])
    tasks = []
    # Simple markdown table parser
    lines = content.split("\n")
    for line in lines:
        if "|" in line and "ID" not in line and "---" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 3:
                tasks.append(
                    {
                        "id": parts[0],
                        "name": parts[1],
                        "schedule": parts[2],  # Interval in minutes or cron string
                    }
                )
    return tasks


def get_cron_logs_from_md() -> List[Dict[str, Any]]:
    """Parse CRON_LOG.md and return recent history."""
    content = load_workspace_file(CORE_FILES["CRON_LOG"])
    logs = []
    # Each entry starts with "### ["
    parts = re.split(r"(?=^### \[)", content, flags=re.MULTILINE)

    for part in parts:
        if not part.strip() or not part.startswith("### ["):
            continue

        try:
            # Format: ### [2026-03-07 05:32:11] Log Cleanup (`log-cleanup`) | [View Chat](/chat_id)
            header_match = re.search(
                r"^### \[(.*?)\] (.*?) \(`(.*?)`\)(?: \| \[View Chat\]\((.*?)\))?", part
            )
            if header_match:
                ts = header_match.group(1)
                name = header_match.group(2)
                tid = header_match.group(3)
                cid = header_match.group(4) if header_match.lastindex >= 4 else None

                # Output is after the header line, up to the separator "---"
                body = part.split("\n\n", 1)[1] if "\n\n" in part else ""
                output = body.split("\n---")[0].strip()

                logs.append(
                    {
                        "timestamp": ts,
                        "task_id": tid,
                        "task_name": name,
                        "status": "success",  # Default for now
                        "output": output,
                        "chat_id": cid.lstrip("/") if cid else None,
                    }
                )
        except Exception as e:
            logger.debug(f"Error parsing log entry: {e}")

    return logs[::-1]  # Newest first


def save_chat_to_disk(chat_id: str, messages: List[Dict[str, Any]]):
    """Save a chat conversation to a JSON file in the chats directory."""
    chats_dir = get_workspace_path(CORE_FILES["CHATS"])
    chats_dir.mkdir(parents=True, exist_ok=True)
    path = chats_dir / f"{chat_id}.json"

    # Add a timestamp if not present in metadata
    chat_data = {
        "id": chat_id,
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
    }

    path.write_text(json.dumps(chat_data, indent=2), encoding="utf-8")
    logger.debug(f"Saved chat {chat_id} to disk")


def list_chats_from_disk() -> List[Dict[str, Any]]:
    """List all chats stored in the workspace."""
    chats_dir = get_workspace_path(CORE_FILES["CHATS"])
    if not chats_dir.exists():
        return []

    chats = []
    for f in chats_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            message_text = ""
            if data.get("messages") and len(data["messages"]) > 0:
                first_msg = data["messages"][0]
                if isinstance(first_msg.get("content"), str):
                    message_text = first_msg["content"]
                elif isinstance(first_msg.get("content"), list):
                    # Handle multimodal or complex content
                    message_text = str(first_msg["content"][0])

            chats.append(
                {
                    "id": data.get("id", f.stem),
                    "firstMessage": message_text[:100],
                    "timestamp": data.get(
                        "timestamp",
                        datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                    ),
                }
            )
        except Exception as e:
            logger.debug(f"Error loading chat file {f}: {e}")

    return sorted(chats, key=lambda x: x["timestamp"], reverse=True)


def get_chat_from_disk(chat_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a specific chat from disk."""
    path = get_workspace_path(CORE_FILES["CHATS"]) / f"{chat_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error reading chat {chat_id}: {e}")
    return None


def delete_chat_from_disk(chat_id: str) -> bool:
    """Delete a chat file from workspace."""
    path = get_workspace_path(CORE_FILES["CHATS"]) / f"{chat_id}.json"
    if path.exists():
        try:
            path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error deleting chat {chat_id}: {e}")
    return False


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


def extract_agent_metadata(content: str) -> Dict[str, str]:
    """
    Extracts basic agent metadata from IDENTITY.md or returns defaults.
    """
    import re

    data = {
        "name": "Agent",
        "description": "AI Agent",
        "emoji": "🤖",
        "vibe": "",
        "content": content.strip(),
    }

    metadata_patterns = {
        "name": r"\* \*\*Name:\*\* (.*)",
        "description": r"\* \*\*Role:\*\* (.*)",
        "emoji": r"\* \*\*Emoji:\*\* (.*)",
        "vibe": r"\* \*\*Vibe:\*\* (.*)",
    }

    for key, pattern in metadata_patterns.items():
        match = re.search(pattern, content)
        if match:
            data[key] = match.group(1).strip()

    return data


def load_identity(tag: Optional[str] = None) -> Dict[str, str]:
    """
    Load IDENTITY.md and return metadata for the agent.
    """
    content = load_workspace_file("IDENTITY.md")
    if not content:
        return {"name": "Agent", "description": "AI Agent", "content": ""}

    return extract_agent_metadata(content)


# --- GLOBAL CONFIGURATIONS ---
# Note: initialize_workspace() is NO LONGER called at module level to avoid
# global state poisoning during import.
meta = {"name": "Agent", "description": "AI Agent"}

DEFAULT_AGENT_NAME = os.getenv("DEFAULT_AGENT_NAME", meta["name"])
DEFAULT_AGENT_DESCRIPTION = os.getenv("AGENT_DESCRIPTION", meta["description"])
# Don't call build_system_prompt_from_workspace() at module level to allow logging setup first
DEFAULT_AGENT_SYSTEM_PROMPT = os.getenv("AGENT_SYSTEM_PROMPT", None)
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = to_integer(os.getenv("PORT", "9000"))
DEFAULT_DEBUG = to_boolean(os.getenv("DEBUG", "False"))
DEFAULT_PROVIDER = os.getenv("PROVIDER", "openai")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "nvidia/nemotron-3-super")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
DEFAULT_MCP_URL = os.getenv("MCP_URL", None)
DEFAULT_MCP_CONFIG = os.getenv("MCP_CONFIG", get_mcp_config_path())
DEFAULT_CUSTOM_SKILLS_DIRECTORY = os.getenv("CUSTOM_SKILLS_DIRECTORY", None)
DEFAULT_ENABLE_WEB_UI = to_boolean(os.getenv("ENABLE_WEB_UI", "False"))
DEFAULT_ENABLE_OTEL = to_boolean(os.getenv("ENABLE_OTEL", "False"))
DEFAULT_ENABLE_OTEL = to_boolean(os.getenv("ENABLE_OTEL", "False"))
DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", None)
DEFAULT_OTEL_EXPORTER_OTLP_HEADERS = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", None)
DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY = os.getenv("OTEL_EXPORTER_OTLP_PUBLIC_KEY", None)
DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY = os.getenv("OTEL_EXPORTER_OTLP_SECRET_KEY", None)
DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL = os.getenv(
    "OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"
)
DEFAULT_SSL_VERIFY = GET_DEFAULT_SSL_VERIFY()

DEFAULT_A2A_BROKER = os.getenv("A2A_BROKER", "in-memory")
DEFAULT_A2A_BROKER_URL = os.getenv("A2A_BROKER_URL", None)
DEFAULT_A2A_STORAGE = os.getenv("A2A_STORAGE", "in-memory")
DEFAULT_A2A_STORAGE_URL = os.getenv("A2A_STORAGE_URL", None)

DEFAULT_MAX_TOKENS = to_integer(os.getenv("MAX_TOKENS", "16384"))
DEFAULT_TEMPERATURE = to_float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_TOP_P = to_float(os.getenv("TOP_P", "1.0"))
DEFAULT_TIMEOUT = to_float(os.getenv("TIMEOUT", "32400.0"))
DEFAULT_TOOL_TIMEOUT = to_float(os.getenv("TOOL_TIMEOUT", "32400.0"))
DEFAULT_PARALLEL_TOOL_CALLS = to_boolean(os.getenv("PARALLEL_TOOL_CALLS", "True"))
DEFAULT_SEED = to_integer(os.getenv("SEED", None))
DEFAULT_PRESENCE_PENALTY = to_float(os.getenv("PRESENCE_PENALTY", "0.0"))
DEFAULT_FREQUENCY_PENALTY = to_float(os.getenv("FREQUENCY_PENALTY", "0.0"))
DEFAULT_LOGIT_BIAS = to_dict(os.getenv("LOGIT_BIAS", None))
DEFAULT_STOP_SEQUENCES = to_list(os.getenv("STOP_SEQUENCES", None))
DEFAULT_EXTRA_HEADERS = to_dict(os.getenv("EXTRA_HEADERS", None))
DEFAULT_EXTRA_BODY = to_dict(os.getenv("EXTRA_BODY", None))


_otel_initialized = False


def setup_otel(
    service_name: Optional[str] = None,
    endpoint: Optional[str] = DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    headers: Optional[str] = DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    public_key: Optional[str] = DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    secret_key: Optional[str] = DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    protocol: Optional[str] = DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
):
    """
    Setup OpenTelemetry tracing using Logfire and instrument pydantic_ai.
    """
    global _otel_initialized

    if not HAS_LOGFIRE:
        logger.warning(
            "OpenTelemetry is enabled but logfire is not installed. Trace logging is disabled."
        )
        return

    # Helper for Langfuse OTLP headers
    if not (headers or os.getenv("OTEL_EXPORTER_OTLP_HEADERS")) and (
        (public_key or os.getenv("OTEL_EXPORTER_OTLP_PUBLIC_KEY"))
        and (secret_key or os.getenv("OTEL_EXPORTER_OTLP_SECRET_KEY"))
    ):
        pk = public_key or os.getenv("OTEL_EXPORTER_OTLP_PUBLIC_KEY")
        sk = secret_key or os.getenv("OTEL_EXPORTER_OTLP_SECRET_KEY")
        auth_string = f"{pk}:{sk}"
        auth_encoded = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_encoded}"
        logger.debug("Generated OTLP Basic Auth headers from public/secret keys")

    # Use logfire for instrumentation
    target_service_name = service_name or retrieve_package_name()

    if _otel_initialized:
        logger.debug(f"Re-configuring OTel for service: {target_service_name}")

    # Log configuration for debugging (masked)
    target_endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    target_headers = headers or os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    logger.debug(
        f"OTel Config: endpoint={target_endpoint}, headers={'REDACTED' if target_headers else 'None'}"
    )

    logfire.configure(
        send_to_logfire=False,
        service_name=target_service_name,
        distributed_tracing=True,
    )

    # Instrument pydantic_ai
    logfire.instrument_pydantic_ai()
    Agent.instrument_all()

    _otel_initialized = True
    logger.info(
        f"OpenTelemetry logging enabled via logfire for service: {target_service_name}"
    )


# if DEFAULT_ENABLE_OTEL:
#     setup_otel()


class ReloadableApp:
    """
    ASGI wrapper that allows hot-swapping the underlying FastAPI application.
    Used for dynamic reloading of agents, skills, and MCP servers.
    """

    def __init__(self, factory: Callable[[], FastAPI]):
        self.factory = factory
        self.app: FastAPI = self.factory()

    async def __call__(self, scope, receive, sender):
        await self.app(scope, receive, sender)

    def reload(self):
        """Re-run the factory to create a fresh app instance."""
        logger.info("Hot-reloading agent application...")
        self.app = self.factory()


def create_agent_parser():
    parser = argparse.ArgumentParser(
        add_help=False, description=f"Run the {DEFAULT_AGENT_NAME} A2A + AG-UI Server"
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to bind the server to"
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DEBUG,
        help="Enable/disable debug mode",
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=[
            "openai",
            "anthropic",
            "google",
            "huggingface",
            "groq",
            "mistral",
            "ollama",
        ],
        help="LLM Provider",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="LLM Model ID")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_LLM_BASE_URL,
        help="LLM Base URL (for OpenAI compatible providers)",
    )
    parser.add_argument("--api-key", default=DEFAULT_LLM_API_KEY, help="LLM API Key")
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL, help="MCP Server URL")
    parser.add_argument(
        "--mcp-config", default=DEFAULT_MCP_CONFIG, help="MCP Server Config"
    )
    parser.add_argument(
        "--custom-skills-directory",
        default=DEFAULT_CUSTOM_SKILLS_DIRECTORY,
        help="Directory containing additional custom agent skills",
    )
    parser.add_argument(
        "--workspace",
        help="Explicit path to the agent workspace directory (contains IDENTITY.md, etc.)",
    )

    parser.add_argument(
        "--web",
        action=argparse.BooleanOptionalAction,
        default=to_boolean(os.getenv("ENABLE_WEB_UI", "False")),
        help="Enable/Disable Agent Web UI",
    )

    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL verification for LLM requests (Use with caution)",
    )

    parser.add_argument(
        "--otel",
        action=argparse.BooleanOptionalAction,
        default=to_boolean(os.getenv("ENABLE_OTEL", "False")),
        help="Enable/Disable OpenTelemetry tracing",
    )
    parser.add_argument(
        "--otel-endpoint",
        default=DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
        help="OpenTelemetry OTLP endpoint",
    )
    parser.add_argument(
        "--otel-headers",
        default=DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
        help="OpenTelemetry OTLP headers",
    )
    parser.add_argument(
        "--otel-public-key",
        default=DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
        help="OpenTelemetry OTLP public key (for Basic Auth)",
    )
    parser.add_argument(
        "--otel-secret-key",
        default=DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
        help="OpenTelemetry OTLP secret key (for Basic Auth)",
    )
    parser.add_argument(
        "--otel-protocol",
        default=DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
        help="OpenTelemetry OTLP protocol",
    )

    parser.add_argument(
        "--a2a-broker",
        default=DEFAULT_A2A_BROKER,
        choices=["in-memory", "redis", "postgres"],
        help="FastA2A Broker type",
    )
    parser.add_argument(
        "--a2a-broker-url",
        default=DEFAULT_A2A_BROKER_URL,
        help="Connection URL for the FastA2A Broker",
    )
    parser.add_argument(
        "--a2a-storage",
        default=DEFAULT_A2A_STORAGE,
        choices=["in-memory", "redis", "postgres"],
        help="FastA2A Storage type",
    )
    parser.add_argument(
        "--a2a-storage-url",
        default=DEFAULT_A2A_STORAGE_URL,
        help="Connection URL for the FastA2A Storage",
    )

    parser.add_argument("--help", action="help", help="Show usage")
    return parser


def create_model(
    provider: str,
    model_id: str,
    base_url: Optional[str],
    api_key: Optional[str],
    ssl_verify: bool = True,
    timeout: float = 300.0,
):
    """
    Create a Pydantic AI model with the specified provider and configuration.

    Args:
        provider: The model provider (openai, anthropic, google, groq, mistral, huggingface, ollama)
        model_id: The specific model ID to use
        base_url: Optional base URL for the API
        api_key: Optional API key
        ssl_verify: Whether to verify SSL certificates (default: True)

    Returns:
        A Pydantic AI Model instance
    """
    http_client = None
    if not ssl_verify:
        http_client = httpx.AsyncClient(verify=False, timeout=timeout)

    if provider == "openai":
        target_base_url = base_url
        target_api_key = api_key

        if http_client and AsyncOpenAI and OpenAIProvider:
            client = AsyncOpenAI(
                api_key=target_api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=target_base_url or os.environ.get("OPENAI_BASE_URL"),
                http_client=http_client,
            )
            provider_instance = OpenAIProvider(openai_client=client)
            return OpenAIChatModel(model_name=model_id, provider=provider_instance)

        if target_base_url:
            os.environ["OPENAI_BASE_URL"] = target_base_url
        if target_api_key:
            os.environ["OPENAI_API_KEY"] = target_api_key
        return OpenAIChatModel(model_name=model_id, provider="openai")

    elif provider == "ollama":
        target_base_url = base_url or "http://localhost:11434/v1"
        target_api_key = api_key or "ollama"

        if http_client and AsyncOpenAI and OpenAIProvider:
            client = AsyncOpenAI(
                api_key=target_api_key,
                base_url=target_base_url,
                http_client=http_client,
            )
            provider_instance = OpenAIProvider(openai_client=client)
            return OpenAIChatModel(model_name=model_id, provider=provider_instance)

        os.environ["OPENAI_BASE_URL"] = target_base_url
        os.environ["OPENAI_API_KEY"] = target_api_key
        return OpenAIChatModel(model_name=model_id, provider="openai")

    elif provider == "anthropic":
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key

        try:
            if http_client and AsyncAnthropic and AnthropicProvider:
                client = AsyncAnthropic(
                    api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
                    http_client=http_client,
                )
                provider_instance = AnthropicProvider(anthropic_client=client)
                return AnthropicModel(model_name=model_id, provider=provider_instance)
        except ImportError:
            pass

        return AnthropicModel(model_name=model_id)

    elif provider == "google":
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        return GoogleModel(model_name=model_id)

    elif provider == "groq":
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

        if http_client and AsyncGroq and GroqProvider:
            client = AsyncGroq(
                api_key=api_key or os.environ.get("GROQ_API_KEY"),
                http_client=http_client,
            )
            provider_instance = GroqProvider(groq_client=client)
            return GroqModel(model_name=model_id, provider=provider_instance)

        return GroqModel(model_name=model_id)

    elif provider == "mistral":
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key

        if http_client and Mistral and MistralProvider:
            pass

        return MistralModel(model_name=model_id)

    elif provider == "huggingface":
        if api_key:
            os.environ["HUGGING_FACE_API_KEY"] = api_key
        return HuggingFaceModel(model_name=model_id)

    return OpenAIChatModel(model_name=model_id, provider="openai")


def create_agent(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    custom_skills_directory: Optional[str] = DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
    name: Optional[str] = DEFAULT_AGENT_NAME,
    system_prompt: Optional[str] = DEFAULT_AGENT_SYSTEM_PROMPT,
    debug: bool = DEFAULT_DEBUG,
) -> Agent:
    """
    Create a Pydantic AI Agent

    Args:
        provider: LLM provider (openai, anthropic, google, etc.)
        model_id: Model identifier
        base_url: Optional base URL for LLM API
        api_key: Optional API key
        mcp_url: Optional single MCP server URL
        mcp_config: Path to MCP config file (JSON)
        custom_skills_directory: Path to additional skills
        ssl_verify: Whether to verify SSL certificates
        name: Name of the agent (or supervisor)
        system_prompt: System prompt for the agent (or supervisor)

    Returns:
        A Pydantic AI Agent instance
    """

    # ── Static MCP toolsets (created once, reused across runs) ──
    agent_toolsets = []

    async def context_aware_tool_guard(ctx, call_tool, name, tool_args):
        return await global_tool_guard(ctx, call_tool, name, tool_args)

    if mcp_url:
        if "sse" in mcp_url.lower():
            server = MCPServerSSE(
                mcp_url,
                http_client=httpx.AsyncClient(
                    verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                ),
                process_tool_call=context_aware_tool_guard,
                elicitation_callback=global_elicitation_callback,
            )
        else:
            server = MCPServerStreamableHTTP(
                mcp_url,
                http_client=httpx.AsyncClient(
                    verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                ),
                process_tool_call=context_aware_tool_guard,
                elicitation_callback=global_elicitation_callback,
            )
        agent_toolsets.append(server)
        logger.info(f"Connected to MCP Server: {mcp_url}")
    if mcp_config:
        try:
            # Prioritize local package config if mcp_config is just a filename
            if not os.path.isabs(mcp_config) and "/" not in mcp_config:
                pkg = retrieve_package_name()
                if pkg and pkg != "agent_utilities":
                    local_pkg_config = Path.cwd() / pkg / mcp_config
                    if local_pkg_config.exists():
                        mcp_config = str(local_pkg_config)
                        logger.info(f"Prioritizing Local Package Config: {mcp_config}")

            mcp_toolset = load_mcp_servers(mcp_config)
            for server in mcp_toolset:
                if hasattr(server, "http_client"):
                    server.http_client = httpx.AsyncClient(
                        verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                    )

                if hasattr(server, "elicitation_callback"):

                    def make_callback(srv):
                        async def cb(ctx, p):
                            # Ensure queue is in contextvars even if task lost it
                            q = (
                                getattr(srv, "elicitation_queue", None)
                                or elicitation_queue_var.get()
                            )
                            token = None
                            if q:
                                token = elicitation_queue_var.set(q)
                            try:
                                return await global_elicitation_callback(ctx, p)
                            finally:
                                if token:
                                    elicitation_queue_var.reset(token)

                        return cb

                    server.elicitation_callback = make_callback(server)

                if hasattr(server, "process_tool_call"):
                    server.process_tool_call = context_aware_tool_guard
                    logger.info(
                        f"Set process_tool_call (Tool Guard) on MCP Server: {server}"
                    )
                else:
                    logger.warning(
                        f"MCP Server {server} does not have process_tool_call attribute"
                    )
            agent_toolsets.extend(mcp_toolset)
            logger.info(f"Connected to MCP Config: {mcp_config}")
        except Exception as e:
            logger.warning(f"Could not load MCP config {mcp_config}: {e}")

    model = create_model(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        ssl_verify=ssl_verify,
        timeout=DEFAULT_TIMEOUT,
    )

    settings = ModelSettings(
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        timeout=DEFAULT_TIMEOUT,
        parallel_tool_calls=DEFAULT_PARALLEL_TOOL_CALLS,
        seed=DEFAULT_SEED,
        presence_penalty=DEFAULT_PRESENCE_PENALTY,
        frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
        logit_bias=DEFAULT_LOGIT_BIAS,
        stop_sequences=DEFAULT_STOP_SEQUENCES,
        extra_headers=DEFAULT_EXTRA_HEADERS,
        extra_body=DEFAULT_EXTRA_BODY,
    )

    from universal_skills.skill_utilities import get_universal_skills_path
    from pydantic_ai_skills import SkillsToolset

    # Always load default skills
    skill_dirs = []
    if skills_path := get_skills_path():
        skill_dirs.append(skills_path)
    skill_dirs.extend(get_universal_skills_path())

    # Load custom skills if provided
    if custom_skills_directory and os.path.exists(custom_skills_directory):
        logger.debug(f"Loading custom skills {custom_skills_directory}")
        skill_dirs.append(str(custom_skills_directory))
        logger.info(f"Loaded Custom Skills at {custom_skills_directory}")

    skills = SkillsToolset(directories=skill_dirs)
    agent_toolsets.append(skills)
    logger.info(f"Loaded Skills at {get_universal_skills_path()}")

    # Finalize prompt if not provided statically.
    # Note: registering a dynamic prompt function doesn't guarantee the LLM will see it if tools aren't invoked in some providers, it's safer to have a static base prompt.
    if system_prompt is None:
        logger.info(
            "No system_prompt provided to create_agent. Building from workspace..."
        )
        system_prompt_str = build_system_prompt_from_workspace()
    else:
        logger.debug(f"Custom Agent System Prompt provided: {system_prompt[:100]}...")
        system_prompt_str = system_prompt

    agent = Agent(
        model=model,
        model_settings=settings,
        name=name,
        toolsets=agent_toolsets,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        deps_type=Any,
    )

    # We use agent.instructions to ensure it is always injected into ModelRequests,
    # even when the frontend (e.g. Vercel SDK) provides a full message history.
    @agent.instructions
    def inject_system_prompt() -> str:
        return system_prompt_str

    # Register Universal Tools (Workspace, A2A, Scheduler)
    # This also handles dynamic system prompt registration via register_agent_tools
    from .tools import register_agent_tools

    register_agent_tools(agent)

    return agent


def create_agent_server(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    custom_skills_directory: Optional[str] = DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    debug: Optional[bool] = DEFAULT_DEBUG,
    host: Optional[str] = DEFAULT_HOST,
    port: Optional[int] = DEFAULT_PORT,
    enable_web_ui: Optional[bool] = DEFAULT_ENABLE_WEB_UI,
    custom_web_app: Optional[Callable[[Agent], Any]] = None,
    custom_web_mount_path: str = "/",
    web_ui_instructions: Optional[str] = None,
    html_source: Optional[str | Path] = None,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
    name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    enable_otel: Optional[bool] = DEFAULT_ENABLE_OTEL,
    otel_endpoint: Optional[str] = DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    otel_headers: Optional[str] = DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    otel_public_key: Optional[str] = DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    otel_secret_key: Optional[str] = DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    otel_protocol: Optional[str] = DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    workspace: Optional[str] = None,
    a2a_broker: str = DEFAULT_A2A_BROKER,
    a2a_broker_url: Optional[str] = DEFAULT_A2A_BROKER_URL,
    a2a_storage: str = DEFAULT_A2A_STORAGE,
    a2a_storage_url: Optional[str] = DEFAULT_A2A_STORAGE_URL,
):
    import uvicorn
    from fasta2a import Skill

    print(
        f"Starting {DEFAULT_AGENT_NAME}:"
        f"\tprovider={provider}"
        f"\tmodel={model_id}"
        f"\tbase_url={base_url}"
        f"\tmcp={mcp_url} | {mcp_config}"
        f"\tssl_verify={ssl_verify}"
    )

    _name = name or DEFAULT_AGENT_NAME

    # Set global workspace override if provided
    if workspace:
        global WORKSPACE_DIR
        WORKSPACE_DIR = workspace
        logger.info(f"Workspace override set to: {workspace}")

    def app_factory() -> FastAPI:
        """Internal factory to build the agent and its web apps."""
        nonlocal enable_otel, enable_web_ui
        _system_prompt = system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT

        if enable_otel is None:
            enable_otel = to_boolean(os.getenv("ENABLE_OTEL", "False"))

        if enable_otel:
            setup_otel(
                _name,
                endpoint=otel_endpoint,
                headers=otel_headers,
                public_key=otel_public_key,
                secret_key=otel_secret_key,
                protocol=otel_protocol,
            )

        # Create fresh agent instance
        agent_instance = create_agent(
            provider=provider,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            mcp_url=mcp_url,
            mcp_config=mcp_config,
            custom_skills_directory=custom_skills_directory,
            ssl_verify=ssl_verify,
            name=_name,
            system_prompt=_system_prompt,
            debug=debug,
        )

        # Unify skill discovery
        skill_dirs = []
        if default_skills_path := get_skills_path():
            skill_dirs.append(default_skills_path)

        try:
            from universal_skills.skill_utilities import get_universal_skills_path

            skill_dirs.extend(get_universal_skills_path())
        except ImportError:
            pass

        # Load skill-graphs if available
        try:
            from skill_graphs.skill_graph_utilities import get_skill_graphs_path

            skill_dirs.extend(get_skill_graphs_path())
        except ImportError:
            logger.debug(
                "skill-graphs package not found, skipping skill-graphs loading."
            )

        if custom_skills_directory and os.path.exists(custom_skills_directory):
            skill_dirs.append(str(custom_skills_directory))

        skills_list = []
        for d in skill_dirs:
            skills_list.extend(load_skills_from_directory(d))

        # Filter skills based on env vars
        enabled_skills = []
        for s in skills_list:
            sid = s.id if hasattr(s, "id") else s.get("id")
            if sid:
                env_var = f"ENABLE_{sid.upper().replace('-', '_')}"
                if os.environ.get(env_var, "true").lower() != "false":
                    enabled_skills.append(s)
        skills_list = enabled_skills

        if not skills_list:
            skills_list = [
                Skill(
                    id="agent",
                    name=_name,
                    description=f"General access to {_name} tools",
                    tags=["agent"],
                    input_modes=["text"],
                    output_modes=["text"],
                )
            ]

        # A2A Setup
        a2a_kwargs = {}
        if a2a_broker == "redis":
            try:
                from a2a_redis import RedisBroker

                a2a_kwargs["broker"] = RedisBroker(
                    url=a2a_broker_url or "redis://localhost:6379"
                )
            except ImportError:
                pass
        elif a2a_broker == "postgres":
            try:
                from a2a_postgres import PostgresBroker

                a2a_kwargs["broker"] = PostgresBroker(
                    url=a2a_broker_url or "postgresql+asyncpg://localhost:5432/a2a"
                )
            except ImportError:
                pass

        if a2a_storage == "redis":
            try:
                from a2a_redis import RedisStorage

                a2a_kwargs["storage"] = RedisStorage(
                    url=a2a_storage_url or "redis://localhost:6379"
                )
            except ImportError:
                pass
        elif a2a_storage == "postgres":
            try:
                from a2a_postgres import PostgresStorage

                a2a_kwargs["storage"] = PostgresStorage(
                    url=a2a_storage_url or "postgresql+asyncpg://localhost:5432/a2a"
                )
            except ImportError:
                pass

        a2a_app = agent_instance.to_a2a(
            name=_name,
            description=DEFAULT_AGENT_DESCRIPTION,
            version=__version__,
            skills=skills_list,
            debug=debug,
            **a2a_kwargs,
        )

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Pass BOTH agent_instance and reload_callback to background_processor if needed
            # For now, reload_cron_tasks handles the global 'tasks' list
            processor_task = asyncio.create_task(background_processor(agent_instance))
            try:
                if hasattr(a2a_app, "router") and hasattr(
                    a2a_app.router, "lifespan_context"
                ):
                    async with a2a_app.router.lifespan_context(a2a_app):
                        yield
                else:
                    yield
            finally:
                processor_task.cancel()
                try:
                    await processor_task
                except asyncio.CancelledError:
                    pass

        app = FastAPI(
            title=f"{DEFAULT_AGENT_NAME} - A2A + AG-UI Server",
            description=DEFAULT_AGENT_DESCRIPTION,
            debug=debug,
            lifespan=lifespan,
        )

        # Store reload_app reference in state if needed
        # We'll set this later in create_agent_server
        app.state.reload_app = None

        @app.get("/health")
        async def health_check():
            return {"status": "OK"}

        app.mount("/a2a", a2a_app)

        @app.post("/api/elicit")
        async def resolve_elicit(request: Request):
            try:
                data = await request.json()
                rid = data.get("id")
                result = data.get("result")
                if await elicitation_manager.resolve_request(rid, result):
                    return {"status": "OK"}
                return Response(
                    content=json.dumps({"error": "Request not found"}),
                    status_code=404,
                    media_type="application/json",
                )
            except Exception as e:
                logger.exception("Chat API Error")
                return JSONResponse(status_code=500, content={"error": str(e)})

        @app.post("/ag-ui")
        async def ag_ui_endpoint(request: Request) -> Response:
            from pydantic_ai.ui import SSE_CONTENT_TYPE
            from pydantic_ai.ui.ag_ui import AGUIAdapter

            accept = request.headers.get("accept", SSE_CONTENT_TYPE)
            try:
                run_input = AGUIAdapter.build_run_input(await request.body())
            except ValidationError as e:
                return Response(
                    content=json.dumps(e.json()),
                    media_type="application/json",
                    status_code=422,
                )

            if hasattr(run_input, "messages"):
                run_input.messages = prune_large_messages(run_input.messages)

            adapter = AGUIAdapter(
                agent=agent_instance, run_input=run_input, accept=accept
            )

            async def merged_event_stream():
                queue = asyncio.Queue()
                logger.debug(f"merged_event_stream: created queue with ID: {id(queue)}")
                output_queue = asyncio.Queue()

                async def run_agent():
                    token = elicitation_queue_var.set(queue)

                    # Set the queue on all MCP servers used by this agent
                    # This ensures the callback can find it even if task context is lost
                    for tool in agent_instance.tools.values():
                        # MCPServer doesn't have a common base class we can easily check here
                        # without importing, but we can check for elicitation_callback
                        if hasattr(tool, "elicitation_callback"):
                            tool.elicitation_queue = queue

                    logger.debug(
                        f"run_agent: set elicitation_queue_var to ID: {id(queue)}"
                    )
                    try:
                        logger.debug("run_agent task started")
                        async for event in adapter.run_stream(
                            deps={"elicitation_queue": queue}
                        ):
                            logger.debug(
                                f"run_agent yielded event: {getattr(event, 'type', type(event))}"
                            )
                            await output_queue.put(event)
                    except Exception as e:
                        logger.error(f"Error in agent run task: {e}")
                    finally:
                        elicitation_queue_var.reset(token)
                        logger.debug("run_agent task finished")
                        await output_queue.put(None)

                async def listen_queue():
                    try:
                        logger.debug("listen_queue task started")
                        while True:
                            event = await queue.get()
                            logger.debug(
                                f"listen_queue received item: {event.get('type')}"
                            )
                            await output_queue.put(event)
                    except asyncio.CancelledError:
                        logger.debug("listen_queue task cancelled")
                    except Exception as e:
                        logger.error(f"Error in queue listener task: {e}")

                agent_task = asyncio.create_task(run_agent())
                queue_task = asyncio.create_task(listen_queue())

                try:
                    while True:
                        event = await output_queue.get()
                        logger.debug(
                            f"merged_event_stream: got event from output_queue: {getattr(event, 'type', event.get('type') if isinstance(event, dict) else type(event))}"
                        )
                        if event is None:
                            break
                        yield event
                finally:
                    agent_task.cancel()
                    queue_task.cancel()

            event_stream = merged_event_stream()

            async def custom_encode_stream(stream):
                async for event in stream:
                    logger.debug(
                        f"custom_encode_stream processing event type: {getattr(event, 'type', event.get('type') if isinstance(event, dict) else type(event))}"
                    )
                    if isinstance(event, dict):
                        # Use Vercel AI SDK Data Part (d: JSON_ARRAY) mapping or standard format expected by agent_webui Part.tsx
                        # A standard data parts array looks like this in Vercel `ai/react` format:
                        yield f"data: {json.dumps([event])}\n\n".encode("utf-8")
                    else:
                        if hasattr(event, "model_dump_json"):
                            yield f"data: {event.model_dump_json()}\n\n".encode("utf-8")
                        else:
                            yield f"data: {json.dumps(event)}\n\n".encode("utf-8")

            sse_stream = custom_encode_stream(event_stream)

            return StreamingResponse(
                sse_stream,
                media_type=accept,
            )

        # Mount Web UI
        if enable_web_ui is None:
            enable_web_ui = to_boolean(os.getenv("ENABLE_WEB_UI", "False"))

        if custom_web_app is not None:
            web_app = custom_web_app(agent_instance)
            app.mount(custom_web_mount_path, web_app)
            logger.info(f"Mounted custom web UI at {custom_web_mount_path}")
        elif enable_web_ui:
            try:
                from agent_webui.server import create_agent_web_app

                identity_meta = load_identity()
                helpers = {
                    "agent_name": _name,
                    "agent_description": identity_meta.get(
                        "description", DEFAULT_AGENT_DESCRIPTION
                    ),
                    "agent_emoji": identity_meta.get("emoji", "🤖"),
                    "get_workspace_path": get_workspace_path,
                    "load_workspace_file": load_workspace_file,
                    "write_workspace_file": write_workspace_file,
                    "write_md_file": write_md_file,
                    "list_workspace_files": list_workspace_files,
                    "initialize_workspace": initialize_workspace,
                    "toggle_skill": lambda sid: f"Skill {sid} toggled (not implemented)",
                    "list_skills": lambda: [
                        {
                            "id": s.id if hasattr(s, "id") else s.get("id"),
                            "name": s.name if hasattr(s, "name") else s.get("name"),
                            "description": (
                                s.description
                                if hasattr(s, "description")
                                else s.get("description")
                            ),
                            "enabled": True,  # For now
                        }
                        for s in skills_list
                    ],
                    "get_cron_calendar": get_cron_tasks_from_md,
                    "get_cron_logs": get_cron_logs_from_md,
                    "get_agent_icon_path": get_agent_icon_path,
                    "save_chat": save_chat_to_disk,
                    "list_chats": list_chats_from_disk,
                    "get_chat": get_chat_from_disk,
                    "delete_chat": delete_chat_from_disk,
                    "reload_callback": lambda: (
                        reloadable.reload() if reloadable else None
                    ),
                }

                web_app = create_agent_web_app(
                    agent_instance,
                    workspace_helpers=helpers,
                    html_source=html_source,
                )
                # Inject reload callback into the web app's state if needed
                web_app.state.reload_app = None  # Will be set below
                app.mount("/", web_app)
                logger.debug("Mounted new standalone agent-web UI dashboard at /")
            except ImportError:
                logger.error(
                    "agent-web package not found. Enhanced UI dashboard disabled."
                )
                # Fallback or error

        return app

    # Create the reloadable wrapper
    reloadable = ReloadableApp(app_factory)

    # Recursive injection of the reloadable reference into all mounted app states
    def inject_reload_app(fast_app: FastAPI, wrapper: ReloadableApp):
        fast_app.state.reload_app = wrapper
        from fastapi.routing import Mount

        for route in fast_app.routes:
            if (
                isinstance(route, Mount)
                and hasattr(route, "app")
                and isinstance(route.app, FastAPI)
            ):
                inject_reload_app(route.app, wrapper)

    inject_reload_app(reloadable.app, reloadable)

    logger.info(
        "Starting server on %s:%s (A2A at /a2a, AG-UI at /ag-ui, Web UI: %s)",
        host,
        port,
        "Enabled (Dashboard)" if enable_web_ui else "Disabled",
    )

    uvicorn.run(
        reloadable,
        host=host,
        port=port,
        timeout_keep_alive=1800,
        timeout_graceful_shutdown=60,
        log_level="debug" if debug else "info",
    )


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


def filter_tools_by_tag(tools: List[Any], tag: str) -> List[Any]:
    """
    Filters a list of tools for a given tag.
    """
    return [t for t in tools if tool_in_tag(t, tag)]


def prune_large_messages(messages: list[Any], max_length: int = 5000) -> list[Any]:
    """
    Summarize large tool outputs in the message history to save context window.
    Keeps the most recent tool outputs intact if they are the very last message,
    but generally we want to prune history.
    """
    pruned_messages = []
    for i, msg in enumerate(messages):
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        if isinstance(content, str) and len(content) > max_length:
            summary = (
                f"{content[:200]} ... "
                f"[Output truncated, original length {len(content)} characters] "
                f"... {content[-200:]}"
            )

            if isinstance(msg, dict):
                msg["content"] = summary
                pruned_messages.append(msg)
            elif hasattr(msg, "content"):
                try:
                    from copy import copy

                    new_msg = copy(msg)
                    new_msg.content = summary
                    pruned_messages.append(new_msg)
                except Exception:
                    pruned_messages.append(msg)
            else:
                pruned_messages.append(msg)
        else:
            pruned_messages.append(msg)

    return pruned_messages


def append_to_file(filename: str, text: str, section_header: Optional[str] = None):
    """Append text, optionally under a header."""
    path = get_workspace_path(filename)
    content = path.read_text(encoding="utf-8") if path.exists() else ""

    if section_header:
        if section_header not in content:
            content += f"\n\n## {section_header}\n"
        content += f"\n{text}\n"
    else:
        content += f"\n\n{text}\n"

    path.write_text(content.strip() + "\n", encoding="utf-8")


def update_cron_task_in_cron_md(task: dict):
    """
    Update/add one row in CRON.md table.
    task = {"id": "daily-news", "name": "...", "interval_min": 1440, ...}
    """
    path = get_workspace_path("CRON.md")
    if not path.exists():
        initialize_workspace()

    lines = path.read_text(encoding="utf-8").splitlines()
    table_start = -1
    for i, line in enumerate(lines):
        if "| ID" in line and "| Name" in line:
            table_start = i
            break

    if table_start == -1:
        # No table → append simple one
        append_to_file(
            "CRON.md",
            "\n## Active Tasks\n\n| ID | Name | Interval (min) | Prompt starts with | Last run | Next approx |\n|----|------|----------------|--------------------|----------|-------------|",
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        table_start = len(lines) - 2  # header + separator

    new_row = (
        f"| {task.get('id','?')} "
        f"| {task.get('name','?')} "
        f"| {task.get('interval_min','?')} "
        f"| {task.get('prompt','?')[:40]}... "
        f"| {datetime.now().strftime('%H:%M')} "
        f"| — |"
    )

    id_found = False
    for i in range(table_start + 2, len(lines)):
        if (
            lines[i].strip().startswith(f"| {task['id']} ")
            or f"| {task['id']} |" in lines[i]
        ):
            lines[i] = new_row
            id_found = True
            break

    if not id_found:
        lines.insert(table_start + 3, new_row)

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def schedule_task(task_id: str, name: str, interval_minutes: int, prompt: str) -> str:
    """Consolidated tool to schedule a task persistently."""
    if interval_minutes < 1:
        return "Interval must be ≥ 1 minute"

    task_data = {
        "id": task_id,
        "name": name,
        "interval_min": interval_minutes,
        "prompt": prompt,
    }
    update_cron_task_in_cron_md(task_data)

    # Immediately update memory for reactive execution
    global tasks
    found = False
    for t in tasks:
        if t.id == task_id:
            t.name = name
            t.interval_minutes = interval_minutes
            t.prompt = prompt
            t.last_run = datetime.now() - timedelta(minutes=interval_minutes + 1)
            found = True
            break

    if not found:
        tasks.append(
            PeriodicTask(
                id=task_id,
                name=name,
                interval_minutes=interval_minutes,
                prompt=prompt,
                last_run=datetime.now() - timedelta(minutes=interval_minutes + 1),
            )
        )

    return f"✅ Scheduled '{name}' (ID: {task_id}) every {interval_minutes} min"


def delete_scheduled_task(task_id: str) -> str:
    """Remove a task from CRON.md and memory."""
    path = get_workspace_path("CRON.md")
    if not path.exists():
        return "CRON.md not found."

    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    found_in_md = False
    for line in lines:
        if line.strip().startswith(f"| {task_id} ") or f"| {task_id} |" in line:
            found_in_md = True
            continue
        new_lines.append(line)

    if found_in_md:
        path.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")

    # Remove from memory
    global tasks
    found_in_mem = False
    tasks_to_keep = []
    for t in tasks:
        if t.id == task_id:
            found_in_mem = True
            continue
        tasks_to_keep.append(t)

    tasks[:] = tasks_to_keep  # Update global list in place

    if found_in_md or found_in_mem:
        return f"✅ Deleted scheduled task '{task_id}'"
    return f"ℹ️ Task '{task_id}' not found."


def list_scheduled_tasks() -> str:
    """List all active periodic tasks from memory."""
    global tasks
    if not tasks:
        return "No periodic tasks scheduled."

    lines = ["Active periodic tasks:"]
    now = datetime.now()
    for t in tasks:
        if t.active:
            mins_since = (now - t.last_run).total_seconds() / 60
            next_in = max(0, int(t.interval_minutes - mins_since))
            lines.append(
                f"• {t.id}: {t.name} (every {t.interval_minutes} min, next ≈ {next_in} min)"
            )
    return "\n".join(lines)


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
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return f"Written {path}"


def append_to_md_file(filename: str, text: str):
    """Safe append to markdown file."""
    if not filename.lower().endswith(".md"):
        raise ValueError("Only .md files allowed")
    append_to_file(filename, text)
    return f"Appended to {filename}"


def create_memory(text: str) -> str:
    """
    Save important decisions, outcomes, user preferences, critical
    information, or information the user explicitly requests to long-term memory (MEMORY.md).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    note = f"- [{timestamp}] {text}"
    append_to_md_file("MEMORY.md", note)
    return "Saved to memory."


def search_memory(query: str) -> str:
    """Search MEMORY.md for a query string."""
    content = load_workspace_file(CORE_FILES["MEMORY"])
    if not content:
        return "Memory is empty."

    lines = content.splitlines()
    results = []
    # Identify entries, usually starting with "- [" or a bullet
    for i, line in enumerate(lines):
        if query.lower() in line.lower():
            results.append(f"Line {i+1}: {line.strip()}")

    if not results:
        return f"No entries found matching '{query}' in memory."
    return "\n".join(results)


def delete_memory_entry(index: int) -> str:
    """Delete a memory entry by line number (1-indexed)."""
    path = get_workspace_path(CORE_FILES["MEMORY"])
    if not path.exists():
        return "Memory file not found."

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    if index < 1 or index > len(lines):
        return f"❌ Invalid index {index}. Memory has {len(lines)} lines."

    # Validate that we are deleting an actual entry (starts with - or *)
    line_to_delete = lines[index - 1].strip()
    if not (
        line_to_delete.startswith("-")
        or line_to_delete.startswith("*")
        or line_to_delete.startswith("|")
    ):
        return f"⚠️ Line {index} does not look like a data entry: '{line_to_delete}'. Deletion aborted for safety."

    deleted_text = lines.pop(index - 1)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return f"✅ Deleted memory entry {index}: {deleted_text}"


def compress_memory(max_entries: int = 50) -> str:
    """
    Compress MEMORY.md by pruning old entries.
    In a future version this could use an LLM to summarize.
    """
    path = get_workspace_path(CORE_FILES["MEMORY"])
    if not path.exists():
        return "Memory file not found."

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Identify where the log starts (usually after ## Log of Important Events)
    log_start = -1
    for i, line in enumerate(lines):
        if "## Log of Important Events" in line:
            log_start = i
            break

    if log_start == -1:
        return "❌ Could not find '## Log of Important Events' section in MEMORY.md"

    header = lines[: log_start + 1]
    entries = [line for line in lines[log_start + 1 :] if line.strip()]

    if len(entries) <= max_entries:
        return f"ℹ️ Memory consists of {len(entries)} entries, which is below the limit of {max_entries}. No compression needed."

    pruned = len(entries) - max_entries
    kept_entries = entries[-max_entries:]

    new_content = "\n".join(header).strip() + "\n\n"
    new_content += "> [!NOTE]\n"
    new_content += f"> Memory was compressed on {datetime.now().strftime('%Y-%m-%d')}. {pruned} older entries were pruned.\n\n"
    new_content += "\n".join(kept_entries)

    path.write_text(new_content.strip() + "\n", encoding="utf-8")
    return f"✅ Compressed memory. Pruned {pruned} old entries, kept the most recent {max_entries}."


def load_a2a_peers() -> List[Dict[str, str]]:
    """Parse A2A_AGENTS.md table into list of dicts."""
    content = load_workspace_file(CORE_FILES["AGENTS"])
    if not content:
        return []

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
            if len(parts) >= 5:
                peers.append(
                    {
                        "name": parts[0],
                        "url": parts[1],
                        "description": parts[2],
                        "capabilities": parts[3],
                        "auth": parts[4] if len(parts) > 4 else "none",
                        "notes": parts[5] if len(parts) > 5 else "",
                    }
                )
    return peers


def register_a2a_peer(
    name: str,
    url: str,
    description: str = "",
    capabilities: str = "",
    auth: str = "none",
    notes: str = "",
) -> str:
    """Add or update a peer in A2A_AGENTS.md table."""
    path = get_workspace_path(CORE_FILES["AGENTS"])
    if not path.exists():
        initialize_workspace()

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    table_start = -1
    for i, line in enumerate(lines):
        if "| Name" in line or "| ID" in line:
            table_start = i
            break

    if table_start == -1:
        new_table = (
            "\n## Registered A2A Peers\n\n"
            "| Name | Endpoint URL | Description | Capabilities | Auth | Notes / Last Connected |\n"
            "|------|--------------|-------------|--------------|------|------------------------|\n"
        )
        content += new_table
        lines = content.splitlines()
        table_start = len(lines) - 3

    new_row = f"| {name} | {url} | {description} | {capabilities} | {auth} | {notes or datetime.now().strftime('%Y-%m-%d')} |"

    updated = False
    for i in range(table_start + 2, len(lines)):
        if lines[i].strip().startswith(f"| {name} ") or f"| {name} |" in lines[i]:
            lines[i] = new_row
            updated = True
            break

    if not updated:
        lines.insert(table_start + 3, new_row)

    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return f"✅ Registered/updated A2A peer '{name}' at {url}"


def get_a2a_peer(name: str) -> Optional[Dict[str, str]]:
    """Return single peer by name (case-insensitive)."""
    peers = load_a2a_peers()
    name_lower = name.lower()
    for p in peers:
        if p.get("name", "").lower() == name_lower:
            return p
    return None


def list_a2a_peers() -> str:
    """List all registered A2A peers formatted for the LLM."""
    peers = load_a2a_peers()
    if not peers:
        return "No A2A peers registered yet."
    lines = ["## Known A2A Peers"]
    for p in peers:
        lines.append(f"- **{p['name']}** → {p['url']}  ({p['capabilities']})")
    return "\n".join(lines)


def delete_a2a_peer(name: str) -> str:
    """Remove a peer from A2A_AGENTS.md registry."""
    path = get_workspace_path(CORE_FILES["AGENTS"])
    if not path.exists():
        return f"❌ {CORE_FILES['AGENTS']} not found."

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    new_lines = []
    found = False

    for line in lines:
        # Check if line is a table row for this peer name
        if line.strip().startswith(f"| {name} ") or f"| {name} |" in line:
            found = True
            continue
        new_lines.append(line)

    if found:
        path.write_text("\n".join(new_lines).strip() + "\n", encoding="utf-8")
        return f"✅ Removed A2A peer '{name}' from registry."
    return f"ℹ️ A2A peer '{name}' not found in registry."


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


DEFAULT_MAX_CRON_LOG_ENTRIES = 50


def append_cron_log(
    task_id: str, task_name: str, output: str, chat_id: Optional[str] = None
):
    """Append a timestamped entry to CRON_LOG.md."""
    path = get_workspace_path("CRON_LOG.md")
    if not path.exists():
        initialize_workspace()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_info = f" | [View Chat](/{chat_id})" if chat_id else ""
    entry = (
        f"\n### [{ts}] {task_name} (`{task_id}`){chat_info}\n\n"
        f"{output.strip()}\n\n"
        f"---\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)
    logger.debug(f"Appended cron log entry for {task_id}")


def cleanup_cron_log(max_entries: int = DEFAULT_MAX_CRON_LOG_ENTRIES):
    """Keep only the last `max_entries` log entries in CRON_LOG.md."""
    path = get_workspace_path("CRON_LOG.md")
    if not path.exists():
        return

    content = path.read_text(encoding="utf-8")
    # Each entry starts with "### ["
    parts = re.split(r"(?=^### \[)", content, flags=re.MULTILINE)

    # parts[0] is the header (before any ### entry)
    header = parts[0] if parts else ""
    entries = [p for p in parts[1:] if p.strip()]

    if len(entries) <= max_entries:
        return  # nothing to prune

    kept = entries[-max_entries:]
    pruned_count = len(entries) - max_entries
    new_content = header.rstrip() + "\n\n" + "".join(kept)
    path.write_text(new_content.strip() + "\n", encoding="utf-8")
    logger.debug(f"Pruned {pruned_count} old cron log entries, kept {max_entries}")


async def reload_cron_tasks():
    """Reload all tasks from CRON.md.

    Every row in the table becomes a PeriodicTask.  Prompts starting with
    '@' are resolved to workspace file contents at execution time (not here).
    """
    content = load_workspace_file("CRON.md")
    if not content:
        return

    parsed_tasks = []
    lines = content.splitlines()
    in_table = False
    for line in lines:
        if "| ID" in line and "| Name" in line:
            in_table = True
            continue
        if (
            in_table
            and line.strip().startswith("|")
            and not (line.strip().startswith("|---") or "| ID" in line)
        ):
            parts = [p.strip() for p in line.strip("| ").split("|")]
            if len(parts) >= 4:
                try:
                    parsed_tasks.append(
                        PeriodicTask(
                            id=parts[0],
                            name=parts[1],
                            interval_minutes=int(parts[2]),
                            prompt=parts[3],
                            last_run=datetime.now(),  # wait for full interval before first run
                            # last_run=datetime.now() - timedelta(minutes=int(parts[2])),  # run soonish
                        )
                    )
                except Exception:
                    continue

    async with lock:
        global tasks
        new_list = []
        for pt in parsed_tasks:
            # Preserve last_run and active state for existing tasks
            existing = next((t for t in tasks if t.id == pt.id), None)
            if existing and existing.interval_minutes == pt.interval_minutes:
                pt.last_run = existing.last_run
                pt.active = existing.active
            new_list.append(pt)
        tasks = new_list


async def background_processor(agent: Any):
    """Background processor for periodic tasks."""

    logger = logging.getLogger(__name__)
    logger.debug("In-memory periodic processor started (checks every 60 s)")

    while True:
        try:
            await reload_cron_tasks()
        except Exception as e:
            logger.error(f"Error reloading cron tasks: {e}")

        await asyncio.sleep(60)
        now = datetime.now()
        due: list[PeriodicTask] = []
        async with lock:
            for t in tasks:
                if (
                    t.active
                    and (now - t.last_run).total_seconds() / 60 >= t.interval_minutes
                ):
                    due.append(t)
                    t.last_run = now  # prevent double-trigger

        for task in due:
            try:
                # Handle internal tasks that don't need an LLM call
                if task.prompt.startswith("__internal:"):
                    cmd = task.prompt.split(":", 1)[1]
                    if cmd == "cleanup_cron_log":
                        cleanup_cron_log()
                        logger.debug("Cron log cleanup completed")
                    continue

                # Resolve @file references in prompts
                resolved_prompt = resolve_prompt(task.prompt)

                logger.info(f"Running periodic task → {task.name} (ID: {task.id})")
                result = await agent.run(resolved_prompt)

                # Capture output
                output = str(result.output or "")
                if output:
                    logger.info(f"Task result: {output[:200]}...")

                # Create a persistent chat entry for this run
                try:
                    chat_id = (
                        f"cron-{task.id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    )
                    messages = []

                    # Add user message (the prompt)
                    messages.append(
                        {
                            "id": "msg-u-1",
                            "role": "user",
                            "content": resolved_prompt,
                            "parts": [{"type": "text", "text": resolved_prompt}],
                        }
                    )

                    # Add assistant response
                    messages.append(
                        {
                            "id": "msg-a-1",
                            "role": "assistant",
                            "content": output,
                            "parts": [{"type": "text", "text": output}],
                        }
                    )

                    save_chat_to_disk(chat_id, messages)
                except Exception as e:
                    logger.error(f"Failed to save cron chat: {e}")
                    chat_id = None

                append_cron_log(
                    task_id=task.id,
                    task_name=task.name,
                    output=output or "(no output)",
                    chat_id=chat_id,
                )
            except Exception as e:
                logger.error(f"Error running periodic task {task.id}: {e}")
                append_cron_log(
                    task_id=task.id,
                    task_name=task.name,
                    output=f"❌ ERROR: {e}",
                )

        await asyncio.sleep(60)


def load_mcp_config() -> dict:
    """Load MCP config from workspace."""
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"mcpServers": {}}
    return {"mcpServers": {}}


def save_mcp_config(config: dict):
    """Save MCP config to workspace."""
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def create_new_skill(
    name: str,
    description: str,
    when_to_use: str = "",
    how_to_use: str = "",
    tags: str = "custom",
) -> str:
    """Helper to scaffold a new skill."""
    safe_name = re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "-"))
    skills_dir = get_workspace_path("skills")
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
    skills_dir = get_workspace_path("skills")
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
    skills_dir = get_workspace_path("skills")
    skill_file = skills_dir / safe_name / "SKILL.md"

    if skill_file.exists():
        return skill_file.read_text(encoding="utf-8")
    return f"❌ SKILL.md not found for skill '{safe_name}'"


def write_skill_md(name: str, content: str) -> str:
    """Overwrite the SKILL.md content of a workspace skill."""
    safe_name = re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "-"))
    skills_dir = get_workspace_path("skills")
    skill_dir = skills_dir / safe_name
    skill_file = skill_dir / "SKILL.md"

    if not skill_dir.exists():
        return f"❌ Skill '{safe_name}' folder does not exist."

    try:
        skill_file.write_text(content.strip() + "\n", encoding="utf-8")
        return f"✅ Updated SKILL.md for skill '{safe_name}'."
    except Exception as e:
        return f"❌ Error writing SKILL.md for skill '{safe_name}': {e}"


async def chat(agent: Agent, prompt: str):
    result = await agent.run(prompt)
    print(f"Response:\n\n{result.output}")


async def node_chat(agent: Agent, prompt: str) -> List:
    nodes = []
    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            nodes.append(node)
            print(node)
    return nodes


async def stream_chat(agent: Agent, prompt: str) -> None:
    async with agent.run_stream(prompt) as result:
        async for text_chunk in result.stream_text(delta=True):
            print(text_chunk, end="", flush=True)
        print("\nDone!")
