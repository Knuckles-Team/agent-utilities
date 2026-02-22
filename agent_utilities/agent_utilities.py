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

# import uvicorn  # Optional
from typing import List, Any, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from fasta2a import Skill
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from importlib.resources import files, as_file

# from fastapi import FastAPI, Request  # Optional
# from starlette.responses import Response, StreamingResponse  # Optional
from pydantic import ValidationError

from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.mcp import (
    load_mcp_servers,
    MCPServerStreamableHTTP,
    MCPServerSSE,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.huggingface import HuggingFaceModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.mistral import MistralModel

# from pydantic_ai.ui import SSE_CONTENT_TYPE  # Optional
# from pydantic_ai.ui.ag_ui import AGUIAdapter  # Optional

# from fasta2a import Skill  # Optional
# from pydantic_ai_skills import SkillsToolset  # Optional
# from universal_skills.skill_utilities import get_universal_skills_path  # Optional
from agent_utilities.base_utilities import (
    to_boolean,
    to_integer,
    to_float,
    to_list,
    to_dict,
    retrieve_package_name,
)

# from .tools import register_agent_tools  # Breaks circular import

from .models import PeriodicTask
from .agent.templates import (
    CORE_FILES,
    tasks,
    lock,
    TEMPLATES,
    NEW_SKILL_TEMPLATE,
)

try:
    from openai import AsyncOpenAI
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:
    AsyncOpenAI = None
    OpenAIProvider = None

try:
    from groq import AsyncGroq
    from pydantic_ai.providers.groq import GroqProvider
except ImportError:
    AsyncGroq = None
    GroqProvider = None

try:
    from mistralai import Mistral
    from pydantic_ai.providers.mistral import MistralProvider
except ImportError:
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
__version__ = "0.1.3"


def get_skills_path() -> str:
    skills_dir = files(retrieve_package_name()) / "skills"
    with as_file(skills_dir) as path:
        skills_path = str(path)
    return skills_path


def get_mcp_config_path() -> str:
    mcp_config_file = files(retrieve_package_name()) / "mcp_config.json"
    with as_file(mcp_config_file) as path:
        mcp_config_path = str(path)
    return mcp_config_path


def load_skills_from_directory(directory: str) -> List[Skill]:
    from fasta2a import Skill

    skills = []
    base_path = Path(directory)

    if not base_path.exists():
        print(f"Skills directory not found: {directory}")
        return skills

    for item in base_path.iterdir():
        if item.is_dir():
            skill_file = item / "SKILL.md"
            if skill_file.exists():
                try:
                    with open(skill_file, "r") as f:
                        content = f.read()
                        if content.startswith("---"):
                            _, frontmatter, _ = content.split("---", 2)
                            data = yaml.safe_load(frontmatter)

                            skill_id = item.name
                            skill_name = data.get("name", skill_id)
                            skill_desc = data.get(
                                "description", f"Access to {skill_name} tools"
                            )
                            skills.append(
                                Skill(
                                    id=skill_id,
                                    name=skill_name,
                                    description=skill_desc,
                                    tags=[skill_id],
                                    input_modes=["text"],
                                    output_modes=["text"],
                                )
                            )
                except Exception as e:
                    print(f"Error loading skill from {skill_file}: {e}")

    return skills


def get_http_client(
    ssl_verify: bool = True, timeout: float = 300.0
) -> httpx.AsyncClient | None:
    if not ssl_verify:
        return httpx.AsyncClient(verify=False, timeout=timeout)
    return None


def get_workspace_path(filename: str) -> Path:
    """Return full path for a file in workspace."""
    workspace_dir = files(retrieve_package_name()) / "agent"
    with as_file(workspace_dir) as path:
        return path / filename


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
            logger.info(f"Initialized {path}")


def load_workspace_file(filename: str, default: str = "") -> str:
    """Read markdown file content. Returns default if missing."""
    path = get_workspace_path(filename)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default


def load_all_core_files() -> Dict[str, str]:
    """Load all core markdown files into a dict."""
    return {k: load_workspace_file(v) for k, v in CORE_FILES.items()}


def build_system_prompt_from_workspace(fallback_prompt: str = "") -> str:
    """
    Combine core files into a rich system prompt.
    Order matters — IDENTITY → USER → AGENTS → MEMORY → custom fallback
    """
    parts = []
    for key in ["IDENTITY", "USER", "AGENTS", "MEMORY"]:
        content = load_workspace_file(CORE_FILES[key])
        if content.strip():
            parts.append(f"---\n# {CORE_FILES[key]}\n{content}\n---")

    if fallback_prompt:
        parts.append(fallback_prompt)

    return "\n\n".join(parts).strip()


def parse_identity_md(content: str) -> Dict[str, Dict[str, str]]:
    """
    Parse IDENTITY.md into a dictionary of agent definitions.
    Supports tagged sections: ## [tag]
    Each section parses bullet points for Name, Role, Emoji, Vibe.
    """
    import re

    # Split into sections by ## [tag]
    sections = re.split(r"^##\s+\[(.*?)\]", content, flags=re.MULTILINE)

    identities = {}

    if len(sections) <= 1:
        # Backward compatibility: treat entire file as single 'default' agent
        identities["default"] = _parse_section_content(content)
    else:
        # First part before any ## [tag] is skipped (usually headers/intro)
        for i in range(1, len(sections), 2):
            tag = sections[i].strip()
            section_content = sections[i + 1]
            identities[tag] = _parse_section_content(section_content)

    return identities


def _parse_section_content(content: str) -> Dict[str, str]:
    """Helper to extract metadata and prompt from a Markdown section."""
    import re

    data = {
        "name": "Agent",
        "description": "An AI agent.",
        "emoji": "🤖",
        "vibe": "",
        "content": content.strip(),
    }

    # Extract bullet points
    name_match = re.search(r"\* \*\*Name:\*\* (.*)", content)
    if name_match:
        data["name"] = name_match.group(1).strip()

    role_match = re.search(r"\* \*\*Role:\*\* (.*)", content)
    if role_match:
        data["description"] = role_match.group(1).strip()

    emoji_match = re.search(r"\* \*\*Emoji:\*\* (.*)", content)
    if emoji_match:
        data["emoji"] = emoji_match.group(1).strip()

    vibe_match = re.search(r"\* \*\*Vibe:\*\* (.*)", content)
    if vibe_match:
        data["vibe"] = vibe_match.group(1).strip()

    return data


def load_identities() -> Dict[str, Dict[str, str]]:
    """
    Load IDENTITY.md and return all identity definitions as a dictionary.
    """
    content = load_workspace_file("IDENTITY.md")
    return parse_identity_md(content)


def load_identity(tag: Optional[str] = None) -> Dict[str, str]:
    """
    Load IDENTITY.md and return metadata for a specific agent tag.
    If no tag is provided, returns the first identity found.
    """
    content = load_workspace_file("IDENTITY.md")
    identities = parse_identity_md(content)

    if not identities:
        return {"name": "Agent", "description": "AI Agent", "content": ""}

    if tag and tag in identities:
        return identities[tag]

    # Return 'default' if it exists, otherwise the first entry
    if "default" in identities:
        return identities["default"]

    return next(iter(identities.values()))


# --- GLOBAL CONFIGURATIONS ---
try:
    initialize_workspace()
    meta = load_identity()
except Exception:
    meta = {"name": "Agent", "description": "AI Agent"}

DEFAULT_AGENT_NAME = os.getenv("DEFAULT_AGENT_NAME", meta["name"])
DEFAULT_AGENT_DESCRIPTION = os.getenv("AGENT_DESCRIPTION", meta["description"])
DEFAULT_AGENT_SYSTEM_PROMPT = os.getenv(
    "AGENT_SYSTEM_PROMPT", build_system_prompt_from_workspace()
)
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = to_integer(os.getenv("PORT", "9000"))
DEFAULT_DEBUG = to_boolean(os.getenv("DEBUG", "False"))
DEFAULT_PROVIDER = os.getenv("PROVIDER", "openai")
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "qwen/qwen3-coder-next")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
DEFAULT_MCP_URL = os.getenv("MCP_URL", None)
DEFAULT_MCP_CONFIG = os.getenv("MCP_CONFIG", get_mcp_config_path())
DEFAULT_CUSTOM_SKILLS_DIRECTORY = os.getenv("CUSTOM_SKILLS_DIRECTORY", None)
DEFAULT_ENABLE_WEB_UI = to_boolean(os.getenv("ENABLE_WEB_UI", "False"))
DEFAULT_SSL_VERIFY = to_boolean(os.getenv("SSL_VERIFY", "True"))

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
    parser.add_argument("--debug", type=bool, default=DEFAULT_DEBUG, help="Debug mode")
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
        "--web",
        action="store_true",
        default=DEFAULT_ENABLE_WEB_UI,
        help="Enable Pydantic AI Web UI",
    )

    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL verification for LLM requests (Use with caution)",
    )

    parser.add_argument("--help", action="store_true", help="Show usage")
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
    agent_defs: Optional[Dict[str, tuple[str, str]]] = None,
) -> Agent:
    """
    Create a Pydantic AI Agent with optional multi-agent supervisor support.

    If agent_defs is provided, it creates a supervisor agent that can delegate tasks
    to child agents specified in the dictionary.

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
        agent_defs: Dictionary of child agent definitions:
                   {"tag": (child_prompt, child_name)}
                   If provided, triggers multi-agent supervisor pattern.

    Returns:
        A Pydantic AI Agent instance (can be a supervisor or a single agent).
    """

    # ── Static MCP toolsets (created once, reused across runs) ──
    agent_toolsets = []

    if mcp_url:
        if "sse" in mcp_url.lower():
            server = MCPServerSSE(
                mcp_url,
                http_client=httpx.AsyncClient(
                    verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                ),
            )
        else:
            server = MCPServerStreamableHTTP(
                mcp_url,
                http_client=httpx.AsyncClient(
                    verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                ),
            )
        agent_toolsets.append(server)
        logger.info(f"Connected to MCP Server: {mcp_url}")
    elif mcp_config:
        try:
            mcp_toolset = load_mcp_servers(mcp_config)
            for server in mcp_toolset:
                if hasattr(server, "http_client"):
                    server.http_client = httpx.AsyncClient(
                        verify=ssl_verify, timeout=DEFAULT_TIMEOUT
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

    if agent_defs:
        from universal_skills.skill_utilities import get_universal_skills_path
        from pydantic_ai_skills import SkillsToolset

        # ── Multi-Agent Pattern ──
        logger.info(f"Initializing Multi-Agent System with {len(agent_defs)} agents")

        # 1. Identify Universal Skills
        universal_skills_path = get_universal_skills_path()
        universal_skill_dirs = []
        if os.path.exists(universal_skills_path):
            for item in os.listdir(universal_skills_path):
                item_path = os.path.join(universal_skills_path, item)
                if os.path.isdir(item_path):
                    universal_skill_dirs.append(item_path)
                    logger.debug(f"Identified universal skill: {item}")

        child_agents = {}
        package_prefix = retrieve_package_name().replace("_", "-") + "-"

        for tag, (child_prompt, child_name) in agent_defs.items():
            tag_toolsets = []

            # Filter MCP servers by tag
            for ts in agent_toolsets:
                if hasattr(ts, "filtered"):
                    tag_toolsets.append(
                        ts.filtered(
                            lambda ctx, tool_def, t=tag: tool_in_tag(tool_def, t)
                        )
                    )

            # Specialized skills for this tag
            skill_dir_name = f"{package_prefix}{tag.replace('_', '-')}"
            child_skills_directories = []

            # Check default skills directory
            default_skill_path = os.path.join(get_skills_path(), skill_dir_name)
            if os.path.exists(default_skill_path):
                child_skills_directories.append(default_skill_path)

            # Check custom skills directory
            if custom_skills_directory:
                custom_skill_path = os.path.join(
                    custom_skills_directory, skill_dir_name
                )
                if os.path.exists(custom_skill_path):
                    child_skills_directories.append(custom_skill_path)

            # Append Universal Skills to ALL child agents
            if universal_skill_dirs:
                child_skills_directories.extend(universal_skill_dirs)

            if child_skills_directories:
                ts = SkillsToolset(directories=child_skills_directories)
                tag_toolsets.append(ts)
                logger.debug(f"Loaded skills for {tag} from {child_skills_directories}")

            child_agent = Agent(
                name=child_name,
                system_prompt=child_prompt,
                model=model,
                model_settings=settings,
                toolsets=tag_toolsets,
                tool_timeout=DEFAULT_TOOL_TIMEOUT,
            )
            child_agents[tag] = child_agent

        # Create Supervisor Agent
        supervisor_skills_dirs = [get_skills_path()]
        if custom_skills_directory:
            supervisor_skills_dirs.append(custom_skills_directory)
        if universal_skill_dirs:
            supervisor_skills_dirs.extend(universal_skill_dirs)

        supervisor = Agent(
            model=model,
            model_settings=settings,
            system_prompt=system_prompt,
            name=name,
            toolsets=[SkillsToolset(directories=supervisor_skills_dirs)],
            tool_timeout=DEFAULT_TOOL_TIMEOUT,
            deps_type=Any,
        )

        # Register Delegation Tools
        def create_delegation_tool(tag, child_agent, child_name):
            async def delegate(ctx: RunContext[Any], task: str) -> str:
                result = await child_agent.run(task, usage=ctx.usage, deps=ctx.deps)
                return str(result.output)

            delegate.__name__ = f"assign_task_to_{tag}_agent"
            delegate.__doc__ = f"Assign a task related to {tag} to the {child_name}."
            return delegate

        for tag, child_agent in child_agents.items():
            _, child_name = agent_defs[tag]
            tool_func = create_delegation_tool(tag, child_agent, child_name)
            supervisor.tool(tool_func)

        # Register Universal Tools (Workspace, A2A, Scheduler)
        from .tools import register_agent_tools

        register_agent_tools(supervisor)
        return supervisor

    # ── Single Agent Pattern ──
    from universal_skills.skill_utilities import get_universal_skills_path
    from pydantic_ai_skills import SkillsToolset

    # Always load default skills
    skill_dirs = [get_skills_path(), get_universal_skills_path()]

    # Load custom skills if provided
    if custom_skills_directory and os.path.exists(custom_skills_directory):
        logger.debug(f"Loading custom skills {custom_skills_directory}")
        skill_dirs.append(str(custom_skills_directory))
        logger.info(f"Loaded Custom Skills at {custom_skills_directory}")

    skills = SkillsToolset(directories=skill_dirs)
    agent_toolsets.append(skills)
    logger.info(f"Loaded Default Skills at {get_skills_path()}")

    agent = Agent(
        model=model,
        model_settings=settings,
        system_prompt=system_prompt,
        name=name,
        toolsets=agent_toolsets,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        deps_type=Any,
    )

    # Register Universal Tools (Workspace, A2A, Scheduler)
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
    enable_web_ui: bool = DEFAULT_ENABLE_WEB_UI,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
):
    import uvicorn
    from fastapi import FastAPI, Request
    from starlette.responses import Response, StreamingResponse
    from fasta2a import Skill

    print(
        f"Starting {DEFAULT_AGENT_NAME}:"
        f"\tprovider={provider}"
        f"\tmodel={model_id}"
        f"\tbase_url={base_url}"
        f"\tmcp={mcp_url} | {mcp_config}"
        f"\tssl_verify={ssl_verify}"
    )

    agent = create_agent(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        mcp_url=mcp_url,
        mcp_config=mcp_config,
        custom_skills_directory=custom_skills_directory,
        ssl_verify=ssl_verify,
        name=DEFAULT_AGENT_NAME,
        system_prompt=DEFAULT_AGENT_SYSTEM_PROMPT,
    )

    # Always load default skills
    skills = load_skills_from_directory(get_skills_path())
    logger.info(f"Loaded {len(skills)} default skills from {get_skills_path()}")

    # Load custom skills if provided
    if custom_skills_directory and os.path.exists(custom_skills_directory):
        custom_skills = load_skills_from_directory(custom_skills_directory)
        skills.extend(custom_skills)
        logger.info(
            f"Loaded {len(custom_skills)} custom skills from {custom_skills_directory}"
        )

    if not skills:
        skills = [
            Skill(
                id="searxng_agent",
                name="Vector Agent Agent",
                description="General access to Vector Agent search tools",
                tags=["searxng", "search"],
                input_modes=["text"],
                output_modes=["text"],
            )
        ]

    a2a_app = agent.to_a2a(
        name=DEFAULT_AGENT_NAME,
        description=DEFAULT_AGENT_DESCRIPTION,
        version=__version__,
        skills=skills,
        debug=debug,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        processor_task = asyncio.create_task(background_processor(agent))
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
            logger.info("In-memory periodic processor stopped")

    app = FastAPI(
        title=f"{DEFAULT_AGENT_NAME} - A2A + AG-UI Server",
        description=DEFAULT_AGENT_DESCRIPTION,
        debug=debug,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        return {"status": "OK"}

    app.mount("/a2a", a2a_app)

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

        adapter = AGUIAdapter(agent=agent, run_input=run_input, accept=accept)
        event_stream = adapter.run_stream()
        sse_stream = adapter.encode_stream(event_stream)

        return StreamingResponse(
            sse_stream,
            media_type=accept,
        )

    if enable_web_ui:
        web_ui = agent.to_web(instructions=DEFAULT_AGENT_SYSTEM_PROMPT)
        app.mount("/", web_ui)
        logger.info(
            "Starting server on %s:%s (A2A at /a2a, AG-UI at /ag-ui, Web UI: %s)",
            host,
            port,
            "Enabled at /" if enable_web_ui else "Disabled",
        )

    uvicorn.run(
        app,
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
    if isinstance(tags_list, list) and tags_list:
        return tags_list

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


def load_a2a_peers() -> List[Dict[str, str]]:
    """Parse AGENTS.md table into list of dicts."""
    content = load_workspace_file("AGENTS.md")
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
    """Add or update a peer in AGENTS.md table."""
    path = get_workspace_path("AGENTS.md")
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


def append_cron_log(task_id: str, task_name: str, output: str):
    """Append a timestamped entry to CRON_LOG.md."""
    path = get_workspace_path("CRON_LOG.md")
    if not path.exists():
        initialize_workspace()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n### [{ts}] {task_name} (`{task_id}`)\n\n" f"{output.strip()}\n\n" f"---\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)
    logger.info(f"Appended cron log entry for {task_id}")


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
    logger.info(f"Pruned {pruned_count} old cron log entries, kept {max_entries}")


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
    logger.info("In-memory periodic processor started (checks every 60 s)")

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
                        logger.info("Cron log cleanup completed")
                    continue

                # Resolve @file references in prompts
                resolved_prompt = resolve_prompt(task.prompt)

                logger.info(f"Running periodic task → {task.name} (ID: {task.id})")
                result = await agent.run(resolved_prompt)
                output = str(result.output or "")
                if output:
                    logger.info(f"Task result: {output[:200]}...")
                append_cron_log(
                    task_id=task.id,
                    task_name=task.name,
                    output=output or "(no output)",
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
