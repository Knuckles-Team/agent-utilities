#!/usr/bin/python

from __future__ import annotations

import os
import logging
import argparse
import httpx
from typing import Any, List, Optional, Union
from pathlib import Path

from pydantic_ai import Agent, ModelSettings, DeferredToolRequests
from pydantic_ai.mcp import (
    load_mcp_servers,
    MCPServerStreamableHTTP,
    MCPServerSSE,
)
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

from universal_skills.skill_utilities import (
    get_universal_skills_path,
)

from .config import (
    DEFAULT_PROVIDER,
    DEFAULT_MODEL_ID,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_MCP_URL,
    DEFAULT_MCP_CONFIG,
    DEFAULT_AGENT_NAME,
    DEFAULT_AGENT_SYSTEM_PROMPT,
    DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    DEFAULT_DEBUG,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_SSL_VERIFY,
    DEFAULT_VALIDATION_MODE,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_PARALLEL_TOOL_CALLS,
    DEFAULT_SEED,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_LOGIT_BIAS,
    DEFAULT_STOP_SEQUENCES,
    DEFAULT_EXTRA_HEADERS,
    DEFAULT_EXTRA_BODY,
    DEFAULT_LOAD_UNIVERSAL_SKILLS,
    DEFAULT_TOOL_TIMEOUT,
    DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
)
from .workspace import (
    get_workspace_path,
    get_skills_path,
)
from .base_utilities import (
    to_boolean,
    retrieve_package_name,
    is_loopback_url,
)
from .model_factory import create_model
from .tool_guard import apply_tool_guard_approvals
from .tool_filtering import (
    skill_matches_tags,
    filter_tools_by_tag,
)
from .prompt_builder import build_system_prompt_from_workspace
from .models import AgentDeps

logger = logging.getLogger(__name__)


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
        "--load-universal-skills",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LOAD_UNIVERSAL_SKILLS,
        help="Enable/Disable loading of all universal-skills by default",
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

    parser.add_argument("--help", action="help", help="Show usage")
    return parser


def create_agent(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = DEFAULT_LLM_BASE_URL,
    api_key: str = DEFAULT_LLM_API_KEY,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: str = DEFAULT_MCP_CONFIG,
    mcp_toolsets: list[Any] = None,
    custom_skills_directory: str = DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
    enable_skills: bool = True,
    enable_universal_tools: bool = True,
    name: str = DEFAULT_AGENT_NAME,
    system_prompt: str = DEFAULT_AGENT_SYSTEM_PROMPT,
    debug: bool = False,
    load_universal_skills: bool = False,
    load_skill_graphs: bool = False,
    tool_tags: Optional[List[str]] = None,
    graph_bundle: tuple = None,
    output_type: Optional[Any] = None,
    current_host: str = None,
    current_port: int = None,
    tool_guard_mode: str = "on",
) -> Agent:
    """
    Create a Pydantic AI Agent
    """

    agent_toolsets = []

    if mcp_url:
        if DEFAULT_VALIDATION_MODE:
            logger.info(f"VALIDATION_MODE: Skipping MCP connection to {mcp_url}")
        elif is_loopback_url(mcp_url, current_host, current_port):
            logger.warning(
                f"Loopback Guard: Skipping self-referential MCP connection to {mcp_url}"
            )
        else:
            try:
                if mcp_url.lower().endswith("/sse"):
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
                agent_toolsets.append(
                    filter_tools_by_tag(server, tool_tags) if tool_tags else server
                )
                logger.info(f"Connected to MCP Server: {mcp_url}")
            except Exception as e:
                logger.error(f"Failed to connect to MCP Server {mcp_url}: {e}")

    if mcp_config:
        if DEFAULT_VALIDATION_MODE:
            logger.info(
                f"VALIDATION_MODE: Skipping MCP config loading from {mcp_config}"
            )
        else:
            try:
                if not os.path.isabs(mcp_config) and "/" not in mcp_config:
                    ws_config = get_workspace_path(mcp_config)
                    if ws_config.exists():
                        mcp_config = str(ws_config)
                        logger.info(f"Loaded MCP config from workspace: {mcp_config}")
                    else:
                        pkg = retrieve_package_name()
                        if pkg and pkg != "agent_utilities":
                            local_pkg_config = (
                                Path.cwd() / pkg / "agent_data" / mcp_config
                            )
                            if local_pkg_config.exists():
                                mcp_config = str(local_pkg_config)
                                logger.info(
                                    f"Prioritizing Local Package Config (agent_data): {mcp_config}"
                                )
                            else:
                                local_pkg_dir = Path.cwd() / pkg / mcp_config
                                if local_pkg_dir.exists():
                                    mcp_config = str(local_pkg_dir)
                                    logger.info(
                                        f"Prioritizing Local Package Config (root): {mcp_config}"
                                    )
                        else:
                            local_config = Path.cwd() / mcp_config
                            if local_config.exists():
                                mcp_config = str(local_config)
                                logger.info(f"Loaded MCP config from cwd: {mcp_config}")

                mcp_toolset = load_mcp_servers(mcp_config)
                for server in mcp_toolset:
                    if hasattr(server, "http_client"):
                        server.http_client = httpx.AsyncClient(
                            verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                        )

                if tool_tags:
                    mcp_toolset = [
                        filter_tools_by_tag(s, tool_tags) for s in mcp_toolset
                    ]

                agent_toolsets.extend(mcp_toolset)
                logger.info(f"Connected to MCP Config: {mcp_config}")
            except Exception as e:
                logger.warning(f"Could not load MCP config {mcp_config}: {e}")

    if mcp_toolsets:
        if DEFAULT_VALIDATION_MODE:
            logger.info("VALIDATION_MODE: Skipping external mcp_toolsets connection")
        else:
            for server in mcp_toolsets:
                if server is None:
                    continue
                if hasattr(server, "http_client") and not getattr(
                    server, "http_client", None
                ):
                    server.http_client = httpx.AsyncClient(
                        verify=ssl_verify, timeout=DEFAULT_TIMEOUT
                    )
            for server in mcp_toolsets:
                if server is None:
                    continue

                ts = None
                if type(server).__name__ == "FastMCP":
                    ts = FastMCPToolset(server)
                else:
                    ts = server

                if tool_tags:
                    ts = filter_tools_by_tag(ts, tool_tags)

                agent_toolsets.append(ts)

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

    from pydantic_ai_skills import SkillsToolset

    if enable_skills:
        skill_dirs = []
        if skills_path := get_skills_path():
            skill_dirs.append(skills_path)

        if load_universal_skills:
            skill_dirs.extend(get_universal_skills_path())

        if load_skill_graphs:
            try:
                from skill_graphs.skill_graph_utilities import get_skill_graphs_path

                skill_dirs.extend(get_skill_graphs_path(default_enabled=True))
            except ImportError:
                pass

        if tool_tags:
            skill_dirs = [d for d in skill_dirs if skill_matches_tags(d, tool_tags)]

        if custom_skills_directory:
            if isinstance(custom_skills_directory, (list, tuple)):
                for d in custom_skills_directory:
                    if d and os.path.exists(d):
                        skill_dirs.append(str(d))
                        logger.info(f"Loaded Custom Skills at {d}")
            elif os.path.exists(custom_skills_directory):
                logger.debug(f"Loading custom skills {custom_skills_directory}")
                skill_dirs.append(str(custom_skills_directory))
                logger.info(f"Loaded Custom Skills at {custom_skills_directory}")

        skills = SkillsToolset(directories=skill_dirs)
        agent_toolsets.append(skills)
        logger.info(f"Loaded {len(skill_dirs)} Skills")

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
        output_type=(
            Union[str, DeferredToolRequests] if output_type is None else output_type
        ),
        toolsets=agent_toolsets,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        deps_type=AgentDeps,
    )

    @agent.instructions
    def inject_system_prompt() -> str:
        return system_prompt_str

    if enable_universal_tools:
        from .tool_registry import register_agent_tools

        register_agent_tools(agent, graph_bundle=graph_bundle)

    if tool_guard_mode != "off":
        apply_tool_guard_approvals(agent)

    return agent
