#!/usr/bin/python
               
from __future__ import annotations

import os
import sys
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

from universal_skills.skill_utilities import (
    resolve_mcp_reference,
    get_universal_skills_path,
)


from .config import *
from .workspace import *
from .scheduler import background_processor
from .base_utilities import (
    to_boolean,
    to_integer,
    to_float,
    to_list,
    to_dict,
    retrieve_package_name,
    GET_DEFAULT_SSL_VERIFY,
    load_env_vars,
    is_loopback_url,
    __version__,
)

                                                                   

from .graph_orchestration import (
    create_graph_agent, 
    get_graph_mermaid, 
    run_graph
)
from .custom_observability import setup_otel
from .model_factory import create_model
from .a2a import discover_agents
from .workspace import get_agent_workspace, initialize_workspace
from .tool_guard import apply_tool_guard_approvals
from .tool_filtering import load_skills_from_directory, skill_matches_tags, filter_tools_by_tag
from .prompt_builder import load_identity, build_system_prompt_from_workspace
from .scheduler import get_cron_tasks_from_md, get_cron_logs_from_md
from .chat_persistence import (
    save_chat_to_disk,
    list_chats_from_disk,
    get_chat_from_disk,
    delete_chat_from_disk,
)

from .models import PeriodicTask

                                 
tasks: List[PeriodicTask] = []
lock = asyncio.Lock()


from pydantic_ai.toolsets.fastmcp import FastMCPToolset

import logging
import warnings
import sys

                                        
warnings.filterwarnings("ignore", message=".*")

logger = logging.getLogger(__name__)

def agent_template():
    """Satisfy repository-manager static validation."""
    print("Agent template accessed", file=sys.stderr)
    return None

def get_http_client(
    ssl_verify: bool = True, timeout: float = 300.0
) -> httpx.AsyncClient | None:
    if not ssl_verify:
        return httpx.AsyncClient(verify=False, timeout=timeout)
    return None



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

    Args:
        provider: LLM provider (openai, anthropic, google, etc.)
        model_id: Model identifier
        base_url: Optional base URL for LLM API
        api_key: Optional API key
        mcp_url: Optional single MCP server URL
        mcp_config: Path to MCP config file (JSON)
        mcp_toolsets: Pre-loaded ToolSets to inject
        custom_skills_directory: Path to additional skills
        ssl_verify: Whether to verify SSL certificates
        name: Name of the agent (or supervisor)
        system_prompt: System prompt for the agent (or supervisor)
        tool_guard_mode: "on" to enable global tool guard, "off" to skip

    Returns:
        A Pydantic AI Agent instance
    """

                                                                  
    agent_toolsets = []

    if mcp_url:
        if DEFAULT_VALIDATION_MODE:
            logger.info(f"VALIDATION_MODE: Skipping MCP connection to {mcp_url}")
        elif is_loopback_url(mcp_url, current_host, current_port):
            logger.warning(f"Loopback Guard: Skipping self-referential MCP connection to {mcp_url}")
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
            logger.info(f"VALIDATION_MODE: Skipping MCP config loading from {mcp_config}")
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
                            local_pkg_config = Path.cwd() / pkg / "agent_data" / mcp_config
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
                    mcp_toolset = [filter_tools_by_tag(s, tool_tags) for s in mcp_toolset]
                
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

    from .models import AgentDeps

    from pydantic_ai import DeferredToolRequests
    from typing import Union, Any

    agent = Agent(
        model=model,
        model_settings=settings,
        name=name,
        output_type=Union[str, DeferredToolRequests] if output_type is None else output_type,
        toolsets=agent_toolsets,
        tool_timeout=DEFAULT_TOOL_TIMEOUT,
        deps_type=AgentDeps,
    )

                                                                                   
                                                                               
    @agent.instructions
    def inject_system_prompt() -> str:
        return system_prompt_str

                                                                     
    if enable_universal_tools:
                                                              
                                                                                       
        from agent_utilities.tools import register_agent_tools

        register_agent_tools(agent, graph_bundle=graph_bundle)

    if tool_guard_mode != "off":
                                                                                   
        apply_tool_guard_approvals(agent)

    return agent



def create_agent_server(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_url: str = DEFAULT_MCP_URL,
    mcp_config: Optional[str] = None,
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
    load_universal_skills: bool = DEFAULT_LOAD_UNIVERSAL_SKILLS,
    load_skill_graphs: bool = DEFAULT_LOAD_SKILL_GRAPHS,
    agent_instance: Optional[Agent] = None,
    graph_bundle: Optional[tuple] = None,
    persistence_type: str = "file",
    persistence_path: Optional[str] = None,
    persistence_dsn: Optional[str] = None,
    persistence_url: Optional[str] = None,
):
    """
    Create and run an agent server with FastAPI and FastMCP.

    If agent_instance is provided, generation attributes (provider, model_id, etc.)
    are bypassed in favor of the existing instantiated agent.
    """
    import uvicorn
    from fasta2a import Skill

    import warnings

                                                                                      
                                                                                                
    warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")

    print(
        f"Starting {DEFAULT_AGENT_NAME}:"
        f"\tprovider={provider}"
        f"\tmodel={model_id}"
        f"\tbase_url={base_url}"
        f"\tmcp={mcp_url} | {mcp_config}"
        f"\tssl_verify={ssl_verify}",
        file=sys.stderr,
    )

    _name = name or DEFAULT_AGENT_NAME

                                               
    if workspace:
        global WORKSPACE_DIR
        WORKSPACE_DIR = workspace
        logger.info(f"Workspace override set to: {workspace}")

    def app_factory() -> FastAPI:
        """Internal factory to build the agent and its web apps."""
        nonlocal enable_otel, enable_web_ui
        skill_dirs = []

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

                                                         
        _agent_instance = agent_instance
        if _agent_instance is None:
            _agent_instance = create_agent(
                provider=provider,
                model_id=model_id,
                base_url=base_url,
                api_key=api_key,
                mcp_url=mcp_url,
                mcp_config=mcp_config,
                custom_skills_directory=custom_skills_directory,
                ssl_verify=ssl_verify,
                name=_name,
                system_prompt=system_prompt,
                debug=debug,
                load_universal_skills=load_universal_skills,
                load_skill_graphs=load_skill_graphs,
                graph_bundle=graph_bundle,
                current_host=host,
                current_port=port,
            )

                                                                           
        if hasattr(_agent_instance, "tools"):
            skills_list = list(_agent_instance.tools.values())
        elif hasattr(_agent_instance, "_function_toolset") and hasattr(
            _agent_instance._function_toolset, "tools"
        ):
            skills_list = list(_agent_instance._function_toolset.tools.values())
        else:
            skills_list = []

                               
        if default_skills_path := get_skills_path():
            skill_dirs.append(default_skills_path)

        try:
            from universal_skills.skill_utilities import get_universal_skills_path

            if load_universal_skills:
                skill_dirs.extend(get_universal_skills_path())
        except ImportError:
            pass

                                        
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

        a2a_app = _agent_instance.to_a2a(
            name=_name,
            description=DEFAULT_AGENT_DESCRIPTION,
            version=__version__,
            skills=skills_list,
            debug=debug,
            **a2a_kwargs,
        )

        @asynccontextmanager
        async def lifespan(app: FastAPI):
                                                                                            
                                                                        
            processor_task = asyncio.create_task(background_processor(_agent_instance))
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

                                                       
                                                     
        app.state.reload_app = None

        @app.get("/health")
        async def health_check():
            return {"status": "OK"}

        app.mount("/a2a", a2a_app)

        @app.post("/ag-ui")
        async def ag_ui_endpoint(request: Request) -> Response:
            """AG-UI endpoint with sideband graph activity support and resumption."""
            from pydantic_ai.ui.ag_ui import AGUIAdapter
            from fastapi.responses import StreamingResponse
            import json
            from uuid import uuid4

                                                                                
                                                                         
            run_id = uuid4().hex
            try:
                                                                                       
                                                                                                      
                body = await request.json()
                if body and (session_id := body.get("session_id") or body.get("run_id")):
                    run_id = session_id
                    logger.info(f"Resuming AG-UI session: {run_id}")
            except Exception:
                pass

                                                                  
            graph_event_queue = asyncio.Queue()
            elicitation_queue = asyncio.Queue()
            
            from .models import AgentDeps
            deps = AgentDeps(
                workspace_path=Path(WORKSPACE_DIR),
                graph_event_queue=graph_event_queue,
                elicitation_queue=elicitation_queue,
                request_id=run_id,
                approval_timeout=DEFAULT_APPROVAL_TIMEOUT,
                provider=DEFAULT_PROVIDER,
                model_id=DEFAULT_MODEL_ID,
                base_url=DEFAULT_LLM_BASE_URL,
                api_key=DEFAULT_LLM_API_KEY
            )
            logger.info(f"AG-UI session context: {run_id}")

                                                                                              
            async def merged_stream():
                adapter = AGUIAdapter(agent=_agent_instance)
                
                                                                     
                                                                              
                
                                                  
                agent_response = await adapter.dispatch_request(request, deps=deps)
                if not isinstance(agent_response, StreamingResponse):
                    yield agent_response.body
                    return

                                
                                                                                                     
                combined_queue = asyncio.Queue()

                async def poll_agent():
                    try:
                        async for chunk in agent_response.body_iterator:
                                                                                                
                                                                                              
                            if chunk.startswith(b'2:') or chunk.startswith(b'9:') or b'"tool_calls"' in chunk:
                                await combined_queue.put(("chunk", chunk))
                                                                      
                                await combined_queue.put(("chunk", b'0 ""\n'))
                            else:
                                await combined_queue.put(("chunk", chunk))
                    except Exception as e:
                        logger.error(f"Agent stream error: {e}")
                    finally:
                        await combined_queue.put(("done", None))

                async def poll_sideband():
                    while True:
                        try:
                                                                                     
                            tasks = [
                                asyncio.create_task(graph_event_queue.get()),
                                asyncio.create_task(elicitation_queue.get()),
                            ]
                            done, pending = await asyncio.wait(
                                tasks, 
                                return_when=asyncio.FIRST_COMPLETED
                            )

                            for task in done:
                                try:
                                    ev = await task
                                                                                                           
                                                                                                                 
                                    if ev:
                                        packet = f"8:{json.dumps(ev)}\n".encode("utf-8")
                                        await combined_queue.put(("chunk", packet))
                                        
                                                                          
                                                                                                            
                                                                                
                                        await combined_queue.put(("chunk", b'0 " "\n'))
                                except Exception as e:
                                    logger.error(f"Error processing sideband event: {e}")
                            
                                                                       
                            for task in pending:
                                task.cancel()
                                try:
                                    await task
                                except asyncio.CancelledError:
                                    pass

                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            logger.error(f"Sideband poller error: {e}")
                            break

                agent_task = asyncio.create_task(poll_agent())
                sideband_task = asyncio.create_task(poll_sideband())

                try:
                    while True:
                        try:
                            msg_type, data = await asyncio.wait_for(combined_queue.get(), timeout=0.1)
                            if msg_type == "done":
                                                                                                  
                                await asyncio.sleep(0.1)
                                if not graph_event_queue.empty() or not elicitation_queue.empty():
                                    continue
                                break
                            yield data
                            combined_queue.task_done()
                        except asyncio.TimeoutError:
                                                                            
                                                                                                 
                                                                    
                            yield b'0 " "\n'
                            if agent_task.done() and combined_queue.empty():
                                break
                            continue
                finally:
                    agent_task.cancel()
                    sideband_task.cancel()

            return StreamingResponse(merged_stream(), content_type="text/plain; charset=utf-8")

        @app.post("/stream")
        async def stream_endpoint(request: Request) -> Response:
            """Generic SSE stream endpoint for graph agents."""
            from fastapi.responses import StreamingResponse
            
            data = await request.json()
            query = data.get("query", data.get("prompt", ""))
            
            if graph_bundle:
                from .graph_orchestration import run_graph_stream
                graph, config = graph_bundle
                return StreamingResponse(
                    run_graph_stream(graph, config, query),
                    media_type="text/event-stream"
                )
            else:
                return JSONResponse({"error": "No graph bundle provided for streaming"}, status_code=400)

                      
        if enable_web_ui is None:
            enable_web_ui = to_boolean(os.getenv("ENABLE_WEB_UI", "False"))

        if custom_web_app is not None:
            web_app = custom_web_app(_agent_instance)
            app.mount(custom_web_mount_path, web_app)
            logger.info(f"Mounted custom web UI at {custom_web_mount_path}")
        elif enable_web_ui:
            try:
                from agent_webui.server import create_agent_web_app

                                                       
                _provider_ui = provider or os.environ.get("PROVIDER") or "openai"
                _model_id_ui = (
                    model_id or os.environ.get("MODEL_ID") or "nvidia/nemotron-3-super"
                )

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
                            "enabled": True,           
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
                    _agent_instance,
                    workspace_helpers=helpers,
                    html_source=html_source,
                    models={_model_id_ui: f"{_provider_ui}:{_model_id_ui}"},
                )
                                                                           
                web_app.state.reload_app = None                     
                app.mount("/", web_app)
                logger.debug("Mounted new standalone agent-web UI dashboard at /")
            except ImportError:
                logger.error(
                    "agent-web package not found. Enhanced UI dashboard disabled."
                )
                                   

        return app

                                   
    reloadable = ReloadableApp(app_factory)

                                                                                 
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



def create_graph_agent_server(
    tag_prompts: dict[str, str] | None = None,
    tag_env_vars: dict[str, str] | None = None,
    mcp_url: str | None = None,
    graph_name: str = "GraphAgent",
    router_model: str = DEFAULT_ROUTER_MODEL,
    agent_model: str = DEFAULT_GRAPH_AGENT_MODEL,
    min_confidence: float = 0.6,
                                                   
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_config: Optional[str] = None,
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
    graph_bundle: Optional[tuple] = None,
    sub_agents: Optional[dict] = None,
    persistence_type: str = "file",
    persistence_path: Optional[str] = None,
    persistence_dsn: Optional[str] = None,
    persistence_url: Optional[str] = None,
):
    """Create and start a graph-based agent server.

    This is the graph equivalent of create_agent_server(). It builds a
    pydantic-graph from the tag→prompt mapping, enhances the system prompt
    with graph routing information, and delegates to create_agent_server().

    Args:
        tag_prompts: Maps domain tag → system prompt for the domain sub-agent.
        tag_env_vars: Maps domain tag → env var name. Auto-generated if None.
        mcp_url: URL of the MCP server. Defaults to http://localhost:{port}/mcp.
        graph_name: Name for the graph.
        router_model: Model for the router (cheap/fast).
        agent_model: Model for domain executors.
        min_confidence: Minimum confidence threshold for routing.
        **kwargs: All remaining args are forwarded to create_agent_server().
    """

    import warnings

                                                                                      
                                                                                                
    warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")

    _mcp_url = mcp_url or os.getenv("MCP_URL")
    if _mcp_url:
        logger.info(f"Graph Agent: Using external MCP server at {_mcp_url}")
    else:
        logger.debug("Graph Agent: No external MCP URL provided.")

    if graph_bundle:
        graph, graph_config = graph_bundle
        tag_prompts = graph_config.get("tag_prompts", {})
        tag_env_vars = graph_config.get("tag_env_vars", {})
        sub_agents = graph_config.get("sub_agents", {})
    else:
        if tag_prompts is None:
            raise ValueError("tag_prompts is required if graph_bundle is not provided")
        graph, graph_config = create_graph_agent(
            tag_prompts=tag_prompts,
            tag_env_vars=tag_env_vars,
            mcp_url=_mcp_url,
            mcp_config=mcp_config,
            name=graph_name,
            router_model=router_model,
            agent_model=agent_model,
            min_confidence=min_confidence,
            sub_agents=sub_agents,
        )

    logger.info(
        f"Graph Agent '{graph_name}' initialized with "
        f"{len(tag_prompts)} domain nodes"
    )
    logger.info(f"Mermaid diagram:\n{get_graph_mermaid(graph, graph_config)}")

                                                   
    domain_list = ", ".join(graph_config["valid_domains"])
    base_prompt = system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT
    graph_prompt = (
        f"{base_prompt}\n\n"
        f"## Graph Orchestration Mode\n"
        f"You have a `run_graph_flow` tool that routes queries through a graph "
        f"orchestrator with specialized domain nodes for: {domain_list}.\n"
        f"Use this tool for domain-specific operations. The graph automatically "
        f"classifies the query, routes to the correct domain node, and executes "
        f"with only that domain's tools loaded for efficiency."
    )

    create_agent_server(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        mcp_url=_mcp_url,
        mcp_config=mcp_config,
        load_universal_skills=True,
        load_skill_graphs=True,
        custom_skills_directory=custom_skills_directory,
        debug=debug,
        host=host,
        port=port,
        enable_web_ui=enable_web_ui,
        custom_web_app=custom_web_app,
        custom_web_mount_path=custom_web_mount_path,
        web_ui_instructions=web_ui_instructions,
        html_source=html_source,
        ssl_verify=ssl_verify,
        name=name,
        system_prompt=graph_prompt,
        enable_otel=enable_otel,
        otel_endpoint=otel_endpoint,
        otel_headers=otel_headers,
        otel_public_key=otel_public_key,
        otel_secret_key=otel_secret_key,
        otel_protocol=otel_protocol,
        workspace=workspace,
        a2a_broker=a2a_broker,
        a2a_broker_url=a2a_broker_url,
        a2a_storage=a2a_storage,
        a2a_storage_url=a2a_storage_url,
        graph_bundle=(graph, graph_config),
        persistence_type=persistence_type,
        persistence_path=persistence_path,
        persistence_dsn=persistence_dsn,
        persistence_url=persistence_url,
    )
