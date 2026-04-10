#!/usr/bin/python

from __future__ import annotations

import os
import sys
import json
import logging
import asyncio
import anyio
import httpx
from typing import Any, List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response
from pydantic_ai import Agent
from .config import (
    DEFAULT_ROUTER_MODEL,
    DEFAULT_GRAPH_AGENT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL_ID,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_MCP_URL,
    DEFAULT_AGENT_NAME,
    DEFAULT_AGENT_DESCRIPTION,
    DEFAULT_AGENT_SYSTEM_PROMPT,
    DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    DEFAULT_DEBUG,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_ENABLE_WEB_UI,
    DEFAULT_SSL_VERIFY,
    DEFAULT_ENABLE_OTEL,
    DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    DEFAULT_A2A_BROKER,
    DEFAULT_A2A_BROKER_URL,
    DEFAULT_A2A_STORAGE,
    DEFAULT_A2A_STORAGE_URL,
    DEFAULT_APPROVAL_TIMEOUT,
    DEFAULT_MCP_CONFIG,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_GRAPH_PERSISTENCE_TYPE,
    DEFAULT_GRAPH_PERSISTENCE_PATH,
    DEFAULT_ENABLE_TERMINAL_UI,
)
from .workspace import (
    WORKSPACE_DIR,
    initialize_workspace,
    get_workspace_path,
    load_workspace_file,
    write_workspace_file,
    write_md_file,
    list_workspace_files,
    get_agent_icon_path,
    get_skills_path,
)
from .scheduler import background_processor
from .base_utilities import (
    to_boolean,
    __version__,
)
from .graph_orchestration import (
    create_graph_agent,
    get_graph_mermaid,
)
from .custom_observability import setup_otel
from .tool_filtering import (
    load_skills_from_directory,
)
from .prompt_builder import load_identity
from .scheduler import get_cron_tasks_from_md, get_cron_logs_from_md
from .chat_persistence import (
    save_chat_to_disk,
    list_chats_from_disk,
    get_chat_from_disk,
    delete_chat_from_disk,
)
from .agent_factory import create_agent
from .models import AgentDeps

import warnings

# Filter RequestsDependencyWarning early to prevent log spam
try:
    from requests.exceptions import RequestsDependencyWarning

    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except ImportError:
    pass

# General urllib3/chardet mismatch warnings
warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")
warnings.filterwarnings("ignore", message=".*")

logger = logging.getLogger(__name__)


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


def inject_reload_app(app: FastAPI, reload_app: ReloadableApp):
    """Recursively inject the reloadable app reference into all mounted sub-apps."""
    app.state.reload_app = reload_app
    if hasattr(app, "routes"):
        for route in app.routes:
            if hasattr(route, "app") and isinstance(route.app, FastAPI):
                inject_reload_app(route.app, reload_app)


def build_agent_app(
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
    skill_types: Optional[List[str]] = None,
    agent_instance: Optional[Agent] = None,
    graph_bundle: Optional[tuple] = None,
    persistence_type: str = "file",
    persistence_path: Optional[str] = None,
    persistence_dsn: Optional[str] = None,
    persistence_url: Optional[str] = None,
    enable_terminal_ui: bool = False,
    isolate_mcp: bool = False,
    mcp_toolsets: Optional[List[Any]] = None,
):
    """
    Create and run an agent server with FastAPI and FastMCP.

    If agent_instance is provided, generation attributes (provider, model_id, etc.)
    are bypassed in favor of the existing instantiated agent.
    """
    from fasta2a import Skill

    import warnings

    warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")

    _name = name or DEFAULT_AGENT_NAME

    if workspace:
        from . import workspace as _ws_mod

        _ws_mod.WORKSPACE_DIR = workspace
        logger.info(f"Workspace override set to: {workspace}")

    reloadable: ReloadableApp = None  # Forward declaration for closure

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

        identity_meta = load_identity()
        _agent_description = identity_meta.get("description", DEFAULT_AGENT_DESCRIPTION)
        _agent_emoji = identity_meta.get("emoji", "🤖")

        _agent_instance = agent_instance
        _initialized_mcp_toolsets = []
        if _agent_instance is None:
            _agent_instance, _initialized_mcp_toolsets = create_agent(
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
                skill_types=skill_types,
                graph_bundle=graph_bundle,
                tool_guard_mode="on",
                isolate_mcp=isolate_mcp,
                mcp_toolsets=mcp_toolsets,
            )

        if hasattr(_agent_instance, "tools"):
            skills_list = list(_agent_instance.tools.values())
        elif hasattr(_agent_instance, "_function_toolset") and hasattr(
            _agent_instance._function_toolset, "tools"
        ):
            skills_list = list(_agent_instance._function_toolset.tools.values())
        else:
            skills_list = []

        _skill_types = skill_types or []
        if default_skills_path := get_skills_path():
            skill_dirs.append(default_skills_path)

        if "universal" in _skill_types:
            try:
                from universal_skills.skill_utilities import get_universal_skills_path

                skill_dirs.extend(get_universal_skills_path())
            except ImportError:
                pass

        if "graphs" in _skill_types:
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
            """Lifespan context manager for the agent server."""
            from .mcp_agent_manager import sync_mcp_agents, should_sync
            from .workspace import (
                CORE_FILES,
                get_workspace_path,
                resolve_mcp_config_path,
            )

            try:
                # Use environment-provided MCP config if available, otherwise default workspace path
                _mcp_path = resolve_mcp_config_path(mcp_config)
                _agents_path = get_workspace_path(CORE_FILES["NODE_AGENTS"])

                if _mcp_path and should_sync(_mcp_path, _agents_path):
                    logger.info(
                        f"Registry is stale or missing. Synchronizing MCP agents from {_mcp_path}..."
                    )
                    await sync_mcp_agents(config_path=_mcp_path)
            except Exception as e:
                logger.error(f"Automatic MCP sync failed on startup: {e}")

            processor_task = asyncio.create_task(background_processor(_agent_instance))
            shutdown_event = anyio.Event()

            async def connect_server(server, ready_event):
                srv_id = getattr(server, "id", getattr(server, "name", str(server)))
                if (
                    hasattr(server, "__aenter__")
                    and getattr(server, "_ag_connected", False) is False
                ):
                    try:
                        logger.info(
                            f"Server Startup: Connecting MCP server '{srv_id}'..."
                        )
                        # enter_async_context must happen in the task that will exit it,
                        # OR we use Structured Concurrency (async with) inside this task.
                        async with server:
                            server._ag_connected = True
                            logger.info(
                                f"Server Startup: Successfully connected '{srv_id}'"
                            )
                            ready_event.set()
                            await shutdown_event.wait()
                    except Exception as e:
                        logger.error(
                            f"Server Startup: Failed to connect to MCP server '{srv_id}': {e}"
                        )
                        # Ensure we don't hang if it fails
                        ready_event.set()
                else:
                    ready_event.set()

            try:
                async with anyio.create_task_group() as tg:
                    ready_events = []
                    for s in _initialized_mcp_toolsets:
                        re = anyio.Event()
                        ready_events.append(re)
                        tg.start_soon(connect_server, s, re)

                    # Wait for all servers to signal they are connected (or failed)
                    for re in ready_events:
                        await re.wait()

                    logger.info(
                        f"Server Startup: Connected valid MCP toolsets ({len(_initialized_mcp_toolsets)} total)."
                    )

                    if hasattr(a2a_app, "router") and hasattr(
                        a2a_app.router, "lifespan_context"
                    ):
                        async with a2a_app.router.lifespan_context(a2a_app):
                            yield
                    else:
                        yield

                    # Signal all connection tasks to wrap up
                    shutdown_event.set()
            finally:
                processor_task.cancel()
                try:
                    await processor_task
                except asyncio.CancelledError:
                    pass

        app = FastAPI(
            title=f"{_agent_emoji} {_name} - Agent Server",
            description=_agent_description,
            version=__version__,
            debug=debug,
            lifespan=lifespan,
            openapi_tags=[
                {"name": "Core", "description": "Essential agent lifecycle endpoints"},
                {
                    "name": "Agent UI",
                    "description": "Standard AG-UI and streaming protocols",
                },
                {
                    "name": "Interoperability",
                    "description": "A2A and external bridge endpoints",
                },
            ],
        )

        app.state.reload_app = None

        @app.get("/health", tags=["Core"], summary="Health Check")
        async def health_check():
            """Returns the current status of the agent server."""
            health_info = {
                "status": "OK",
                "agent": _name,
                "version": __version__,
            }
            # Add graph info if available
            if graph_bundle:
                try:
                    from .graph_orchestration import (
                        NODE_SKILL_MAP,
                        load_mcp_agent_registry,
                    )

                    registry = load_mcp_agent_registry()
                    health_info["graph"] = {
                        "skill_agents": len(NODE_SKILL_MAP),
                        "mcp_agents": len(registry.agents),
                        "mcp_tools": len(registry.tools),
                    }
                except Exception:
                    pass
            return health_info

        app.mount("/a2a", a2a_app)

        @app.post("/ag-ui", tags=["Agent UI"], summary="AG-UI Streaming Endpoint")
        async def ag_ui_endpoint(request: Request) -> Response:
            """
            Primary endpoint for the Agent UI. Supports sideband graph activity
            annotations and session resumption.
            """
            from pydantic_ai.ui.ag_ui import AGUIAdapter
            from fastapi.responses import StreamingResponse
            from uuid import uuid4

            run_id = uuid4().hex
            try:
                body = await request.json()
                if body and (
                    session_id := body.get("session_id") or body.get("run_id")
                ):
                    run_id = session_id
                    logger.info(f"Resuming AG-UI session: {run_id}")
            except Exception:
                pass

            graph_event_queue = asyncio.Queue()
            elicitation_queue = asyncio.Queue()

            deps = AgentDeps(
                workspace_path=Path(WORKSPACE_DIR or "."),
                graph_event_queue=graph_event_queue,
                elicitation_queue=elicitation_queue,
                request_id=run_id,
                approval_timeout=DEFAULT_APPROVAL_TIMEOUT,
                provider=DEFAULT_PROVIDER,
                model_id=DEFAULT_MODEL_ID,
                base_url=DEFAULT_LLM_BASE_URL,
                api_key=DEFAULT_LLM_API_KEY,
                mcp_toolsets=_initialized_mcp_toolsets,
            )
            logger.info(f"AG-UI session context: {run_id}")

            async def merged_stream():
                query = ""
                try:
                    body = await request.json()
                    query = body.get("query", body.get("prompt", ""))
                except Exception:
                    pass
                adapter = AGUIAdapter(agent=_agent_instance, run_input=query)

                agent_response = await adapter.dispatch_request(
                    request, agent=_agent_instance, deps=deps
                )
                if not isinstance(agent_response, StreamingResponse):
                    yield agent_response.body
                    return

                combined_queue = asyncio.Queue()

                async def poll_agent():
                    try:
                        async for chunk in agent_response.body_iterator:
                            # Handle both bytes and str chunks gracefully
                            chunk_str = (
                                chunk.decode("utf-8")
                                if isinstance(chunk, bytes)
                                else str(chunk)
                            )

                            if (
                                chunk_str.startswith("2:")
                                or chunk_str.startswith("9:")
                                or '"tool_calls"' in chunk_str
                            ):
                                await combined_queue.put(
                                    (
                                        "chunk",
                                        (
                                            chunk
                                            if isinstance(chunk, bytes)
                                            else chunk.encode("utf-8")
                                        ),
                                    )
                                )
                                # Force immediate flush with an explicit heartbeat
                                await combined_queue.put(("chunk", b'0 " "\n'))
                                await asyncio.sleep(0.01)  # Yield to event loop
                            else:
                                await combined_queue.put(
                                    (
                                        "chunk",
                                        (
                                            chunk
                                            if isinstance(chunk, bytes)
                                            else chunk.encode("utf-8")
                                        ),
                                    )
                                )
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
                                tasks, return_when=asyncio.FIRST_COMPLETED
                            )

                            for task in done:
                                try:
                                    ev = await task
                                    if ev:
                                        packet = f"8:{json.dumps(ev)}\n".encode("utf-8")
                                        await combined_queue.put(("chunk", packet))
                                        # Force immediate flush for sideband annotations
                                        await combined_queue.put(("chunk", b'0 " "\n'))
                                        await asyncio.sleep(0.01)  # Yield to event loop
                                except Exception as e:
                                    logger.error(
                                        f"Error processing sideband event: {e}"
                                    )
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
                            msg_type, data = await asyncio.wait_for(
                                combined_queue.get(), timeout=0.1
                            )
                            if msg_type == "done":
                                await asyncio.sleep(0.1)
                                if (
                                    not graph_event_queue.empty()
                                    or not elicitation_queue.empty()
                                ):
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

            return StreamingResponse(
                merged_stream(),
                media_type="text/plain; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        @app.post("/stream", tags=["Agent UI"], summary="SSE Stream Endpoint")
        async def stream_endpoint(request: Request) -> Response:
            """
            Generic SSE stream endpoint for high-fidelity graph agent execution.
            """
            from fastapi.responses import StreamingResponse

            data = await request.json()
            query = data.get("query", data.get("prompt", ""))
            mode = data.get("mode", "ask")
            topology = data.get("topology", "basic")

            if graph_bundle:
                from .graph_orchestration import run_graph_stream

                graph, config = graph_bundle
                return StreamingResponse(
                    run_graph_stream(
                        graph,
                        config,
                        query,
                        mode=mode,
                        topology=topology,
                        mcp_toolsets=_initialized_mcp_toolsets,
                    ),
                    media_type="text/event-stream",
                )
            else:
                return JSONResponse(
                    {"error": "No graph bundle provided for streaming"}, status_code=400
                )

        @app.post("/bridge", tags=["Interoperability"], summary="IDE Bridge Endpoint")
        async def bridge_endpoint(request: Request) -> JSONResponse:
            """
            Handles asynchronous interactions between the agent and IDE/Human-in-the-loop.
            """
            data = await request.json()
            action = data.get("action")
            run_id = data.get("run_id") or data.get("session_id")

            logger.info(f"Bridge Action: {action} (run_id: {run_id})")

            # TODO: Logic to resume graph persistence or update task status
            return JSONResponse(
                {"status": "received", "action": action, "run_id": run_id}
            )

        @app.get("/chats", tags=["Core"], summary="List Chat History")
        async def list_chats():
            """Returns a list of all stored chat sessions."""
            return list_chats_from_disk()

        @app.get("/chats/{chat_id}", tags=["Core"], summary="Get Chat Details")
        async def get_chat(chat_id: str):
            """Returns the full message history for a specific chat."""
            chat_data = get_chat_from_disk(chat_id)
            if not chat_data:
                return JSONResponse({"error": "Chat not found"}, status_code=404)
            return chat_data

        @app.get(
            "/mcp/config", tags=["Interoperability"], summary="Get MCP Configuration"
        )
        async def get_mcp_config():
            """Returns the current mcp_config.json contents."""
            from .workspace import CORE_FILES as _cf

            mcp_config_path = get_workspace_path(
                _cf.get("MCP_CONFIG", "mcp_config.json")
            )
            if not mcp_config_path.exists():
                # Fallback to local agent_data/mcp_config.json if not in workspace

                mcp_config_path = (
                    Path(__file__).parent / "agent_data" / "mcp_config.json"
                )

            if mcp_config_path.exists():
                try:
                    return json.loads(mcp_config_path.read_text(encoding="utf-8"))
                except Exception:
                    return {"mcpServers": {}}
            return {"mcpServers": {}}

        @app.get(
            "/mcp/tools", tags=["Interoperability"], summary="List Available MCP Tools"
        )
        async def list_mcp_tools():
            """Returns a list of all tools from all connected MCP servers."""
            tools = []
            if hasattr(_agent_instance, "toolsets"):
                for ts in _agent_instance.toolsets:
                    # Skip the SkillsToolset which is handled separately via A2A if needed
                    if type(ts).__name__ == "SkillsToolset":
                        continue

                    # For MCPServer toolsets, we can extract tool info
                    if hasattr(ts, "get_tools"):
                        try:
                            # Some toolsets might be async or require a context
                            ts_tools = ts.get_tools()
                            for t in ts_tools:
                                tools.append(
                                    {
                                        "name": getattr(t, "name", str(t)),
                                        "description": getattr(t, "description", ""),
                                        "tag": getattr(
                                            ts, "name", "mcp"
                                        ),  # Use toolset name as tag
                                    }
                                )
                        except Exception:
                            pass
            return tools

        app.post(
            "/mcp/reload",
            tags=["Interoperability"],
            summary="Hot-reload MCP servers and rebuild graph",
        )

        async def reload_mcp_config():
            """Re-sync MCP agents from config and rebuild graph without restarting"""
            try:
                from .mcp_agent_manager import sync_mcp_agents
                from .workspace import resolve_mcp_config_path
                from .graph_orchestration import load_mcp_agents_registry

                _mcp_cfg_path = resolve_mcp_config_path(mcp_config or "mcp_config.json")
                if _mcp_cfg_path:
                    await sync_mcp_agents(config_path=_mcp_cfg_path)
                registry = load_mcp_agents_registry()
                return {
                    "status": "reloaded",
                    "agents": len(registry.agents),
                    "tools": len(registry.tools),
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}

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
    inject_reload_app(reloadable.app, reloadable)
    return reloadable.app


def create_agent_server(
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_url: Optional[str] = DEFAULT_MCP_URL,
    mcp_config: Optional[str] = DEFAULT_MCP_CONFIG,
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
    skill_types: Optional[List[str]] = None,
    agent_instance: Optional[Agent] = None,
    graph_bundle: Optional[tuple] = None,
    persistence_type: str = "file",
    persistence_path: Optional[str] = None,
    persistence_dsn: Optional[str] = None,
    persistence_url: Optional[str] = None,
    enable_terminal_ui: bool = False,
    isolate_mcp: bool = False,
    mcp_toolsets: Optional[List[Any]] = None,
):
    """
    Create and run an agent server with FastAPI and FastMCP.
    """
    import uvicorn

    print(
        f"Starting {DEFAULT_AGENT_NAME}:"
        f"\tprovider={provider}"
        f"\tmodel={model_id}"
        f"\tbase_url={base_url}"
        f"\tmcp={mcp_url} | {mcp_config}"
        f"\tssl_verify={ssl_verify}",
        file=sys.stderr,
    )

    app = build_agent_app(
        provider=provider,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        mcp_url=mcp_url,
        mcp_config=mcp_config,
        custom_skills_directory=custom_skills_directory,
        debug=debug,
        enable_web_ui=enable_web_ui,
        custom_web_app=custom_web_app,
        custom_web_mount_path=custom_web_mount_path,
        web_ui_instructions=web_ui_instructions,
        html_source=html_source,
        ssl_verify=ssl_verify,
        name=name,
        system_prompt=system_prompt,
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
        skill_types=skill_types,
        agent_instance=agent_instance,
        graph_bundle=graph_bundle,
        persistence_type=persistence_type,
        persistence_path=persistence_path,
        persistence_dsn=persistence_dsn,
        persistence_url=persistence_url,
        isolate_mcp=isolate_mcp,
        mcp_toolsets=mcp_toolsets,
    )

    reloadable = app.state.reload_app

    logger.info(
        "Starting server on %s:%s (A2A at /a2a, AG-UI at /ag-ui, Web UI: %s)",
        host,
        port,
        "Enabled (Dashboard)" if enable_web_ui else "Disabled",
    )

    if enable_terminal_ui:
        import threading
        import time
        import subprocess

        def run_server():
            uvicorn.run(
                reloadable,
                host=host,
                port=port,
                timeout_keep_alive=1800,
                timeout_graceful_shutdown=60,
                log_level="error",  # Suppress server logs in CLI mode
            )

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to be healthy
        url = f"http://{host}:{port}/health"
        max_retries = 10
        for i in range(max_retries):
            try:
                import requests

                resp = requests.get(url, timeout=1)
                if resp.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)

        logger.info(f"Launching Agent Terminal UI connecting to {url}...")
        env = os.environ.copy()
        env["AGENT_URL"] = f"http://{host}:{port}"
        try:
            subprocess.call(["agent-terminal-ui"], env=env)
        except FileNotFoundError:
            print(
                "\nError: 'agent-terminal-ui' command not found. Please install the agent-terminal-ui package."
            )
        except Exception as e:
            print(f"Error launching TUI: {e}")

        return

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
    mcp_url: str | None = DEFAULT_MCP_URL,
    graph_name: str = "GraphAgent",
    router_model: str = DEFAULT_ROUTER_MODEL,
    agent_model: str = DEFAULT_GRAPH_AGENT_MODEL,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    provider: str = DEFAULT_PROVIDER,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL,
    api_key: Optional[str] = DEFAULT_LLM_API_KEY,
    mcp_config: Optional[str] = DEFAULT_MCP_CONFIG,
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
    persistence_type: str = DEFAULT_GRAPH_PERSISTENCE_TYPE,
    persistence_path: Optional[str] = DEFAULT_GRAPH_PERSISTENCE_PATH,
    persistence_dsn: Optional[str] = None,
    persistence_url: Optional[str] = None,
    enable_terminal_ui: bool = DEFAULT_ENABLE_TERMINAL_UI,
    skill_types: Optional[List[str]] = None,
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

    from .workspace import WORKSPACE_DIR as _ws_sentinel

    if workspace:
        from . import workspace as _ws_mod

        _ws_mod.WORKSPACE_DIR = workspace
        logger.info(f"Graph Agent: Workspace set early to {workspace}")
    elif not _ws_sentinel:
        from .workspace import get_agent_workspace

        _auto_ws = get_agent_workspace()
        logger.info(f"Graph Agent: Auto-detected workspace {_auto_ws}")

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
            from .workspace import (
                CORE_FILES,
                get_workspace_path,
                resolve_mcp_config_path,
            )
            from .mcp_agent_manager import should_sync
            from .graph_orchestration import initialize_graph_from_workspace

            _mcp_cfg_path = resolve_mcp_config_path(mcp_config)
            _agents_reg_path = get_workspace_path(CORE_FILES["NODE_AGENTS"])

            if _mcp_cfg_path and should_sync(_mcp_cfg_path, _agents_reg_path):
                from .mcp_agent_manager import sync_mcp_agents

                logger.info(f"Regenerating MCP registry from {_mcp_cfg_path}...")
                asyncio.run(sync_mcp_agents(config_path=_mcp_cfg_path))

            graph, graph_config = initialize_graph_from_workspace(
                mcp_config=mcp_config,
                router_model=router_model,
                agent_model=agent_model,
                workspace=workspace,
            )
            # Fetch tag_prompts from graph_config for logging below
            tag_prompts = graph_config.get("tag_prompts", {})
        else:
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
        f"Graph Agent '{graph_name}' initialized with {len(tag_prompts)} domain nodes"
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
        mcp_config=None,  # Handled robustly by graph_config
        skill_types=skill_types,
        custom_skills_directory=custom_skills_directory,
        debug=debug,
        host=host,
        port=port,
        enable_web_ui=enable_web_ui,
        enable_terminal_ui=enable_terminal_ui,
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
        isolate_mcp=True,
        mcp_toolsets=graph_config.get("mcp_toolsets", []),
    )
