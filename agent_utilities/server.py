#!/usr/bin/python
"""Agent Server Module.

This module provides the core web server functionality for the agent ecosystem.
It uses FastAPI to expose endpoints for AG-UI streaming, A2A communication,
ACP protocol adapters, and management interfaces. It dynamically handles
agent lifecycle, workspace initialization, and observability setup.

CONCEPT:AU-004 Protocol Layer
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import anyio
import httpx

if TYPE_CHECKING:
    from fastapi import FastAPI
import base64
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic_ai import Agent, BinaryContent
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response

from .agent_factory import create_agent
from .approval_manager import ApprovalManager
from .base_utilities import (
    __version__,
    to_boolean,
)
from .chat_persistence import (
    delete_chat_from_disk,
    get_chat_from_disk,
    list_chats_from_disk,
    save_chat_to_disk,
)
from .config import (
    DEFAULT_A2A_BROKER,
    DEFAULT_A2A_BROKER_URL,
    DEFAULT_A2A_STORAGE,
    DEFAULT_A2A_STORAGE_URL,
    DEFAULT_ACP_SESSION_ROOT,
    DEFAULT_AGENT_DESCRIPTION,
    DEFAULT_AGENT_NAME,
    DEFAULT_AGENT_SYSTEM_PROMPT,
    DEFAULT_APPROVAL_TIMEOUT,
    DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    DEFAULT_DEBUG,
    DEFAULT_ENABLE_ACP,
    DEFAULT_ENABLE_OTEL,
    DEFAULT_ENABLE_TERMINAL_UI,
    DEFAULT_ENABLE_WEB_LOGS,
    DEFAULT_ENABLE_WEB_UI,
    DEFAULT_GRAPH_AGENT_MODEL,
    DEFAULT_GRAPH_PERSISTENCE_PATH,
    DEFAULT_GRAPH_PERSISTENCE_TYPE,
    DEFAULT_HOST,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_MCP_CONFIG,
    DEFAULT_MCP_URL,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MODEL_ID,
    DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    DEFAULT_PORT,
    DEFAULT_PROVIDER,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_SSL_VERIFY,
    config,
)
from .custom_observability import setup_otel
from .graph_orchestration import (
    create_graph_agent,
    get_graph_mermaid,
)
from .models import AgentDeps, ModelDefinition, ModelRegistry
from .prompt_builder import load_identity
from .scheduler import (
    background_processor,
    get_cron_logs,
    get_cron_tasks,
)
from .tool_filtering import (
    load_skills_from_directory,
)
from .workspace import (
    WORKSPACE_DIR,
    get_agent_icon_path,
    get_skills_path,
    get_workspace_path,
    initialize_workspace,
    list_workspace_files,
    load_workspace_file,
    write_md_file,
    write_workspace_file,
)

logger = logging.getLogger(__name__)

# Singleton approval manager shared between the graph executor and
# the /api/approve endpoint.  Created once at module import.
_approval_manager = ApprovalManager()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Legacy security dependency — delegates to the unified auth module.

    Kept for backward compatibility.  New code should use
    ``auth.verify_credentials`` instead.

    CONCEPT:AU-011 — Secrets & Authentication
    """

    # Construct a minimal request-like object for the unified verifier
    if not config.enable_api_auth and not config.auth_jwt_jwks_uri:
        return
    if config.enable_api_auth and config.agent_api_key:
        if api_key != config.agent_api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Could not validate credentials",
            )


def setup_server_file_logging(workspace: str | None = None) -> str | None:
    """Configure a file handler for the root logger to capture all server logs.

    Args:
        workspace: Optional workspace directory path.

    Returns:
        The path to the log file if successfully configured, else None.
    """
    from .workspace import WORKSPACE_DIR

    ws = workspace or WORKSPACE_DIR or "."
    log_dir = Path(ws) / "agent_data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "server.log"

    # Setup file handler on root logger to capture everything
    root_logger = logging.getLogger()

    # Remove existing file handlers if any (to avoid duplicates on reload)
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)
    logger.info(f"Server logs redirected to: {log_file}")
    return str(log_file)


async def process_parts(parts: list[dict[str, Any]]) -> list[Any]:
    """Process incoming message parts from the Agent UI.

    Handles text, images, and binary attachments. Images are automatically
    persisted to the active workspace with unique identifiers.

    Args:
        parts: List of raw message part dictionaries from the request.

    Returns:
        A list of processed Pydantic AI message part objects (TextPart, BinaryContent).

    """
    processed = []
    # Avoid circular/heavy imports at top level if needed
    from pydantic_ai.messages import TextPart

    for part in parts:
        if "text" in part:
            processed.append(TextPart(part["text"]))
        elif "image" in part or "binary" in part:
            # Handle base64 image
            img_data = part.get("image") or part.get("binary")
            if not img_data:
                continue
            media_type = part.get("media_type", "image/png")
            if isinstance(img_data, str) and img_data.startswith("data:"):
                # Strip data:image/png;base64,
                _, img_data = img_data.split(",", 1)

            if isinstance(img_data, str):
                raw_bytes = base64.b64decode(img_data)
            else:
                raw_bytes = img_data

            if len(raw_bytes) > config.max_upload_size:
                logger.warning(
                    f"Upload rejected: size {len(raw_bytes)} exceeds limit {config.max_upload_size}"
                )
                continue

            # Save to workspace for persistence
            try:
                from .workspace import WORKSPACE_DIR

                img_dir = Path(WORKSPACE_DIR or ".") / "agent_data" / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                from uuid import uuid4

                img_filename = f"{uuid4().hex}.{media_type.split('/')[-1]}"
                img_path = img_dir / img_filename
                img_path.write_bytes(raw_bytes)
                logger.debug(f"Saved uploaded image to: {img_path}")
            except Exception as e:
                logger.warning(f"Failed to save image to disk: {e}")

            processed.append(BinaryContent(data=raw_bytes, media_type=media_type))
    return processed


def get_http_client(
    ssl_verify: bool = True, timeout: float = 300.0
) -> httpx.AsyncClient | None:
    """Create a configured HTTPX AsyncClient for internal requests.

    Args:
        ssl_verify: Whether to verify SSL certificates.
        timeout: Request timeout in seconds.

    Returns:
        An AsyncClient instance if ssl_verify is False; otherwise None (allowing
        default handlers to handle verification).

    """
    if not ssl_verify:
        return httpx.AsyncClient(verify=False, timeout=timeout)  # nosec B501
    return None


from pydantic import BaseModel


class CodemapRequest(BaseModel):
    """Request schema for codebase codemap generation."""

    prompt: str
    mode: Literal["fast", "smart"] = "smart"


class ReloadableApp:
    """ASGI application wrapper that supports manual hot-reloading.

    This wrapper allows swapping the underlying FastAPI application instance
    at runtime without restarting the physical server process.
    """

    def __init__(self, factory: Callable[[], FastAPI]):
        """Initialize the reloadable application.

        Args:
            factory: A function that returns a fresh FastAPI instance.

        """
        self.factory = factory
        self.app: FastAPI = self.factory()

    async def __call__(self, scope, receive, sender):
        """Standard ASGI entry point."""
        await self.app(scope, receive, sender)

    def reload(self):
        """Execute the factory to replace the current application state."""
        logger.info("Hot-reloading agent application...")
        self.app = self.factory()


def inject_reload_app(app: FastAPI, reload_app: ReloadableApp):
    """Recursively inject a ReloadableApp reference into FastAPI state.

    Args:
        app: The FastAPI application to inject into.
        reload_app: The ReloadableApp instance to reference.

    """
    app.state.reload_app = reload_app
    if hasattr(app, "routes"):
        for route in app.routes:
            if hasattr(route, "app") and isinstance(route.app, FastAPI):
                inject_reload_app(route.app, reload_app)


def _build_model_from_registry(
    registry: ModelRegistry | None, model_id: str | None
) -> Any | None:
    """Resolve ``model_id`` against ``registry`` and build a pydantic-ai Model.

    Used by the protocol-level endpoints (``/ag-ui``, ``/stream``) to build
    a per-turn model override that can be applied via
    ``agent.override(model=...)``. Returns ``None`` when the id is absent
    or missing from the registry, or when the model factory fails — the
    caller is expected to fall through to the agent's default in that
    case, so the header is always soft-honoured.
    """
    if not model_id or registry is None or not getattr(registry, "models", None):
        return None
    definition = registry.get_by_id(model_id)
    if definition is None:
        logger.debug(
            "Requested model id '%s' not found in registry; using default.",
            model_id,
        )
        return None
    try:
        from .model_factory import create_model

        api_key = os.getenv(definition.api_key_env) if definition.api_key_env else None
        return create_model(
            provider=definition.provider,
            model_id=definition.model_id,
            base_url=definition.base_url,
            api_key=api_key,
        )
    except Exception as e:
        logger.warning(
            "Failed to build override model for '%s'; falling back: %s",
            model_id,
            e,
        )
        return None


def resolve_model_registry(
    *,
    registry: ModelRegistry | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> ModelRegistry:
    """Resolve the active model registry.

    Priority order:

    1. Explicit ``registry`` argument (caller-supplied).
    2. ``MODELS_CONFIG`` env var pointing at a JSON/YAML file.
    3. Fallback bootstrap from classic single-model kwargs (``provider``,
       ``model_id``, ``base_url``, ``api_key_env``). The single entry is
       marked ``is_default=True`` and placed in tier ``medium``.
    4. An empty registry if nothing above resolves.

    Args:
        registry: Pre-built registry to use verbatim.
        provider: Fallback provider string for single-model bootstrap.
        model_id: Fallback model identifier for single-model bootstrap.
        base_url: Optional base URL for single-model bootstrap.
        api_key_env: Env var name for the single-model API key. The raw
            key value is not stored in the registry.

    Returns:
        A ``ModelRegistry`` (possibly empty).
    """
    if registry is not None:
        return registry

    cfg_path = os.getenv("MODELS_CONFIG")
    if cfg_path:
        p = Path(cfg_path)
        if p.is_file():
            try:
                return ModelRegistry.load_from_file(p)
            except Exception as e:
                logger.error("Failed to load MODELS_CONFIG from %s: %s", cfg_path, e)

    if model_id:
        _id = f"{provider}:{model_id}" if provider else model_id
        return ModelRegistry(
            models=[
                ModelDefinition(
                    id=_id,
                    name=model_id,
                    provider=provider or "openai",
                    model_id=model_id,
                    base_url=base_url,
                    api_key_env=api_key_env,
                    tier="medium",
                    is_default=True,
                )
            ]
        )

    return ModelRegistry()


def build_agent_app(
    provider: str | None = DEFAULT_PROVIDER,
    model_id: str | None = DEFAULT_MODEL_ID,
    base_url: str | None = DEFAULT_LLM_BASE_URL,
    api_key: str | None = DEFAULT_LLM_API_KEY,
    mcp_url: str | None = DEFAULT_MCP_URL,
    mcp_config: str | None = None,
    custom_skills_directory: str | None = DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    debug: bool | None = DEFAULT_DEBUG,
    host: str | None = DEFAULT_HOST,
    port: int | None = DEFAULT_PORT,
    enable_web_ui: bool | None = DEFAULT_ENABLE_WEB_UI,
    custom_web_app: Callable[[Agent], Any] | None = None,
    custom_web_mount_path: str = "/",
    web_ui_instructions: str | None = None,
    html_source: str | Path | None = None,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
    name: str | None = None,
    system_prompt: str | None = None,
    enable_otel: bool | None = DEFAULT_ENABLE_OTEL,
    otel_endpoint: str | None = DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    otel_headers: str | None = DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    otel_public_key: str | None = DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    otel_secret_key: str | None = DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    otel_protocol: str | None = DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    workspace: str | None = None,
    a2a_broker: str = DEFAULT_A2A_BROKER,
    a2a_broker_url: str | None = DEFAULT_A2A_BROKER_URL,
    a2a_storage: str = DEFAULT_A2A_STORAGE,
    a2a_storage_url: str | None = DEFAULT_A2A_STORAGE_URL,
    skill_types: list[str] | None = None,
    agent_instance: Agent | None = None,
    graph_bundle: tuple[Any, ...] | None = None,
    persistence_type: str = "file",
    persistence_path: str | None = None,
    persistence_dsn: str | None = None,
    persistence_url: str | None = None,
    enable_terminal_ui: bool = False,
    enable_acp: bool = DEFAULT_ENABLE_ACP,
    acp_session_root: str | None = DEFAULT_ACP_SESSION_ROOT,
    isolate_mcp: bool = False,
    mcp_toolsets: list[Any] | None = None,
    model_registry: ModelRegistry | None = None,
) -> FastAPI:
    """Construct and configure a complete Agent FastAPI application.

    This function orchestrates the initialization of the agent, A2A protocol
    discovery, ACP adapters, Web UI mounts, and observability settings. It
    supports dynamic reloading via a factory pattern to handle configuration
    changes without restarting the process.

    Args:
        provider: The primary LLM provider (e.g., 'anthropic', 'openai').
        model_id: Specific model identifier (e.g., 'claude-3-5-sonnet-latest').
        base_url: Optional override for the LLM API base URL.
        api_key: Optional secret key for the LLM provider.
        mcp_url: Optional URL of a single standalone MCP server.
        mcp_config: Path to the mcp_config.json for multiple servers.
        custom_skills_directory: Directory containing local skill definitions.
        debug: Enable verbose logging and debug features.
        host: Host interface to bind the server to.
        port: TCP port to listen on.
        enable_web_ui: Mount the standard Agent Web UI (FastAG-UI).
        custom_web_app: Optional custom Starlette/FastAPI app for the frontend.
        custom_web_mount_path: Path where the frontend should be served.
        web_ui_instructions: Natural language instructions for the UI agent.
        html_source: Path to custom HTML source for the web frontend.
        ssl_verify: Whether to enforce SSL certificate verification.
        name: Human-friendly name for this agent instance.
        system_prompt: Base instructions for the agent's behavior.
        enable_otel: Enable OpenTelemetry tracing and metrics.
        otel_endpoint: OTLP exporter endpoint.
        otel_headers: Custom Opaque headers for the OTLP exporter.
        otel_public_key: Public key for OTLP authentication.
        otel_secret_key: Secret key for OTLP authentication.
        otel_protocol: OTLP transport protocol ('http/protobuf' or 'grpc').
        workspace: Path to the directory used for persistent storage.
        a2a_broker: Protocol for A2A message exchange ('redis', 'postgres').
        a2a_broker_url: Connection string for the A2A broker.
        a2a_storage: Protocol for A2A state storage.
        a2a_storage_url: Connection string for the A2A storage.
        skill_types: List of built-in skill catalogs to load.
        agent_instance: Pre-instantiated Agent to bypass factory creation.
        graph_bundle: Tuple defining state nodes and transitions for the graph.
        persistence_type: Backend for state persistence.
        persistence_path: Directory or DSN for persisting data.
        persistence_dsn: Full database connection string for persistence.
        persistence_url: Alias for persistence_dsn.
    """
    from fasta2a import Skill

    _name = name or DEFAULT_AGENT_NAME

    # For ACP, we want to ensure universal skills and graphs are loaded by default
    # to provide a rich set of slash commands and domain experts.
    if enable_acp and skill_types is None:
        skill_types = [
            "universal",
            "graphs",
            "tdd-methodology",
            "manual_testing",
            "walkthroughs",
        ]
        logger.info(f"ACP Enabled: defaulting skill_types to {skill_types}")
    if workspace:
        from . import workspace as _ws_mod

        _ws_mod.WORKSPACE_DIR = workspace
        logger.info(f"Workspace override set to: {workspace}")

    reloadable: ReloadableApp | None = None  # Forward declaration for closure

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
        _initialized_mcp_toolsets: list[Any] = []
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

        # Enumerate function tools via the public toolsets property
        skills_list = []
        try:
            from pydantic_ai.toolsets.function import FunctionToolset

            for ts in _agent_instance.toolsets:
                if isinstance(ts, FunctionToolset) and hasattr(ts, "tools"):
                    skills_list = list(ts.tools.values())
                    break
        except ImportError:
            pass

        _skill_types = skill_types or []
        if default_skills_path := get_skills_path():
            skill_dirs.extend(default_skills_path)

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
            debug=debug or False,
            **a2a_kwargs,
        )

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Lifespan context manager for the agent server."""
            from .mcp_agent_manager import should_sync, sync_mcp_agents
            from .workspace import (
                resolve_mcp_config_path,
            )

            try:
                # Use environment-provided MCP config if available, otherwise default workspace path
                _mcp_path = resolve_mcp_config_path(mcp_config)

                # Trigger sync on startup if ACP is enabled or if config has changed
                if _mcp_path and (enable_acp or should_sync(_mcp_path)):
                    logger.info(
                        f"Startup Sync: Ingesting MCP tools from {_mcp_path} to Knowledge Graph..."
                    )
                    asyncio.create_task(sync_mcp_agents(config_path=_mcp_path))
            except Exception as e:
                logger.error(
                    f"Automatic Knowledge Graph ingestion failed on startup: {e}"
                )

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
            debug=debug or False,
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
            dependencies=[Depends(verify_api_key)] if config.enable_api_auth else [],
        )

        _origins = (
            [o.strip() for o in config.allowed_origins.split(",")]
            if config.allowed_origins
            else ["*"]
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        _hosts = (
            [h.strip() for h in config.allowed_hosts.split(",")]
            if config.allowed_hosts
            else ["*"]
        )
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=_hosts,
        )

        @app.middleware("http")
        async def _model_override_middleware(request: Request, call_next):
            """Capture the ``x-agent-model-id`` per-turn model override.

            Stores the raw header value on ``request.state.requested_model_id``
            for endpoint handlers (AG-UI, ``/stream``) and in the
            ``REQUESTED_MODEL_ID_CTX`` ContextVar for code paths that have no
            direct request access (ACP's ``run_graph_flow`` closure). Both
            channels are populated unconditionally — validation against
            ``app.state.model_registry`` happens downstream in
            :func:`pick_specialist_model` so an unknown id falls back to
            the default without raising.
            """
            from .graph.state import REQUESTED_MODEL_ID_CTX

            header_value = request.headers.get("x-agent-model-id") or None
            request.state.requested_model_id = header_value
            token = REQUESTED_MODEL_ID_CTX.set(header_value)
            try:
                return await call_next(request)
            finally:
                REQUESTED_MODEL_ID_CTX.reset(token)

        app.state.reload_app = None

        _resolved_registry = resolve_model_registry(
            registry=model_registry,
            provider=provider,
            model_id=model_id,
            base_url=base_url,
        )
        app.state.model_registry = _resolved_registry
        _default_model = _resolved_registry.get_default()
        logger.info(
            "Model registry bootstrapped with %d model(s); default=%s",
            len(_resolved_registry.models),
            _default_model.id if _default_model else None,
        )
        # Propagate the registry into the graph config so that the
        # orchestrator's specialist-spawning path sees the same registry as
        # the /models endpoint.
        if graph_bundle is not None and len(_resolved_registry.models) > 0:
            _graph_obj, _graph_cfg = graph_bundle
            if isinstance(_graph_cfg, dict):
                _graph_cfg.setdefault("model_registry", _resolved_registry)

        @app.get(
            "/models",
            tags=["Core"],
            summary="List Configured Models",
        )
        async def list_configured_models() -> dict[str, Any]:
            """Return the configured model registry.

            Consumers: web-UI model picker + cost table, terminal-UI
            ``/model list``, graph orchestrator's specialist spawner.
            """
            reg = getattr(app.state, "model_registry", None)
            if reg is None:
                return {"models": [], "default_id": None}
            return reg.to_api_payload()

        @app.get("/health", tags=["Core"], summary="Health Check")
        async def health_check():
            """Returns the current status of the agent server."""
            health_info: dict[str, Any] = {
                "status": "OK",
                "agent": _name,
                "version": __version__,
            }
            # Add graph info if available
            if graph_bundle:
                with suppress(Exception):
                    from .graph.config_helpers import get_discovery_registry

                    registry = get_discovery_registry()
                    skill_agents = [
                        a for a in registry.agents if a.agent_type == "prompt"
                    ]
                    mcp_agents = [a for a in registry.agents if a.agent_type == "mcp"]
                    a2a_agents = [a for a in registry.agents if a.agent_type == "a2a"]

                    health_info["graph"] = {
                        "skill_agents": len(skill_agents),
                        "mcp_agents": len(mcp_agents),
                        "a2a_agents": len(a2a_agents),
                        "mcp_tools": sum(len(a.tools) for a in registry.agents),
                    }
            return health_info

        if enable_acp:
            from .acp_adapter import (
                build_acp_config,
                create_graph_acp_app,
                is_acp_available,
            )

            if is_acp_available():
                logger.info("Mounting ACP protocol layer at /acp")
                # Build the standard adapter config
                acp_config = build_acp_config(
                    session_root=Path(acp_session_root) if acp_session_root else None
                )
                # Create graph-backed ACP app so that ACP requests route
                # through the full HSM pipeline (routing, planning, specialists,
                # verification) instead of running a flat agent.
                acp_app = create_graph_acp_app(
                    _agent_instance,
                    acp_config,
                    graph_bundle=graph_bundle,
                    mcp_toolsets=_initialized_mcp_toolsets,
                )
                app.mount("/acp", acp_app)
            else:
                logger.warning("ACP requested but pydantic-acp not installed.")

        app.mount("/a2a", a2a_app)

        @app.post("/ag-ui", tags=["Agent UI"], summary="AG-UI Streaming Endpoint")
        async def ag_ui_endpoint(request: Request) -> Response:
            """Primary streaming endpoint for the Agent UI (FastAG-UI).

            Supports sideband graph activity annotations, session resumption,
            and rich media attachments. This endpoint handles high-fidelity
            SSE streaming with sideband data.

            Returns:
                A StreamingResponse for continuous interaction or a JSONResponse
                on failure.

            """
            try:
                from pydantic_ai.ui.ag_ui import AGUIAdapter
            except ImportError:
                logger.error(
                    "AG-UI: AGUIAdapter not found in pydantic_ai. Ensure pydantic-ai[ag-ui] is installed."
                )
                return JSONResponse(
                    {"status": "error", "message": "AG-UI not available"},
                    status_code=501,
                )
            from uuid import uuid4

            from fastapi.responses import StreamingResponse

            run_id = uuid4().hex
            logger.info(
                f"[LAYER:ACP] AG-UI Request Received. Assigned internal run_id: {run_id}"
            )
            with suppress(Exception):
                body = await request.json()
                if body:
                    session_id = body.get("session_id") or body.get("run_id")
                    if session_id:
                        run_id = session_id
                        logger.info(f"[LAYER:ACP] AG-UI: Resuming session: {run_id}")

            graph_event_queue: asyncio.Queue[Any] = asyncio.Queue()
            elicitation_queue: asyncio.Queue[Any] = asyncio.Queue()

            from .patterns.manager import PatternManager

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
            deps.patterns = PatternManager(deps)
            logger.info(f"AG-UI session context: {run_id}")

            requested_model_id = getattr(request.state, "requested_model_id", None)
            override_model = _build_model_from_registry(
                getattr(app.state, "model_registry", None),
                requested_model_id,
            )

            async def merged_stream():
                from contextlib import nullcontext

                query = ""
                query_parts = []
                with suppress(Exception):
                    body = await request.json()
                    query = body.get("query", body.get("prompt", ""))
                    raw_parts = body.get("parts", [])
                    if raw_parts:
                        query_parts = await process_parts(raw_parts)

                # ── FAST PATH: Direct graph execution (no LLM tool-call hop) ──
                # Controlled by the GRAPH_DIRECT_EXECUTION env var (default: true)
                # Only activates when graph_bundle contains a real Graph with iter()
                from .config import DEFAULT_GRAPH_DIRECT_EXECUTION

                _use_fast_path = False
                if graph_bundle and DEFAULT_GRAPH_DIRECT_EXECUTION:
                    _graph_obj, _ = graph_bundle
                    _use_fast_path = hasattr(_graph_obj, "iter")

                if _use_fast_path:
                    from .agui_emitter import AGUIGraphEmitter
                    from .graph.unified import execute_graph_iter

                    logger.info(
                        "[LAYER:AG-UI] Direct graph execution fast-path "
                        f"for query: '{query[:50]}...'"
                    )
                    graph, graph_cfg = graph_bundle
                    emitter = AGUIGraphEmitter()
                    try:
                        async for event in execute_graph_iter(
                            graph=graph,
                            config=graph_cfg,
                            query=query,
                            run_id=run_id,
                            mode="ask",
                            mcp_toolsets=_initialized_mcp_toolsets,
                            requested_model_id=requested_model_id,
                        ):
                            # Translate graph events to AG-UI wire format
                            for chunk in emitter.translate(event):
                                yield chunk
                            # Forward sideband events from the graph
                            # event queue to the AG-UI stream
                            while not graph_event_queue.empty():
                                ev = graph_event_queue.get_nowait()
                                if ev:
                                    for chunk in emitter._format_sideband(ev):
                                        yield chunk
                    except Exception as e:
                        logger.exception(f"AG-UI direct graph error: {e}")
                        error_data = json.dumps({"type": "error", "error": str(e)})
                        yield f"data: {error_data}\n\n".encode()
                    return

                # ── FALLBACK: Standard AGUIAdapter path (non-graph agents) ──
                run_input = query_parts if query_parts else query
                override_ctx = (
                    _agent_instance.override(model=override_model)
                    if override_model is not None
                    else nullcontext()
                )
                try:
                    with override_ctx:
                        adapter = AGUIAdapter(
                            agent=_agent_instance, run_input=run_input
                        )
                        _q_preview = query[:50]
                        logger.info(
                            "[LAYER:ACP] AG-UI: Dispatching request for "
                            f"query: '{_q_preview}...'"
                        )
                        if override_model is not None:
                            logger.info(
                                "AG-UI: Applying per-turn model override '%s'",
                                requested_model_id,
                            )
                        agent_response = await adapter.dispatch_request(
                            request, agent=_agent_instance, deps=deps
                        )
                    logger.info(
                        "[LAYER:ACP] AG-UI: Dispatch successful. Stream established."
                    )
                except Exception as e:
                    logger.exception(f"AG-UI: Dispatch error: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                    return

                if not isinstance(agent_response, StreamingResponse):
                    yield agent_response.body
                    return

                combined_queue: asyncio.Queue = asyncio.Queue()

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
                                        packet = f"8:{json.dumps(ev)}\n".encode()
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
                        except TimeoutError:
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
            """Generic SSE stream endpoint for high-fidelity graph agent execution."""
            from fastapi.responses import StreamingResponse

            data = await request.json()
            query = data.get("query", data.get("prompt", ""))
            raw_parts = data.get("parts", [])
            query_parts = await process_parts(raw_parts) if raw_parts else []
            mode = data.get("mode", "ask")
            topology = data.get("topology", "basic")
            requested_model_id = getattr(request.state, "requested_model_id", None)

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
                        query_parts=query_parts,
                        requested_model_id=requested_model_id,
                    ),
                    media_type="text/event-stream",
                )
            else:
                return JSONResponse(
                    {"error": "No graph bundle provided for streaming"}, status_code=400
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

        @app.post(
            "/api/approve",
            tags=["Human-in-the-Loop"],
            summary="Resolve a pending tool approval or elicitation",
        )
        async def resolve_approval(request: Request):
            """Resolve a pending approval request from the graph executor.

            Expected JSON body::

                {
                    "request_id": "<id from approval_required event>",
                    "decisions": {
                        "<tool_call_id>": "accept" | "deny",
                        ...
                    },
                    "feedback": "optional text"
                }

            """
            try:
                data = await request.json()
                rid = data.get("request_id") or data.get("id")
                if not rid:
                    return JSONResponse(
                        {"error": "request_id is required"}, status_code=400
                    )
                if _approval_manager.resolve(rid, data):
                    return {"status": "resolved", "request_id": rid}
                return JSONResponse(
                    {"error": "Request not found or already resolved"},
                    status_code=404,
                )
            except Exception as e:
                logger.exception("Approval resolution error")
                return JSONResponse({"error": str(e)}, status_code=500)

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
                        with suppress(Exception):
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
            return tools

        @app.post("/api/codemap", tags=["Core"], summary="Generate a codebase codemap")
        async def generate_codemap_endpoint(payload: CodemapRequest):
            """Generate a task-specific hierarchical codemap artifact."""
            from .knowledge_graph.codemaps import CodemapGenerator
            from .knowledge_graph.engine import IntelligenceGraphEngine

            kg = IntelligenceGraphEngine.get_active()
            if not kg:
                return JSONResponse(
                    {"status": "error", "message": "Knowledge Graph not initialized"},
                    status_code=503,
                )

            generator = CodemapGenerator(kg)
            try:
                artifact = await generator.create(
                    prompt=payload.prompt, mode=payload.mode
                )
                return {
                    "status": "success",
                    "codemap_id": artifact.id,
                    "artifact": artifact.model_dump(),
                }
            except Exception as e:
                logger.exception("Failed to generate codemap")
                return JSONResponse(
                    {"status": "error", "message": str(e)},
                    status_code=500,
                )

        @app.post(
            "/mcp/reload",
            tags=["Interoperability"],
            summary="Hot-reload MCP servers and rebuild graph",
        )
        async def reload_mcp_config():
            """Re-sync MCP agents from config and rebuild graph without restarting."""
            try:
                from .graph_orchestration import load_node_agents_registry
                from .mcp_agent_manager import sync_mcp_agents
                from .workspace import resolve_mcp_config_path

                _mcp_cfg_path = resolve_mcp_config_path(mcp_config or "mcp_config.json")
                if _mcp_cfg_path:
                    await sync_mcp_agents(config_path=_mcp_cfg_path)
                registry = load_node_agents_registry()
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
                    model_id or os.environ.get("MODEL_ID") or "google/gemma-4-31b"
                )

                def _graph_native_list_skills():
                    from .knowledge_graph.backends import create_backend

                    ws_path = get_workspace_path("")
                    db_path = str(ws_path / "knowledge_graph.db")
                    backend = create_backend(db_path=db_path)
                    if backend is None:
                        return []

                    skills = []
                    with suppress(Exception):
                        prompts = backend.execute(
                            "MATCH (p:Prompt) RETURN p.id AS id, "
                            "p.name AS name, "
                            "p.description AS description_text"
                        )
                        for p in prompts:
                            skills.append(
                                {
                                    "id": p.get("id"),
                                    "name": p.get("name"),
                                    "description": p.get("description_text", ""),
                                    "enabled": True,
                                    "type": "prompt",
                                }
                            )

                    with suppress(Exception):
                        tools = backend.execute(
                            "MATCH (t:Tool) RETURN t.id AS id, "
                            "t.name AS name, "
                            "t.description AS description_text, "
                            "t.mcp_server AS server"
                        )
                        for t in tools:
                            server_label = t.get("server", "mcp")
                            desc_text = t.get("description_text", "")
                            skills.append(
                                {
                                    "id": t.get("id"),
                                    "name": t.get("name"),
                                    "description": f"[{server_label}] {desc_text}",
                                    "enabled": True,
                                    "type": "tool",
                                }
                            )

                    return sorted(skills, key=lambda x: x.get("name", "").lower())

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
                    "list_skills": _graph_native_list_skills,
                    "get_cron_calendar": get_cron_tasks,
                    "get_cron_logs": get_cron_logs,
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
                web_app.state.model_registry = _resolved_registry
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
    provider: str | None = DEFAULT_PROVIDER,
    model_id: str | None = DEFAULT_MODEL_ID,
    base_url: str | None = DEFAULT_LLM_BASE_URL,
    api_key: str | None = DEFAULT_LLM_API_KEY,
    mcp_url: str | None = DEFAULT_MCP_URL,
    mcp_config: str | None = DEFAULT_MCP_CONFIG,
    custom_skills_directory: str | None = DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    debug: bool | None = DEFAULT_DEBUG,
    host: str | None = DEFAULT_HOST,
    port: int | None = DEFAULT_PORT,
    enable_web_ui: bool | None = DEFAULT_ENABLE_WEB_UI,
    custom_web_app: Callable[[Agent], Any] | None = None,
    custom_web_mount_path: str = "/",
    web_ui_instructions: str | None = None,
    html_source: str | Path | None = None,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
    name: str | None = None,
    system_prompt: str | None = None,
    enable_otel: bool | None = DEFAULT_ENABLE_OTEL,
    otel_endpoint: str | None = DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    otel_headers: str | None = DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    otel_public_key: str | None = DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    otel_secret_key: str | None = DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    otel_protocol: str | None = DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    workspace: str | None = None,
    a2a_broker: str = DEFAULT_A2A_BROKER,
    a2a_broker_url: str | None = DEFAULT_A2A_BROKER_URL,
    a2a_storage: str = DEFAULT_A2A_STORAGE,
    a2a_storage_url: str | None = DEFAULT_A2A_STORAGE_URL,
    skill_types: list[str] | None = None,
    agent_instance: Agent | None = None,
    graph_bundle: tuple[Any, ...] | None = None,
    persistence_type: str = "file",
    persistence_path: str | None = None,
    persistence_dsn: str | None = None,
    persistence_url: str | None = None,
    enable_terminal_ui: bool = False,
    enable_acp: bool = DEFAULT_ENABLE_ACP,
    acp_session_root: str | None = DEFAULT_ACP_SESSION_ROOT,
    isolate_mcp: bool = False,
    mcp_toolsets: list[Any] | None = None,
    model_registry: ModelRegistry | None = None,
    enable_web_logs: bool = DEFAULT_ENABLE_WEB_LOGS,
):
    """Create and run an agent server with FastAPI and FastMCP."""
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
        mcp_url=mcp_url or "",
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
        model_registry=model_registry,
    )

    reloadable = app.state.reload_app

    logger.info(
        "Enabled (Dashboard)" if enable_web_ui else "Disabled",
    )

    log_file_path = None
    if enable_terminal_ui and enable_web_logs:
        log_file_path = setup_server_file_logging(workspace)

    if enable_terminal_ui:
        import subprocess
        import threading
        import time

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
            with suppress(Exception):
                import requests

                resp = requests.get(url, timeout=1)
                if resp.status_code == 200:
                    break
            time.sleep(0.5)

        logger.info(f"Launching Agent Terminal UI connecting to {url}...")
        env = os.environ.copy()
        env["AGENT_URL"] = f"http://{host}:{port}"
        if log_file_path:
            env["AGENT_LOG_FILE"] = log_file_path
        try:
            subprocess.call(["agent-terminal-ui"], env=env)  # nosec B607
        except FileNotFoundError:
            print(
                "\nError: 'agent-terminal-ui' command not found. Please install the agent-terminal-ui package."
            )
        except Exception as e:
            print(f"Error launching TUI: {e}")

        return

    uvicorn.run(
        reloadable,
        host=host or "0.0.0.0",  # nosec B104
        port=port or 9000,
        timeout_keep_alive=1800,
        timeout_graceful_shutdown=60,
        log_level="debug" if debug else "info",
    )


def create_graph_agent_server(
    tag_prompts: dict[str, str] | None = None,
    tag_env_vars: dict[str, str] | None = None,
    mcp_url: str | None = DEFAULT_MCP_URL,
    graph_name: str = "GraphAgent",
    router_model: str | None = DEFAULT_ROUTER_MODEL,
    agent_model: str | None = DEFAULT_GRAPH_AGENT_MODEL,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    provider: str | None = DEFAULT_PROVIDER,
    model_id: str | None = DEFAULT_MODEL_ID,
    base_url: str | None = DEFAULT_LLM_BASE_URL,
    api_key: str | None = DEFAULT_LLM_API_KEY,
    mcp_config: str | None = DEFAULT_MCP_CONFIG,
    custom_skills_directory: str | None = DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    debug: bool | None = DEFAULT_DEBUG,
    host: str | None = DEFAULT_HOST,
    port: int | None = DEFAULT_PORT,
    enable_web_ui: bool | None = DEFAULT_ENABLE_WEB_UI,
    custom_web_app: Callable[[Agent], Any] | None = None,
    custom_web_mount_path: str = "/",
    web_ui_instructions: str | None = None,
    html_source: str | Path | None = None,
    ssl_verify: bool = DEFAULT_SSL_VERIFY,
    name: str | None = None,
    system_prompt: str | None = None,
    enable_otel: bool | None = DEFAULT_ENABLE_OTEL,
    otel_endpoint: str | None = DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    otel_headers: str | None = DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    otel_public_key: str | None = DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    otel_secret_key: str | None = DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    otel_protocol: str | None = DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    workspace: str | None = None,
    a2a_broker: str = DEFAULT_A2A_BROKER,
    a2a_broker_url: str | None = DEFAULT_A2A_BROKER_URL,
    a2a_storage: str = DEFAULT_A2A_STORAGE,
    a2a_storage_url: str | None = DEFAULT_A2A_STORAGE_URL,
    graph_bundle: tuple[Any, ...] | None = None,
    sub_agents: dict[str, Any] | None = None,
    persistence_type: str = DEFAULT_GRAPH_PERSISTENCE_TYPE,
    persistence_path: str | None = DEFAULT_GRAPH_PERSISTENCE_PATH,
    persistence_dsn: str | None = None,
    persistence_url: str | None = None,
    enable_terminal_ui: bool = DEFAULT_ENABLE_TERMINAL_UI,
    skill_types: list[str] | None = None,
    custom_headers: dict[str, Any] | None = None,
    model_registry: ModelRegistry | None = None,
    enable_web_logs: bool = DEFAULT_ENABLE_WEB_LOGS,
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
        # Inject the singleton approval manager so the graph executor can
        # pause for human-in-the-loop tool approvals.
        graph_config.setdefault("approval_manager", _approval_manager)
        # Attach the resolved model registry so the dispatcher can route
        # per-specialist via pick_for_task().
        if model_registry is not None:
            graph_config["model_registry"] = model_registry
        tag_prompts = graph_config.get("tag_prompts", {})
        tag_env_vars = graph_config.get("tag_env_vars", {})
        sub_agents = graph_config.get("sub_agents", {})
    else:
        if tag_prompts is None:
            from .graph_orchestration import initialize_graph_from_workspace
            from .mcp_agent_manager import should_sync
            from .workspace import (
                get_agent_workspace,
                resolve_mcp_config_path,
            )

            _mcp_cfg_path = resolve_mcp_config_path(mcp_config or "")

            if _mcp_cfg_path and should_sync(_mcp_cfg_path):
                from .mcp_agent_manager import sync_mcp_agents

                logger.info(
                    f"Ingesting MCP tools from {_mcp_cfg_path} to Knowledge Graph..."
                )
                try:
                    asyncio.get_running_loop().create_task(
                        sync_mcp_agents(config_path=_mcp_cfg_path)
                    )
                except RuntimeError:
                    asyncio.run(sync_mcp_agents(config_path=_mcp_cfg_path))

            graph, graph_config = initialize_graph_from_workspace(
                mcp_config=mcp_config,
                router_model=router_model,
                agent_model=agent_model,
                api_key=api_key,
                base_url=_mcp_url,
                custom_headers=custom_headers,
                workspace=workspace,
                ssl_verify=ssl_verify,
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

        if model_registry is not None:
            graph_config["model_registry"] = model_registry

    if tag_prompts:
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
        enable_web_logs=enable_web_logs,
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
        model_registry=model_registry,
    )
