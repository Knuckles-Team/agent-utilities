import asyncio
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import anyio
from fastapi import Depends, FastAPI, Request
from pydantic_ai import Agent
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from agent_utilities.agent.factory import create_agent
from agent_utilities.core.config import (
    DEFAULT_A2A_BROKER,
    DEFAULT_A2A_BROKER_URL,
    DEFAULT_A2A_CONFIG,
    DEFAULT_A2A_REFRESH_INTERVAL,
    DEFAULT_A2A_STORAGE,
    DEFAULT_A2A_STORAGE_URL,
    DEFAULT_ACP_SESSION_ROOT,
    DEFAULT_AGENT_DESCRIPTION,
    DEFAULT_AGENT_NAME,
    DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    DEFAULT_DEBUG,
    DEFAULT_ENABLE_ACP,
    DEFAULT_ENABLE_OTEL,
    DEFAULT_ENABLE_WEB_UI,
    DEFAULT_HOST,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MCP_URL,
    DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    DEFAULT_PORT,
    DEFAULT_SSL_VERIFY,
    config,
    setting,
)
from agent_utilities.core.scheduler import background_processor
from agent_utilities.core.workspace import get_skills_path
from agent_utilities.observability.custom_observability import setup_otel
from agent_utilities.prompting.builder import load_identity
from agent_utilities.tools.tool_filtering import load_skills_from_directory

from ..base_utilities import __version__, to_boolean
from .concurrency import AsyncioConcurrencyManager, RedisConcurrencyManager
from .dependencies import inject_reload_app, resolve_model_registry, verify_api_key
from .models import ReloadableApp
from .routers import agent_ui, ard, commands, core, human, interop, proxy

logger = logging.getLogger(__name__)


def build_agent_app(
    provider: str | None = DEFAULT_LLM_PROVIDER,
    model_id: str | None = DEFAULT_LLM_MODEL_ID,
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
    model_registry: Any | None = None,
    a2a_config: str | None = DEFAULT_A2A_CONFIG,
) -> FastAPI:
    """Construct and configure a complete Agent FastAPI application."""
    from fasta2a import Skill

    _name = name or DEFAULT_AGENT_NAME

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
        from agent_utilities.core import workspace as _ws_mod

        _ws_mod.WORKSPACE_DIR = workspace
        logger.info(f"Workspace override set to: {workspace}")

    reloadable: ReloadableApp | None = None

    def app_factory() -> FastAPI:
        nonlocal enable_otel, enable_web_ui
        skill_dirs = []

        if enable_otel is None:
            enable_otel = to_boolean(setting("ENABLE_OTEL", "False"))

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
                tool_guard_mode=setting("TOOL_GUARD_MODE", "on"),
                isolate_mcp=isolate_mcp,
                mcp_toolsets=mcp_toolsets,
            )

        _skill_types = skill_types or []
        from agent_utilities.core.config import DEFAULT_VALIDATION_MODE

        if not DEFAULT_VALIDATION_MODE and (default_skills_path := get_skills_path()):
            skill_dirs.extend(default_skills_path)

        if not DEFAULT_VALIDATION_MODE:
            if "universal" in _skill_types:
                try:
                    from universal_skills.skill_utilities import (
                        get_universal_skills_path,  # type: ignore
                    )

                    skill_dirs.extend(get_universal_skills_path())
                except ImportError:
                    pass

            if "graphs" in _skill_types:
                try:
                    from skill_graphs.skill_graph_utilities import (
                        get_skill_graphs_path,  # type: ignore
                    )

                    skill_dirs.extend(get_skill_graphs_path())
                except ImportError:
                    logger.debug("skill-graphs package not found.")

            if custom_skills_directory and os.path.exists(custom_skills_directory):
                skill_dirs.append(custom_skills_directory)

        skills_list = []
        for d in skill_dirs:
            skills_list.extend(load_skills_from_directory(d))

        enabled_skills = []
        for s in skills_list:
            sid = s.id if hasattr(s, "id") else s.get("id")
            if sid:
                env_var = f"ENABLE_{sid.upper().replace('-', '_')}"
                if setting(env_var, "true").lower() != "false":
                    enabled_skills.append(s)
        skills_list = enabled_skills

        if not skills_list:
            # CONCEPT:ECO-4.0 — Register PlannerGraphSkill when graph_bundle is available
            if graph_bundle is not None:
                try:
                    from ..protocols.a2a_graph_skill import PlannerGraphSkill

                    _graph_obj, _graph_cfg = graph_bundle
                    planner_skill = PlannerGraphSkill(
                        graph=_graph_obj,
                        graph_config=_graph_cfg,
                        mcp_toolsets=_initialized_mcp_toolsets,
                        skill_id="planner",
                        name=f"{_name} Planner",
                        description=f"Graph-backed planning agent for {_name}",
                        tags=["agent", "planner", "graph"],
                    )
                    skills_list.append(planner_skill)
                    logger.info(
                        "[CONCEPT:ECO-4.0] Registered PlannerGraphSkill as A2A-native skill"
                    )
                except Exception as e:
                    logger.warning(f"PlannerGraphSkill registration failed: {e}")

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
                from a2a_redis import RedisBroker  # type: ignore

                a2a_kwargs["broker"] = RedisBroker(
                    url=a2a_broker_url or "redis://localhost:6379"
                )
            except ImportError:
                pass
        elif a2a_broker == "postgres":
            try:
                from a2a_postgres import PostgresBroker  # type: ignore

                a2a_kwargs["broker"] = PostgresBroker(
                    url=a2a_broker_url or "postgresql+asyncpg://localhost:5432/a2a"
                )
            except ImportError:
                pass

        if a2a_storage == "redis":
            try:
                from a2a_redis import RedisStorage  # type: ignore

                a2a_kwargs["storage"] = RedisStorage(
                    url=a2a_storage_url or "redis://localhost:6379"
                )
            except ImportError:
                pass
        elif a2a_storage == "postgres":
            try:
                from a2a_postgres import PostgresStorage  # type: ignore

                a2a_kwargs["storage"] = PostgresStorage(
                    url=a2a_storage_url or "postgresql+asyncpg://localhost:5432/a2a"
                )
            except ImportError:
                pass

        # pydantic-ai v2 removed Agent.to_a2a(); the bridge now lives in fasta2a.
        from fasta2a.pydantic_ai import agent_to_a2a

        a2a_app = agent_to_a2a(
            _agent_instance,
            name=_name,
            description=DEFAULT_AGENT_DESCRIPTION,
            version=__version__,
            skills=skills_list,
            debug=debug or False,
            **a2a_kwargs,
        )

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            from agent_utilities.core.workspace import resolve_mcp_config_path
            from agent_utilities.mcp.agent_manager import should_sync, sync_mcp_agents

            try:
                _mcp_path = resolve_mcp_config_path(mcp_config)
                if _mcp_path and (enable_acp or should_sync(_mcp_path)):
                    logger.info(
                        f"Startup Sync: Ingesting MCP tools from {_mcp_path} to Knowledge Graph..."
                    )
                    asyncio.create_task(sync_mcp_agents(config_path=_mcp_path))
            except Exception as e:
                logger.error(
                    f"Automatic Knowledge Graph ingestion failed on startup: {e}"
                )

            # CONCEPT:ECO-4.0: A2A agent sync and periodic refresh
            _a2a_cfg = a2a_config or config.a2a_config
            if _a2a_cfg:
                try:
                    from agent_utilities.protocols.a2a_config import (
                        periodic_a2a_refresh,
                        sync_a2a_agents,
                    )

                    asyncio.create_task(sync_a2a_agents(config_path=_a2a_cfg))
                    asyncio.create_task(
                        periodic_a2a_refresh(
                            config_path=_a2a_cfg,
                            interval_seconds=DEFAULT_A2A_REFRESH_INTERVAL,
                        )
                    )
                except Exception as e:
                    logger.warning(f"A2A startup sync failed: {e}")

            processor_task = asyncio.create_task(background_processor(_agent_instance))

            # CONCEPT:OS-5.8 Boot SynthesisEngine daemon
            async def run_synthesis_daemon():
                import asyncio

                from agent_utilities.ecosystem.governance_agent import (
                    GraphGovernanceAgent,
                )
                from agent_utilities.knowledge_graph.core.engine import (
                    IntelligenceGraphEngine,
                )
                from agent_utilities.knowledge_graph.memory import (
                    SynthesisEngine,
                )

                engine = IntelligenceGraphEngine()
                synthesis = SynthesisEngine(engine=engine)

                # CONCEPT:KG-2.7 — preload the bundled ontology TBox into the local
                # OWL store at startup so OWL reasoning + the local SPARQL endpoint
                # have the schema immediately (best-effort; a no-op when owlready2
                # /rdflib aren't installed, e.g. the most minimal tiny profile).
                try:
                    from pathlib import Path as _Path

                    import agent_utilities.knowledge_graph as _kgpkg
                    from agent_utilities.knowledge_graph.backends.owl import (
                        create_owl_backend,
                    )

                    _owl = create_owl_backend()
                    _core_ttl = _Path(_kgpkg.__file__).parent / "ontology.ttl"
                    if _owl is not None and _core_ttl.exists():
                        _owl.load_ontology(str(_core_ttl))
                        logger.info(
                            "Preloaded bundled ontology TBox into local OWL store"
                        )
                except Exception as _tbox_e:  # noqa: BLE001 — best-effort
                    logger.debug("TBox preload skipped: %s", _tbox_e)

                # Boot Phase 5 Daemon using the SAME engine
                gov_agent = GraphGovernanceAgent(
                    engine=engine, workspace=workspace or "."
                )
                asyncio.create_task(gov_agent.start())

                while True:
                    try:
                        # Wait 10 seconds before first run to let system boot
                        await asyncio.sleep(10)
                        synthesis.run(dry_run=False)
                        await asyncio.sleep(600)  # Run every 10 minutes
                    except asyncio.CancelledError:
                        break
                    except Exception as ce:
                        logger.error("SynthesisEngine error: %s", ce)
                        await asyncio.sleep(60)

            synthesis_task = asyncio.create_task(run_synthesis_daemon())

            shutdown_event = anyio.Event()

            try:
                # We no longer connect to all MCP servers on startup.
                # Servers are now lazy-loaded on demand during graph execution.
                if hasattr(a2a_app, "router") and hasattr(
                    a2a_app.router, "lifespan_context"
                ):
                    async with a2a_app.router.lifespan_context(a2a_app):
                        yield
                else:
                    yield

                shutdown_event.set()
            finally:
                processor_task.cancel()
                synthesis_task.cancel()
                try:
                    await processor_task
                    await synthesis_task
                except asyncio.CancelledError:
                    pass

        app = FastAPI(
            title=f"{_agent_emoji} {_name} - Agent Server",
            description=_agent_description or "",
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
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=_hosts)

        @app.middleware("http")
        async def _model_override_middleware(request: Request, call_next):
            from ..graph.state import REQUESTED_MODEL_ID_CTX

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

        # Share variables with routers
        app.state.agent_instance = _agent_instance
        app.state.mcp_toolsets = _initialized_mcp_toolsets
        app.state.graph_bundle = graph_bundle
        app.state.agent_name = _name
        app.state.mcp_config = mcp_config

        # OS-5.3 Session Concurrency Manager
        if "broker" in a2a_kwargs and hasattr(a2a_kwargs["broker"], "redis"):
            app.state.concurrency_manager = RedisConcurrencyManager(
                a2a_kwargs["broker"].redis
            )
        else:
            app.state.concurrency_manager = AsyncioConcurrencyManager()

        _default_model = _resolved_registry.get_default()
        logger.info(
            "Model registry bootstrapped with %d model(s); default=%s",
            len(_resolved_registry.models),
            _default_model.id if _default_model else None,
        )
        if graph_bundle is not None and len(_resolved_registry.models) > 0:
            _graph_obj, _graph_cfg = graph_bundle
            if isinstance(_graph_cfg, dict):
                _graph_cfg.setdefault("model_registry", _resolved_registry)

        app.include_router(core.router)
        app.include_router(agent_ui.router)
        app.include_router(interop.router)
        # ARD registry surface (ECO-4.95): /.well-known/ai-catalog.json + /search.
        # Mounted before the optional SPA "/" mount so the well-known path resolves.
        app.include_router(ard.router)
        app.include_router(human.router)
        app.include_router(commands.router)
        # CONCEPT:ORCH-1.34 — BYOK provider-normalizing proxy (/api/proxy/{provider}/stream).
        app.include_router(proxy.router)
        # CONCEPT:KG-2.24 — Live Refreshable Artifacts (/api/artifacts...).
        from agent_utilities.gateway.artifacts_api import artifacts_router
        from agent_utilities.knowledge_graph.live_artifacts.kg_source import (
            install_kg_artifact_source,
        )

        app.include_router(artifacts_router)
        # Wire artifact refresh to re-derive from the live KG (falls back to preserve-prior on failure).
        install_kg_artifact_source()
        # CONCEPT:AHE-3.12 — LongMemEval-S validation harness (Quarq HTTP runner compatible).
        from .routers import benchmark

        app.include_router(benchmark.router)

        # CONCEPT:OS-5.33 / ORCH-1.46 — developer-workspace runtime HTTP surface
        # (/api/runtime/* — create session, post typed actions, SSE the event log).
        from .routers import runtime as runtime_router

        app.include_router(runtime_router.router)

        # SWE-bench harness + failure-driven remediation (AHE-3.22 / AHE-3.23).
        from .routers import swebench as swebench_router

        app.include_router(swebench_router.router)

        # CONCEPT:ECO-4.43 — git issue/PR -> SWE task resolver + webhook ingress.
        from .routers import git_webhooks as git_router

        app.include_router(git_router.router)

        try:
            from agent_utilities.gateway.api import dashboard_router
            from agent_utilities.gateway.graph_api import register_graph_routes
            from agent_utilities.gateway.usage_api import usage_router

            app.include_router(dashboard_router, prefix="/api/dashboard")
            # The full Knowledge Graph REST surface is centralized here (graph-os
            # MCP is now a thin FastMCP wrapper). Routes are mounted under /api/*.
            register_graph_routes(app, prefix="/api")
            # CONCEPT:ECO-4.41 — usage/cost/observability surface for all 3 UIs.
            app.include_router(usage_router, prefix="/api/observability")
            logger.info(
                "Mounted centralized Gateway API "
                "(Dashboard + Knowledge Graph + Observability)"
            )
        except ImportError as e:
            logger.error(f"Failed to load Gateway APIs: {e}")

        if enable_acp:
            from agent_utilities.protocols.acp_adapter import (
                build_acp_config,
                create_graph_acp_app,
                is_acp_available,
            )

            if is_acp_available():
                logger.info("Mounting ACP protocol layer at /acp")
                acp_config = build_acp_config(
                    session_root=Path(acp_session_root) if acp_session_root else None
                )
                acp_app = create_graph_acp_app(
                    _agent_instance,
                    acp_config,
                    graph_bundle=graph_bundle,
                    mcp_toolsets=_initialized_mcp_toolsets,
                    concurrency_manager=app.state.concurrency_manager,
                )
                app.mount("/acp", acp_app)
            else:
                logger.warning("ACP requested but pydantic-acp not installed.")

        app.mount("/a2a", a2a_app)

        if enable_web_ui is None:
            enable_web_ui = to_boolean(setting("ENABLE_WEB_UI", "False"))

        if custom_web_app is not None:
            web_app = custom_web_app(_agent_instance)
            app.mount(custom_web_mount_path, web_app)
            logger.info(f"Mounted custom web UI at {custom_web_mount_path}")
        elif enable_web_ui:
            try:
                from .routers import enhanced

                app.include_router(enhanced.router)

                from agent_webui.server import create_agent_web_app

                from agent_utilities.core.chat_persistence import (
                    delete_chat_from_disk,
                    get_chat_from_disk,
                    list_chats_from_disk,
                    save_chat_to_disk,
                )
                from agent_utilities.core.scheduler import get_cron_logs, get_cron_tasks
                from agent_utilities.core.workspace import (
                    get_agent_icon_path,
                    get_workspace_path,
                    initialize_workspace,
                    list_workspace_files,
                    load_workspace_file,
                    write_md_file,
                    write_workspace_file,
                )

                _provider_ui = provider or setting("PROVIDER") or "openai"
                _model_id_ui = model_id or setting("MODEL_ID") or "google/gemma-4-31b"

                def _graph_native_list_skills():
                    from ..knowledge_graph.backends import create_backend

                    backend = create_backend()
                    if backend is None:
                        return []

                    skills = []
                    with suppress(Exception):
                        prompts = backend.execute(
                            "MATCH (p:Prompt) RETURN p.id AS id, p.name AS name, p.description AS descriptionription_text"
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
                            "MATCH (t:Tool) RETURN t.id AS id, t.name AS name, t.description AS descriptionription_text, t.mcp_server AS server"
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
