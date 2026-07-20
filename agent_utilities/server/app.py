import asyncio
import ipaddress
import logging
import os
import re
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from agent_utilities.agent.factory import create_agent
from agent_utilities.core.config import (
    DEFAULT_A2A_BROKER,
    DEFAULT_A2A_CONFIG,
    DEFAULT_A2A_REFRESH_INTERVAL,
    DEFAULT_A2A_STORAGE,
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
    DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    DEFAULT_PORT,
    config,
    setting,
)
from agent_utilities.core.scheduler import background_processor
from agent_utilities.core.workspace import get_skills_path
from agent_utilities.gateway.rate_limit import GatewayRateLimitMiddleware
from agent_utilities.observability.custom_observability import setup_otel
from agent_utilities.prompting.builder import load_identity
from agent_utilities.security.http_boundary import (
    AuthenticationBoundaryMiddleware,
    BoundedRequestBodyMiddleware,
    OriginPolicyMiddleware,
)
from agent_utilities.tools.tool_filtering import load_skills_from_directory

from ..base_utilities import __version__, to_boolean
from .concurrency import AsyncioConcurrencyManager
from .dependencies import inject_reload_app, resolve_model_registry
from .models import ReloadableApp
from .routers import agent_ui, ard, commands, core, human, interop, proxy

logger = logging.getLogger(__name__)
_MODEL_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")


def _csv_values(value: str | None) -> list[str]:
    """Return non-empty, de-duplicated comma-separated configuration values."""
    return list(
        dict.fromkeys(part.strip() for part in (value or "").split(",") if part.strip())
    )


def _is_loopback_listener(host: str | None) -> bool:
    """Return whether *host* can only bind a loopback interface."""
    candidate = (host or "").strip().lower()
    if candidate == "localhost":
        return True
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _http_boundary_settings(host: str | None) -> tuple[list[str], list[str]]:
    """Validate the REST listener boundary and return CORS/Host allowlists.

    This validation deliberately runs while the app is constructed so a remote,
    unauthenticated or Host-header-wildcard deployment fails before it listens.
    """
    jwt_auth = bool(config.auth_jwt_jwks_uri)

    loopback = _is_loopback_listener(host)
    if not loopback and not jwt_auth:
        raise RuntimeError("A non-loopback REST listener requires JWT authentication")
    if not loopback:
        direct_tls = bool(config.server_tls_certfile and config.server_tls_keyfile)
        if bool(config.server_tls_certfile) != bool(config.server_tls_keyfile):
            raise RuntimeError("SERVER_TLS_CERTFILE and SERVER_TLS_KEYFILE are a pair")
        if direct_tls and not (
            Path(str(config.server_tls_certfile)).is_file()
            and Path(str(config.server_tls_keyfile)).is_file()
        ):
            raise RuntimeError("Server TLS material is unavailable")
        if config.server_tls_terminated and not config.server_trusted_proxy_cidrs:
            raise RuntimeError(
                "SERVER_TLS_TERMINATED requires SERVER_TRUSTED_PROXY_CIDRS"
            )
        if not direct_tls and not config.server_tls_terminated:
            raise RuntimeError(
                "A non-loopback REST listener requires direct TLS or a trusted "
                "TLS-terminating ingress"
            )
    if jwt_auth and not (config.auth_jwt_issuer and config.auth_jwt_audience):
        raise RuntimeError(
            "JWT authentication requires explicit AUTH_JWT_ISSUER and AUTH_JWT_AUDIENCE"
        )

    origins = _csv_values(config.allowed_origins)
    if config.cors_allow_credentials and (not origins or "*" in origins):
        raise RuntimeError(
            "CORS_ALLOW_CREDENTIALS requires explicit ALLOWED_ORIGINS without '*'"
        )

    hosts = _csv_values(config.allowed_hosts)
    if not hosts:
        if not loopback:
            raise RuntimeError(
                "A non-loopback REST listener requires an explicit ALLOWED_HOSTS allowlist"
            )
        # ``testserver`` is Starlette's in-process test authority; it does not
        # broaden the network bind, which remains loopback-only here.
        hosts = ["localhost", "127.0.0.1", "[::1]", "testserver"]
    if any(
        "*" in value
        or len(value) > 253
        or any(character in value for character in "/\\@?#\r\n\t ")
        for value in hosts
    ):
        raise RuntimeError("Host-header trust requires exact authorities")
    return origins, hosts


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
    custom_web_app: Callable[[Any], Any] | None = None,
    custom_web_mount_path: str = "/",
    web_ui_instructions: str | None = None,
    html_source: str | Path | None = None,
    name: str | None = None,
    system_prompt: str | None = None,
    enable_otel: bool | None = DEFAULT_ENABLE_OTEL,
    otel_endpoint: str | None = DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    otel_headers: str | None = None,
    otel_public_key: str | None = None,
    otel_secret_key: str | None = None,
    otel_protocol: str | None = DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    workspace: str | None = None,
    a2a_broker: str = DEFAULT_A2A_BROKER,
    a2a_storage: str = DEFAULT_A2A_STORAGE,
    skill_types: list[str] | None = None,
    agent_instance: Any | None = None,
    graph_bundle: tuple[Any, ...] | None = None,
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
        logger.info("Workspace override configured")

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
                name=_name,
                system_prompt=system_prompt,
                debug=debug,
                skill_types=skill_types,
                graph_bundle=graph_bundle,
                isolate_mcp=isolate_mcp,
                mcp_toolsets=mcp_toolsets,
            )
        else:
            from agent_utilities.security.tool_guard import (
                apply_tool_guard_approvals,
            )

            toolsets = list(getattr(_agent_instance, "toolsets", ()) or ())
            if any(
                hasattr(toolset, "list_tools") or hasattr(toolset, "direct_call_tool")
                for toolset in toolsets
            ):
                raise RuntimeError(
                    "prebuilt served agents cannot bind MCP toolsets; use the "
                    "governed agent factory"
                )
            if hasattr(_agent_instance, "toolsets"):
                apply_tool_guard_approvals(_agent_instance)

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
            # CONCEPT:AU-ECO.messaging.native-backend-abstraction — Register PlannerGraphSkill when graph_bundle is available
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
                        "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Registered PlannerGraphSkill as A2A-native skill"
                    )
                except Exception as exc:
                    logger.warning(
                        "PlannerGraphSkill registration failed (exception_type=%s)",
                        type(exc).__name__,
                    )

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

        # CONCEPT:AU-KB-CURRENCY (A2A projection, `04-five-intersections.md`
        # item 1/4: "no epistemic descriptors" on the AgentCard). Advertised
        # unconditionally and additively (never replaces the skills a
        # deployment already registers above) — every agent-utilities server
        # shares the SAME one-engine KG, so `epistemic_status`/`why`/
        # `what_changed` over it (the `graph-query-and-explanation` skill's
        # `explain_provenance_by_ids`/`explain_belief`/`epistemic_status`/
        # `explain_policy` actions) is a real, already-implemented
        # capability of every deployment, not an aspirational one.
        skills_list.append(
            Skill(
                id="epistemic-answer",
                name=f"{_name} Epistemic Answer",
                description=(
                    "Answers epistemic_status/why/what_changed queries over the "
                    "shared knowledge graph: calibrated confidence, evidence/"
                    "source citations, belief justification trees, bitemporal "
                    "valid/tx history, and policy-redaction-aware provenance."
                ),
                tags=["epistemic", "provenance", "confidence", "kg"],
                examples=[
                    "Why do you believe X?",
                    "What is the epistemic status of Y?",
                    "What changed about Z since last week?",
                ],
                input_modes=["text"],
                output_modes=["text"],
            )
        )

        if a2a_broker != "epistemic_graph" or a2a_storage != "epistemic_graph":
            raise ValueError(
                "A2A_BROKER and A2A_STORAGE must both select 'epistemic_graph'"
            )
        from agent_utilities.protocols.a2a_epistemic import (
            agent_to_epistemic_a2a,
            build_epistemic_graph_a2a_backends,
        )

        native_broker, native_storage = build_epistemic_graph_a2a_backends(config)
        a2a_app = agent_to_epistemic_a2a(
            _agent_instance,
            broker=native_broker,
            storage=native_storage,
            name=_name,
            description=DEFAULT_AGENT_DESCRIPTION,
            version=__version__,
            skills=skills_list,
            debug=debug or False,
        )

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            from agent_utilities.core.workspace import resolve_mcp_config_path
            from agent_utilities.mcp.agent_manager import should_sync, sync_mcp_agents

            try:
                _mcp_path = resolve_mcp_config_path(mcp_config)
                if _mcp_path and (enable_acp or should_sync(_mcp_path)):
                    logger.info("Startup sync: ingesting MCP tools")
                    asyncio.create_task(sync_mcp_agents(config_path=_mcp_path))
            except Exception as exc:
                logger.error(
                    "Automatic Knowledge Graph ingestion failed on startup "
                    "(exception_type=%s)",
                    type(exc).__name__,
                )

            # CONCEPT:AU-ECO.messaging.native-backend-abstraction: A2A agent sync and periodic refresh
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
                except Exception as exc:
                    logger.warning(
                        "A2A startup sync failed (exception_type=%s)",
                        type(exc).__name__,
                    )

            processor_task = asyncio.create_task(background_processor(_agent_instance))

            # CONCEPT:AU-OS.scaling.epistemic-dynamic-priority-quota Boot SynthesisEngine daemon
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

                engine = IntelligenceGraphEngine.get_or_create()
                synthesis = SynthesisEngine(engine=engine)

                # CONCEPT:AU-KG.ontology.preload-tbox — preload the bundled ontology TBox into the local
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
                    logger.debug(
                        "TBox preload skipped (exception_type=%s)",
                        type(_tbox_e).__name__,
                    )

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
                        logger.error(
                            "SynthesisEngine error (exception_type=%s)",
                            type(ce).__name__,
                        )
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

        _origins, _hosts = _http_boundary_settings(host)
        if debug and not _is_loopback_listener(host):
            raise RuntimeError("Debug exception responses are restricted to loopback")
        if not _origins and _is_loopback_listener(host):
            scheme = "https" if config.server_tls_certfile else "http"
            listen_port = int(port or DEFAULT_PORT)
            default_port = {"http": 80, "https": 443}[scheme]
            suffix = "" if listen_port == default_port else f":{listen_port}"
            _origins = [
                f"{scheme}://localhost{suffix}",
                f"{scheme}://127.0.0.1{suffix}",
                f"{scheme}://[::1]{suffix}",
            ]
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
        )

        if _origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=_origins,
                allow_credentials=config.cors_allow_credentials,
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=[
                    "Authorization",
                    "Content-Type",
                    "X-Agent-Model-Id",
                ],
            )

        effective_rate = (
            config.gateway_rate_limit
            if config.gateway_rate_limit > 0
            else (50.0 if not _is_loopback_listener(host) else 0.0)
        )

        @app.middleware("http")
        async def _model_override_middleware(request: Request, call_next):
            from ..graph.state import REQUESTED_MODEL_ID_CTX

            values = [
                value
                for key, value in request.scope.get("headers", ())
                if key.lower() == b"x-agent-model-id"
            ]
            if len(values) > 1:
                return JSONResponse(
                    {"error": "invalid model override"},
                    status_code=400,
                    headers={"Cache-Control": "no-store"},
                )
            try:
                header_value = values[0].decode("ascii") if values else None
            except UnicodeDecodeError:
                header_value = None
                valid = False
            else:
                valid = header_value is None or bool(
                    _MODEL_ID_RE.fullmatch(header_value)
                )
            if not valid:
                return JSONResponse(
                    {"error": "invalid model override"},
                    status_code=400,
                    headers={"Cache-Control": "no-store"},
                )
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

        # OS-5.3 process-local run coordination is separate from the durable
        # native A2A broker.  No removed Redis broker shape is detected here.
        app.state.concurrency_manager = AsyncioConcurrencyManager()

        logger.info(
            "Model registry bootstrapped with %d model(s)",
            len(_resolved_registry.models),
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
        # CONCEPT:AU-ORCH.adapter.byok-provider-proxy — BYOK provider-normalizing proxy (/api/proxy/{provider}/stream).
        app.include_router(proxy.router)
        # CONCEPT:AU-KG.memory.live-refreshable-artifact-models — Live Refreshable Artifacts (/api/artifacts...).
        from agent_utilities.gateway.artifacts_api import artifacts_router
        from agent_utilities.knowledge_graph.live_artifacts.kg_source import (
            install_kg_artifact_source,
        )

        app.include_router(artifacts_router)
        # Wire artifact refresh to re-derive from the live KG (falls back to preserve-prior on failure).
        install_kg_artifact_source()
        # CONCEPT:AU-AHE.evaluation.longmemeval-validation-harness — LongMemEval-S validation harness (Quarq HTTP runner compatible).
        from .routers import benchmark

        app.include_router(benchmark.router)

        # CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating / ORCH-1.46 — developer-workspace runtime HTTP surface
        # (/api/runtime/* — create session, post typed actions, SSE the event log).
        from .routers import runtime as runtime_router

        app.include_router(runtime_router.router)

        # SWE-bench harness + failure-driven remediation (AHE-3.22 / AHE-3.23).
        from .routers import swebench as swebench_router

        app.include_router(swebench_router.router)

        # CONCEPT:AU-ECO.connector.git-task-resolver — git issue/PR -> SWE task resolver + webhook ingress.
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
            # CONCEPT:AU-ECO.mcp.usage-cost-observability-surface — usage/cost/observability surface for all 3 UIs.
            app.include_router(usage_router, prefix="/api/observability")
            logger.info(
                "Mounted centralized Gateway API "
                "(Dashboard + Knowledge Graph + Observability)"
            )
        except ImportError as exc:
            logger.error(
                "Failed to load Gateway APIs (exception_type=%s)",
                type(exc).__name__,
            )

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
            logger.info("Mounted custom web UI")
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
                    from ..knowledge_graph.core.engine import (
                        IntelligenceGraphEngine,
                    )

                    backend = IntelligenceGraphEngine.get_or_create().backend
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

        # Add these last so Starlette places them outside every router,
        # middleware installed by the centralized graph API, and mounted
        # A2A/ACP/custom application. Route dependencies do not cover mounts.
        from agent_utilities.security.request_identity import HEALTH_PATHS

        app.add_middleware(
            BoundedRequestBodyMiddleware,
            max_bytes=config.max_upload_size,
        )
        app.add_middleware(
            AuthenticationBoundaryMiddleware,
            exempt_paths=HEALTH_PATHS,
        )
        if effective_rate > 0:
            app.add_middleware(
                GatewayRateLimitMiddleware,
                rate=effective_rate,
                burst=(config.gateway_rate_burst or None),
            )
        app.add_middleware(OriginPolicyMiddleware, allowed_origins=_origins)
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=_hosts)
        if config.server_tls_terminated:
            from agent_utilities.security.http_boundary import (
                TrustedProxyPeerMiddleware,
            )

            app.add_middleware(
                TrustedProxyPeerMiddleware,
                trusted_cidrs=config.server_trusted_proxy_cidrs,
            )

        return app

    reloadable = ReloadableApp(app_factory)
    inject_reload_app(reloadable.app, reloadable)
    return reloadable.app
