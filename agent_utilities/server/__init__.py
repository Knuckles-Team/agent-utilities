import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from agent_utilities.core.config import (
    DEFAULT_A2A_BROKER,
    DEFAULT_A2A_BROKER_URL,
    DEFAULT_A2A_STORAGE,
    DEFAULT_A2A_STORAGE_URL,
    DEFAULT_ACP_SESSION_ROOT,
    DEFAULT_AGENT_NAME,
    DEFAULT_AGENT_SYSTEM_PROMPT,
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
)

from ..models import ModelRegistry
from .app import build_agent_app
from .dependencies import setup_server_file_logging
from .routers.human import _approval_manager

logger = logging.getLogger(__name__)


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
    from contextlib import suppress

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
            subprocess.run(["agent-terminal-ui"], env=env, check=False)  # nosec B607
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
    """
    from agent_utilities.core.workspace import WORKSPACE_DIR as _ws_sentinel

    if workspace:
        from agent_utilities.core import workspace as _ws_mod

        _ws_mod.WORKSPACE_DIR = workspace
        logger.info(f"Graph Agent: Workspace set early to {workspace}")
    elif not _ws_sentinel:
        from agent_utilities.core.workspace import get_agent_workspace

        _auto_ws = get_agent_workspace()
        logger.info(f"Graph Agent: Auto-detected workspace {_auto_ws}")

    _mcp_url = mcp_url or os.getenv("MCP_URL")
    if _mcp_url:
        logger.info(f"Graph Agent: Using external MCP server at {_mcp_url}")
    else:
        logger.debug("Graph Agent: No external MCP URL provided.")

    if graph_bundle:
        graph, graph_config = graph_bundle
        graph_config.setdefault("approval_manager", _approval_manager)
        if model_registry is not None:
            graph_config["model_registry"] = model_registry
        tag_prompts = graph_config.get("tag_prompts", {})
        tag_env_vars = graph_config.get("tag_env_vars", {})
        sub_agents = graph_config.get("sub_agents", {})
    else:
        if tag_prompts is None:
            import asyncio

            from agent_utilities.core.workspace import resolve_mcp_config_path
            from agent_utilities.mcp.agent_manager import should_sync

            from ..graph_orchestration import initialize_graph_from_workspace

            _mcp_cfg_path = resolve_mcp_config_path(mcp_config or "")

            if _mcp_cfg_path and should_sync(_mcp_cfg_path):
                from agent_utilities.mcp.agent_manager import sync_mcp_agents

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
            tag_prompts = graph_config.get("tag_prompts", {})
        else:
            from ..graph import create_graph_agent

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

    from ..graph import get_graph_mermaid

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
