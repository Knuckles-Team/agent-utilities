import atexit
import logging
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from agent_utilities.core.config import (
    DEFAULT_A2A_BROKER,
    DEFAULT_A2A_BROKER_URL,
    DEFAULT_A2A_CONFIG,
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
    DEFAULT_GRAPH_PERSISTENCE_PATH,
    DEFAULT_GRAPH_PERSISTENCE_TYPE,
    DEFAULT_HOST,
    DEFAULT_LITE_LLM_MODEL_ID,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MCP_CONFIG,
    DEFAULT_MCP_URL,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT,
    DEFAULT_OTEL_EXPORTER_OTLP_HEADERS,
    DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL,
    DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY,
    DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY,
    DEFAULT_PORT,
    DEFAULT_ROUTER_MODEL,
    DEFAULT_SSL_VERIFY,
)
from agent_utilities.core.config import (
    DEFAULT_A2A_REFRESH_INTERVAL as DEFAULT_A2A_REFRESH_INTERVAL,
)

from ..models import ModelRegistry
from .app import build_agent_app
from .dependencies import setup_server_file_logging
from .routers.human import _approval_manager

logger = logging.getLogger(__name__)


_CLEANUP_REGISTERED = False


def _resolve_gateway_workers(is_pytest: bool, enable_terminal_ui: bool) -> int:
    """Resolve the effective gateway worker count (CONCEPT:OS-5.23).

    ``GATEWAY_WORKERS`` (AgentConfig ``gateway_workers``) defaults to 1 —
    single process, single event loop, in-process KG daemon: exactly the
    historical behaviour. Forced to 1 under pytest (no forking inside the
    test runner) and with the terminal UI (it owns the foreground process).
    """
    from agent_utilities.core.config import config

    try:
        workers = int(getattr(config, "gateway_workers", 1) or 1)
    except (TypeError, ValueError):
        workers = 1
    if workers <= 1:
        return 1
    if is_pytest:
        return 1
    if enable_terminal_ui:
        logger.warning(
            "GATEWAY_WORKERS=%d ignored: the terminal UI runs a single "
            "in-process server.",
            workers,
        )
        return 1
    return workers


def _bind_gateway_socket(host: str, port: int):
    """Bind the shared pre-fork listen socket (CONCEPT:OS-5.23).

    Bound once in the parent BEFORE forking; every worker serves on the
    inherited socket (the classic pre-fork model — uvicorn's own multiprocess
    supervisor works the same way, but requires an import-string app, which
    the dynamically-built gateway app cannot provide).
    """
    import socket as _socket

    family = _socket.AF_INET6 if ":" in (host or "") else _socket.AF_INET
    sock = _socket.socket(family, _socket.SOCK_STREAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.set_inheritable(True)
    return sock


def _fork_gateway_workers(workers: int, host: str, port: int):
    """Pre-fork ``workers-1`` children sharing one listen socket.

    Returns ``(shared_socket, child_pids)`` — ``child_pids`` is empty in the
    children (each child serves as a worker; the parent is worker 0 and reaps
    the children when its server exits).

    Per-process state notice (CONCEPT:OS-5.23): each worker builds its OWN
    app/engine connections. The KG host role is serialized by the advisory
    flock in :mod:`agent_utilities.knowledge_graph.core.host_lock` — the first
    worker to resolve wins ``host`` (consolidated daemon/ticks); the rest
    self-heal to ``client``. Prometheus metrics and rate-limit buckets are
    per-worker. See ``docs/architecture/gateway_scaling.md``.
    """
    shared_socket = _bind_gateway_socket(host, port)
    logger.warning(
        "GATEWAY_WORKERS=%d: pre-forking %d gateway workers on a shared "
        "listen socket (%s:%s). State is PER-PROCESS: exactly ONE worker wins "
        "the KG host flock and runs the daemon/ticks (the rest are clients); "
        "/metrics scrapes sample one worker; GATEWAY_RATE_LIMIT is effectively "
        "multiplied by the worker count. (CONCEPT:OS-5.23)",
        workers,
        workers,
        host,
        port,
    )
    child_pids: list[int] = []
    for _ in range(workers - 1):
        pid = os.fork()
        if pid == 0:
            return shared_socket, []  # child: build + serve, then os._exit
        child_pids.append(pid)
    return shared_socket, child_pids


def _serve_on_socket(app: Any, sock: Any, host: str, port: int, debug: bool) -> None:
    """Run a single uvicorn server on an already-bound (shared) socket."""
    import uvicorn

    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        timeout_keep_alive=1800,
        timeout_graceful_shutdown=60,
        log_level="debug" if debug else "info",
    )
    uvicorn.Server(server_config).run(sockets=[sock])


def _run_agent_server(
    provider: str | None = DEFAULT_LLM_PROVIDER,
    model_id: str | None = DEFAULT_LLM_MODEL_ID,
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
    a2a_config: str | None = DEFAULT_A2A_CONFIG,
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
    # Force disable terminal UI in tests or non-interactive environments to prevent hangs
    is_pytest = (
        "pytest" in sys.modules
        or "py.test" in sys.modules
        or os.getenv("PYTEST_CURRENT_TEST") is not None
    )
    is_non_interactive = not sys.stdin or not sys.stdin.isatty()
    if is_pytest or is_non_interactive:
        enable_terminal_ui = False

    import uvicorn

    global _CLEANUP_REGISTERED
    if not _CLEANUP_REGISTERED:
        _CLEANUP_REGISTERED = True
        _main_pid = os.getpid()
        _cleanup_done = False

        def _cleanup_child_processes(signum=None, frame=None):
            """Kill all child processes spawned by this server on exit."""
            nonlocal _cleanup_done
            if _cleanup_done:
                return
            _cleanup_done = True
            try:
                import subprocess

                # Find all child PIDs of this process
                result = subprocess.run(  # nosec B603 B607
                    ["pgrep", "-P", str(_main_pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.stdout.strip():
                    for pid_str in result.stdout.strip().split("\n"):
                        try:
                            child_pid = int(pid_str.strip())
                            os.kill(child_pid, signal.SIGTERM)
                            logger.debug(f"Cleaned up child process {child_pid}")
                        except (ProcessLookupError, PermissionError, ValueError):
                            pass  # nosec B110
            except Exception:
                pass  # nosec B110 — best-effort cleanup

        # Register cleanup for both graceful and forced exits
        atexit.register(_cleanup_child_processes)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)
                if prev in (signal.SIG_DFL, signal.SIG_IGN, None):
                    signal.signal(sig, _cleanup_child_processes)
                else:
                    # Chain: call previous handler after cleanup
                    def _chained(signum, frame, _prev=prev):
                        _cleanup_child_processes(signum, frame)
                        if callable(_prev):
                            _prev(signum, frame)

                    signal.signal(sig, _chained)
            except (OSError, ValueError):
                pass  # nosec B110 — signal registration may fail in threads

    print(
        f"Starting {DEFAULT_AGENT_NAME}:"
        f"\tprovider={provider}"
        f"\tmodel={model_id}"
        f"\tbase_url={base_url}"
        f"\tmcp={mcp_url} | {mcp_config}"
        f"\tssl_verify={ssl_verify}",
        file=sys.stderr,
    )

    # Multi-worker readiness (CONCEPT:OS-5.23): fork BEFORE building the app
    # so every worker constructs its own app, engine connections and daemon
    # role (the host flock elects exactly one KG host among the workers).
    # Default GATEWAY_WORKERS=1 keeps the historical single-process path.
    workers = _resolve_gateway_workers(is_pytest, enable_terminal_ui)
    shared_socket = None
    child_pids: list[int] = []
    if workers > 1:
        shared_socket, child_pids = _fork_gateway_workers(
            workers,
            host or "0.0.0.0",
            port or 9000,  # nosec B104
        )
    is_worker_child = shared_socket is not None and not child_pids

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
        a2a_config=a2a_config,
    )

    reloadable = app.state.reload_app

    logger.info(
        "Enabled (Dashboard)" if enable_web_ui else "Disabled",
    )

    log_file_path = None
    if enable_terminal_ui and enable_web_logs:
        log_file_path = setup_server_file_logging(workspace)

    if enable_terminal_ui:
        # TUI is now launched in create_agent_server to avoid blocking.
        # This block is preserved if _run_agent_server is called directly.
        import subprocess
        import threading

        def run_server():
            uvicorn.run(
                reloadable,
                host=host or "0.0.0.0",
                port=port or 8000,
                timeout_keep_alive=1800,
                timeout_graceful_shutdown=60,
                log_level="error",  # Suppress server logs in CLI mode
            )

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        logger.info(
            f"Launching Agent Terminal UI connecting to http://{host}:{port}..."
        )
        env = os.environ.copy()
        env["AGENT_URL"] = f"http://{host}:{port}"
        if log_file_path:
            env["AGENT_LOG_FILE"] = log_file_path

        try:
            subprocess.run(["agent-terminal-ui"], env=env, check=False)  # nosec B607
        except FileNotFoundError:
            print("\nError: 'agent-terminal-ui' command not found.")
        except Exception as e:
            print(f"Error launching TUI: {e}")

        return

    if shared_socket is not None:
        # Pre-fork worker pool (CONCEPT:OS-5.23): every process (parent =
        # worker 0 + forked children) serves on the shared inherited socket.
        try:
            _serve_on_socket(
                reloadable,
                shared_socket,
                host or "0.0.0.0",  # nosec B104
                port or 9000,
                bool(debug),
            )
        finally:
            if is_worker_child:
                os._exit(0)  # never fall back into the caller's stack
            for pid in child_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for pid in child_pids:
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
        return

    uvicorn.run(
        reloadable,
        host=host or "0.0.0.0",  # nosec B104
        port=port or 9000,
        timeout_keep_alive=1800,
        timeout_graceful_shutdown=60,
        log_level="debug" if debug else "info",
    )


def create_agent_server(
    tag_prompts: dict[str, str] | None = None,
    tag_env_vars: dict[str, str] | None = None,
    mcp_url: str | None = DEFAULT_MCP_URL,
    graph_name: str = "GraphAgent",
    router_model: str | None = DEFAULT_ROUTER_MODEL,
    agent_model: str | None = DEFAULT_LITE_LLM_MODEL_ID,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    provider: str | None = DEFAULT_LLM_PROVIDER,
    model_id: str | None = DEFAULT_LLM_MODEL_ID,
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
    """Create and start an agent server with graph-based orchestration.

    This is the standard entry point for all agent servers. It builds a
    pydantic-graph from the tag→prompt mapping, enhances the system prompt
    with graph routing information, and starts the server.
    """
    # Force disable terminal UI in tests or non-interactive environments to prevent hangs
    is_pytest = (
        "pytest" in sys.modules
        or "py.test" in sys.modules
        or os.getenv("PYTEST_CURRENT_TEST") is not None
    )
    is_non_interactive = not sys.stdin or not sys.stdin.isatty()
    if is_pytest or is_non_interactive:
        enable_terminal_ui = False

    from agent_utilities.core.workspace import WORKSPACE_DIR as _ws_sentinel

    if workspace:
        from agent_utilities.core import workspace as _ws_mod

        _ws_mod.WORKSPACE_DIR = workspace
        logger.info(f"Graph Agent: Workspace set early to {workspace}")
    elif not _ws_sentinel:
        from agent_utilities.core.workspace import get_agent_workspace

        _auto_ws = get_agent_workspace()
        logger.info(f"Graph Agent: Auto-detected workspace {_auto_ws}")

    if enable_terminal_ui:
        import subprocess
        import threading

        logger.info(
            f"Launching Agent Terminal UI connecting to http://{host or '0.0.0.0'}:{port or 9000}..."
        )
        env = os.environ.copy()
        env["AGENT_URL"] = f"http://{host or '0.0.0.0'}:{port or 9000}"

        if enable_web_logs:
            from agent_utilities.server.dependencies import setup_server_file_logging

            log_file_path = setup_server_file_logging(workspace)
            if log_file_path:
                env["AGENT_LOG_FILE"] = log_file_path

        # Preserve original terminal descriptors for the TUI
        try:
            terminal_stdout_fd = os.dup(1)
            terminal_stderr_fd = os.dup(2)
            terminal_stdin_fd = os.dup(0)
        except OSError:
            terminal_stdout_fd, terminal_stderr_fd, terminal_stdin_fd = None, None, None

        def run_tui():
            try:
                kwargs: dict[str, Any] = {}
                if (
                    terminal_stdout_fd is not None
                    and terminal_stdin_fd is not None
                    and terminal_stderr_fd is not None
                ):
                    kwargs["stdin"] = terminal_stdin_fd
                    kwargs["stdout"] = terminal_stdout_fd
                    kwargs["stderr"] = terminal_stderr_fd
                    kwargs["pass_fds"] = (
                        terminal_stdin_fd,
                        terminal_stdout_fd,
                        terminal_stderr_fd,
                    )

                subprocess.run(["agent-terminal-ui"], env=env, check=False, **kwargs)  # type: ignore[call-overload]  # nosec B607
            except FileNotFoundError:
                print("Error: agent-terminal-ui command not found.")
            import os
            import signal

            os.kill(os.getpid(), signal.SIGTERM)

        tui_thread = threading.Thread(target=run_tui, daemon=False)
        tui_thread.start()
        # Disable it in create_agent_server so it's not launched twice
        enable_terminal_ui = False

        if enable_web_logs and "log_file_path" in locals() and log_file_path:
            import time

            time.sleep(0.5)  # Allow TUI to start
            try:
                fd = os.open(log_file_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
                os.dup2(fd, 1)
                os.dup2(fd, 2)
                os.close(fd)
            except OSError:
                pass

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

            from agent_utilities.core.config import DEFAULT_VALIDATION_MODE

            if (
                _mcp_cfg_path
                and should_sync(_mcp_cfg_path)
                and not DEFAULT_VALIDATION_MODE
            ):
                from agent_utilities.mcp.agent_manager import sync_mcp_agents

                logger.info(
                    f"Ingesting MCP tools from {_mcp_cfg_path} to Knowledge Graph in background..."
                )
                try:
                    asyncio.get_running_loop().create_task(
                        sync_mcp_agents(config_path=_mcp_cfg_path)
                    )
                except RuntimeError:
                    import threading

                    def _run_sync():
                        asyncio.run(sync_mcp_agents(config_path=_mcp_cfg_path))

                    t = threading.Thread(target=_run_sync, daemon=True)
                    t.start()

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

    _run_agent_server(
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
