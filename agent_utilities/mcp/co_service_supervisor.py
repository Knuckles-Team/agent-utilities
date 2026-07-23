"""Self-composing co-service supervisor for the ``graph-os`` entrypoint.

``uvx agent-utilities graph-os`` (equally the plain ``graph-os`` console script)
brings up graph-os PLUS whatever co-services the already-loaded ``AgentConfig``
says are configured — today that is the messaging inbound router
(:mod:`agent_utilities.messaging.daemon`) and, only when this process would
otherwise leave the KG without ANY host, the consolidated KG host daemon
(:mod:`agent_utilities.gateway.daemon`). No separate deployment, no manual second
command, and — critically — **no new environment variable**: every decision below
reads a setting that already exists for another reason (``TELEGRAM_BOT_TOKEN`` et
al. via :func:`agent_utilities.messaging.daemon.configured_platforms`,
``KG_DAEMON_ROLE`` via :func:`agent_utilities.knowledge_graph.core.engine_tasks.daemon_role`,
and the live host-lock file via
:func:`agent_utilities.knowledge_graph.core.host_lock.host_daemon_running`).

Detection signals
-----------------
* **messaging** — configured iff :func:`agent_utilities.messaging.daemon.configured_platforms`
  returns at least one platform (a real token/app id is present). This is the
  exact same check the standalone ``agent-utilities-messaging`` process already
  uses to decide whether to serve or exit immediately.
* **host daemon** — needed iff this process's *configured* ``KG_DAEMON_ROLE`` is
  explicitly ``client`` (the production/deployed convention — see
  ``deploy/swarm/graphos.stack.yml``, which pairs a client-role ``graph-os`` with a
  dedicated ``graph-os-daemon`` host) **and** no host is currently holding the
  singleton lock (:func:`host_daemon_running`). An ad hoc local/``uvx`` run shares
  that same client-role config (inherited from the user's XDG ``config.json``) but
  has no separately-running ``graph-os-daemon`` process — without this, the graph
  would silently receive no queue drain / maintenance sweeps / KG_LOOP evolution
  cycle for the entire session. When the role is left at its default (``auto``) or
  explicitly ``host``, nothing is needed here: the engine's own construction
  already self-elects host via the same lock (CONCEPT:EG-KG.storage.nonblocking-checkpoint / OS-5.9)
  the moment ``graph-os`` builds it.
* **agent-webui** — configured iff ``config.enable_web_ui`` (the existing
  ``ENABLE_WEB_UI`` field). It is a separate Node/Vite frontend, not a Python
  asyncio task, so it can never be started IN-PROCESS here; it is still reported
  as part of the composition (:func:`detect_composition`) so the multi-backend
  deployment planners (container/kubernetes) can include it, but this in-process
  supervisor only logs that it is configured and external.

STDIO safety
------------
On the ``stdio`` transport, stdout IS the JSON-RPC channel. Every co-service here
is driven on its OWN dedicated thread with its OWN asyncio event loop — never
sharing the FastMCP server's loop — and relies on
:func:`agent_utilities.mcp.server_factory.protect_stdio_jsonrpc` (already applied
by ``kg_server.mcp_server()`` before any co-service is started) to keep every
``print``/warning process-wide pinned to stderr. Co-service loggers use the
standard library's default ``StreamHandler`` target (stderr), so nothing here
writes to stdout under either transport.

Supervision + shutdown
-----------------------
A co-service that exits (cleanly or via an exception) while the supervisor has
not been asked to stop is a bug we must never let hide silently — this project
has already had two services die unnoticed for days (one 9 days / 2434
restarts). :class:`CoServiceSupervisor` logs every crash loudly and restarts the
service with bounded, backed-off retries (:data:`_MAX_RESTARTS` within
:data:`_RESTART_WINDOW_SECONDS`); once the bound is hit it stops retrying and
leaves a loud error in the log rather than crash-looping forever.
:meth:`CoServiceSupervisor.stop_all` is invoked from ``kg_server.py``'s shutdown
path (SIGTERM/SIGINT via ``mcp.run`` and the process ``finally`` block) and joins
every co-service thread before the process exits.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Bounded restart: at most this many restarts within the rolling window before a
# crash-looping co-service is left stopped (loudly) instead of spun forever.
_MAX_RESTARTS = 5
_RESTART_WINDOW_SECONDS = 300.0
_MAX_BACKOFF_SECONDS = 30.0


@dataclasses.dataclass(frozen=True)
class CompositionPlan:
    """What THIS composition run detected as configured, independent of backend.

    Reused by both the in-process supervisor below and the multi-backend
    deployment planners (:mod:`agent_utilities.deployment.backends`) so "is
    messaging/host-daemon/webui configured" is decided in exactly one place.
    """

    messaging_platforms: tuple[str, ...] = ()
    host_daemon_needed: bool = False
    web_ui_enabled: bool = False

    @property
    def messaging_configured(self) -> bool:
        return bool(self.messaging_platforms)

    def co_service_names(self) -> tuple[str, ...]:
        """Names of co-services this composition would bring up (any backend)."""
        names: list[str] = []
        if self.messaging_configured:
            names.append("messaging")
        if self.host_daemon_needed:
            names.append("graph-os-daemon")
        if self.web_ui_enabled:
            names.append("agent-webui")
        return tuple(names)


def host_daemon_needed() -> bool:
    """True iff this deployment is configured ``client`` and nobody is hosting.

    Never true for the default ``auto`` (the engine self-elects host on its own
    construction) or an explicit ``host`` (same reason). See the module docstring
    for the full reasoning.
    """
    from agent_utilities.knowledge_graph.core.engine_tasks import daemon_role
    from agent_utilities.knowledge_graph.core.host_lock import host_daemon_running

    return daemon_role() == "client" and not host_daemon_running()


def detect_composition(engine: Any = None) -> CompositionPlan:
    """Detect the configured co-services from the already-loaded AgentConfig.

    Pure/side-effect-free: safe to call before any engine or process identity
    exists (used by the deployment planners as well as this supervisor).
    """
    from agent_utilities.core.config import config
    from agent_utilities.messaging.daemon import configured_platforms

    return CompositionPlan(
        messaging_platforms=tuple(configured_platforms(engine)),
        host_daemon_needed=host_daemon_needed(),
        web_ui_enabled=bool(getattr(config, "enable_web_ui", False)),
    )


def bring_up_host_daemon_if_needed() -> dict[str, Any]:
    """Start the consolidated KG host daemon in-process, only if genuinely needed.

    A one-shot, synchronous, best-effort call — the daemon's own background
    threads (graph writer / maintenance scheduler / embedding backfill / KG_LOOP)
    already carry their own internal resilience once started
    (:mod:`agent_utilities.knowledge_graph.core.engine_tasks`); this function's
    only job is to win (or lose) the singleton host-lock race exactly once at
    startup. Losing the race (another host won it between our liveness check and
    the actual attempt) is NOT an error — the self-healing lock already handles
    it; we log and continue as a client.
    """
    if not host_daemon_needed():
        return {"started": False, "reason": "not_needed"}
    from agent_utilities.gateway.daemon import start_host_daemon
    from agent_utilities.knowledge_graph.core.host_lock import KGHostAlreadyRunning

    try:
        start_host_daemon()
    except KGHostAlreadyRunning as exc:
        logger.info(
            "graph-os-daemon co-service: a host appeared during startup "
            "(exception_type=%s) — continuing as a client.",
            type(exc).__name__,
        )
        return {"started": False, "reason": "lost_race"}
    logger.info(
        "graph-os-daemon co-service started in-process (KG_DAEMON_ROLE was "
        "configured client with no live host — bringing up the consolidated "
        "daemon here so this run is never silently unhosted)."
    )
    return {"started": True}


class CoServiceSupervisor:
    """Owns every co-service thread this ``graph-os`` process composed.

    One instance per served process. Each co-service is a blocking
    ``run(stop_event)`` callable (e.g. :func:`agent_utilities.messaging.daemon.run_forever`
    partially applied over its engine/platforms) driven on its own
    ``_authorized_background_thread`` — the SAME verified-session-carrying thread
    helper the KG host daemon uses for its own background threads, so every
    co-service inherits the process's verified actor/session for its whole
    lifetime, including across restarts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._services: dict[str, tuple[threading.Event, threading.Thread]] = {}

    def start_service(
        self,
        name: str,
        run: Callable[[threading.Event], None],
        session: Any,
    ) -> None:
        """Start one supervised co-service thread under the process's session."""
        from agent_utilities.knowledge_graph.core.engine_tasks import (
            _authorized_background_thread,
        )

        with self._lock:
            if name in self._services:
                raise RuntimeError(f"co-service {name!r} is already running")
            stop_event = threading.Event()
            thread = _authorized_background_thread(
                session,
                self._run_supervised,
                name=f"CoService-{name}",
                args=(name, run, stop_event),
            )
            self._services[name] = (stop_event, thread)
            thread.start()
        logger.info("co-service %s started.", name)

    def _run_supervised(
        self,
        name: str,
        run: Callable[[threading.Event], None],
        stop_event: threading.Event,
    ) -> None:
        """Bounded-restart loop for one co-service (runs inside its own thread)."""
        restarts: list[float] = []
        while not stop_event.is_set():
            try:
                run(stop_event)
            except Exception as exc:  # noqa: BLE001 — a co-service crash must never take the process down
                logger.error(
                    "co-service %s crashed (exception_type=%s) — will restart "
                    "unless the bound is exceeded.",
                    name,
                    type(exc).__name__,
                )
            else:
                if stop_event.is_set():
                    return
                logger.error(
                    "co-service %s exited on its own without a stop request — "
                    "treating this as a crash and restarting (bounded).",
                    name,
                )
            now = time.monotonic()
            restarts = [t for t in restarts if now - t < _RESTART_WINDOW_SECONDS]
            restarts.append(now)
            if len(restarts) > _MAX_RESTARTS:
                logger.error(
                    "co-service %s exceeded %d restarts within %.0fs — giving up "
                    "and leaving it STOPPED. This requires operator attention.",
                    name,
                    _MAX_RESTARTS,
                    _RESTART_WINDOW_SECONDS,
                )
                return
            backoff = min(2.0 ** len(restarts), _MAX_BACKOFF_SECONDS)
            stop_event.wait(backoff)

    def stop_all(self, timeout: float = 10.0) -> None:
        """Signal every co-service to stop and join its thread (clean shutdown)."""
        with self._lock:
            services = list(self._services.items())
            self._services.clear()
        for name, (stop_event, thread) in services:
            stop_event.set()
        for name, (_stop_event, thread) in services:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.error(
                    "co-service %s did not stop within %.0fs of shutdown.",
                    name,
                    timeout,
                )
            else:
                logger.info("co-service %s stopped.", name)

    def running(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                name for name, (_e, t) in self._services.items() if t.is_alive()
            )


def start_co_services(session: Any, engine: Any) -> CoServiceSupervisor:
    """Bring up every remaining configured co-service for THIS ``graph-os`` process.

    Called once from ``kg_server.mcp_server()`` — AFTER
    :func:`bring_up_host_daemon_if_needed` (which must run BEFORE the process's
    own engine bootstrap so it can win, or lose, the host-lock race as the first
    constructor; it is a one-shot call, not a supervised thread, so it is invoked
    separately and earlier) and after ``_start_engine_bootstrap`` so a real
    ``engine`` is available. Messaging is started here as a supervised co-service
    thread using that engine + the process's verified session.
    """
    plan = detect_composition(engine)
    supervisor = CoServiceSupervisor()

    if plan.messaging_configured:
        from agent_utilities.messaging.daemon import run_forever

        platforms = list(plan.messaging_platforms)
        supervisor.start_service(
            "messaging",
            lambda stop_event: run_forever(engine, platforms, stop_event),
            session,
        )
    else:
        logger.debug(
            "messaging co-service not configured — no platform tokens present."
        )

    if plan.web_ui_enabled:
        # agent-webui is a separate Node/Vite frontend, not a Python asyncio task —
        # it cannot be started in-process here. It is still part of the composition
        # (ENABLE_WEB_UI is real config), so the multi-backend deployment planners
        # (container/kubernetes) include it as its own service; this in-process
        # backend can only report that it is configured and external.
        logger.info(
            "agent-webui is configured (ENABLE_WEB_UI) but is an external "
            "frontend process — run it separately or via the container/"
            "kubernetes deployment backend, not the in-process composition."
        )

    return supervisor
