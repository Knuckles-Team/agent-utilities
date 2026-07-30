"""Self-composing co-service supervisor for the ``graph-os`` entrypoint.

``uvx agent-utilities graph-os`` (equally the plain ``graph-os`` console script)
brings up graph-os PLUS the messaging inbound router when the already-loaded
``AgentConfig`` contains a configured messaging platform
(:mod:`agent_utilities.messaging.daemon`). The explicit ``KG_DAEMON_ROLE``
boundary is never changed here: a ``client`` serves requests and submits durable
work, while the gateway or standalone ``graph-os-daemon`` process owns background
workers, maintenance schedules, and autonomous loops.

Detection signals
-----------------
* **messaging** — configured iff :func:`agent_utilities.messaging.daemon.configured_platforms`
  returns at least one platform (a real token/app id is present). This is the
  exact same check the standalone ``agent-utilities-messaging`` process already
  uses to decide whether to serve or exit immediately.
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
    web_ui_enabled: bool = False

    @property
    def messaging_configured(self) -> bool:
        return bool(self.messaging_platforms)

    def co_service_names(self) -> tuple[str, ...]:
        """Names of co-services this composition would bring up (any backend)."""
        names: list[str] = []
        if self.messaging_configured:
            names.append("messaging")
        if self.web_ui_enabled:
            names.append("agent-webui")
        return tuple(names)


def detect_composition(engine: Any = None) -> CompositionPlan:
    """Detect the configured co-services from the already-loaded AgentConfig.

    Pure/side-effect-free: safe to call before any engine or process identity
    exists (used by the deployment planners as well as this supervisor).
    """
    from agent_utilities.core.config import config
    from agent_utilities.messaging.daemon import configured_platforms

    return CompositionPlan(
        messaging_platforms=tuple(configured_platforms(engine)),
        web_ui_enabled=bool(getattr(config, "enable_web_ui", False)),
    )


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

    Called once from ``kg_server.mcp_server()`` after ``_start_engine_bootstrap``
    so a real ``engine`` is available. Messaging is started here as a supervised
    co-service thread using that engine + the process's verified session.
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
