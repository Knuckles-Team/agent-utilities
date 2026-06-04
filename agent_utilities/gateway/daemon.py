"""Gateway-hosted unified KG daemon (CONCEPT:KG-2.8 / OS-5.9).

The API gateway is the single process that runs the ONE consolidated KG
background daemon (queue drain + graph writer + task workers + maintenance
scheduler + file-watch poll). Every other entry point — the MCP server, CLI,
and one-shot scripts — runs as a ``client`` (``KG_DAEMON_ROLE=client``) and
spawns NO background threads; their work is enqueued to the durable task queue
that this host daemon drains.

Mount the daemon with ``start_host_daemon()`` from the gateway's lifespan/startup
and surface its state via the ``/daemon/*`` routes on ``dashboard_router``.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_engine: Any = None
_lock = threading.Lock()


def start_host_daemon() -> Any:
    """Start (once) the single consolidated KG daemon in this gateway process.

    Forces ``KG_DAEMON_ROLE=host`` so constructing the engine launches the
    consolidated daemon threads, then returns the engine. Idempotent: repeated
    calls return the same engine.
    """
    global _engine
    with _lock:
        if _engine is not None:
            return _engine
        # The gateway process is the authoritative daemon host.
        os.environ["KG_DAEMON_ROLE"] = "host"
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )

        _engine = IntelligenceGraphEngine.get_active() or IntelligenceGraphEngine()
        try:
            # Ensure the on-demand task-worker pool is up in the host.
            if hasattr(_engine, "start_task_workers"):
                _engine.start_task_workers()
        except Exception as e:  # noqa: BLE001
            logger.warning("host daemon: start_task_workers failed: %s", e)
        logger.info("Gateway host daemon started: %s", daemon_status())
        return _engine


def daemon_status() -> dict[str, Any]:
    """Return the consolidated daemon's status (role + live threads + jobs)."""
    eng = _engine
    if eng is None:
        return {"running": False, "role": os.environ.get("KG_DAEMON_ROLE", "auto")}
    try:
        return eng.unified_daemon_status()
    except Exception as e:  # noqa: BLE001
        return {"running": False, "error": str(e)}


def main() -> None:
    """Run the single consolidated KG daemon as a standalone host process.

    This is the daemon ``host`` when the full API gateway (agent-webui) isn't
    run as a long-lived service: it starts the one consolidated daemon and
    blocks, draining the durable task queue that ``KG_DAEMON_ROLE=client``
    processes (MCP server / CLI / scripts) submit to. Console entry point:
    ``graph-os-daemon``. (CONCEPT:KG-2.8 / OS-5.9)
    """
    import logging
    import signal

    logging.basicConfig(
        level=os.environ.get("KG_DAEMON_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    start_host_daemon()
    logger.info("graph-os host daemon running: %s", daemon_status())

    stop = threading.Event()

    def _handle(signum: int, _frame: Any) -> None:
        logger.info("Received signal %s — shutting down host daemon.", signum)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):  # not in main thread / unsupported
            pass

    # Daemon threads do the work; this just keeps the process alive and logs a
    # heartbeat with queue depth periodically.
    while not stop.wait(timeout=60.0):
        try:
            logger.debug("host daemon heartbeat: %s", daemon_status())
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main()
