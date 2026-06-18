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

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

_engine: Any = None
_lock = threading.Lock()
_messaging_thread: threading.Thread | None = None


# CONCEPT:ECO-4.51 — Inbound messaging router with graph-agent reply in the host daemon
def start_messaging_router(engine: Any) -> None:
    """Start the inbound messaging router for any configured backend.

    Auto-detected: runs only when a messaging backend has a token/app id configured (so
    it is opt-out, not a manual knob). The router runs on its OWN event loop in a daemon
    thread, so it works both inside the async gateway and in the thread-based standalone
    daemon ``main()``. Idempotent.
    """
    global _messaging_thread
    with _lock:
        if _messaging_thread is not None and _messaging_thread.is_alive():
            return
        from agent_utilities.messaging.service import MessagingService

        platforms = MessagingService.instance(engine).configured_platforms()
        if not platforms:
            logger.info(
                "messaging: no backend configured — inbound router not started "
                "(set e.g. TELEGRAM_BOT_TOKEN to enable)."
            )
            return
        _messaging_thread = threading.Thread(
            target=_run_messaging_router,
            args=(engine, platforms),
            name="messaging-inbound-router",
            daemon=True,
        )
        _messaging_thread.start()
        logger.info("messaging: inbound router starting for backends %s", platforms)


def _run_messaging_router(engine: Any, platforms: list[str]) -> None:
    """Connect configured backends and run the InboundRouter on a private loop."""
    import asyncio

    from agent_utilities.messaging.commands import command_specs
    from agent_utilities.messaging.router import (
        InboundRouter,
        create_planner_handler,
    )
    from agent_utilities.messaging.service import MessagingService

    async def _serve() -> None:
        svc = MessagingService.instance(engine)
        router = InboundRouter()
        for pid in platforms:
            backend = await svc.get_backend(pid)
            if backend is None:
                continue
            svc.register_connected(backend)
            # Publish OUR universal command set on every backend (CONCEPT:ECO-4.57);
            # each registers it where the platform supports a runtime menu, else no-ops.
            try:
                await backend.register_commands(command_specs())
            except Exception as exc:  # noqa: BLE001
                logger.warning("messaging: register_commands(%s) failed: %s", pid, exc)
            router.register_backend(backend)
        router.set_default_handler(await create_planner_handler(engine))
        await router.start()  # blocks on the listener tasks

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_serve())
    except Exception as e:  # noqa: BLE001
        logger.error("messaging inbound router stopped: %s", e)
    finally:
        loop.close()


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
    # Auto-start the inbound messaging router (CONCEPT:ECO-4.51) outside the lock —
    # it takes its own lock and no-ops when no backend is configured.
    try:
        start_messaging_router(_engine)
    except Exception as e:  # noqa: BLE001
        logger.warning("host daemon: messaging router start failed: %s", e)
    return _engine


def daemon_status() -> dict[str, Any]:
    """Return the consolidated daemon's status (role + live threads + jobs)."""
    eng = _engine
    if eng is None:
        return {"running": False, "role": setting("KG_DAEMON_ROLE", "auto")}
    try:
        return eng.unified_daemon_status()
    except Exception as e:  # noqa: BLE001
        return {"running": False, "error": str(e)}


def drain_task_queue() -> list[str]:
    """Purge the durable task-queue store (recovery from a corrupt/stuck queue).

    A broken queue can't be safely drained by reading it, so we remove the
    store files outright; the host then starts with an empty queue. SQLite
    backend = ``kg_task_queue.db`` (+ WAL/SHM siblings). Returns removed paths.
    """
    from pathlib import Path

    from agent_utilities.core.paths import data_dir

    base = data_dir() / "kg_task_queue.db"
    removed: list[str] = []
    for p in (base, Path(f"{base}-wal"), Path(f"{base}-shm")):
        try:
            if p.exists():
                p.unlink()
                removed.append(str(p))
        except OSError as e:
            logger.warning("could not remove %s: %s", p, e)
    logger.warning("Drained task queue (removed %s).", removed or "nothing")
    return removed


def stop_host_daemon() -> None:
    """Graceful shutdown: best-effort engine checkpoint, then release the lock.

    The durable task queue persists on its own (SQLite/NATS/Kafka), so a
    restarted host resumes it. We trigger an engine checkpoint (snapshot to
    ``GRAPH_SERVICE_PERSIST_DIR`` when enabled) for a fast warm restart, then
    release the singleton host lock so a replacement host can take over.
    """
    global _engine
    eng = _engine
    if eng is not None:
        try:
            gc = getattr(eng, "graph_compute", None) or getattr(eng, "graph", None)
            client = getattr(gc, "_client", None)
            if client is not None and hasattr(client, "checkpoint"):
                client.checkpoint()
                logger.info("Engine checkpoint written on shutdown.")
        except Exception as e:  # noqa: BLE001
            logger.debug("engine checkpoint on shutdown skipped: %s", e)
    try:
        from agent_utilities.knowledge_graph.core.host_lock import release_host_lock

        release_host_lock()
    except Exception as e:  # noqa: BLE001
        logger.debug("host lock release skipped: %s", e)
    _engine = None
    logger.info("Host daemon stopped.")


def main() -> None:
    """Run the single consolidated KG daemon as a standalone host process.

    This is the daemon ``host`` when the full API gateway (agent-webui) isn't
    run as a long-lived service: it starts the one consolidated daemon and
    blocks, draining the durable task queue that ``KG_DAEMON_ROLE=client``
    processes (MCP server / CLI / scripts) submit to. The singleton host lock
    (``host_lock.py``) guarantees only one host runs; a second start raises a
    descriptive ``KGHostAlreadyRunning``. Console entry point: ``graph-os-daemon``.
    (CONCEPT:KG-2.8 / OS-5.9)

    Flags: ``--drain-queue`` purges the durable queue before starting (recovery);
    ``--status`` prints the live host + daemon status and exits.
    """
    import argparse
    import json
    import logging
    import signal

    from agent_utilities.knowledge_graph.core.host_lock import (
        KGHostAlreadyRunning,
        host_lock_holder,
    )

    ap = argparse.ArgumentParser(prog="graph-os-daemon")
    ap.add_argument(
        "--drain-queue",
        action="store_true",
        help="Purge the durable task queue before starting (corrupt-queue recovery).",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="Print the live host-lock holder + daemon status, then exit.",
    )
    args, _ = ap.parse_known_args()

    logging.basicConfig(
        level=setting("KG_DAEMON_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.status:
        print(
            json.dumps(
                {"host_lock_holder": host_lock_holder(), "daemon": daemon_status()},
                indent=2,
                default=str,
            )
        )
        return

    if args.drain_queue:
        drain_task_queue()

    try:
        start_host_daemon()
    except KGHostAlreadyRunning as e:
        logger.error("%s", e)
        raise SystemExit(2) from e

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

    stop_host_daemon()


if __name__ == "__main__":
    main()
