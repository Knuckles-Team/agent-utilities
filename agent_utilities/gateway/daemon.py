"""Gateway-hosted unified KG daemon (CONCEPT:EG-KG.storage.nonblocking-checkpoint / OS-5.9).

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
import time
from dataclasses import replace
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.security.error_surface import public_error_payload

logger = logging.getLogger(__name__)

_engine: Any = None
_lock = threading.Lock()

_metrics_lock = threading.Lock()
_metrics_started = False


def start_daemon_metrics_listener() -> bool:
    """Expose this daemon's ``agent_utilities_*`` Prometheus registry over HTTP.

    CONCEPT:AU-OS.observability.daemon-metrics-listener (D-OG-4). The
    consolidated KG host daemon this module runs records the
    ``SCHEDULED_JOB_*``/``LANE_*``/``KG_INGEST_*``/``DISPATCH_*`` families
    (``observability/gateway_metrics.py``) into the SAME per-process
    ``prometheus_client`` registry every other metric in this package uses —
    but unlike graph-os (``server_factory.py``, port 8000) or the full
    gateway (``gateway/graph_api.py``), the standalone entry point this
    module's :func:`main` runs (``python3 -m agent_utilities.gateway.daemon``
    — the ``graph-os-host`` container in the live pod) has NO HTTP server of
    any kind. Half of the package's defined metric series were therefore
    structurally uncollectable: recorded, but never scrapeable.
    ``prometheus_client.start_http_server`` is deliberately the entire fix —
    it starts a small background HTTP thread serving ``generate_latest()``
    over the default registry; no new route table, no framework.

    Only called from the standalone :func:`main` path, never from
    :func:`start_host_daemon` directly — when this daemon instead runs
    in-process inside the full gateway (mounted from its lifespan, per this
    module's own docstring), that process already serves ``GET /metrics``
    on its own HTTP port via ``graph_api.py``, reading the identical shared
    registry; a second listener there would be redundant, not merely
    harmless.

    Bind address / auth — read the justification, this is a deliberate
    judgment call, not a default:

    Binds **pod-local** (``0.0.0.0`` inside the container; default port
    ``9110``, ``KG_DAEMON_METRICS_PORT``) with **no bearer token** — unlike
    graph-os's own ``/metrics`` (``server_factory.py``), which IS
    bearer-gated on a non-loopback listener. That is not an inconsistency;
    it is the same exposure judgment reaching the opposite answer because
    the exposure itself differs:

    * graph-os's ``/metrics`` (port 8000) lives on the SAME listener as the
      public/Ingress-facing MCP tool-call surface (``graph-os.arpa``) — an
      unauthenticated route there is reachable by anyone who can reach that
      hostname, hence the bearer gate.
    * This listener has no Service port and no Ingress route. Its only
      intended consumer is Prometheus's ``kubernetes-pods-metrics-port`` job
      (``role: pod``, zero-annotation discovery keyed on a container port
      literally named ``metrics``), which itself only runs inside the
      cluster's pod network — the same reachability tier as
      node-exporter/cAdvisor, both already scraped unauthenticated
      fleet-wide (see docs/architecture/observability.md's topology table).
      This is that tier, not a broader one.
    * The content is pure operational counts/durations/labels (schedule
      names, lane names, queue depths) — never a secret, a tool-call
      payload, or PII.
    * A strictly loopback (``127.0.0.1``) bind — the posture the bundled
      ``epistemic-graph-server`` engine itself uses for its own native
      metrics listener — would be UNREACHABLE from Prometheus, which scrapes
      via the Pod IP (the shared pod network's routable interface), not via
      a process that merely shares the pod's network namespace. Pod-local
      is the narrowest bind Prometheus's pod-IP scrape model can reach.

    Never blocks daemon startup. Two distinct failure modes are both logged
    at ERROR — loud, never swallowed — and both leave the daemon running
    with no metrics listener, exactly as it behaved before this function
    existed:

    * the optional ``metrics`` extra (``prometheus-client``) is not
      installed;
    * the bind itself fails (port already in use, permission denied, ...).
    """
    global _metrics_started
    if not setting("KG_DAEMON_METRICS", True):
        logger.info("Daemon metrics listener disabled (KG_DAEMON_METRICS=false).")
        return False
    with _metrics_lock:
        if _metrics_started:
            return True
        try:
            from prometheus_client import start_http_server
        except ImportError as exc:
            logger.error(
                "Daemon metrics listener NOT started: the optional 'metrics' "
                "extra (prometheus-client) is not installed (%s). "
                "SCHEDULED_JOB_*/LANE_*/KG_INGEST_*/DISPATCH_* series recorded "
                "by this process will not be scrapeable until it is.",
                exc,
            )
            return False
        host = str(setting("KG_DAEMON_METRICS_HOST", "0.0.0.0") or "0.0.0.0")
        port = int(setting("KG_DAEMON_METRICS_PORT", 9110) or 9110)
        try:
            start_http_server(port, addr=host)
        except OSError as exc:
            logger.error(
                "Daemon metrics listener NOT started: could not bind %s:%d "
                "(%s). The daemon continues running WITHOUT "
                "a scrapeable metrics endpoint — SCHEDULED_JOB_*/LANE_*/"
                "KG_INGEST_*/DISPATCH_* series stay uncollectable until this "
                "is resolved.",
                host,
                port,
                exc,
            )
            return False
        _metrics_started = True
        logger.info(
            "Daemon metrics listener started on %s:%d "
            "(CONCEPT:AU-OS.observability.daemon-metrics-listener).",
            host,
            port,
        )
        return True


def mint_process_identity() -> Any:
    """Mint the standalone host daemon's verified graph authority.

    The daemon is a long-lived, non-request process, so there is no HTTP
    middleware to establish an actor or :class:`GraphSession`.  Resolve and
    validate the configured process credential before constructing any graph
    backend, and attach a renewable in-memory lease so worker threads retain
    bounded authority without retaining token material.
    """
    from agent_utilities.core.config import config
    from agent_utilities.security.brain_context import CredentialLease
    from agent_utilities.security.request_identity import (
        acquire_process_identity_token,
        mint_actor_from_token_sync,
        mint_graph_session,
    )

    token = acquire_process_identity_token(config)
    actor = mint_actor_from_token_sync(token)
    del token
    expires_at = getattr(actor, "credential_expires_at", None)
    if expires_at is None:
        raise RuntimeError("Graph host process identity has no bounded expiry")
    actor = replace(
        actor,
        credential_lease=CredentialLease(int(expires_at)),
    )
    session = mint_graph_session(actor)
    session.ensure_authority_current(minimum_ttl_seconds=30)
    session.engine_verified_context()
    logger.info("Verified graph host process authority minted")
    return session


def refresh_process_identity(session: Any) -> Any:
    """Renew ``session`` in place before its validated credential expires.

    Only the shared expiry lease changes.  Subject, tenant, roles, groups, and
    actor type must remain byte-for-byte equivalent to the authority captured
    by the background worker threads.
    """
    from agent_utilities.core.config import config
    from agent_utilities.knowledge_graph.core.session import SessionExpiredError
    from agent_utilities.security.request_identity import (
        acquire_process_identity_token,
        mint_actor_from_token_sync,
    )

    try:
        session.ensure_authority_current(minimum_ttl_seconds=30)
        return session
    except SessionExpiredError:
        pass

    lease = getattr(getattr(session, "actor", None), "credential_lease", None)
    if lease is None:
        raise RuntimeError("Graph host process authority is not renewable")
    token = acquire_process_identity_token(config)
    renewed_actor = mint_actor_from_token_sync(token)
    del token
    identity_fields = (
        "actor_id",
        "actor_type",
        "roles",
        "tenant_id",
        "authenticated",
        "groups",
    )
    if any(
        getattr(session.actor, field, None) != getattr(renewed_actor, field, None)
        for field in identity_fields
    ):
        raise RuntimeError("Graph host process authority changed during renewal")
    expires_at = getattr(renewed_actor, "credential_expires_at", None)
    if expires_at is None or int(expires_at) <= int(time.time()) + 30:
        raise RuntimeError("Graph host process authority renewal is too short-lived")
    lease.renew(int(expires_at))
    session.ensure_authority_current(minimum_ttl_seconds=30)
    return session


def start_host_daemon(*, defer_background_start: bool = False) -> Any:
    """Start (once) the single consolidated KG daemon in this gateway process.

    Forces ``KG_DAEMON_ROLE=host`` so constructing the engine launches the
    consolidated daemon threads, then returns the engine. Idempotent: repeated
    calls return the same engine.
    """
    global _engine
    with _lock:
        if _engine is not None:
            return _engine
        # CONCEPT:AU-OS.identity.per-agent-on-behalf-delegation (decision 3) — fail closed at
        # startup when delegated identity is on but no run-token signing secret resolves, so the
        # daemon refuses to boot (config-contract) instead of failing lazily on the first spawn.
        # No-op in the shipped `warn`/`off` posture.
        from agent_utilities.security.run_token import require_token_secret

        require_token_secret()
        # The gateway process is the authoritative daemon host.
        os.environ["KG_DAEMON_ROLE"] = "host"
        from agent_utilities.knowledge_graph.core.engine import (
            IntelligenceGraphEngine,
        )

        _engine = IntelligenceGraphEngine.get_or_create(
            defer_background_start=defer_background_start
        )
        if not defer_background_start:
            try:
                # Ensure the on-demand task-worker pool is up in the host.
                if hasattr(_engine, "start_task_workers"):
                    _engine.start_task_workers()
            except Exception as exc:  # noqa: BLE001 — best-effort pool start; the daemon must still come up, but the real cause is logged (not just its type) so a persistently-failing worker pool is diagnosable
                logger.warning("host daemon: start_task_workers failed: %s", exc)
        # Always-on KG-native observability (CONCEPT:AU-OS.config.model-factory-passthrough): install the trace sink
        # backed by this host's engine, so every traced agent call persists a
        # Trace/Span/Generation subgraph that is graph-queryable. One-time injection;
        # best-effort (a failure here must never block the daemon).
        try:
            from agent_utilities.harness.trace_backend import KGTraceBackend
            from agent_utilities.harness.tracing import set_kg_trace_sink

            set_kg_trace_sink(KGTraceBackend(backend=_engine))
        except Exception as exc:  # noqa: BLE001 — best-effort install; the daemon must still start without the trace sink, but the real cause is logged (not just its type) so a persistently-failing sink is diagnosable
            logger.warning("host daemon: KG trace sink install failed: %s", exc)
        # Durable error-detail persistence (D-24, CONCEPT:AU-KG.audit.durable-error-detail):
        # install the KG-backed sink so a caller's `error.detail_ref` stays resolvable past
        # this process's lifetime and across replicas. One-time injection; best-effort
        # (a failure here must never block the daemon or degrade error reporting — the
        # in-process detail store keeps working on its own either way).
        try:
            from agent_utilities.observability.error_detail_sink import (
                GraphErrorDetailSink,
            )
            from agent_utilities.security.error_surface import (
                register_detail_persistence_sink,
            )

            register_detail_persistence_sink(GraphErrorDetailSink(engine=_engine))
        except Exception as exc:  # noqa: BLE001 — best-effort install; the in-process detail store keeps working, but the real cause is logged (not just its type) so a persistently-failing sink is diagnosable
            logger.warning(
                "host daemon: durable error-detail sink install failed: %s", exc
            )
        logger.info("Gateway host daemon started")
    # CONCEPT:AU-ECO.messaging.inbound-messaging-router-runs — the inbound messaging router runs in its OWN process
    # (``agent-utilities-messaging`` / ``agent_utilities.messaging.daemon``), NOT here, so
    # the host's CPU-bound maintenance (codebase ingestion / relevance sweeps) can never
    # starve the inbound reply loop. It connects to this same shared engine as a client.
    return _engine


def daemon_status() -> dict[str, Any]:
    """Return the consolidated daemon's status (role + live threads + jobs)."""
    eng = _engine
    if eng is None:
        return {"running": False, "role": setting("KG_DAEMON_ROLE", "auto")}
    try:
        return eng.unified_daemon_status()
    except Exception as exc:  # noqa: BLE001
        return {"running": False, **public_error_payload(exc, logger=logger)}


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
        except OSError as exc:
            logger.warning(
                "could not remove a task-queue artifact (exception_type=%s)",
                type(exc).__name__,
            )
    logger.warning("Drained task queue (removed_count=%d).", len(removed))
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
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "engine checkpoint on shutdown skipped (exception_type=%s)",
                type(exc).__name__,
            )
    try:
        # CONCEPT:AU-OS.host.so-they-are-idle — tear down any warm-fork parents this host was pooling.
        from agent_utilities.runtime.warm_registry import WarmParentRegistry

        reaped = WarmParentRegistry.drain_active()
        if reaped:
            logger.info("Drained %d warm-fork parent(s) on shutdown.", len(reaped))
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "warm-parent drain on shutdown skipped (exception_type=%s)",
            type(exc).__name__,
        )
    try:
        from agent_utilities.knowledge_graph.core.host_lock import release_host_lock

        release_host_lock()
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "host lock release skipped (exception_type=%s)", type(exc).__name__
        )
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
    (CONCEPT:EG-KG.storage.nonblocking-checkpoint / OS-5.9)

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

    # CONCEPT:AU-KG.ingest.attaching-this-root-logger — self-ingest telemetry: graph-os ships its own logs into
    # the epistemic-graph engine obs store. Opt-in (default-off); no-op when unset.
    try:
        from agent_utilities.observability.self_ingest import (
            install_self_ingest_logging,
        )

        install_self_ingest_logging()
    except Exception:  # noqa: BLE001 — telemetry must never block the daemon
        logger.debug("self-ingest logging not installed")

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

    stop = threading.Event()

    def _handle(signum: int, _frame: Any) -> None:
        logger.info("Received signal %s — shutting down host daemon.", signum)
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except (ValueError, OSError):  # not in main thread / unsupported
            pass

    session = mint_process_identity()
    from agent_utilities.knowledge_graph.core.session import use_session
    from agent_utilities.security.brain_context import use_actor

    with use_actor(session.actor), use_session(session):
        try:
            start_host_daemon()
        except KGHostAlreadyRunning as exc:
            logger.error(
                "KG host is already running (exception_type=%s)", type(exc).__name__
            )
            raise SystemExit(2) from None

        # D-OG-4: the standalone daemon has no HTTP server of its own — start
        # one, best-effort, purely to expose the metrics this process already
        # records (see start_daemon_metrics_listener's docstring for the
        # bind/auth decision). Never fatal: logs loudly and continues on any
        # failure.
        start_daemon_metrics_listener()

        logger.info("graph-os host daemon running")

        try:
            # Daemon threads do the work; this main-thread heartbeat also
            # renews their shared credential lease before expiry.
            while not stop.wait(timeout=30.0):
                refresh_process_identity(session)
                logger.debug("host daemon heartbeat")
        finally:
            stop_host_daemon()


if __name__ == "__main__":
    main()
