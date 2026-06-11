"""Dedicated ingest engine lifecycle (CONCEPT:KG-2.58, Phase D).

A SECOND, EPHEMERAL epistemic-graph engine that handles ONLY the codebase-ingest
path's heavy compute — stateless tree-sitter parsing and throwaway
community-detection tenants — isolated from the query engine and the background
daemons (embedding backfill / reconcile / task poll) that otherwise contend with
it on the single shared engine. Profiling showed those background workers, not the
ingest itself, dominate a daemon ingest's wall-clock; moving the ingest's parse +
community onto a private engine removes that contention.

The ingest engine runs with NO persist dir: its data is ephemeral (parse is
stateless; community tenants are per-job throwaways deleted after each run), so it
never checkpoints and never competes for snapshot I/O. The durable code/feature
WRITES still go to the query engine (where queries read them).

Opt-in via ``KG_INGEST_ENGINE_ENDPOINT``; unset ⇒ today's single-engine behavior.
Every failure path returns ``None`` so the ingest cleanly falls back to the query
engine — a missing/dead ingest engine never breaks ingestion.
"""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404 — spawns our own engine binary, fixed argv
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _socket_of(endpoint: str) -> str | None:
    """Local unix socket path for an endpoint, or None for non-local endpoints."""
    if endpoint.startswith("unix://"):
        return endpoint[len("unix://") :]
    if endpoint.startswith("/"):
        return endpoint
    return None


def _reachable(socket_path: str, auth_secret: str | None) -> bool:
    """True iff an engine is actually listening on ``socket_path``.

    Checks ``os.path.exists`` FIRST: the epistemic-graph client silently falls back
    to the default ``/tmp/epistemic-graph.sock`` when the requested socket is
    absent, so a bare connect would report the QUERY engine as reachable and route
    ingest there. Guarding on existence avoids that footgun.
    """
    if not os.path.exists(socket_path):
        return False
    try:
        from epistemic_graph.client import SyncEpistemicGraphClient

        client = SyncEpistemicGraphClient.connect(
            socket_path=socket_path,
            auth_secret=auth_secret,
            graph_name="__ingest_probe__",
        )
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_ingest_engine(
    endpoint: str | None,
    auth_secret: str | None,
    *,
    insecure: bool = False,
) -> str | None:
    """Ensure the ephemeral ingest engine is up at ``endpoint``.

    Returns the endpoint when it is reachable (so the caller routes ingest compute
    there), or ``None`` on any problem (so the caller uses the query engine).
    Spawning is single-instance-guarded per socket (shared with the autostart
    guard), so concurrent callers never double-spawn.
    """
    if not endpoint:
        return None
    sock = _socket_of(endpoint)
    if not sock:
        logger.debug("ingest engine: non-local endpoint %s not spawnable", endpoint)
        return None
    if _reachable(sock, auth_secret):
        return endpoint

    from agent_utilities.knowledge_graph.core.engine_lock import engine_spawn_guard

    with engine_spawn_guard(sock):
        # Double-check inside the guard: a peer may have just spawned it.
        if _reachable(sock, auth_secret):
            return endpoint
        server = str(Path(sys.executable).parent / "epistemic-graph-server")
        if not os.path.exists(server):
            logger.warning(
                "ingest engine: binary not found at %s; using query engine", server
            )
            return None
        cmd = [server, "--socket-path", sock]  # NO --persist-dir: ephemeral
        child_env = dict(os.environ)
        if insecure:
            child_env["EPISTEMIC_GRAPH_ALLOW_INSECURE"] = "1"
            child_env.pop("GRAPH_SERVICE_AUTH_SECRET", None)
        else:
            child_env["GRAPH_SERVICE_AUTH_SECRET"] = auth_secret or ""
        try:
            subprocess.Popen(  # nosec B603 — fixed argv, our own binary
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=child_env,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("ingest engine spawn failed (%s); using query engine", e)
            return None
        for _ in range(50):  # ~10s for it to bind + accept
            if _reachable(sock, auth_secret):
                logger.info(
                    "dedicated ingest engine up at %s (ephemeral, no persist)", sock
                )
                return endpoint
            time.sleep(0.2)
    logger.warning("ingest engine did not come up at %s; using query engine", sock)
    return None
