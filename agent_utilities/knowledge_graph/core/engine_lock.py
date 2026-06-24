"""Single-instance spawn guard for the local epistemic-graph engine.

CONCEPT:KG-2.8 / OS-5.9 — companion to :mod:`host_lock`. The host-lock elects ONE
*daemon* owner; this elects ONE *engine* process per socket path.

Engine autostart (``graph_compute`` connect → ``subprocess.Popen`` of
``epistemic-graph-server``) was previously unguarded: any process that found the
engine down would spawn one, with no mutual exclusion. Two spawners racing — or a
client spawning while a displaced-but-alive engine still held an unlinked socket —
produced a **split-brain**: two engines bound to the same socket path,
checkpointing to the same ``--persist-dir`` and clobbering each other's snapshots.
That cost a 107GB orphan and corrupted persistence in practice.

The fix is double-checked locking around the spawn, keyed by the socket path:

* An exclusive ``flock`` on ``engine-<sha8(socket)>.lock`` serializes all spawners
  for a given socket. The lock auto-releases when the holder dies (no stale-PID bug).
* The guard is held across *spawn + wait-until-reachable*, so a concurrent spawner
  blocks until the engine is confirmed up, then its own re-check connect succeeds —
  it never spawns a second engine.
* Callers MUST re-attempt the connect after acquiring the guard (the double check):
  the engine may have come up while they waited for the lock.

This is advisory and process-local to spawning; it does not stop an operator from
launching a second engine by hand. It does make our own autostart path single-instance.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import socket as _socket
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.knowledge_graph.core.file_lock import (
    LockUnavailable,
    lock_exclusive_nb,
    unlock,
)

logger = logging.getLogger(__name__)

# Default socket if a caller passes None (matches host_lock's default).
_DEFAULT_SOCKET = "/tmp/epistemic-graph.sock"  # nosec B108 — default UDS path, overridable

# How long to wait for the spawn guard before giving up and proceeding without it
# (a slow peer spawn should never wedge a connect forever).
_GUARD_TIMEOUT_SECS = 30.0


def _runtime_base() -> Path:
    """Return the user runtime dir (created if needed) — same base as host_lock."""
    try:
        import platformdirs

        base = Path(platformdirs.user_runtime_dir("agent-utilities"))
    except Exception:  # pragma: no cover - platformdirs present in practice
        base = Path(setting("XDG_RUNTIME_DIR") or "/tmp") / "agent-utilities"  # nosec B108 — XDG fallback
    base.mkdir(parents=True, exist_ok=True)
    return base


def engine_lock_path(socket_path: str | None) -> Path:
    """Per-socket engine lock path: ``engine-<sha8(socket)>.lock``."""
    sock = socket_path or _DEFAULT_SOCKET
    digest = hashlib.sha256(str(sock).encode("utf-8")).hexdigest()[:8]
    return _runtime_base() / f"engine-{digest}.lock"


def _record_holder(fd: int, socket_path: str | None) -> None:
    """Stamp the lock file with this spawner's identity (best-effort)."""
    meta = {
        "pid": os.getpid(),
        "host": _socket.gethostname(),
        "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "socket": socket_path or _DEFAULT_SOCKET,
    }
    try:
        data = json.dumps(meta).encode("utf-8")
        os.ftruncate(fd, 0)
        os.pwrite(fd, data, 0)
        os.fsync(fd)
    except Exception:  # pragma: no cover - metadata is advisory
        pass


def engine_lock_holder(socket_path: str | None) -> dict[str, Any] | None:
    """Return the recorded spawner identity for a socket's lock, if any."""
    path = engine_lock_path(socket_path)
    try:
        return json.loads(path.read_text() or "{}") or None
    except Exception:
        return None


@contextlib.contextmanager
def engine_spawn_guard(
    socket_path: str | None, timeout: float = _GUARD_TIMEOUT_SECS
) -> Iterator[bool]:
    """Hold the per-socket engine spawn lock for the duration of the block.

    Yields ``True`` if the exclusive lock was acquired (this process is the sole
    spawner), or ``False`` if it could not be acquired within ``timeout`` (a peer
    is spawning; the caller should still re-check connectivity and avoid spawning).

    The lock is intentionally held across the caller's spawn+wait so concurrent
    spawners serialize behind it and find the engine already up on re-check.
    """
    path = engine_lock_path(socket_path)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    acquired = False
    deadline = time.monotonic() + max(0.0, timeout)
    try:
        while True:
            try:
                lock_exclusive_nb(fd)
                acquired = True
                break
            except (LockUnavailable, BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    h = engine_lock_holder(socket_path) or {}
                    logger.warning(
                        "engine spawn guard for %s held by pid=%s since %s; "
                        "proceeding without the guard after %.0fs.",
                        socket_path or _DEFAULT_SOCKET,
                        h.get("pid", "?"),
                        h.get("acquired_at", "?"),
                        timeout,
                    )
                    break
                time.sleep(0.2)
        if acquired:
            _record_holder(fd, socket_path)
        yield acquired
    finally:
        try:
            if acquired:
                unlock(fd)
        except Exception:  # pragma: no cover
            pass
        with contextlib.suppress(Exception):
            os.close(fd)
