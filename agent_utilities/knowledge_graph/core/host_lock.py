"""Singleton lock + role resolution for the consolidated KG host daemon.

CONCEPT:KG-2.8 / OS-5.9 — the KG runs exactly ONE consolidated background daemon
(queue drain + graph writer + task workers + maintenance scheduler). Every other
entry point (the ``graph-os`` MCP server, CLI, one-shot scripts) must run as a
``client`` that spawns NO daemon threads and lets the host drain the durable task
queue. Historically this was advisory (``KG_DAEMON_ROLE``), so multiple processes
in ``auto`` role each started the daemon threads and spun the CPU contending for
the one engine.

This module makes it enforceable with an **advisory ``flock``**:

* The lock auto-releases when the holder dies, so a crashed host never leaves a
  stale lock blocking restart (the classic PID-file staleness bug).
* ``auto`` is **self-healing**: the first process to start becomes the host; any
  later duplicate (e.g. a multiplexer respawned by a reconnect) fails the lock and
  silently becomes a client.
* An explicit ``host`` that finds the lock held raises :class:`KGHostAlreadyRunning`
  with the holder's pid / start time / socket — a loud, descriptive blocking error.

The decision is cached per-process: :func:`resolve_daemon_role` is called once from
the engine's task-manager ``__init__`` and the result is read everywhere via
:func:`effective_daemon_role`.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import socket as _socket
import time
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)


class KGHostAlreadyRunning(RuntimeError):
    """Raised when an explicit ``host`` start finds another host already running."""


def _lock_path() -> Path:
    """Return the host-lock path under the user runtime dir (created if needed)."""
    try:
        import platformdirs

        base = Path(platformdirs.user_runtime_dir("agent-utilities"))
    except Exception:  # pragma: no cover - platformdirs always present in practice
        base = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / "agent-utilities"  # nosec B108 — XDG fallback, not a security-sensitive temp path
    base.mkdir(parents=True, exist_ok=True)
    return base / "kg_daemon_host.lock"


_lock_fd: Any = None
_effective_role: str | None = None


def _read_holder(path: Path) -> dict[str, Any]:
    """Read the current lock holder's identity metadata (best-effort)."""
    try:
        return json.loads(path.read_text() or "{}")
    except Exception:
        return {}


def _try_acquire() -> bool:
    """Attempt a non-blocking exclusive flock; record our identity on success.

    Uses os-level fd writes (``ftruncate`` + ``pwrite`` at offset 0) so the
    winner's metadata cleanly overwrites a previous (now-dead) holder's record —
    append-mode (``"a+"``) would write at EOF regardless of ``seek``.
    """
    global _lock_fd
    if _lock_fd is not None:
        return True  # already held by this process
    path = _lock_path()
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        os.close(fd)
        return False
    meta = {
        "pid": os.getpid(),
        "host": _socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "socket": setting("EPISTEMIC_GRAPH_SOCKET", "/tmp/epistemic-graph.sock"),  # nosec B108 — default UDS path, overridable via env
        "role": setting("KG_DAEMON_ROLE", "auto"),
    }
    try:
        data = json.dumps(meta).encode("utf-8")
        os.ftruncate(fd, 0)
        os.pwrite(fd, data, 0)
        os.fsync(fd)
    except Exception:  # pragma: no cover
        pass
    _lock_fd = fd
    atexit.register(release_host_lock)
    return True


def resolve_daemon_role(requested: str | None = None) -> str:
    """Decide the EFFECTIVE daemon role for this process via the singleton lock.

    - ``client`` -> ``client`` (never locks, never runs daemon threads).
    - ``host``   -> acquire the lock or raise :class:`KGHostAlreadyRunning`.
    - ``auto``   -> ``host`` if the lock is free, else ``client`` (self-healing).

    Cached: the first decision is reused for the process lifetime. Under
    ``AGENT_UTILITIES_TESTING`` no lock is taken (host/auto resolve to ``host`` so
    explicit ``start_task_workers()`` in tests still works).
    """
    global _effective_role
    if _effective_role is not None:
        return _effective_role
    role = (requested or setting("KG_DAEMON_ROLE") or "auto").strip().lower()
    if role not in {"host", "client", "auto"}:
        role = "auto"

    if os.environ.get("AGENT_UTILITIES_TESTING"):
        _effective_role = "client" if role == "client" else "host"
        return _effective_role

    if role == "client":
        _effective_role = "client"
    elif role == "host":
        if _try_acquire():
            _effective_role = "host"
        else:
            h = _read_holder(_lock_path())
            raise KGHostAlreadyRunning(
                "A KG host daemon is already running — "
                f"pid={h.get('pid', '?')} host={h.get('host', '?')} "
                f"started={h.get('started_at', '?')} socket={h.get('socket', '?')}. "
                "Refusing to start a second host. Run with KG_DAEMON_ROLE=client to "
                "connect to the running engine, or stop that process (SIGTERM) first."
            )
    else:  # auto
        if _try_acquire():
            _effective_role = "host"
        else:
            h = _read_holder(_lock_path())
            logger.info(
                "KG host daemon already running (pid=%s, started=%s) — this process "
                "runs as a client (no daemon threads).",
                h.get("pid", "?"),
                h.get("started_at", "?"),
            )
            _effective_role = "client"
    return _effective_role


def effective_daemon_role() -> str:
    """Return the cached effective role, resolving lazily from the env if unset."""
    return _effective_role if _effective_role is not None else resolve_daemon_role()


def is_host() -> bool:
    """True iff this process holds the host lock (runs the consolidated daemon)."""
    return effective_daemon_role() == "host"


def host_lock_holder() -> dict[str, Any] | None:
    """Return the recorded host identity metadata, or None if no lockfile yet."""
    return _read_holder(_lock_path()) or None


def release_host_lock() -> None:
    """Release the host lock (idempotent). Called on graceful shutdown + atexit."""
    global _lock_fd
    fd = _lock_fd
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:  # pragma: no cover
        pass
    try:
        os.close(fd)
    except Exception:  # pragma: no cover
        pass
    _lock_fd = None
    logger.info("Released KG host daemon lock.")
