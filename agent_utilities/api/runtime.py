"""Public AU process-runtime port: engine lifecycle and the host lock (AU-3).

A hosting process (GraphOS) opens exactly one AU runtime per process and hands
the runtime's engine to AU's other public ports (``catalog_read_ports``, the
co-services). It never imports AU's knowledge-graph internals.

``role`` selects the consolidated-daemon role through the singleton host
lock: ``"host"`` takes the lock or raises :class:`HostAlreadyRunning`, while
``"client"`` never locks and never starts daemon threads.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Literal

from agent_utilities.knowledge_graph.core.host_lock import (
    KGHostAlreadyRunning as HostAlreadyRunning,
)

RuntimeRole = Literal["host", "client"]

_OPEN_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class DrainResult:
    """Outcome of draining the runtime's shared engine transport."""

    timed_out: bool
    active_requests: int | None


class AgentRuntime:
    """The one process AU runtime; a thin, typed view over its engine."""

    def __init__(self, engine: Any, role: RuntimeRole) -> None:
        self._engine = engine
        self.role: RuntimeRole = role

    @property
    def engine(self) -> Any:
        """The engine object AU's other public ports take as their authority."""
        return self._engine

    def graph_client(self, graph: str) -> Any:
        """The awaitable, session-routed EG client view for ``graph``.

        No new connection is opened; calls run on the process transport.
        """
        compute = getattr(self._engine, "graph_compute", None)
        if compute is None:
            raise RuntimeError("the AU runtime has no graph transport")
        return compute.for_graph(graph).async_client

    def start_background_daemons(self) -> None:
        self._engine.start_background_daemons()

    def start_task_workers(self, worker_count: int | None = None) -> None:
        self._engine.start_task_workers(worker_count)

    def drain_and_close(self, timeout_s: float | None = None) -> DrainResult:
        """Drain the shared transport, then close it; continuity is never claimed."""
        compute = getattr(self._engine, "graph_compute", None)
        if compute is None:
            return DrainResult(timed_out=False, active_requests=None)
        status = compute.drain(timeout_s)
        compute.close()
        return DrainResult(
            timed_out=bool(getattr(status, "timed_out", False)),
            active_requests=getattr(status, "active_requests", None),
        )


def _create_engine(defer_background_start: bool) -> Any:
    from agent_utilities.core.paths import ensure_dirs
    from agent_utilities.knowledge_graph.backends import create_backend
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    ensure_dirs()
    return IntelligenceGraphEngine(
        backend=create_backend(), defer_background_start=defer_background_start
    )


def open_process_runtime(
    *, role: RuntimeRole, defer_background_start: bool
) -> AgentRuntime:
    """Open (or return) the process AU runtime; idempotent per process.

    The first call fixes the role: ``"host"`` acquires the singleton host lock
    first and raises :class:`HostAlreadyRunning` when another process holds it.
    """
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.host_lock import resolve_daemon_role

    with _OPEN_LOCK:
        active = IntelligenceGraphEngine.get_active()
        if active is not None:
            return AgentRuntime(active, role)
        os.environ["KG_DAEMON_ROLE"] = role
        resolve_daemon_role(role)
        engine = IntelligenceGraphEngine.get_or_create(
            factory=lambda: _create_engine(defer_background_start)
        )
        return AgentRuntime(engine, role)


def acquire_host_lock() -> None:
    """Take the singleton host lock or raise :class:`HostAlreadyRunning`."""
    from agent_utilities.knowledge_graph.core.host_lock import resolve_daemon_role

    resolve_daemon_role("host")


def release_host_lock() -> None:
    from agent_utilities.knowledge_graph.core.host_lock import (
        release_host_lock as _release,
    )

    _release()


def host_lock_holder() -> dict[str, Any] | None:
    """The current lock holder's recorded identity, or ``None``."""
    from agent_utilities.knowledge_graph.core.host_lock import (
        host_lock_holder as _holder,
    )

    return _holder()


__all__ = [
    "AgentRuntime",
    "DrainResult",
    "HostAlreadyRunning",
    "RuntimeRole",
    "acquire_host_lock",
    "host_lock_holder",
    "open_process_runtime",
    "release_host_lock",
]
