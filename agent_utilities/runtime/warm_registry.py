"""CONCEPT:AU-OS.host.so-they-are-idle — Host-singleton warm-parent pool for the warm-fork sandbox family.

Generalises the per-class pool + idle-reap pattern of :class:`~.docker_workspace.DockerWorkspace`
(``_REGISTRY`` + ``reap_idle``) into one substrate-agnostic registry shared by every warm-fork
rung (``forkserver``, ``container_fork``, ``firecracker`` — CONCEPT:AU-ORCH.sandbox.warmforkfanoutcapability/1.86). A rung pays
warm-up once for a :class:`~agent_utilities.rlm.sandboxes.base.WarmSpec`, registers the resulting
parent here keyed by the spec's content hash, and subsequent fan-out *reuses* the parent instead
of re-warming.

Deliberately dependency-light and **opaque**: it stores parent objects + a synchronous ``close``
callable + metadata, and never imports the sandbox layer — so there is no ``runtime`` ↔ ``rlm``
import cycle, and reaping (driven from a synchronous maintenance tick, CONCEPT:AU-OS.state.unified-scheduling-one-intelligent) needs no
event loop. The pool is size-capped by :func:`compute_warm_parent_count` (auto-sized to host
RAM/CPU per *Configuration discipline* — warm parents are memory-heavy); registering past the cap
evicts the least-recently-used parent.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# A warm parent (a loaded process / warmed container / live microVM) idle longer than this is
# reaped. Mirrors DockerWorkspace.reap_idle's 3600s default.
DEFAULT_IDLE_TTL_SECS = 1800.0

# CONCEPT:AU-ORCH.sandbox.warm-parent-lifetime-cap — a HARD wall-clock lifetime cap. A warm parent older than this is reaped even
# if it is continuously borrowed/busy (so its idle clock never advances). Idle-reaping alone cannot
# evict a parent whose forked child is spinning at 100% CPU — that is exactly the runaway a hard age
# cap closes. The backend's own container/process self-expiry is the primary defence; this is the
# in-registry backstop on the same budget.
DEFAULT_MAX_AGE_SECS = 3600.0

# Assumed resident footprint of one warm parent (imports + caches loaded). Used only to bound the
# pool to available RAM; the real number varies by spec, so this is a conservative divisor.
_WARM_PARENT_GIB = 1.5


def compute_warm_parent_count(configured: int | None = None) -> int:
    """Auto-size the warm-parent pool to host RAM + CPU (mirrors ``compute_ingest_worker_count``).

    An explicit ``configured`` value wins outright (the escape hatch). Otherwise bound by
    available memory (``_WARM_PARENT_GIB`` per parent) and CPU (parents are forked/run on demand,
    so cap near core count), with a floor of 2.
    """
    if configured is not None and configured > 0:
        return configured
    try:
        cpu = os.cpu_count() or 4
        max_cpu = max(2, int(cpu))
        try:
            import psutil

            avail = psutil.virtual_memory().available
            max_mem = max(1, int(avail / (_WARM_PARENT_GIB * (1024**3))))
        except Exception:  # noqa: BLE001 - psutil optional; fall back to CPU bound only
            max_mem = max_cpu
        return max(2, min(max_cpu, max_mem))
    except Exception:  # noqa: BLE001
        return 4


@dataclass
class _Entry:
    parent: Any
    close: Callable[[], None]
    kind: str
    last_used: float
    created: float
    borrows: int = field(default=0)


class WarmParentRegistry:
    """Process-wide pool of warm parents keyed by :attr:`WarmSpec.key`.

    Thread-safe: the asyncio backends warm/acquire from the event-loop thread while the
    maintenance tick reaps from a worker thread.
    """

    _instance: WarmParentRegistry | None = None
    _instance_lock = threading.Lock()

    def __init__(self, *, max_parents: int | None = None) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._max = compute_warm_parent_count(max_parents)

    # ── singleton ────────────────────────────────────────────────────────────
    @classmethod
    def get(cls) -> WarmParentRegistry:
        """The host-wide registry singleton (instantiated by the host daemon, OS-5.58)."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reap_active(
        cls,
        max_idle_secs: float = DEFAULT_IDLE_TTL_SECS,
        max_age_secs: float = DEFAULT_MAX_AGE_SECS,
    ) -> list[str]:
        """Reap the singleton if it exists; no-op (and don't instantiate) otherwise."""
        return (
            []
            if cls._instance is None
            else cls._instance.reap(max_idle_secs, max_age_secs)
        )

    @classmethod
    def drain_active(cls) -> list[str]:
        """Drain the singleton if it exists; no-op (and don't instantiate) otherwise."""
        return [] if cls._instance is None else cls._instance.drain()

    # ── pool ops ─────────────────────────────────────────────────────────────
    def acquire(self, key: str) -> Any | None:
        """Return the warm parent for ``key`` (refreshing its idle clock), or ``None`` if absent."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            entry.last_used = time.time()
            entry.borrows += 1
            return entry.parent

    def register(
        self, key: str, parent: Any, *, close: Callable[[], None], kind: str
    ) -> None:
        """Store a freshly-warmed ``parent``. Evicts the LRU parent if the pool is at capacity.

        ``close`` is a *synchronous* teardown for the parent (terminate the process, rm the
        container, kill the microVM) — invoked on reap/drain/eviction.
        """
        now = time.time()
        with self._lock:
            if key in self._entries:  # someone else warmed concurrently; keep the first
                self._entries[key].last_used = now
                self._safe_close(close, kind)  # drop the duplicate we were handed
                return
            while len(self._entries) >= self._max and self._entries:
                lru = min(self._entries.items(), key=lambda kv: kv[1].last_used)[0]
                self._evict_locked(lru, reason="capacity")
            self._entries[key] = _Entry(
                parent=parent, close=close, kind=kind, last_used=now, created=now
            )

    def reap(
        self,
        max_idle_secs: float = DEFAULT_IDLE_TTL_SECS,
        max_age_secs: float = DEFAULT_MAX_AGE_SECS,
    ) -> list[str]:
        """Close + drop parents idle longer than ``max_idle_secs`` OR older than ``max_age_secs``.

        The age cap (CONCEPT:AU-ORCH.sandbox.warm-parent-lifetime-cap) is the strict-lifetime backstop: a parent whose forked
        child is busy-looping refreshes neither ``last_used`` nor frees CPU, so idle-reaping never
        evicts it — the hard age check does. Returns the reaped keys. Wired into the host
        maintenance tick (``_tick_warm_parent_reap``, CONCEPT:AU-OS.host.so-they-are-idle).
        """
        now = time.time()
        with self._lock:
            stale = [
                k
                for k, e in self._entries.items()
                if now - e.last_used > max_idle_secs or now - e.created > max_age_secs
            ]
            for key in stale:
                aged = now - self._entries[key].created > max_age_secs
                self._evict_locked(key, reason="max_age" if aged else "idle")
            return stale

    def drain(self) -> list[str]:
        """Close + drop every warm parent (host-daemon shutdown)."""
        with self._lock:
            keys = list(self._entries)
            for key in keys:
                self._evict_locked(key, reason="drain")
            return keys

    def stats(self) -> dict[str, Any]:
        """Snapshot for the doctor check / metrics: pool size, cap, per-kind counts."""
        with self._lock:
            by_kind: dict[str, int] = {}
            for e in self._entries.values():
                by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
            return {
                "warm_parents": len(self._entries),
                "max_parents": self._max,
                "by_kind": by_kind,
            }

    # ── internals (call with the lock held) ───────────────────────────────────
    def _evict_locked(self, key: str, *, reason: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        logger.debug(
            "warm-parent evict key=%s kind=%s reason=%s", key, entry.kind, reason
        )
        self._safe_close(entry.close, entry.kind)

    @staticmethod
    def _safe_close(close: Callable[[], None], kind: str) -> None:
        try:
            close()
        except Exception as e:  # noqa: BLE001 - teardown is best-effort; never raise from reap
            logger.warning("warm-parent close failed (kind=%s): %s", kind, e)
