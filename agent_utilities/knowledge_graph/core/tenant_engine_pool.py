# CONCEPT:AU-KG.sharding.elastic-over-kg-shard - Elastic per-tenant L1 engine pool: a bounded warm set of tenant-graph engine clients with LRU eviction (cold tenants snapshotted/dropped) and hydrate-on-miss layered over the KG-2.58 shard routing
"""Bounded, warm pool of per-tenant engine clients with LRU eviction.

KG-2.58 routes each tenant to a named graph on a (statically configured) shard;
this is the **elastic** layer that makes "thousands of tenants" tractable in one
process: only the ``capacity`` most-recently-used tenant graphs are kept *warm*
(an open engine client + its resident working set). When a new tenant is touched
past capacity, the least-recently-used tenant is **evicted** — snapshotted/dropped
and its client closed — and a later access **hydrates** it again.

Eviction is never lossy: the tiered backend mirrors every write to the durable
L3 tier (Postgres/pggraph), which remains the source of truth, so an evicted
tenant's data is reloaded from L3 on the next access. The pool only governs L1
*residency*.

Disabled by default (``KG_ENGINE_POOL_SIZE=0`` → engines are constructed per use
exactly as today). The engine factory, hydrate, and evict steps are injectable
so the LRU policy is unit-testable without a live engine, and so the eviction
hook can call a future engine-side per-graph unload when one ships.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from ...security.brain_context import current_actor
from .shard_topology import default_graph_name, tenant_graph_name

logger = logging.getLogger(__name__)

__all__ = ["TenantEnginePool", "acquire_engine", "get_engine_pool", "reset_engine_pool"]


def _default_factory(graph_name: str) -> Any:
    """Construct an engine client bound to ``graph_name`` (KG-2.58 routing)."""
    from .graph_compute import GraphComputeEngine

    return GraphComputeEngine(graph_name=graph_name)


def _drop_on_evict_enabled() -> bool:
    """Whether eviction unloads the tenant graph from the engine (needs L3)."""
    try:
        from agent_utilities.core.config import config

        return bool(getattr(config, "kg_engine_pool_drop_on_evict", False))
    except Exception:  # noqa: BLE001 — default to the safe (no-drop) behaviour
        return False


def _default_evict(graph_name: str, engine: Any) -> None:
    """Release an evicted tenant's L1 residency.

    Always closes the process-local client. When ``KG_ENGINE_POOL_DROP_ON_EVICT``
    is set (safe only when the data is durably mirrored to L3, which re-hydrates
    on the next access), it also unloads the tenant's named graph from the engine
    via :meth:`GraphComputeEngine.drop_graph` to reclaim engine memory.
    """
    if _drop_on_evict_enabled():
        drop = getattr(engine, "drop_graph", None)
        if callable(drop):
            try:
                drop()
            except Exception as exc:  # noqa: BLE001 — best-effort unload
                logger.debug("drop_graph %s skipped: %s", graph_name, exc)
    for method in ("close", "disconnect"):
        fn = getattr(engine, method, None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 — eviction is best-effort
                logger.debug("evict %s via %s skipped: %s", graph_name, method, exc)
            break


class TenantEnginePool:
    """An LRU-bounded set of warm per-tenant engine clients.

    Thread-safe (front pods serve many tenants concurrently). ``capacity <= 0``
    means *unbounded passthrough*: every :meth:`acquire` builds a fresh engine
    and nothing is cached or evicted — byte-for-byte today's behaviour.
    """

    def __init__(
        self,
        capacity: int,
        factory: Callable[[str], Any] | None = None,
        on_evict: Callable[[str, Any], None] | None = None,
        on_hydrate: Callable[[str, Any], None] | None = None,
    ) -> None:
        self.capacity = int(capacity)
        self._factory = factory or _default_factory
        self._on_evict = on_evict or _default_evict
        self._on_hydrate = on_hydrate
        self._warm: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # -- resolution --------------------------------------------------------
    def _graph_for(self, tenant: str | None, graph_name: str | None) -> str:
        if graph_name:
            return graph_name
        base = default_graph_name()
        t = tenant if tenant is not None else current_actor().tenant_id
        return tenant_graph_name(t, base=base)

    # -- core --------------------------------------------------------------
    def acquire(self, tenant: str | None = None, graph_name: str | None = None) -> Any:
        """Return a warm engine for the tenant's graph, hydrating on a miss.

        Passthrough (capacity <= 0) builds and returns a fresh engine without
        caching. Otherwise the warm set is consulted; a miss constructs +
        hydrates the engine and may evict the LRU tenant to stay within capacity.
        """
        graph = self._graph_for(tenant, graph_name)
        if self.capacity <= 0:
            return self._factory(graph)

        with self._lock:
            engine = self._warm.get(graph)
            if engine is not None and self._is_live(engine):
                self._warm.move_to_end(graph)  # most-recently-used
                self.hits += 1
                return engine
            if engine is not None:
                # Warm engine failed its liveness probe (dead socket) — discard it
                # and rebuild below, so a stale connection isn't handed out.
                self._warm.pop(graph, None)
                try:
                    self._on_evict(graph, engine)
                except Exception:  # noqa: BLE001 — eviction must not raise
                    pass

            self.misses += 1
            engine = self._factory(graph)
            if self._on_hydrate is not None:
                try:
                    self._on_hydrate(graph, engine)
                except Exception as exc:  # noqa: BLE001 — hydrate is best-effort
                    logger.debug("hydrate %s skipped: %s", graph, exc)
            self._warm[graph] = engine
            self._evict_to_capacity()
            return engine

    @staticmethod
    def _is_live(engine: Any) -> bool:
        """Best-effort liveness check before handing out a warm engine.

        Probes a ``ping``/``health`` method if the engine exposes one; otherwise
        trusts the circuit-breaker to fast-fail a dead endpoint on first use (so
        this is a safe no-op for engine types without a probe). Keeps a stale
        warm connection from being silently reused after the socket died.
        """
        probe = getattr(engine, "ping", None) or getattr(engine, "health", None)
        if not callable(probe):
            return True
        try:
            probe()
            return True
        except Exception:  # noqa: BLE001 — any failure ⇒ treat as dead
            return False

    def _evict_to_capacity(self) -> None:
        while len(self._warm) > self.capacity:
            old_graph, old_engine = self._warm.popitem(last=False)  # LRU
            self.evictions += 1
            try:
                self._on_evict(old_graph, old_engine)
            except Exception as exc:  # noqa: BLE001 — eviction must not raise
                logger.debug("evict %s failed: %s", old_graph, exc)

    # -- introspection / lifecycle ----------------------------------------
    def warm_tenants(self) -> list[str]:
        """Currently-resident graph names, LRU-first."""
        with self._lock:
            return list(self._warm.keys())

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "capacity": self.capacity,
                "warm": len(self._warm),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def clear(self) -> None:
        """Evict everything (shutdown / test reset)."""
        with self._lock:
            while self._warm:
                graph, engine = self._warm.popitem(last=False)
                try:
                    self._on_evict(graph, engine)
                except Exception:  # noqa: BLE001
                    pass


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_POOL: TenantEnginePool | None = None
_POOL_LOCK = threading.Lock()


def get_engine_pool() -> TenantEnginePool:
    """The lazily-built, process-wide pool sized from ``KG_ENGINE_POOL_SIZE``."""
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                try:
                    from agent_utilities.core.config import config

                    cap = int(getattr(config, "kg_engine_pool_size", 0) or 0)
                except Exception:  # noqa: BLE001 — default to passthrough
                    cap = 0
                _POOL = TenantEnginePool(capacity=cap)
                if cap > 0:
                    logger.info("Tenant engine pool active (capacity=%d)", cap)
    return _POOL


def acquire_engine(tenant: str | None = None, graph_name: str | None = None) -> Any:
    """Convenience: acquire a (possibly warm) engine from the process pool."""
    return get_engine_pool().acquire(tenant=tenant, graph_name=graph_name)


def reset_engine_pool() -> None:
    """Drop the singleton (test helper; clears the warm set first)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.clear()
        _POOL = None
