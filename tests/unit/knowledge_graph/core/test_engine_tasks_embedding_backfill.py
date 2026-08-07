"""D-EMB: the pgvector-only embedding backfill daemon must fall back to a
native-engine reconciliation path instead of silently no-op'ing forever.

Root cause (see `TaskManagerMixin._tick_embedding_backfill_generic`'s
docstring): `_tick_embedding_backfill` returns 0 immediately on any backend
that isn't `PostgreSQLBackend` -- which is every production topology this
codebase actually runs on (native Ladybug/redb engine via
`BrainGuardedBackend`). The "dedicated vector-embedding backfill drain"
thread has been alive and ticking the whole time; it just never did anything.
"""

from __future__ import annotations

import time

from agent_utilities.knowledge_graph.core.engine_tasks import (
    _EMBED_BACKFILL_GENERIC_INTERVAL_S,
    TaskManagerMixin,
)


class _NonPgvectorBackend:
    """A backend with none of the pgvector-only attributes."""


class _NativeBackendWithHydrate:
    def __init__(self, count: int = 0, *, raises: bool = False) -> None:
        self.count = count
        self.raises = raises
        self.calls = 0

    def hydrate_engine_embeddings(self, batch_log_every: int = 5000) -> int:
        self.calls += 1
        if self.raises:
            raise RuntimeError("engine unreachable")
        return self.count


def _bare_task_manager() -> TaskManagerMixin:
    """A TaskManagerMixin instance without running its (heavy) __init__."""
    return object.__new__(TaskManagerMixin)  # type: ignore[return-value]


def test_tick_embedding_backfill_returns_zero_for_backend_without_hydrate():
    """Neither pgvector attributes NOR `hydrate_engine_embeddings` -> 0, not
    an exception (e.g. a bare in-memory test double)."""
    mgr = _bare_task_manager()
    mgr.backend = _NonPgvectorBackend()

    assert mgr._tick_embedding_backfill() == 0


def test_tick_embedding_backfill_falls_back_to_generic_hydrate():
    """The pgvector gate no-ops, but the native-engine fallback actually
    reconciles `embedding`-property rows into the ANN index."""
    mgr = _bare_task_manager()
    backend = _NativeBackendWithHydrate(count=7)
    mgr.backend = backend

    embedded = mgr._tick_embedding_backfill()

    assert embedded == 7
    assert backend.calls == 1


def test_generic_backfill_is_rate_limited():
    """A full `hydrate_engine_embeddings` scan must not run every tick (that
    would add a full-graph scan to an already-contended engine every cycle,
    D-PERF-2) -- only once per _EMBED_BACKFILL_GENERIC_INTERVAL_S."""
    mgr = _bare_task_manager()
    backend = _NativeBackendWithHydrate(count=3)
    mgr.backend = backend

    first = mgr._tick_embedding_backfill_generic()
    second = mgr._tick_embedding_backfill_generic()

    assert first == 3
    assert second == 0  # rate-limited: no second hydrate call yet
    assert backend.calls == 1


def test_generic_backfill_reruns_after_interval_elapses(monkeypatch):
    mgr = _bare_task_manager()
    backend = _NativeBackendWithHydrate(count=1)
    mgr.backend = backend

    t = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])
    assert mgr._tick_embedding_backfill_generic() == 1
    t[0] += _EMBED_BACKFILL_GENERIC_INTERVAL_S + 1
    assert mgr._tick_embedding_backfill_generic() == 1
    assert backend.calls == 2


def test_generic_backfill_failure_is_best_effort():
    """A hydrate failure (engine unreachable) must degrade to 0, never raise
    -- this runs inside a daemon loop that must never die."""
    mgr = _bare_task_manager()
    mgr.backend = _NativeBackendWithHydrate(raises=True)

    assert mgr._tick_embedding_backfill_generic() == 0
