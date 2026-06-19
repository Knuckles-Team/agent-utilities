"""Per-session memento pre-prime cache (CONCEPT:KG-2.131).

``get_recent_mementos(source=session, limit=3)`` is a synchronous backend round-trip.
Running it inline on the async reply path (inside ``_build_execution_config``) blocked
the event loop on every chat turn. This module keeps a small per-source LRU cache of the
recently-recalled mementos so a turn reads them from memory (zero I/O on the hot path).

The cache is refreshed by the existing background ``_persist_and_enrich`` pass
(CONCEPT:ECO-4.74): after each turn that pass already *writes* the new memento under the
session source, so it also refreshes this cache for that source — turn ``N+1`` reads turn
``N``'s memento from memory, never from a blocking query. A cold miss fetches once (the
caller does so off the loop via ``to_thread``) and populates.

It lives in ``knowledge_graph/memory/`` (core) so every entrypoint inherits it, keyed by
the same ``memento_source`` the universal path already threads (Universal capability).
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

# Bounded LRU: chat sessions come and go; cap memory. 512 distinct active sources is far
# more than any single gateway holds at once, and each entry is a tiny list of strings.
_MAX_SESSIONS = 512

# How many mementos the universal path primes per run (matches the prior inline limit).
DEFAULT_MEMENTO_LIMIT = 3


class SessionMementoCache:
    """Process-local LRU of ``{source -> (mementos, fetched_at)}`` (CONCEPT:KG-2.131).

    Thread-safe: the background refresh pass and the reply path touch it concurrently.
    A process-singleton (:func:`instance`) so the reply path and the background enrich
    pass share one cache.
    """

    _instance: SessionMementoCache | None = None
    _instance_lock = threading.Lock()

    def __init__(self, max_sessions: int = _MAX_SESSIONS) -> None:
        self._max = max_sessions
        self._lock = threading.Lock()
        self._store: OrderedDict[str, tuple[list[str], float]] = OrderedDict()

    @classmethod
    def instance(cls) -> SessionMementoCache:
        """Return the process-wide singleton cache."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get(self, source: str) -> list[str] | None:
        """Return the cached mementos for ``source``, or ``None`` on a miss.

        A hit moves the entry to most-recently-used. Returns a *copy* so callers can't
        mutate the cached list.
        """
        if not source:
            return None
        with self._lock:
            entry = self._store.get(source)
            if entry is None:
                return None
            self._store.move_to_end(source)
            return list(entry[0])

    def put(self, source: str, mementos: list[str]) -> None:
        """Cache ``mementos`` for ``source`` (most-recently-used), evicting the LRU."""
        if not source:
            return
        with self._lock:
            self._store[source] = (list(mementos), time.monotonic())
            self._store.move_to_end(source)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def invalidate(self, source: str) -> None:
        """Drop the cached entry for ``source`` (e.g. after a wipe)."""
        with self._lock:
            self._store.pop(source, None)

    def clear(self) -> None:
        """Drop all cached entries (tests / reset)."""
        with self._lock:
            self._store.clear()


def refresh_session_memento_cache(
    engine, source: str, limit: int = DEFAULT_MEMENTO_LIMIT
) -> list[str]:
    """Re-read the recent mementos for ``source`` and store them in the cache.

    Called from the background ``_persist_and_enrich`` pass (off the reply path), so a
    synchronous fetch here is fine — it never touches the latency-critical turn. Returns
    the freshly-fetched mementos (also cached). CONCEPT:KG-2.131 / ECO-4.74.
    """
    from agent_utilities.knowledge_graph.memory.memento_compressor import (
        get_recent_mementos,
    )

    mementos = get_recent_mementos(engine, source=source, limit=limit)
    SessionMementoCache.instance().put(source, mementos)
    return mementos
