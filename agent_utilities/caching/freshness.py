"""Declared cache freshness driven by the engine's invalidation feed (AU-SEC-R003).

CONCEPT:AU-KG.memory.semantic-response-cache — the freshness half of the cache contract.

The engine publishes, per graph, one invalidation event for every committed write (the classes
and edge types it touched, or ``all`` for a write it could not attribute) and the declared
``eg:volatilityClass`` policy of its classes (``FreshnessFeed``, AU-SEC-R003). This module turns that
into two things every AU cache layer shares:

* **Invalidation by class.** A :class:`FreshnessHub` applies each feed page to the caches attached
  to it (anything implementing :class:`ClassInvalidatable`): an event naming class ``C`` drops
  exactly the entries that depend on ``C``; a coarse event, a gap in the feed or a new engine
  epoch drops everything cached for the graph.
* **TTL from the schema.** :meth:`FreshnessHub.ttl_for` derives an entry's time bound from the
  declared volatility of the classes it depends on, instead of each caller inventing one.

The learned signal is monotone-safe: an observed change rate (:class:`ChangeRateEstimator`) may
only SHORTEN a declared TTL (:func:`shorten_only`); lengthening one takes an edit to the declared
policy in the graph. Every uncertain state falls back, never forward: an undeclared class has no
default TTL, a ``live`` class is never cached, and a hub whose feed has not been read recently
enough (``max_feed_lag_s``) grants no declared TTL — events it has not seen may already have
retired an entry — so a cache falls back to the caller's own tolerance (the contract before
AU-SEC-R003) or refuses. A feed that could not be read drops everything the hub guards.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "ChangeRateEstimator",
    "ClassFreshness",
    "ClassInvalidatable",
    "FreshnessHub",
    "VolatilityPolicy",
    "combine_tolerance",
    "freshness_hub",
    "poll_engine",
    "shorten_only",
    "store_scoped",
]

#: The engine wire method this module consumes.
FEED_METHOD = "FreshnessFeed"


@dataclass(frozen=True)
class ClassFreshness:
    """One class's declared freshness: its volatility and time bound (``math.inf`` = none)."""

    volatility: str
    max_staleness_s: float


class VolatilityPolicy:
    """The declared class policy of one graph, as the engine resolved it."""

    def __init__(
        self,
        classes: Mapping[str, ClassFreshness] | None = None,
        version: int | None = None,
    ) -> None:
        self.classes: dict[str, ClassFreshness] = dict(classes or {})
        self.version = version

    @classmethod
    def from_feed(
        cls, rows: Iterable[Mapping[str, Any]], version: int
    ) -> VolatilityPolicy:
        """Build from the feed's ``policy`` rows (``class``/``volatility``/``max_staleness_ms``)."""
        classes = {str(row["class"]): _class_freshness(row) for row in rows}
        return cls(classes, version)

    def declared_ttl(self, classes: Iterable[str]) -> float | None:
        """The tightest declared bound over ``classes``; ``None`` when any is undeclared (or
        none is given) — an undeclared class has no default, so the caller must say."""
        bounds = [self.classes.get(name) for name in classes]
        if not bounds or any(bound is None for bound in bounds):
            return None
        return min(bound.max_staleness_s for bound in bounds if bound is not None)


def _class_freshness(row: Mapping[str, Any]) -> ClassFreshness:
    bound_ms = row.get("max_staleness_ms")
    bound = math.inf if bound_ms is None else float(bound_ms) / 1000.0
    return ClassFreshness(
        volatility=str(row.get("volatility", "live")), max_staleness_s=bound
    )


def shorten_only(declared: float, learned: float | None) -> float:
    """Fold a learned TTL into a declared one. The learned value may only shorten it."""
    if learned is None:
        return declared
    return min(declared, learned)


def combine_tolerance(caller: float | None, declared: float | None) -> float | None:
    """A cache entry's effective freshness tolerance in seconds, or ``None`` to refuse caching.

    ``caller`` is what the caller declared (``None`` = unsaid, negative = invalid); ``declared``
    is the class-derived bound (``None`` = no declaration applies). A declared bound only
    CAPS the caller's; a declared bound of zero (a ``live`` class, or a hub too far behind its
    feed) refuses outright.
    """
    if caller is not None and caller < 0:
        return None
    if declared is None:
        return caller
    if declared <= 0:
        return None
    return declared if caller is None else min(caller, declared)


class ChangeRateEstimator:
    """The observed change interval of each class, from the invalidation events it receives.

    An exponentially weighted mean of the seconds between consecutive events naming a class.
    Its TTL suggestion is ``safety × mean interval`` once ``min_observations`` intervals were
    seen — a class that changes every ten seconds should not be served from a one-minute-old
    entry even if its declaration allows one. The suggestion is only ever used through
    :func:`shorten_only`.
    """

    def __init__(
        self, *, alpha: float = 0.2, safety: float = 0.5, min_observations: int = 3
    ) -> None:
        self._alpha = alpha
        self._safety = safety
        self._min_observations = max(1, int(min_observations))
        self._last_seen: dict[str, float] = {}
        self._mean_interval: dict[str, float] = {}
        self._observations: dict[str, int] = {}

    def observe(self, name: str, at: float) -> None:
        """Record that ``name`` changed at monotonic time ``at``."""
        previous = self._last_seen.get(name)
        self._last_seen[name] = at
        if previous is None:
            return
        interval = max(0.0, at - previous)
        mean = self._mean_interval.get(name, interval)
        self._mean_interval[name] = mean + self._alpha * (interval - mean)
        self._observations[name] = self._observations.get(name, 0) + 1

    def learned_ttl(self, name: str) -> float | None:
        """The suggested bound for ``name``, or ``None`` before enough observations."""
        if self._observations.get(name, 0) < self._min_observations:
            return None
        return self._safety * self._mean_interval[name]


class ClassInvalidatable(Protocol):
    """A cache the hub can invalidate. Both return how many entries were dropped."""

    def invalidate_classes(self, graph: str, classes: frozenset[str]) -> int: ...

    def invalidate_graph(self, graph: str) -> int: ...


class FreshnessHub:
    """One graph's feed cursor, declared policy and learned change rates, and the caches that
    subscribe to its invalidation events."""

    def __init__(
        self,
        graph: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        rates: ChangeRateEstimator | None = None,
        max_feed_lag_s: float = 60.0,
    ) -> None:
        self.graph = graph
        self.cursor = 0
        self.epoch: int | None = None
        self.policy = VolatilityPolicy()
        self._clock = clock
        self._rates = rates or ChangeRateEstimator()
        self._max_feed_lag_s = max_feed_lag_s
        self._last_read: float | None = None
        self._targets: list[ClassInvalidatable] = []
        self._lock = threading.Lock()

    def attach(self, target: ClassInvalidatable) -> None:
        """Subscribe ``target`` to this graph's invalidation events (idempotent)."""
        with self._lock:
            if all(existing is not target for existing in self._targets):
                self._targets.append(target)

    def feed_request(self) -> dict[str, Any]:
        """The parameters of the next ``FreshnessFeed`` read."""
        return {
            "after_version": self.cursor,
            "limit": 0,
            "policy_after": self.policy.version,
        }

    def apply_feed(self, feed: Mapping[str, Any]) -> int:
        """Apply one feed page; returns the number of cache entries invalidated."""
        epoch = int(feed.get("epoch", 0))
        restarted = bool(feed.get("gap")) or (
            self.epoch is not None and epoch != self.epoch
        )
        head = int(feed.get("head_version", 0))
        if restarted:
            dropped = self._invalidate_graph()
            self.cursor = head
        else:
            dropped = self._apply_events(feed)
        self._adopt_policy(feed, reset=restarted)
        self.epoch = epoch
        self._last_read = self._clock()
        return dropped

    def mark_unreachable(self) -> int:
        """The feed could not be read: events may have been missed, so drop everything and grant
        no TTL until the next successful read."""
        self._last_read = None
        return self._invalidate_graph()

    def ttl_for(self, classes: Iterable[str]) -> float | None:
        """Seconds an entry depending on ``classes`` may be served; ``None`` = no declaration
        applies — an undeclared class, or a feed not read within ``max_feed_lag_s`` — so the
        caller's own tolerance governs; ``0.0`` = do not serve (a ``live`` class)."""
        if not self._feed_is_current():
            return None
        names = frozenset(classes)
        declared = self.policy.declared_ttl(names)
        if declared is None:
            return None
        learned = [
            ttl for ttl in map(self._rates.learned_ttl, names) if ttl is not None
        ]
        return shorten_only(declared, min(learned, default=None))

    def _feed_is_current(self) -> bool:
        if self._last_read is None:
            return False
        return self._clock() - self._last_read <= self._max_feed_lag_s

    def _apply_events(self, feed: Mapping[str, Any]) -> int:
        """Apply a page's events and advance the cursor past them. An empty page means nothing
        follows the cursor up to ``head_version``; a non-empty one may be a partial page, so the
        cursor moves only past what was applied."""
        events = list(feed.get("events") or [])
        now = self._clock()
        dropped = sum(self._apply_event(event, now) for event in events)
        if events:
            self.cursor = max(self.cursor, int(events[-1].get("version", 0)))
        else:
            self.cursor = max(self.cursor, int(feed.get("head_version", 0)))
        return dropped

    def _apply_event(self, event: Mapping[str, Any], now: float) -> int:
        if event.get("scope") == "all":
            return self._invalidate_graph()
        names = frozenset(event.get("classes") or ()) | frozenset(
            event.get("edge_types") or ()
        )
        for name in names:
            self._rates.observe(name, now)
        return sum(
            target.invalidate_classes(self.graph, names) for target in self._snapshot()
        )

    def _adopt_policy(self, feed: Mapping[str, Any], *, reset: bool) -> None:
        rows = feed.get("policy")
        if rows is not None:
            self.policy = VolatilityPolicy.from_feed(
                rows, int(feed.get("policy_version", 0))
            )
        elif reset:
            # A new epoch invalidates the policy we hold too: ask for it again next read.
            self.policy = VolatilityPolicy(self.policy.classes, None)

    def _invalidate_graph(self) -> int:
        return sum(target.invalidate_graph(self.graph) for target in self._snapshot())

    def _snapshot(self) -> list[ClassInvalidatable]:
        with self._lock:
            return list(self._targets)


_hubs: dict[str, FreshnessHub] = {}
_hubs_lock = threading.Lock()


def freshness_hub(graph: str) -> FreshnessHub:
    """The process-wide hub of ``graph`` (created on first use)."""
    with _hubs_lock:
        hub = _hubs.get(graph)
        if hub is None:
            hub = _hubs[graph] = FreshnessHub(graph)
        return hub


#: ``send(method, params, graph)`` — the engine client's request coroutine.
FeedSend = Callable[[str, dict[str, Any], str], Awaitable[Any]]


async def poll_engine(hub: FreshnessHub, send: FeedSend) -> int:
    """Read one ``FreshnessFeed`` page for ``hub.graph`` and apply it.

    A failed read marks the hub unreachable (everything it guards is dropped and no TTL is
    granted until the next good read) and the error propagates to the caller.
    """
    read = False
    try:
        raw = await send(FEED_METHOD, hub.feed_request(), hub.graph)
        dropped = hub.apply_feed(feed_mapping(raw))
        read = True
        return dropped
    finally:
        if not read:
            logger.warning("freshness feed read failed for graph %s", hub.graph)
            hub.mark_unreachable()


def feed_mapping(raw: Any) -> Mapping[str, Any]:
    """The feed body of an engine response: a mapping, or a result wrapper carrying one."""
    if isinstance(raw, Mapping):
        return raw
    payload = getattr(raw, "payload", None)
    if isinstance(payload, Mapping):
        return payload
    raise TypeError(f"unexpected FreshnessFeed result: {type(raw).__name__}")


def store_scoped(
    backend: Any, key: str, blob: bytes, graph: str, classes: frozenset[str]
) -> bool:
    """Store ``blob`` under ``key`` in a KV-style cache, class-scoped when the backend supports
    it (``put_scoped``) and the value depends on known classes; a plain ``put`` otherwise."""
    put_scoped = getattr(backend, "put_scoped", None)
    if put_scoped is None or not classes:
        return bool(backend.put(key, blob))
    return bool(put_scoped(key, blob, graph=graph, classes=classes))
