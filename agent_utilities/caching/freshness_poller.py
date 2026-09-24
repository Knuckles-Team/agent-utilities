"""The running driver of the engine invalidation feed (EH-401).

CONCEPT:AU-KG.memory.semantic-response-cache — a :class:`~agent_utilities.caching.freshness.
FreshnessHub` only keeps its caches correct if something keeps reading the feed. A
:class:`FeedPoller` is that something: one asyncio task per graph, started and stopped by the
server's existing ASGI lifespan (:func:`agent_utilities.server.app._app_lifespan`), polling
``FreshnessFeed`` at a bounded interval with jitter and backing off exponentially on errors.

Observable on the two existing surfaces:

* ``agent_utilities_cache_freshness_feed_age_seconds{graph}`` — seconds since the last
  successful read (the gateway ``/metrics`` endpoint);
* the ``cache_freshness`` check of :func:`agent_utilities.observability.runtime_health.
  collect_health` — ``degraded`` when a running poller has not read its feed within its lag
  bound (never a readiness failure: the caches then fall back to callers' own tolerances).

A failed read never ends the poller. It marks the hub unreachable, which drops everything the hub
guards (see :func:`~agent_utilities.caching.freshness.poll_engine`), and the next attempt waits
longer.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

from agent_utilities.caching.freshness import (
    FeedSend,
    FreshnessHub,
    freshness_hub,
    poll_engine,
)

logger = logging.getLogger(__name__)

__all__ = [
    "FeedPoller",
    "poller_statuses",
    "start_engine_feed_poller",
    "start_process_feed_poller",
    "stop_process_feed_poller",
]

_DEFAULT_INTERVAL_S = 5.0
_DEFAULT_JITTER = 0.2
_DEFAULT_MAX_BACKOFF_S = 300.0

#: The running pollers, one per graph.
_pollers: dict[str, FeedPoller] = {}


class FeedPoller:
    """Poll one graph's ``FreshnessFeed`` into its hub until stopped."""

    def __init__(
        self,
        hub: FreshnessHub,
        send: FeedSend,
        *,
        interval_s: float = _DEFAULT_INTERVAL_S,
        jitter: float = _DEFAULT_JITTER,
        max_backoff_s: float = _DEFAULT_MAX_BACKOFF_S,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.hub = hub
        self._send = send
        self._interval_s = max(0.0, interval_s)
        self._jitter = max(0.0, jitter)
        self._max_backoff_s = max(self._interval_s, max_backoff_s)
        self._sleep = sleep
        self._clock = clock
        self._rng = rng
        self.failures = 0
        self.last_success: float | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start the poll loop on the running event loop (idempotent)."""
        if not self.running:
            self._task = asyncio.create_task(
                self._run(), name=f"freshness-feed:{self.hub.graph}"
            )

    async def stop(self) -> None:
        """Cancel the poll loop and wait for it to finish."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def poll_once(self) -> bool:
        """One read. Returns whether it succeeded; a failure is logged, never raised."""
        (outcome,) = await asyncio.gather(
            poll_engine(self.hub, self._send), return_exceptions=True
        )
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, BaseException):
            self.failures += 1
            logger.warning(
                "freshness feed poll failed for graph %s (exception_type=%s, failures=%d)",
                self.hub.graph,
                type(outcome).__name__,
                self.failures,
            )
            return False
        self.failures = 0
        self.last_success = self._clock()
        _record_feed_age(self.hub.graph, 0.0)
        return True

    def next_delay(self) -> float:
        """The wait before the next read: the interval, doubled per consecutive failure up to
        ``max_backoff_s``, stretched by up to ``jitter`` so pollers do not synchronise."""
        base = min(self._max_backoff_s, self._interval_s * (2**self.failures))
        return base * (1.0 + self._jitter * self._rng())

    def status(self) -> dict[str, Any]:
        """The poller's state for the health surface."""
        age = None if self.last_success is None else self._clock() - self.last_success
        lagging = not self.running or age is None or age > self.hub.max_feed_lag_s
        return {
            "graph": self.hub.graph,
            "running": self.running,
            "last_poll_age_s": age,
            "consecutive_failures": self.failures,
            "lagging": lagging,
        }

    async def _run(self) -> None:
        while True:
            await self.poll_once()
            self._publish_age()
            await self._sleep(self.next_delay())

    def _publish_age(self) -> None:
        if self.last_success is not None:
            _record_feed_age(self.hub.graph, self._clock() - self.last_success)


def _record_feed_age(graph: str, age_s: float) -> None:
    from agent_utilities.observability.gateway_metrics import (
        CACHE_FRESHNESS_FEED_AGE_SECONDS,
    )

    CACHE_FRESHNESS_FEED_AGE_SECONDS.labels(graph=graph).set(age_s)


def poller_statuses() -> list[dict[str, Any]]:
    """Every registered poller's status (the health check reads this)."""
    return [poller.status() for poller in _pollers.values()]


def register_poller(poller: FeedPoller) -> FeedPoller:
    """Register ``poller`` as its graph's one poller, replacing none: an existing running
    poller for the graph is returned instead."""
    existing = _pollers.get(poller.hub.graph)
    if existing is not None and existing.running:
        return existing
    _pollers[poller.hub.graph] = poller
    return poller


async def stop_poller(poller: FeedPoller) -> None:
    """Stop ``poller`` and forget it."""
    await poller.stop()
    if _pollers.get(poller.hub.graph) is poller:
        del _pollers[poller.hub.graph]


def start_engine_feed_poller(compute: Any) -> FeedPoller:
    """Start (or return) the poller for the process engine's graph.

    ``compute`` is the process :class:`~agent_utilities.knowledge_graph.core.graph_compute.
    GraphComputeEngine`; reads go through its raw wire escape hatch (``_send_wire``) on a
    worker thread, so the poller shares the process transport and never opens a connection.
    Must be called inside the verified session the reads run under (the task copies it).
    """
    graph = str(getattr(compute, "graph_name", "") or "")

    async def send(method: str, params: dict[str, Any], _graph: str) -> Any:
        return await asyncio.to_thread(compute._send_wire, method, params)

    poller = register_poller(FeedPoller(freshness_hub(graph), send))
    poller.start()
    return poller


async def start_process_feed_poller() -> FeedPoller | None:
    """The server lifespan's hook: start the poller for the process engine's graph under the
    system session (like the synthesis daemon beside it). ``None`` — logged, never raised — when
    no engine is reachable at startup: the caches then keep their callers' own tolerances."""
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.session import use_session
    from agent_utilities.security.brain_context import use_actor
    from agent_utilities.security.request_identity import system_write_session

    session = system_write_session()
    with use_actor(session.actor), use_session(session):
        (engine,) = await asyncio.gather(
            asyncio.to_thread(IntelligenceGraphEngine.get_or_create),
            return_exceptions=True,
        )
        compute = getattr(engine, "graph_compute", None)
        if isinstance(engine, BaseException) or not hasattr(compute, "_send_wire"):
            logger.warning(
                "freshness feed poller not started: no engine client (outcome=%s)",
                type(engine).__name__,
            )
            return None
        return start_engine_feed_poller(compute)


async def stop_process_feed_poller(poller: FeedPoller | None) -> None:
    """The server lifespan's shutdown hook."""
    if poller is not None:
        await stop_poller(poller)
