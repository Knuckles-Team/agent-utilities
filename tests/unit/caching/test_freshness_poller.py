"""EH-401 — the running driver of the engine invalidation feed.

CONCEPT:AU-KG.memory.semantic-response-cache. Proves the poller starts and stops cleanly, keeps
exactly one poller per graph, carries a feed event into a live cache while running, backs off
(bounded, jittered) on errors without dying, and reports its last-poll age on the health surface.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_utilities.caching import freshness_poller
from agent_utilities.caching.freshness import FreshnessHub
from agent_utilities.caching.freshness_poller import (
    FeedPoller,
    poller_statuses,
    start_engine_feed_poller,
    stop_process_feed_poller,
)
from agent_utilities.caching.semantic_cache import (
    SemanticCache,
    SemanticCacheKey,
    SemanticCachePolicy,
)
from agent_utilities.observability import runtime_health
from tests.unit.caching.test_semantic_cache import _embed

_POLICY = [{"class": "Doc", "volatility": "slow", "max_staleness_ms": 600_000}]


def _page(events: list[dict[str, Any]], head: int) -> dict[str, Any]:
    return {
        "events": events,
        "gap": False,
        "head_version": head,
        "epoch": 0,
        "policy_version": 1,
        "policy": _POLICY,
    }


class _ScriptedEngine:
    """Serves feed pages in order, then empty pages; can be told to fail."""

    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = list(pages)
        self.calls: list[dict[str, Any]] = []
        self.failing = False

    async def send(self, method: str, params: dict[str, Any], graph: str) -> Any:
        self.calls.append(params)
        if self.failing:
            raise ConnectionError("engine down")
        if self.pages:
            return self.pages.pop(0)
        return _page([], params["after_version"])

    def _send_wire(self, method: str, params: dict[str, Any]) -> Any:
        return asyncio.run(self.send(method, params, ""))


async def _yield(_delay: float) -> None:
    await asyncio.sleep(0)


async def _until(condition: Any, rounds: int = 200) -> None:
    for _ in range(rounds):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    freshness_poller._pollers.clear()
    yield
    freshness_poller._pollers.clear()


def test_a_running_poller_carries_a_feed_event_into_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AU_SEMANTIC_CACHE", "true")
    hub = FreshnessHub("g")
    engine = _ScriptedEngine([_page([], 0)])
    cache = SemanticCache(embed_fn=_embed, freshness=lambda graph: hub)
    key = SemanticCacheKey(tenant="tenant-a")
    policy = SemanticCachePolicy(
        enabled=True,
        side_effect_free=True,
        graph="g",
        depends_on_classes=frozenset({"Doc"}),
    )

    async def scenario() -> None:
        poller = FeedPoller(hub, engine.send, sleep=_yield)
        poller.start()
        await _until(lambda: hub.policy.version == 1)
        assert cache.store(key, "the doc", "answer", policy=policy)
        assert cache.lookup(key, "the doc", policy=policy).hit
        engine.pages.append(
            _page([{"version": 3, "scope": "classes", "classes": ["Doc"]}], 3)
        )
        await _until(lambda: hub.cursor == 3)
        assert cache.lookup(key, "the doc", policy=policy).outcome == "miss"
        await poller.stop()
        assert not poller.running

    asyncio.run(scenario())


def test_errors_back_off_bounded_and_the_poller_keeps_running() -> None:
    hub = FreshnessHub("g")
    engine = _ScriptedEngine([])
    engine.failing = True
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)
        await asyncio.sleep(0)

    async def scenario() -> None:
        poller = FeedPoller(
            hub,
            engine.send,
            interval_s=1.0,
            max_backoff_s=8.0,
            sleep=record,
            rng=lambda: 1.0,
        )
        poller.start()
        await _until(lambda: len(delays) >= 6)
        assert poller.running, "a failed read must not end the poller"
        engine.failing = False
        await _until(lambda: poller.failures == 0 and poller.last_success is not None)
        await poller.stop()

    asyncio.run(scenario())
    assert delays[:5] == [2.4, 4.8, 9.6, 9.6, 9.6], (
        "doubling, capped at 8s, +20% jitter"
    )


def test_one_poller_per_graph_and_clean_shutdown() -> None:
    engine = _ScriptedEngine([])

    async def scenario() -> None:
        first = start_engine_feed_poller(_Compute(engine))
        second = start_engine_feed_poller(_Compute(engine))
        assert first is second, "one poller per graph"
        await _until(lambda: first.last_success is not None)
        assert [status["graph"] for status in poller_statuses()] == ["gp"]
        await stop_process_feed_poller(first)
        assert not first.running
        assert poller_statuses() == []
        await stop_process_feed_poller(None)

    asyncio.run(scenario())


class _Compute:
    graph_name = "gp"

    def __init__(self, engine: _ScriptedEngine) -> None:
        self._engine = engine

    def _send_wire(self, method: str, params: dict[str, Any]) -> Any:
        return self._engine._send_wire(method, params)


def test_the_health_surface_reports_feed_age() -> None:
    assert runtime_health._check_cache_freshness(None)["status"] == "not_configured"
    hub = FreshnessHub("g", max_feed_lag_s=30.0)
    now = [100.0]
    poller = FeedPoller(hub, _ScriptedEngine([]).send, clock=lambda: now[0])
    freshness_poller.register_poller(poller)
    assert runtime_health._check_cache_freshness(None)["status"] == "degraded"

    async def scenario() -> None:
        poller.start()
        await _until(lambda: poller.last_success is not None)
        report = runtime_health._check_cache_freshness(None)
        assert report["status"] == "ok"
        assert report["detail"]["pollers"][0]["last_poll_age_s"] == 0.0
        now[0] += 31.0
        assert runtime_health._check_cache_freshness(None)["status"] == "degraded"
        await poller.stop()

    asyncio.run(scenario())
