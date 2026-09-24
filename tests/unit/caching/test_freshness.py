"""EH-401 — AU caches follow the engine's per-class invalidation feed and declared volatility.

CONCEPT:AU-KG.memory.semantic-response-cache. Proves:

* an engine event for class ``C`` drops exactly the entries depending on ``C``; a coarse
  event, a feed gap, a new engine epoch or an unreadable feed drops everything for the graph;
* an entry's TTL comes from the declared ``eg:volatilityClass`` of its classes (``live`` is
  never cached, an undeclared class has no default);
* a learned change rate may only SHORTEN a declared TTL, never lengthen it;
* the semantic cache and the in-process context-bundle cache both subscribe.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest

from agent_utilities.caching.freshness import (
    ChangeRateEstimator,
    FreshnessHub,
    VolatilityPolicy,
    combine_tolerance,
    poll_engine,
    shorten_only,
)
from agent_utilities.caching.semantic_cache import (
    SemanticCache,
    SemanticCacheKey,
    SemanticCachePolicy,
)
from tests.unit.caching.test_semantic_cache import _embed


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class _Target:
    """Records what the hub asked it to drop."""

    def __init__(self) -> None:
        self.class_calls: list[frozenset[str]] = []
        self.graph_calls = 0

    def invalidate_classes(self, graph: str, classes: frozenset[str]) -> int:
        self.class_calls.append(classes)
        return 1

    def invalidate_graph(self, graph: str) -> int:
        self.graph_calls += 1
        return 1


def _policy_rows() -> list[dict[str, Any]]:
    return [
        {"class": "Doc", "volatility": "slow", "max_staleness_ms": 600_000},
        {"class": "Quote", "volatility": "live", "max_staleness_ms": 0},
        {"class": "Law", "volatility": "immutable", "max_staleness_ms": None},
    ]


def _feed(events: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    head = max([e["version"] for e in events], default=extra.pop("head", 0))
    return {
        "events": events,
        "gap": False,
        "head_version": head,
        "epoch": 0,
        "policy_version": 7,
        "policy": _policy_rows(),
        **extra,
    }


def _event(version: int, *classes: str, scope: str = "classes") -> dict[str, Any]:
    return {
        "version": version,
        "scope": scope,
        "classes": list(classes),
        "edge_types": [],
    }


def _hub(clock: _Clock) -> FreshnessHub:
    return FreshnessHub("g", clock=clock, max_feed_lag_s=30.0)


def test_declared_ttl_needs_every_class_declared() -> None:
    policy = VolatilityPolicy.from_feed(_policy_rows(), 7)
    assert policy.declared_ttl({"Doc"}) == 600.0
    assert policy.declared_ttl({"Doc", "Law"}) == 600.0
    assert policy.declared_ttl({"Law"}) == math.inf
    assert policy.declared_ttl({"Doc", "Unknown"}) is None
    assert policy.declared_ttl(set()) is None


def test_a_declared_bound_only_caps_the_callers_tolerance() -> None:
    assert combine_tolerance(None, None) is None
    assert combine_tolerance(-1, 600.0) is None
    assert combine_tolerance(30, None) == 30
    assert combine_tolerance(3_600, 600.0) == 600.0
    assert combine_tolerance(None, 600.0) == 600.0
    assert combine_tolerance(3_600, 0.0) is None, "a live class refuses outright"


def test_a_learned_rate_never_lengthens_a_declared_ttl() -> None:
    rates = ChangeRateEstimator(min_observations=2)
    for at in (0.0, 10.0, 20.0):
        rates.observe("Doc", at)
    learned = rates.learned_ttl("Doc")
    assert learned is not None and learned < 10.0
    for declared in (1.0, 5.0, 60.0, math.inf):
        assert shorten_only(declared, learned) <= declared
    assert shorten_only(1.0, 1_000.0) == 1.0, (
        "a slow observed rate cannot extend a bound"
    )
    assert ChangeRateEstimator().learned_ttl("Unseen") is None


def test_class_events_drop_only_their_classes_and_advance_the_cursor() -> None:
    clock = _Clock()
    hub, target = _hub(clock), _Target()
    hub.attach(target)
    hub.attach(target)
    hub.apply_feed(_feed([_event(3, "Doc"), _event(4, "Person")]))
    assert target.class_calls == [frozenset({"Doc"}), frozenset({"Person"})]
    assert target.graph_calls == 0
    assert hub.cursor == 4
    assert hub.feed_request() == {"after_version": 4, "limit": 0, "policy_after": 7}


def test_a_partial_page_does_not_skip_unread_events() -> None:
    hub = _hub(_Clock())
    feed = _feed([_event(3, "Doc")])
    feed["head_version"] = 9
    hub.apply_feed(feed)
    assert hub.cursor == 3, "events 4..9 were not delivered yet"
    hub.apply_feed(_feed([], head=9))
    assert hub.cursor == 9


@pytest.mark.parametrize(
    "restart",
    [
        {"events": [_event(5, scope="all")]},
        {"gap": True},
        {"epoch": 2},
    ],
)
def test_coarse_events_gaps_and_new_epochs_drop_the_whole_graph(
    restart: dict[str, Any],
) -> None:
    hub, target = _hub(_Clock()), _Target()
    hub.attach(target)
    hub.apply_feed(_feed([_event(3, "Doc")]))
    base = _feed([])
    base.update(restart)
    hub.apply_feed(base)
    assert target.graph_calls == 1


def test_ttl_comes_from_the_class_and_needs_a_current_feed() -> None:
    clock = _Clock()
    hub = _hub(clock)
    assert hub.ttl_for({"Doc"}) is None, "no feed read yet: no declaration applies"
    hub.apply_feed(_feed([]))
    assert hub.ttl_for({"Doc"}) == 600.0
    assert hub.ttl_for({"Quote"}) == 0.0
    assert hub.ttl_for({"Unknown"}) is None
    clock.now += 31.0
    assert hub.ttl_for({"Doc"}) is None, (
        "a feed not read within its lag bound grants no declared TTL"
    )


def test_observed_changes_shorten_the_class_ttl() -> None:
    clock = _Clock()
    hub = FreshnessHub("g", clock=clock, rates=ChangeRateEstimator(min_observations=2))
    for version in (1, 2, 3):
        clock.now += 4.0
        hub.apply_feed(_feed([_event(version, "Doc")]))
    ttl = hub.ttl_for({"Doc"})
    assert ttl is not None and ttl < 600.0


def test_an_unreadable_feed_drops_everything_and_reraises() -> None:
    hub, target = _hub(_Clock()), _Target()
    hub.attach(target)

    async def failing(method: str, params: dict[str, Any], graph: str) -> Any:
        raise ConnectionError("engine down")

    with pytest.raises(ConnectionError):
        asyncio.run(poll_engine(hub, failing))
    assert target.graph_calls == 1
    assert hub.ttl_for({"Doc"}) is None

    async def serving(method: str, params: dict[str, Any], graph: str) -> Any:
        assert (method, graph) == ("FreshnessFeed", "g")
        return _feed([_event(2, "Doc")])

    assert asyncio.run(poll_engine(hub, serving)) == 1
    assert hub.ttl_for({"Doc"}) == 600.0


def _key() -> SemanticCacheKey:
    return SemanticCacheKey(tenant="tenant-a", principal="user-1")


def _class_policy(*classes: str, tolerance: int | None = None) -> SemanticCachePolicy:
    return SemanticCachePolicy(
        enabled=True,
        side_effect_free=True,
        freshness_tolerance_seconds=tolerance,
        graph="g",
        depends_on_classes=frozenset(classes),
    )


def test_the_semantic_cache_serves_by_declared_class_and_drops_on_its_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AU_SEMANTIC_CACHE", "true")
    hub = _hub(_Clock())
    hub.apply_feed(_feed([]))
    cache = SemanticCache(embed_fn=_embed, freshness=lambda graph: hub)
    policy = _class_policy("Doc")
    assert cache.store(_key(), "what is the doc", "answer", policy=policy)
    assert cache.lookup(_key(), "what is the doc", policy=policy).hit
    hub.apply_feed(_feed([_event(9, "Person")]))
    assert cache.lookup(_key(), "what is the doc", policy=policy).hit, "disjoint class"
    hub.apply_feed(_feed([_event(10, "Doc")]))
    assert cache.lookup(_key(), "what is the doc", policy=policy).outcome == "miss"


def test_the_semantic_cache_never_caches_a_live_or_undeclared_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AU_SEMANTIC_CACHE", "true")
    hub = _hub(_Clock())
    hub.apply_feed(_feed([]))
    cache = SemanticCache(embed_fn=_embed, freshness=lambda graph: hub)
    live = cache.lookup(_key(), "quote", policy=_class_policy("Quote", tolerance=3_600))
    assert live.outcome == "refused_freshness"
    undeclared = cache.lookup(_key(), "x", policy=_class_policy("Unknown"))
    assert undeclared.outcome == "refused_freshness"


def test_the_bundle_cache_scopes_by_class() -> None:
    from agent_utilities.core import contextual_model

    cache = contextual_model._InProcessBundleCache(ttl_s=300.0)
    hub = contextual_model.freshness_hub("bundle-graph")
    hub.apply_feed(_feed([]))
    assert cache.put_scoped(
        "k1", b"doc", graph="bundle-graph", classes=frozenset({"Doc"})
    )
    assert not cache.put_scoped(
        "k2", b"quote", graph="bundle-graph", classes=frozenset({"Quote"})
    ), "a live class is never cached"
    assert cache.get("k1") == b"doc"
    hub.apply_feed(_feed([_event(4, "Doc")]))
    assert cache.get("k1") is None


def test_a_lagging_feed_stops_serving_event_bounded_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AU_SEMANTIC_CACHE", "true")
    clock = _Clock()
    hub = _hub(clock)
    hub.apply_feed(_feed([]))
    cache = SemanticCache(embed_fn=_embed, freshness=lambda graph: hub)
    policy = _class_policy("Law")
    assert cache.store(_key(), "the statute", "text", policy=policy)
    assert cache.lookup(_key(), "the statute", policy=policy).hit
    clock.now += 31.0
    assert (
        cache.lookup(_key(), "the statute", policy=policy).outcome
        == "refused_freshness"
    ), "an immutable class is bounded only by events; unread events mean no hit"
