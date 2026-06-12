"""Tests for the elastic per-tenant L1 engine pool (CONCEPT:KG-2.62).

Covers LRU warm-set bounds, eviction (with the evict hook), hydrate-on-miss,
hit/miss accounting, passthrough when disabled, and graph-name resolution.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.tenant_engine_pool import TenantEnginePool


class _FakeEngine:
    def __init__(self, graph_name: str):
        self.graph_name = graph_name
        self.closed = False

    def close(self):
        self.closed = True


def _pool(capacity, **kw):
    made: list[str] = []
    evicted: list[str] = []

    def factory(graph):
        made.append(graph)
        return _FakeEngine(graph)

    def on_evict(graph, engine):
        evicted.append(graph)
        engine.close()

    pool = TenantEnginePool(
        capacity=capacity, factory=factory, on_evict=on_evict, **kw
    )
    return pool, made, evicted


def test_passthrough_when_disabled():
    pool, made, evicted = _pool(0)
    e1 = pool.acquire(graph_name="g1")
    e2 = pool.acquire(graph_name="g1")
    # No caching: each acquire builds a fresh engine, nothing evicted.
    assert e1 is not e2
    assert made == ["g1", "g1"]
    assert evicted == []
    assert pool.stats()["warm"] == 0


def test_warm_hit_returns_same_engine():
    pool, made, _ = _pool(4)
    a = pool.acquire(graph_name="g1")
    b = pool.acquire(graph_name="g1")
    assert a is b  # warm hit
    assert made == ["g1"]  # built once
    assert pool.hits == 1 and pool.misses == 1


def test_lru_eviction_past_capacity():
    pool, made, evicted = _pool(2)
    pool.acquire(graph_name="g1")
    pool.acquire(graph_name="g2")
    pool.acquire(graph_name="g3")  # evicts g1 (LRU)
    assert evicted == ["g1"]
    assert set(pool.warm_tenants()) == {"g2", "g3"}
    assert pool.evictions == 1


def test_recency_protects_from_eviction():
    pool, _, evicted = _pool(2)
    pool.acquire(graph_name="g1")
    pool.acquire(graph_name="g2")
    pool.acquire(graph_name="g1")  # touch g1 → now MRU
    pool.acquire(graph_name="g3")  # evicts g2, not g1
    assert evicted == ["g2"]
    assert set(pool.warm_tenants()) == {"g1", "g3"}


def test_evicted_engine_is_closed():
    pool, _, _ = _pool(1)
    e1 = pool.acquire(graph_name="g1")
    pool.acquire(graph_name="g2")  # evicts g1
    assert e1.closed is True  # eviction released the client


def test_hydrate_hook_runs_on_miss_only():
    hydrated: list[str] = []
    pool, _, _ = _pool(4, on_hydrate=lambda g, e: hydrated.append(g))
    pool.acquire(graph_name="g1")
    pool.acquire(graph_name="g1")  # hit → no re-hydrate
    assert hydrated == ["g1"]


def test_tenant_resolves_to_named_graph():
    pool, made, _ = _pool(4)
    # Explicit tenant → tenant__<slug>__<base> (KG-2.58 naming).
    pool.acquire(tenant="acme")
    assert any(g.startswith("tenant__acme__") for g in made), made


def test_clear_evicts_all():
    pool, _, evicted = _pool(4)
    pool.acquire(graph_name="g1")
    pool.acquire(graph_name="g2")
    pool.clear()
    assert set(evicted) == {"g1", "g2"}
    assert pool.warm_tenants() == []
