"""Tests for the deployment-default in-process compiled-bundle cache.

CONCEPT:AU-KG.retrieval.context-compiler-kv-seam (Seam 6) already gives
``ContextCompiler.compile(kv_backend=...)`` a fully tenant/ACL-scoped cache key
(``compute_bundle_cache_key`` — a SHA-256 digest over the pseudonymized tenant,
principal, graph, policy_version, catalog_epoch, and query), but nothing in the
process ever installed a real ``kv_backend`` — ``contextual_model._compiler_cache``
was always ``None``, so every delegated run recompiled its evidence bundle from
scratch even for an exact repeat. ``_InProcessBundleCache`` is a bounded,
TTL-expiring, purely in-process (no network) implementation of the SAME
duck-typed ``get``/``put``/``delete`` shape, wired as the lazily-built
deployment default in :func:`contextual_model._resolve_compiler_cache`.

These tests prove: (1) the cache primitive itself (get/put/delete, TTL expiry,
bounded LRU eviction); (2) the opt-in wiring (`MODEL_CONTEXT_COMPILER_CACHE_ENABLED`,
default OFF — see that function's docstring for why it is opt-in, not
opt-out); (3) an explicit :func:`set_context_compiler_cache` override always
wins and `None` reverts to the default resolution, never a hard disable; and
(4) THE hard-constraint property: two different tenants sharing one process
(and, deliberately, one cache instance) can never observe each other's
compiled bundle through this cache — a cache key computed for tenant A can
never be looked up successfully by tenant B, because
``compute_bundle_cache_key`` folds the pseudonymized tenant/principal into the
digest itself.
"""

from __future__ import annotations

import time

import pytest

from agent_utilities.core import contextual_model as cm
from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.ontology.permissioning import (
    clear_markings,
    use_marking_authority,
)
from agent_utilities.knowledge_graph.retrieval.context_compiler import ContextCompiler
from agent_utilities.models.company_brain import ActorType, DataClassification, NodeACL
from agent_utilities.security.brain_context import ActorContext


class _FakeMarkingStore:
    @staticmethod
    def execute(_query, _params):
        return []


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    reset_company_brain()
    clear_markings()
    # Every test here owns its own cache instance/state explicitly — never let
    # a lazily-built process-wide default leak between tests.
    monkeypatch.setattr(cm, "_compiler_cache", None)
    monkeypatch.setattr(cm, "_default_bundle_cache", None)
    with use_marking_authority(_FakeMarkingStore()):
        yield
    reset_company_brain()
    clear_markings()


# ---------------------------------------------------------------------------
# 1. The cache primitive itself
# ---------------------------------------------------------------------------


def test_get_put_round_trip():
    cache = cm._InProcessBundleCache()
    assert cache.get("k1") is None
    assert cache.put("k1", b"hello") is True
    assert cache.get("k1") == b"hello"


def test_delete_evicts_and_reports_whether_it_existed():
    cache = cm._InProcessBundleCache()
    assert cache.delete("missing") is False
    cache.put("k1", b"v1")
    assert cache.delete("k1") is True
    assert cache.get("k1") is None
    assert cache.delete("k1") is False


def test_ttl_expiry():
    cache = cm._InProcessBundleCache(ttl_s=0.05)
    cache.put("k1", b"v1")
    assert cache.get("k1") == b"v1"
    time.sleep(0.08)
    assert cache.get("k1") is None, "expired entry must not be returned"


def test_bounded_lru_eviction():
    cache = cm._InProcessBundleCache(maxsize=2)
    cache.put("k1", b"v1")
    cache.put("k2", b"v2")
    # Touch k1 so it is the most-recently-used; k2 becomes the eviction victim.
    assert cache.get("k1") == b"v1"
    cache.put("k3", b"v3")
    assert cache.get("k2") is None, "least-recently-used entry must be evicted"
    assert cache.get("k1") == b"v1"
    assert cache.get("k3") == b"v3"


# ---------------------------------------------------------------------------
# 2. Opt-in wiring — default OFF, explicit override always wins
# ---------------------------------------------------------------------------


def test_default_resolution_is_off_unless_opted_in(monkeypatch):
    monkeypatch.delenv("MODEL_CONTEXT_COMPILER_CACHE_ENABLED", raising=False)
    assert cm._resolve_compiler_cache() is None
    assert cm.get_context_compiler_cache() is None


def test_default_resolution_opts_in_via_config(monkeypatch):
    monkeypatch.setenv("MODEL_CONTEXT_COMPILER_CACHE_ENABLED", "true")
    cache = cm._resolve_compiler_cache()
    assert isinstance(cache, cm._InProcessBundleCache)
    # Lazily built ONCE — a second resolution reuses the SAME instance so
    # entries persist across calls within the process, not a fresh cache
    # (and therefore an unconditional miss) every time.
    assert cm._resolve_compiler_cache() is cache


def test_explicit_override_wins_regardless_of_config(monkeypatch):
    monkeypatch.delenv("MODEL_CONTEXT_COMPILER_CACHE_ENABLED", raising=False)
    sentinel = object()
    cm.set_context_compiler_cache(sentinel)
    try:
        assert cm._resolve_compiler_cache() is sentinel
        assert cm.get_context_compiler_cache() is sentinel
    finally:
        cm.set_context_compiler_cache(None)


def test_explicit_none_reverts_to_default_not_a_hard_disable(monkeypatch):
    """``set_context_compiler_cache(None)`` clears the override; it does NOT
    itself force caching off — the config default governs from there, exactly
    as documented (use `MODEL_CONTEXT_COMPILER_CACHE_ENABLED=false` to force
    off)."""
    monkeypatch.setenv("MODEL_CONTEXT_COMPILER_CACHE_ENABLED", "true")
    cm.set_context_compiler_cache(object())
    cm.set_context_compiler_cache(None)
    assert isinstance(cm._resolve_compiler_cache(), cm._InProcessBundleCache)


# ---------------------------------------------------------------------------
# 3. Hard constraint: cross-tenant isolation through a SHARED cache instance
# ---------------------------------------------------------------------------

_NODES = [
    {
        "id": "claim:shared-id",
        "type": "Claim",
        "name": "Claim",
        "description": "Same node id, reused by both tenants below on purpose.",
        "score": 0.9,
        "confidence": 0.9,
        "source_refs": ["doc:1"],
    },
]


class _FakeRetriever:
    def __init__(self, nodes: list[dict]) -> None:
        self._nodes = nodes

    def retrieve_hybrid(self, query, context_window=10, **kwargs):
        return list(self._nodes)[:context_window]


def _grant_public(nodes: list[dict]) -> None:
    from agent_utilities.knowledge_graph.core.company_brain_runtime import (
        get_company_brain,
    )

    permissions = get_company_brain().permissions
    for node in nodes:
        permissions.set_acl(
            NodeACL(node_id=node["id"], classification=DataClassification.PUBLIC)
        )


def _session(tenant: str, principal: str) -> GraphSession:
    actor = ActorContext(
        actor_id=principal,
        actor_type=ActorType.AI_AGENT,
        tenant_id=tenant,
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=tenant,
        policy_version="v1",
        scopes=frozenset({"kg:read"}),
    )


def test_shared_cache_instance_never_leaks_a_bundle_across_tenants():
    """Two tenants querying THE SAME shared ``_InProcessBundleCache`` instance
    with the identical query text and the identical (test-fixture) evidence
    node id must NEVER observe a cache hit against each other's compiled
    bundle — proving the hard constraint: a faster read must return the SAME
    rows an uncached read would, not another tenant's rows.
    """
    _grant_public(_NODES)
    cache = cm._InProcessBundleCache()
    compiler = ContextCompiler(_FakeRetriever(_NODES))

    session_a = _session("tenant-a", "principal:a")
    session_b = _session("tenant-b", "principal:b")

    with use_session(session_a):
        bundle_a1 = compiler.compile(
            "what is the policy?",
            session=session_a,
            top_k=1,
            candidate_pool=1,
            kv_backend=cache,
        )
    with use_session(session_b):
        bundle_b1 = compiler.compile(
            "what is the policy?",
            session=session_b,
            top_k=1,
            candidate_pool=1,
            kv_backend=cache,
        )

    # Different tenants against the same identical query/evidence-id-set never
    # collide on a cache key — each pays its OWN compile (a real miss), not a
    # cross-tenant hit.
    assert bundle_a1.kv_cache_hit is False
    assert bundle_b1.kv_cache_hit is False
    assert bundle_a1.cache_key != bundle_b1.cache_key
    assert bundle_a1.session_tenant != bundle_b1.session_tenant

    # A SECOND compile for tenant A (same session, same query) legitimately
    # hits ITS OWN cached entry...
    with use_session(session_a):
        bundle_a2 = compiler.compile(
            "what is the policy?",
            session=session_a,
            top_k=1,
            candidate_pool=1,
            kv_backend=cache,
        )
    assert bundle_a2.kv_cache_hit is True
    assert bundle_a2.session_tenant == bundle_a1.session_tenant

    # Both tenants' entries are independently present in the shared cache
    # (each cached its OWN compile), but under DIFFERENT keys — tenant A's
    # entry is unreachable via tenant B's key and vice versa, the
    # cryptographic guarantee `compute_bundle_cache_key` gives.
    assert cache.get(bundle_a1.cache_key) is not None
    assert cache.get(bundle_b1.cache_key) is not None
    assert cache.get(bundle_a1.cache_key) != cache.get(bundle_b1.cache_key)
