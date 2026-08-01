"""Unit tests for :mod:`agent_utilities.caching.semantic_cache`.

CONCEPT:AU-KG.memory.semantic-response-cache. Proves the full safety contract from the
module docstring:

* :class:`SemanticCacheKey` hashes every component — the security boundary (mirrors
  :class:`~agent_utilities.kvcache.checkpoint.KVCheckpointKey`).
* Cross-tenant reuse is IMPOSSIBLE by construction (different fingerprint bucket) — an
  explicit test, per the constitution's "add an explicit test proving it" requirement.
* A side-effecting request is NEVER served from cache (default-refused).
* No declared freshness tolerance is NEVER served from cache (default-refused).
* A stale entry (age beyond tolerance) falls back to a live call (miss/stale, never a hit).
* The default-OFF master switch (``AU_SEMANTIC_CACHE``) AND per-call ``policy.enabled`` must
  BOTH be true for a hit to ever be possible.
* Eviction (bucket FIFO bound + per-bucket priority/age bound) and invalidation.
"""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from agent_utilities.caching.semantic_cache import (
    SemanticCache,
    SemanticCacheKey,
    SemanticCachePolicy,
)


def _embed(text: str) -> list[float]:
    """Deterministic bag-of-letters embedding — good enough to distinguish unrelated
    strings without pulling in a real embedding model for unit tests."""
    vector = [0.0] * 26
    for ch in text.lower():
        if "a" <= ch <= "z":
            vector[ord(ch) - 97] += 1.0
    return vector


@pytest.fixture(autouse=True)
def _enable_master_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AU_SEMANTIC_CACHE", "true")


@pytest.fixture
def cache() -> SemanticCache:
    return SemanticCache(embed_fn=_embed)


def _key(**overrides: str) -> SemanticCacheKey:
    fields = {
        "tenant": "tenant-a",
        "principal": "user-1",
        "policy_version": "v1",
        "model_identity": "openai:gpt-x",
        "prompt_version": "pv1",
        "tool_schema_version": "",
        "retrieval_snapshot": "",
        "ontology_version": "1",
        "safety_posture": "strict",
    }
    fields.update(overrides)
    return SemanticCacheKey(**fields)


def _enabled_policy(**overrides: object) -> SemanticCachePolicy:
    fields: dict[str, object] = {
        "enabled": True,
        "side_effect_free": True,
        "freshness_tolerance_seconds": 60,
        "similarity_threshold": 0.9,
    }
    fields.update(overrides)
    return SemanticCachePolicy(**fields)  # type: ignore[arg-type]


class TestSemanticCacheKeyFingerprint:
    def test_fingerprint_deterministic(self) -> None:
        assert _key().fingerprint == _key().fingerprint

    @pytest.mark.parametrize(
        "override",
        [
            {"tenant": "tenant-b"},
            {"principal": "user-2"},
            {"policy_version": "v2"},
            {"model_identity": "anthropic:claude-x"},
            {"prompt_version": "pv2"},
            {"tool_schema_version": "ts2"},
            {"retrieval_snapshot": "snap2"},
            {"ontology_version": "2"},
            {"safety_posture": "on"},
        ],
    )
    def test_any_component_change_invalidates(self, override: dict[str, str]) -> None:
        assert _key().fingerprint != _key(**override).fingerprint

    def test_tenant_is_mandatory(self) -> None:
        with pytest.raises(ValidationError):
            SemanticCacheKey(tenant="")


class TestMissThenHit:
    def test_miss_then_store_then_hit(self, cache: SemanticCache) -> None:
        key = _key()
        policy = _enabled_policy()

        miss = cache.lookup(key, "what is the capital of france", policy=policy)
        assert miss.outcome == "miss"
        assert miss.hit is False

        stored = cache.store(
            key, "what is the capital of france", "Paris", policy=policy
        )
        assert stored is True

        hit = cache.lookup(key, "what is the capital of france", policy=policy)
        assert hit.hit is True
        assert hit.outcome == "hit"
        assert hit.response_text == "Paris"
        assert (
            hit.similarity is not None and hit.similarity >= policy.similarity_threshold
        )
        assert hit.age_seconds is not None and hit.age_seconds >= 0.0

    def test_below_similarity_threshold_is_a_miss(self, cache: SemanticCache) -> None:
        key = _key()
        policy = _enabled_policy(similarity_threshold=0.999)
        cache.store(key, "what is the capital of france", "Paris", policy=policy)
        result = cache.lookup(
            key, "completely unrelated query text here", policy=policy
        )
        assert result.hit is False
        assert result.outcome == "below_threshold"


class TestCrossTenantRefusal:
    """Explicit proof that cross-tenant reuse is impossible — the constitution's
    "add an explicit test proving it" requirement, mirrored from KVCheckpointStore's test."""

    def test_cross_tenant_lookup_never_hits(self, cache: SemanticCache) -> None:
        policy = _enabled_policy()
        owner_key = _key(tenant="tenant-owner")
        cache.store(owner_key, "what is the capital of france", "Paris", policy=policy)

        other_key = _key(tenant="tenant-other")
        result = cache.lookup(other_key, "what is the capital of france", policy=policy)

        assert result.hit is False
        assert result.outcome in ("miss", "stale")
        assert result.response_text is None

    def test_cross_principal_lookup_never_hits(self, cache: SemanticCache) -> None:
        policy = _enabled_policy()
        cache.store(_key(principal="user-1"), "q", "a", policy=policy)
        result = cache.lookup(_key(principal="user-2"), "q", policy=policy)
        assert result.hit is False

    def test_cross_policy_version_lookup_never_hits(self, cache: SemanticCache) -> None:
        policy = _enabled_policy()
        cache.store(_key(policy_version="v1"), "q", "a", policy=policy)
        result = cache.lookup(_key(policy_version="v2"), "q", policy=policy)
        assert result.hit is False


class TestSideEffectingRequestsNeverCached:
    def test_lookup_refuses_when_not_side_effect_free(
        self, cache: SemanticCache
    ) -> None:
        key = _key()
        cache.store(key, "q", "a", policy=_enabled_policy())
        policy = _enabled_policy(side_effect_free=False)
        result = cache.lookup(key, "q", policy=policy)
        assert result.hit is False
        assert result.outcome == "refused_side_effect"

    def test_store_refuses_when_not_side_effect_free(
        self, cache: SemanticCache
    ) -> None:
        policy = _enabled_policy(side_effect_free=False)
        stored = cache.store(_key(), "q", "a", policy=policy)
        assert stored is False
        assert cache.stats()["entries"] == 0


class TestFreshnessToleranceRequired:
    def test_lookup_refuses_with_no_freshness_tolerance(
        self, cache: SemanticCache
    ) -> None:
        key = _key()
        cache.store(key, "q", "a", policy=_enabled_policy())
        policy = _enabled_policy(freshness_tolerance_seconds=None)
        result = cache.lookup(key, "q", policy=policy)
        assert result.hit is False
        assert result.outcome == "refused_freshness"

    def test_store_refuses_with_no_freshness_tolerance(
        self, cache: SemanticCache
    ) -> None:
        policy = _enabled_policy(freshness_tolerance_seconds=None)
        stored = cache.store(_key(), "q", "a", policy=policy)
        assert stored is False


class TestStaleEntryFallsBackToLiveCall:
    def test_stale_entry_never_hits(self, cache: SemanticCache) -> None:
        key = _key()
        store_policy = _enabled_policy(freshness_tolerance_seconds=0)
        cache.store(key, "what is the capital of france", "Paris", policy=store_policy)
        # A tiny sleep guarantees age > 0 > tolerance without flaking on clock resolution.
        time.sleep(0.01)
        lookup_policy = _enabled_policy(freshness_tolerance_seconds=0)
        result = cache.lookup(
            key, "what is the capital of france", policy=lookup_policy
        )
        assert result.hit is False
        assert result.outcome == "stale"

    def test_tightest_of_store_and_lookup_tolerance_wins(
        self, cache: SemanticCache
    ) -> None:
        key = _key()
        # Stored with a generous tolerance...
        cache.store(
            key,
            "what is the capital of france",
            "Paris",
            policy=_enabled_policy(freshness_tolerance_seconds=3600),
        )
        time.sleep(0.01)
        # ...but the LOOKUP declares a near-zero tolerance — must still refuse.
        result = cache.lookup(
            key,
            "what is the capital of france",
            policy=_enabled_policy(freshness_tolerance_seconds=0),
        )
        assert result.hit is False


class TestMasterSwitchAndPolicyBothRequired:
    def test_policy_disabled_never_hits_even_with_master_switch_on(
        self, cache: SemanticCache
    ) -> None:
        key = _key()
        cache.store(key, "q", "a", policy=_enabled_policy())
        result = cache.lookup(key, "q", policy=_enabled_policy(enabled=False))
        assert result.outcome == "disabled"

    def test_master_switch_off_never_hits_even_with_policy_enabled(
        self, cache: SemanticCache, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key = _key()
        cache.store(key, "q", "a", policy=_enabled_policy())
        monkeypatch.setenv("AU_SEMANTIC_CACHE", "false")
        result = cache.lookup(key, "q", policy=_enabled_policy())
        assert result.outcome == "disabled"
        assert result.hit is False


class TestEmbedUnavailableFailsToLiveCall:
    def test_none_embedder_never_raises_and_never_hits(self) -> None:
        cache = SemanticCache(embed_fn=lambda _text: None)
        policy = _enabled_policy()
        result = cache.lookup(_key(), "q", policy=policy)
        assert result.hit is False
        assert result.outcome == "embed_unavailable"
        stored = cache.store(_key(), "q", "a", policy=policy)
        assert stored is False

    def test_raising_embedder_never_raises_out(self) -> None:
        def _boom(_text: str) -> list[float]:
            raise RuntimeError("embedder down")

        cache = SemanticCache(embed_fn=_boom)
        result = cache.lookup(_key(), "q", policy=_enabled_policy())
        assert result.hit is False
        assert result.outcome == "embed_unavailable"


class TestEviction:
    def test_bucket_fifo_eviction_bound(self) -> None:
        cache = SemanticCache(embed_fn=_embed, max_buckets=2, max_entries_per_bucket=8)
        policy = _enabled_policy()
        cache.store(_key(tenant="t1"), "q1", "a1", policy=policy)
        cache.store(_key(tenant="t2"), "q2", "a2", policy=policy)
        assert cache.stats()["buckets"] == 2
        # A third distinct bucket evicts the oldest (t1) under the FIFO bound.
        cache.store(_key(tenant="t3"), "q3", "a3", policy=policy)
        assert cache.stats()["buckets"] == 2
        miss = cache.lookup(_key(tenant="t1"), "q1", policy=policy)
        assert miss.hit is False

    def test_per_bucket_entry_bound_evicts_lowest_priority_first(self) -> None:
        from agent_utilities.core.resource_priority import PriorityClass, priority_scope

        cache = SemanticCache(embed_fn=_embed, max_entries_per_bucket=1)
        policy = _enabled_policy()
        key = _key()
        with priority_scope(PriorityClass.BACKGROUND_INGESTION):
            cache.store(key, "background query alpha", "bg-answer", policy=policy)
        with priority_scope(PriorityClass.INTERACTIVE):
            cache.store(key, "interactive query beta", "fg-answer", policy=policy)
        # Only one entry survives the bound; it must be the higher-priority (interactive) one.
        assert cache.stats()["entries"] == 1
        hit = cache.lookup(key, "interactive query beta", policy=policy)
        assert hit.hit is True
        assert hit.response_text == "fg-answer"


class TestInvalidation:
    def test_invalidate_key_drops_exactly_that_bucket(
        self, cache: SemanticCache
    ) -> None:
        policy = _enabled_policy()
        key_a = _key(tenant="t1")
        key_b = _key(tenant="t2")
        cache.store(key_a, "q", "a", policy=policy)
        cache.store(key_b, "q", "a", policy=policy)
        assert cache.invalidate_key(key_a) is True
        assert cache.stats()["buckets"] == 1
        assert cache.lookup(key_a, "q", policy=policy).hit is False
        assert cache.lookup(key_b, "q", policy=policy).hit is True

    def test_invalidate_by_component_filter(self, cache: SemanticCache) -> None:
        policy = _enabled_policy()
        cache.store(
            _key(tenant="t1", model_identity="openai:gpt-x"), "q", "a", policy=policy
        )
        cache.store(
            _key(tenant="t2", model_identity="openai:gpt-x"), "q", "a", policy=policy
        )
        cache.store(
            _key(tenant="t1", model_identity="anthropic:claude-x"),
            "q",
            "a",
            policy=policy,
        )

        dropped = cache.invalidate(tenant="t1")
        assert dropped == 2
        assert cache.stats()["buckets"] == 1

    def test_invalidate_no_filters_clears_everything(
        self, cache: SemanticCache
    ) -> None:
        policy = _enabled_policy()
        cache.store(_key(tenant="t1"), "q", "a", policy=policy)
        cache.store(_key(tenant="t2"), "q", "a", policy=policy)
        dropped = cache.invalidate()
        assert dropped == 2
        assert cache.stats() == {"buckets": 0, "entries": 0}
