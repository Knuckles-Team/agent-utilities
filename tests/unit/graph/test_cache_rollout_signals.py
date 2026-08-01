"""Cached-rollout reward-shaping tests — CONCEPT:AU-AHE.reward.cache-rollout-signals (CacheRL arXiv:2606.14179)."""

from agent_utilities.graph.training_signals import (
    CACHE_TIER_RELIABILITY,
    cache_tier_aware_reward,
    token_cache_mask,
)


def test_token_cache_mask_masks_observations():
    sources = ["model", "action", "observation", "model", "action", "observation"]
    mask = token_cache_mask(sources)
    assert mask == [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    # Length preserving so it can multiply a per-token loss directly.
    assert len(mask) == len(sources)


def test_token_cache_mask_custom_trainable():
    sources = ["model", "action", "observation"]
    # Train only on reasoning tokens, not the emitted action.
    assert token_cache_mask(sources, trainable={"model"}) == [1.0, 0.0, 0.0]


def test_cache_aware_reward_keeps_successes():
    # A positive (successful) reward is never discounted for using the cache.
    assert cache_tier_aware_reward(1.0, ["fuzzy", "semantic"]) == 1.0
    assert cache_tier_aware_reward(0.0, ["semantic"]) == 0.0


def test_cache_aware_reward_discounts_low_tier_failures():
    # A failure that relied on a low-reliability tier is softened (weakest link).
    base = -1.0
    semantic = cache_tier_aware_reward(base, ["exact", "semantic"])
    fuzzy = cache_tier_aware_reward(base, ["fuzzy"])
    assert semantic == round(base * CACHE_TIER_RELIABILITY["semantic"], 6)  # -0.7
    assert fuzzy == round(base * CACHE_TIER_RELIABILITY["fuzzy"], 6)  # -0.85
    # Semantic is less reliable than fuzzy ⇒ its failure is discounted more.
    assert semantic > fuzzy  # -0.7 > -0.85


def test_cache_aware_reward_full_penalty_for_live():
    # A failure on real (live/exact) observations is fully the model's fault.
    assert cache_tier_aware_reward(-1.0, ["live", "exact"]) == -1.0
    assert cache_tier_aware_reward(-1.0, []) == -1.0  # no cache used → unchanged
