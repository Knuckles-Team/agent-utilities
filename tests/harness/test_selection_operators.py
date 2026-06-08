#!/usr/bin/python
"""Tests for the unified selection / aggregation operators + VariantPool wiring.

CONCEPT:ORCH-1.30
"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.harness.selection_operators import (
    bradley_terry_scores,
    conservative_rating,
    contribution_weighted_vote,
    rank_from_comparisons,
    select_top_k,
)

pytestmark = pytest.mark.concept("ORCH-1.30")


# --- Bradley–Terry ---------------------------------------------------------


def test_bradley_terry_orders_by_wins():
    # a beats b beats c consistently → a > b > c.
    comps = [("a", "b"), ("a", "b"), ("a", "c"), ("b", "c"), ("b", "c")]
    scores = bradley_terry_scores(["a", "b", "c"], comps)
    assert scores["a"] > scores["b"] > scores["c"]
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_bradley_terry_internalises_opponent_strength():
    # x's only win is over the strongest (a); y's only win is over the weakest (c).
    # BT should rate beating the strong opponent at least as highly.
    comps = [("a", "b"), ("a", "c"), ("b", "c"), ("x", "a"), ("y", "c")]
    scores = bradley_terry_scores(["a", "b", "c", "x", "y"], comps)
    assert scores["x"] >= scores["y"]


def test_bradley_terry_empty_and_unknown_ids():
    assert bradley_terry_scores([], []) == {}
    # unknown ids in comparisons are ignored, not crash
    s = bradley_terry_scores(["a", "b"], [("a", "b"), ("zzz", "a")])
    assert set(s) == {"a", "b"}


# --- conservative rating (TrueSkill-LCB spirit) ----------------------------


def test_conservative_rating_penalises_uncertainty():
    comps = [("a", "b")] * 8
    lcb = conservative_rating(["a", "b"], comps, kappa=1.0)
    assert lcb["a"] > lcb["b"]


def test_rank_from_comparisons_dispatch():
    comps = [("a", "b"), ("a", "b"), ("b", "c")]
    ranked = rank_from_comparisons(["a", "b", "c"], comps, method="bradley_terry")
    assert [r[0] for r in ranked][0] == "a"
    with pytest.raises(ValueError):
        rank_from_comparisons(["a"], [], method="nope")


# --- contribution-weighted vote --------------------------------------------


def test_contribution_weighted_vote_credits_contribution():
    # 'x' gets one low-contribution vote; 'y' gets one high-contribution vote.
    tally = contribution_weighted_vote([("x", 0.0), ("y", 3.0)], beta=1.0)
    assert tally["y"] > tally["x"]
    assert tally["x"] == pytest.approx(1.0)
    assert tally["y"] == pytest.approx(4.0)


# --- scalar select_top_k ---------------------------------------------------


def test_select_top_k_score():
    cands = [
        {"id": "a", "fitness": 0.9},
        {"id": "b", "fitness": 0.5},
        {"id": "c", "fitness": 0.7},
    ]
    top = select_top_k(cands, 2, method="score")
    assert [c["id"] for c in top] == ["a", "c"]


def test_select_top_k_lcb_prefers_reliable():
    # a has higher mean but huge variance; b is slightly lower but reliable.
    cands = [
        {"id": "a", "fitness": 0.90, "fitness_std": 0.40},
        {"id": "b", "fitness": 0.80, "fitness_std": 0.02},
    ]
    assert [c["id"] for c in select_top_k(cands, 1, method="lcb")] == ["b"]
    # plain score would pick a
    assert [c["id"] for c in select_top_k(cands, 1, method="score")] == ["a"]


def test_select_top_k_unknown_method():
    with pytest.raises(ValueError):
        select_top_k([{"id": "a"}], 1, method="bogus")


# --- live path: VariantPool.tournament_select(strategy=...) -----------------


def _pool_with_variants(variants):
    from agent_utilities.harness.variant_pool import VariantPool

    pool = VariantPool(MagicMock())
    pool.get_variants = lambda base_id: variants  # type: ignore[assignment]
    return pool


def test_variant_pool_score_strategy_is_deterministic_topk():
    variants = [
        {"id": "v1", "fitness": 0.2},
        {"id": "v2", "fitness": 0.9},
        {"id": "v3", "fitness": 0.5},
        {"id": "v4", "fitness": 0.7},
    ]
    pool = _pool_with_variants(variants)
    assert pool.tournament_select("base", top_k=2, strategy="score") == ["v2", "v4"]


def test_variant_pool_lcb_strategy_prefers_reliable():
    variants = [
        {"id": "lucky", "fitness": 0.95, "fitness_std": 0.5},
        {"id": "steady", "fitness": 0.85, "fitness_std": 0.01},
        {"id": "low", "fitness": 0.3, "fitness_std": 0.0},
    ]
    pool = _pool_with_variants(variants)
    assert pool.tournament_select("base", top_k=1, strategy="lcb") == ["steady"]


def test_variant_pool_default_strategy_unchanged():
    # Default 'tournament' still returns top_k ids (stochastic) without error.
    variants = [{"id": f"v{i}", "fitness": i / 10} for i in range(6)]
    pool = _pool_with_variants(variants)
    out = pool.tournament_select("base", top_k=3)
    assert len(out) == 3
    assert set(out) <= {v["id"] for v in variants}
