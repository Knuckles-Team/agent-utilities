#!/usr/bin/python
"""Tests for DW-GRPO dynamic reward weighting + GEPA pool wiring.

CONCEPT:AU-ORCH.optimization.selection-on-unseen-data
"""

import pytest

from agent_utilities.rlm.dynamic_reward import DynamicRewardWeighter
from agent_utilities.rlm.gepa import Candidate, ParetoCandidatePool

pytestmark = pytest.mark.concept("AU-ORCH.optimization.selection-on-unseen-data")


# --- DynamicRewardWeighter --------------------------------------------------


def test_weights_uniform_before_history():
    w = DynamicRewardWeighter(["accuracy", "efficiency"])
    assert w.ready is False
    assert w.weights() == {"accuracy": 0.5, "efficiency": 0.5}


def test_lagging_objective_is_upweighted():
    w = DynamicRewardWeighter(["accuracy", "efficiency"])
    w.observe({"accuracy": 0.5, "efficiency": 0.5})
    w.observe(
        {"accuracy": 0.9, "efficiency": 0.5}
    )  # accuracy raced ahead; efficiency stalled
    assert w.ready is True
    slopes = w.slopes()
    assert slopes["accuracy"] > slopes["efficiency"]
    weights = w.weights()
    # the stalled objective (efficiency) gets MORE weight (anti-seesaw)
    assert weights["efficiency"] > weights["accuracy"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_scalarize_uses_weights():
    w = DynamicRewardWeighter(["a", "b"])
    w.observe({"a": 0.0, "b": 0.0})
    w.observe({"a": 1.0, "b": 0.0})  # a improving → b up-weighted
    # candidate strong on the up-weighted objective b scores higher
    assert w.scalarize({"a": 0.0, "b": 1.0}) > w.scalarize({"a": 1.0, "b": 0.0})


# --- ParetoCandidatePool DW-GRPO wiring ------------------------------------


def _cand(cid, acc, eff, gen=1):
    return Candidate(
        id=cid,
        prompt_text="p",
        generation=gen,
        scores={"accuracy": acc, "efficiency": eff},
    )


def test_pool_without_dynamic_weighting_is_unchanged():
    pool = ParetoCandidatePool(["accuracy", "efficiency"])
    assert pool.observe() == {}  # no-op
    pool.update([_cand("a", 0.9, 0.1)])
    assert pool.weighted_best().id == "a"  # falls back to primary best


def test_pool_weighted_best_diverges_from_frontier_under_seesaw():
    pool = ParetoCandidatePool(["accuracy", "efficiency"], dynamic_weighting=True)
    # gen 1: low baseline
    pool.update([_cand("g1", 0.5, 0.2)])
    pool.observe()
    # gen 2: accuracy races ahead, efficiency stalls → efficiency becomes lagging
    pool.update([_cand("g2", 0.9, 0.2)])
    pool.observe()
    # introduce an efficiency-strong candidate onto the frontier
    pool.update([_cand("eff", 0.1, 0.9)])

    frontier_top = pool.get_frontier()[0]
    weighted = pool.weighted_best()
    # primary-objective frontier favours the high-accuracy candidate...
    assert frontier_top.scores["accuracy"] >= 0.9
    # ...but DW-GRPO weighting promotes the lagging-objective (efficiency) candidate
    assert weighted.id == "eff"
    assert pool.reward_weights["efficiency"] > pool.reward_weights["accuracy"]


def test_pool_weighted_best_cold_fallback():
    pool = ParetoCandidatePool(["accuracy", "efficiency"], dynamic_weighting=True)
    pool.update([_cand("a", 0.9, 0.1), _cand("b", 0.1, 0.2)])
    # not ready (no observations) → primary-objective best
    assert pool.weighted_best().id == "a"


# --- GEPAOptimizer wiring (live) -------------------------------------------


def test_gepa_optimizer_enables_dynamic_weighting():
    from pydantic import BaseModel

    from agent_utilities.rlm.gepa import GEPAOptimizer

    class Sig(BaseModel):
        answer: str = ""

    opt = GEPAOptimizer(
        signature_class=Sig,
        base_prompt="base",
        evaluator_fn=lambda *a, **k: ({}, ""),
        objectives=["accuracy", "efficiency"],
    )
    # the optimizer's pool has DW-GRPO weighting active
    assert opt.pool._weighter is not None
    assert set(opt.pool.reward_weights) == {"accuracy", "efficiency"}
