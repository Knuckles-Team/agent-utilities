#!/usr/bin/python
"""Test-Time Diversity (VPO) — CONCEPT:AHE-3.16.

Diversity metric, MMR best-of-k selection, and the effort-derived diverse fan-out
width on the live ReasoningBudget path.
"""

import pytest

from agent_utilities.graph.test_time_diversity import (
    diverse_fan_out_width,
    mean_pairwise_distance,
    select_diverse,
)
from agent_utilities.harness.reasoning_effort import get_budget

pytestmark = pytest.mark.concept("AHE-3.16")


def test_mean_pairwise_distance_orthogonal_is_max():
    orthogonal = [[1.0, 0.0], [0.0, 1.0]]
    identical = [[1.0, 0.0], [1.0, 0.0]]
    assert mean_pairwise_distance(orthogonal) == pytest.approx(1.0, abs=1e-6)
    assert mean_pairwise_distance(identical) == pytest.approx(0.0, abs=1e-6)
    assert mean_pairwise_distance([[1.0, 0.0]]) == 0.0  # singleton


def test_select_diverse_prefers_spread_over_pure_quality():
    # Three candidates: c0 best score but near-duplicate of c1; c2 lower score but
    # orthogonal. A diversity-aware best-of-2 should pick c0 + c2, not c0 + c1.
    scores = [1.0, 0.95, 0.6]
    embeddings = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    picked = set(select_diverse(scores, embeddings, k=2, quality_weight=0.5))
    assert picked == {0, 2}


def test_select_diverse_pure_quality_when_weight_one():
    scores = [1.0, 0.95, 0.6]
    embeddings = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    # quality_weight=1 → ignore diversity → top-2 by score = c0, c1
    picked = set(select_diverse(scores, embeddings, k=2, quality_weight=1.0))
    assert picked == {0, 1}


def test_diverse_width_scales_with_effort_on_live_budget():
    # The width comes from the live ReasoningBudget — harder queries fan out wider.
    assert diverse_fan_out_width(0.0) == 1
    assert diverse_fan_out_width(1.0) > diverse_fan_out_width(0.2)
    # and it is exposed on the budget object itself (live path)
    assert get_budget(0.8).diversity_width == diverse_fan_out_width(0.8)
