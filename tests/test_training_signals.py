#!/usr/bin/python
"""Tests for the deterministic reward/dataset spine + RewardDecomposer wiring.

CONCEPT:AHE-3.1
"""

import pytest

from agent_utilities.graph.reward_decomposition import RewardDecomposer
from agent_utilities.graph.training_signals import (
    batch_normalized_advantage,
    composite_reward,
    difficulty_floor_filter,
    failure_point,
)

pytestmark = pytest.mark.concept("AHE-3.1")


# --- batch_normalized_advantage --------------------------------------------


def test_advantage_centers_and_scales():
    adv = batch_normalized_advantage([1.0, 2.0, 3.0])
    assert adv[1] == 0.0  # the mean maps to 0
    assert adv[0] < 0 < adv[2]
    assert adv[2] == pytest.approx(-adv[0])


def test_advantage_degenerate_groups_are_zero():
    assert batch_normalized_advantage([]) == []
    assert batch_normalized_advantage([5.0]) == [0.0]
    assert batch_normalized_advantage([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]


# --- failure_point ---------------------------------------------------------


def test_failure_point_first_divergence():
    assert failure_point([False, False, True, False]) == 2
    assert failure_point([True, True]) == 0
    assert failure_point([False, False]) is None
    assert failure_point([]) is None


# --- composite_reward ------------------------------------------------------


def test_composite_reward_weighted_sum():
    r = composite_reward({"acc": 1.0, "fmt": 0.5}, {"acc": 0.8, "fmt": 0.2})
    assert r == pytest.approx(0.8 * 1.0 + 0.2 * 0.5)


def test_composite_reward_conditional_gating():
    # func reward is gated OFF → contributes 0
    r = composite_reward(
        {"acc": 1.0, "func": 1.0},
        {"acc": 1.0, "func": 1.0},
        conditions={"func": False},
    )
    assert r == pytest.approx(1.0)


# --- difficulty_floor_filter -----------------------------------------------


def test_difficulty_floor_filter_dicts_and_objects():
    items = [{"step_count": 1}, {"step_count": 5}, {"step_count": 3}]
    kept = difficulty_floor_filter(items, min_count=3)
    assert [i["step_count"] for i in kept] == [5, 3]

    class T:
        def __init__(self, n):
            self.step_count = n

    objs = [T(0), T(4)]
    assert difficulty_floor_filter(objs, min_count=2) == [objs[1]]


# --- RewardDecomposer wiring (live path) -----------------------------------


def _decomposer_with_records():
    rd = RewardDecomposer(alpha=0.2)
    # failing trajectory whose 2nd step is incorrect
    rd.decompose(
        "traj_fail",
        steps=[{"action": "a", "success": True}, {"action": "b", "success": False}],
        goal_achieved=False,
    )
    # successful trajectory, all steps correct
    rd.decompose(
        "traj_ok",
        steps=[{"action": "a", "success": True}, {"action": "b", "success": True}],
        goal_achieved=True,
    )
    return rd


def test_batch_advantages_over_records():
    rd = _decomposer_with_records()
    adv = rd.batch_advantages()
    assert len(adv) == 2
    # the successful trajectory has the higher total_reward → higher advantage
    assert adv[1] > adv[0]


def test_failure_points_localize_divergence():
    rd = _decomposer_with_records()
    fps = rd.failure_points()
    assert fps[0] == 1  # failing trajectory diverges at step index 1
    assert fps[1] is None  # successful trajectory has no incorrect step


def test_distillation_insights_surface_spine_signals():
    rd = _decomposer_with_records()
    insights = rd.get_distillation_insights()
    assert "advantage_spread" in insights
    assert insights["localized_failures"] == 1
    assert insights["mean_failure_point"] == 1.0
