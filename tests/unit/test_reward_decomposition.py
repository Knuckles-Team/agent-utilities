"""Tests for CONCEPT:AHE-3.1 — Decomposed Reward Signals.

Validates step-level vs trajectory-level reward separation, credit
assignment accuracy, and distillation insight extraction.
"""

import pytest

from agent_utilities.graph.reward_decomposition import (
    DecomposedRewardRecord,
    RewardDecomposer,
    StepOutcome,
    StepReward,
    TrajectoryOutcome,
    TrajectoryReward,
)

# ── StepReward Tests ─────────────────────────────────────────────────


class TestStepReward:
    """Tests for individual step reward signals."""

    def test_correct_factory(self):
        """correct() produces CORRECT outcome with +0.1 reward."""
        sr = StepReward.correct(step_index=0, action="open_file")
        assert sr.outcome == StepOutcome.CORRECT
        assert sr.reward == 0.1
        assert sr.constraint_satisfied is True

    def test_incorrect_factory(self):
        """incorrect() produces INCORRECT outcome with -0.1 reward."""
        sr = StepReward.incorrect(step_index=1, action="bad_command")
        assert sr.outcome == StepOutcome.INCORRECT
        assert sr.reward == -0.1
        assert sr.constraint_satisfied is False

    def test_neutral_default(self):
        """Default step reward is NEUTRAL with 0.0 reward."""
        sr = StepReward(step_index=0)
        assert sr.outcome == StepOutcome.NEUTRAL
        assert sr.reward == 0.0

    def test_reasoning_field(self):
        """Reasoning explanation is stored."""
        sr = StepReward.correct(0, "test", reasoning="Matched expected output")
        assert "expected output" in sr.reasoning


# ── TrajectoryReward Tests ───────────────────────────────────────────


class TestTrajectoryReward:
    """Tests for trajectory-level reward signals."""

    def test_success_factory(self):
        """success() produces SUCCESS outcome with +1.0 reward."""
        tr = TrajectoryReward.success(subgoals_reached=3, subgoals_total=3)
        assert tr.outcome == TrajectoryOutcome.SUCCESS
        assert tr.reward == 1.0
        assert tr.completion_ratio == 1.0

    def test_failure_factory(self):
        """failure() produces FAILURE outcome with 0.0 reward."""
        tr = TrajectoryReward.failure(subgoals_reached=1, subgoals_total=3)
        assert tr.outcome == TrajectoryOutcome.FAILURE
        assert tr.reward == 0.0
        assert tr.completion_ratio == pytest.approx(1 / 3, abs=0.01)

    def test_partial_completion(self):
        """Completion ratio tracks subgoal progress."""
        tr = TrajectoryReward.failure(subgoals_reached=2, subgoals_total=4)
        assert tr.completion_ratio == 0.5


# ── DecomposedRewardRecord Tests ─────────────────────────────────────


class TestDecomposedRewardRecord:
    """Tests for the combined reward record."""

    def test_successful_trajectory_all_correct(self):
        """Full success: R_total = 1.0 + 0.2 * (3 * 0.1) = 1.06."""
        record = DecomposedRewardRecord(
            trajectory_id="t1",
            trajectory_reward=TrajectoryReward.success(),
            step_rewards=[
                StepReward.correct(0),
                StepReward.correct(1),
                StepReward.correct(2),
            ],
            alpha=0.2,
        )
        assert record.total_reward == pytest.approx(1.06, abs=0.01)
        assert record.step_accuracy == 1.0
        assert record.correct_steps == 3

    def test_failed_trajectory_correct_steps(self):
        """Failed trajectory with correct steps: R_total = 0.0 + 0.2 * (2 * 0.1) = 0.04."""
        record = DecomposedRewardRecord(
            trajectory_id="t2",
            trajectory_reward=TrajectoryReward.failure(),
            step_rewards=[
                StepReward.correct(0),
                StepReward.correct(1),
            ],
            alpha=0.2,
        )
        # Steps contribute +0.04, trajectory contributes 0.0
        assert record.total_reward == pytest.approx(0.04, abs=0.01)
        assert record.step_accuracy == 1.0
        # Key insight: correct steps are NOT penalized in a failed trajectory

    def test_mixed_steps(self):
        """Mixed correct/incorrect steps: proper accounting."""
        record = DecomposedRewardRecord(
            trajectory_id="t3",
            trajectory_reward=TrajectoryReward.success(),
            step_rewards=[
                StepReward.correct(0),
                StepReward.incorrect(1),
                StepReward.correct(2),
            ],
            alpha=0.2,
        )
        # Step sum = 0.1 - 0.1 + 0.1 = 0.1
        # Total = 1.0 + 0.2 * 0.1 = 1.02
        assert record.total_reward == pytest.approx(1.02, abs=0.01)
        assert record.correct_steps == 2
        assert record.incorrect_steps == 1
        assert record.step_accuracy == pytest.approx(2 / 3, abs=0.01)

    def test_add_step_recomputes(self):
        """Adding a step triggers recomputation."""
        record = DecomposedRewardRecord(
            trajectory_id="t4",
            trajectory_reward=TrajectoryReward.success(),
        )
        assert record.total_steps == 0

        record.add_step(StepReward.correct(0))
        assert record.total_steps == 1
        assert record.total_reward == pytest.approx(1.02, abs=0.01)

    def test_id_generation(self):
        """ID is auto-generated with drr_ prefix."""
        record = DecomposedRewardRecord()
        assert record.id.startswith("drr_")

    def test_timestamp_populated(self):
        """Timestamp is auto-populated."""
        record = DecomposedRewardRecord()
        assert record.timestamp  # Non-empty


# ── RewardDecomposer Tests ───────────────────────────────────────────


class TestRewardDecomposer:
    """Tests for the trajectory decomposition engine."""

    def test_decompose_success(self):
        """Successful trajectory with correct steps."""
        decomposer = RewardDecomposer(alpha=0.2)
        record = decomposer.decompose(
            trajectory_id="t1",
            steps=[
                {"action": "open_file", "success": True},
                {"action": "edit_line", "success": True},
                {"action": "save_file", "success": True},
            ],
            goal_achieved=True,
        )
        assert record.trajectory_reward.outcome == TrajectoryOutcome.SUCCESS
        assert record.total_steps == 3
        assert record.correct_steps == 3
        assert record.total_reward > 1.0  # 1.0 + step bonuses

    def test_decompose_failure_with_correct_steps(self):
        """Failed trajectory — correct steps get positive step reward."""
        decomposer = RewardDecomposer(alpha=0.2)
        record = decomposer.decompose(
            trajectory_id="t2",
            steps=[
                {"action": "step1", "success": True},
                {"action": "step2", "success": True},
                {"action": "step3", "success": False},
            ],
            goal_achieved=False,
        )
        assert record.trajectory_reward.outcome == TrajectoryOutcome.FAILURE
        assert record.trajectory_reward.reward == 0.0
        # But total_reward > 0 because correct steps contribute
        assert record.total_reward > 0.0
        assert record.correct_steps == 2

    def test_decompose_with_subgoals(self):
        """Tracks subgoal completion in trajectory reward."""
        decomposer = RewardDecomposer()
        record = decomposer.decompose(
            trajectory_id="t3",
            steps=[{"action": "a", "success": True}],
            goal_achieved=False,
            subgoals_reached=2,
            subgoals_total=5,
        )
        assert record.trajectory_reward.subgoals_reached == 2
        assert record.trajectory_reward.completion_ratio == pytest.approx(0.4, abs=0.01)

    def test_distillation_insights_correct_in_failures(self):
        """Identifies correct steps in failed trajectories."""
        decomposer = RewardDecomposer()

        # A failed trajectory with 2 correct steps
        decomposer.decompose(
            trajectory_id="fail1",
            steps=[
                {"action": "good_step", "success": True},
                {"action": "good_step2", "success": True},
                {"action": "bad_step", "success": False},
            ],
            goal_achieved=False,
        )

        insights = decomposer.get_distillation_insights()
        assert insights["correct_in_failures"] == 2
        assert insights["records_analyzed"] == 1
        assert insights["overall_step_accuracy"] == pytest.approx(2 / 3, abs=0.01)

    def test_distillation_insights_incorrect_in_successes(self):
        """Identifies fragile success patterns."""
        decomposer = RewardDecomposer()

        # A successful trajectory with an incorrect step (fragile)
        decomposer.decompose(
            trajectory_id="fragile1",
            steps=[
                {"action": "ok", "success": True},
                {"action": "bad", "success": False},
                {"action": "ok", "success": True},
            ],
            goal_achieved=True,
        )

        insights = decomposer.get_distillation_insights()
        assert insights["incorrect_in_successes"] == 1

    def test_multiple_trajectories(self):
        """Accumulates records across multiple decompose() calls."""
        decomposer = RewardDecomposer()

        decomposer.decompose(
            "t1", [{"action": "a", "success": True}], goal_achieved=True
        )
        decomposer.decompose(
            "t2", [{"action": "b", "success": False}], goal_achieved=False
        )

        insights = decomposer.get_distillation_insights()
        assert insights["records_analyzed"] == 2
        assert insights["total_steps_analyzed"] == 2

    def test_clear(self):
        """Clear empties accumulated records."""
        decomposer = RewardDecomposer()
        decomposer.decompose(
            "t1", [{"action": "a", "success": True}], goal_achieved=True
        )
        assert decomposer.get_distillation_insights()["records_analyzed"] == 1

        decomposer.clear()
        assert decomposer.get_distillation_insights()["records_analyzed"] == 0

    def test_alpha_weighting(self):
        """Higher alpha gives more weight to step rewards."""
        steps = [
            {"action": "a", "success": True},
            {"action": "b", "success": True},
        ]

        low_alpha = RewardDecomposer(alpha=0.1)
        high_alpha = RewardDecomposer(alpha=0.5)

        record_low = low_alpha.decompose("t1", steps, goal_achieved=True)
        record_high = high_alpha.decompose("t2", steps, goal_achieved=True)

        # Higher alpha → higher total reward (since step rewards are positive)
        assert record_high.total_reward > record_low.total_reward

    def test_empty_steps(self):
        """Empty step list still produces valid record."""
        decomposer = RewardDecomposer()
        record = decomposer.decompose("t1", [], goal_achieved=True)
        assert record.total_steps == 0
        assert record.total_reward == 1.0  # Just trajectory reward

    def test_reasoning_propagation(self):
        """Step reasoning is preserved through decomposition."""
        decomposer = RewardDecomposer()
        record = decomposer.decompose(
            "t1",
            [{"action": "test", "success": True, "reasoning": "Output matched"}],
            goal_achieved=True,
        )
        assert record.step_rewards[0].reasoning == "Output matched"
