"""Tests for CONCEPT:AHE-3.9 — Horizon-Aware Task Curriculum.

Validates macro-action composition, progressive horizon scheduling,
promotion policies, and curriculum state management.
"""

import pytest

from agent_utilities.graph.horizon_curriculum import (
    CurriculumStage,
    HorizonCurriculum,
    HorizonStageConfig,
    MacroAction,
    PromotionPolicy,
    SubgoalCheckpoint,
)


# ── MacroAction Tests ────────────────────────────────────────────────


class TestMacroAction:
    """Tests for the MacroAction composite primitive model."""

    def test_auto_step_count(self):
        """step_count auto-derives from atomic_steps length."""
        macro = MacroAction(
            name="edit_file",
            atomic_steps=["open_file", "navigate", "insert_text"],
        )
        assert macro.step_count == 3

    def test_explicit_step_count(self):
        """Explicit step_count overrides auto-derivation."""
        macro = MacroAction(
            name="deploy",
            atomic_steps=["build", "test", "push"],
            step_count=5,
        )
        assert macro.step_count == 5

    def test_preconditions_postconditions(self):
        """Pre/postconditions are properly stored."""
        macro = MacroAction(
            name="safe_delete",
            atomic_steps=["backup", "delete", "verify"],
            preconditions=["file_exists"],
            postconditions=["file_removed", "backup_exists"],
        )
        assert len(macro.preconditions) == 1
        assert len(macro.postconditions) == 2

    def test_default_success_rate(self):
        """Default success rate is 1.0."""
        macro = MacroAction(name="noop", atomic_steps=["step1"])
        assert macro.success_rate == 1.0

    def test_id_generation(self):
        """ID is auto-generated with macro_ prefix."""
        macro = MacroAction(name="test", atomic_steps=["a"])
        assert macro.id.startswith("macro_")


# ── SubgoalCheckpoint Tests ──────────────────────────────────────────


class TestSubgoalCheckpoint:
    """Tests for intermediate milestone checkpoints."""

    def test_mark_reached(self):
        """Marking a checkpoint records step number."""
        sg = SubgoalCheckpoint(description="Setup complete")
        assert not sg.reached
        assert sg.reached_at_step is None

        sg.mark_reached(step=5)
        assert sg.reached
        assert sg.reached_at_step == 5

    def test_default_reward(self):
        """Default reward is 0.1."""
        sg = SubgoalCheckpoint(description="test")
        assert sg.reward == 0.1

    def test_id_generation(self):
        """ID is auto-generated with sg_ prefix."""
        sg = SubgoalCheckpoint(description="test")
        assert sg.id.startswith("sg_")


# ── HorizonCurriculum Tests ─────────────────────────────────────────


class TestHorizonCurriculum:
    """Tests for the progressive horizon scheduling engine."""

    def test_default_stages(self):
        """Default curriculum has 3 stages: MACRO, INTERMEDIATE, FULL."""
        curriculum = HorizonCurriculum()
        assert len(curriculum.stages) == 3
        assert curriculum.stages[0].stage == CurriculumStage.MACRO
        assert curriculum.stages[1].stage == CurriculumStage.INTERMEDIATE
        assert curriculum.stages[2].stage == CurriculumStage.FULL

    def test_initial_state(self):
        """Curriculum starts at MACRO stage."""
        curriculum = HorizonCurriculum()
        assert curriculum.current_stage_name == "macro"
        assert not curriculum.is_final_stage

    def test_record_episode_no_promotion(self):
        """Single success doesn't trigger promotion (min_episodes=3)."""
        curriculum = HorizonCurriculum()
        result = curriculum.record_episode(success=True)

        assert result["stage"] == "macro"
        assert not result["promoted"]
        assert result["episodes_in_stage"] == 1

    def test_threshold_promotion(self):
        """Promotion occurs after min_episodes with sufficient success rate."""
        curriculum = HorizonCurriculum(
            promotion_policy=PromotionPolicy.THRESHOLD,
        )
        # Record 3 successes → should promote (3/3 = 100% ≥ 70%)
        for i in range(2):
            curriculum.record_episode(success=True)

        result = curriculum.record_episode(success=True)
        assert result["promoted"]
        assert curriculum.current_stage_name == "intermediate"

    def test_no_promotion_below_threshold(self):
        """No promotion when success rate is below threshold."""
        curriculum = HorizonCurriculum(
            promotion_policy=PromotionPolicy.THRESHOLD,
        )
        # Record 1 success, 2 failures → 33% < 70%
        curriculum.record_episode(success=True)
        curriculum.record_episode(success=False)
        result = curriculum.record_episode(success=False)

        assert not result["promoted"]
        assert curriculum.current_stage_name == "macro"

    def test_adaptive_promotion(self):
        """Adaptive promotion uses EMA of success rate."""
        curriculum = HorizonCurriculum(
            promotion_policy=PromotionPolicy.ADAPTIVE,
            ema_alpha=0.5,
        )
        # Several successes to build up EMA
        for _ in range(5):
            curriculum.record_episode(success=True)

        assert curriculum.current_stage_name != "macro"

    def test_full_curriculum_progression(self):
        """Can progress through all 3 stages to completion."""
        curriculum = HorizonCurriculum(
            stages=[
                HorizonStageConfig(
                    stage=CurriculumStage.MACRO,
                    max_horizon=5,
                    promotion_threshold=0.5,
                    min_episodes=2,
                ),
                HorizonStageConfig(
                    stage=CurriculumStage.INTERMEDIATE,
                    max_horizon=15,
                    promotion_threshold=0.5,
                    min_episodes=2,
                ),
                HorizonStageConfig(
                    stage=CurriculumStage.FULL,
                    max_horizon=50,
                    promotion_threshold=0.5,
                    min_episodes=2,
                ),
            ],
            promotion_policy=PromotionPolicy.THRESHOLD,
        )

        # Stage 1 → 2
        curriculum.record_episode(success=True)
        result = curriculum.record_episode(success=True)
        assert result["promoted"]
        assert curriculum.current_stage_name == "intermediate"

        # Stage 2 → 3
        curriculum.record_episode(success=True)
        result = curriculum.record_episode(success=True)
        assert result["promoted"]
        assert curriculum.current_stage_name == "full"
        assert curriculum.is_final_stage

    def test_no_promotion_past_final(self):
        """Cannot promote past the final stage."""
        curriculum = HorizonCurriculum(
            stages=[
                HorizonStageConfig(
                    stage=CurriculumStage.FULL,
                    max_horizon=50,
                    promotion_threshold=0.5,
                    min_episodes=1,
                ),
            ],
        )
        result = curriculum.record_episode(success=True)
        assert not result["promoted"]
        assert curriculum.is_final_stage

    def test_get_macro_actions(self):
        """MACRO stage returns all macros; FULL stage returns none."""
        macros = [
            MacroAction(name="fast_edit", atomic_steps=["a", "b"]),
        ]
        curriculum = HorizonCurriculum(
            stages=[
                HorizonStageConfig(
                    stage=CurriculumStage.MACRO,
                    max_horizon=5,
                    macro_actions=macros,
                    promotion_threshold=1.0,
                    min_episodes=100,
                ),
                HorizonStageConfig(
                    stage=CurriculumStage.FULL,
                    max_horizon=50,
                ),
            ],
        )
        assert len(curriculum.get_macro_actions()) == 1

    def test_horizon_reduction_ratio(self):
        """Ratio is current_horizon / full_horizon."""
        curriculum = HorizonCurriculum()
        ratio = curriculum.compute_horizon_reduction_ratio()
        # Default: macro=5, full=50 → 5/50 = 0.1
        assert ratio == 0.1

    def test_reset(self):
        """Reset returns curriculum to first stage."""
        curriculum = HorizonCurriculum(
            stages=[
                HorizonStageConfig(
                    stage=CurriculumStage.MACRO,
                    max_horizon=5,
                    promotion_threshold=0.5,
                    min_episodes=1,
                ),
                HorizonStageConfig(
                    stage=CurriculumStage.FULL,
                    max_horizon=50,
                ),
            ],
        )
        curriculum.record_episode(success=True)
        assert curriculum.current_stage_name != "macro" or curriculum.is_final_stage

        curriculum.reset()
        assert curriculum.current_stage_name == "macro"

    def test_to_dict(self):
        """Serialization includes all stage stats."""
        curriculum = HorizonCurriculum()
        curriculum.record_episode(success=True)

        d = curriculum.to_dict()
        assert d["current_stage"] == "macro"
        assert d["total_stages"] == 3
        assert d["promotion_policy"] == "threshold"
        assert len(d["stages"]) == 3
        assert d["stages"][0]["episodes"] == 1

    def test_plateau_promotion(self):
        """Plateau policy promotes when improvement stalls."""
        curriculum = HorizonCurriculum(
            stages=[
                HorizonStageConfig(
                    stage=CurriculumStage.MACRO,
                    max_horizon=5,
                    promotion_threshold=0.5,
                    min_episodes=3,
                ),
                HorizonStageConfig(
                    stage=CurriculumStage.FULL,
                    max_horizon=50,
                ),
            ],
            promotion_policy=PromotionPolicy.PLATEAU,
        )
        # Need min_episodes + 3 = 6 episodes for plateau detection
        for _ in range(6):
            curriculum.record_episode(success=True)

        # After 6 consistent successes, plateau should be detected
        assert curriculum.current_stage_name in ("intermediate", "full")

    def test_success_rate_tracking(self):
        """Success rate is correctly computed per stage."""
        curriculum = HorizonCurriculum()
        curriculum.record_episode(success=True)
        curriculum.record_episode(success=False)
        result = curriculum.record_episode(success=True)

        assert result["success_rate"] == pytest.approx(2 / 3, abs=0.01)
