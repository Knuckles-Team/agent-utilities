#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort — Decomposed Reward Signals.

Strangled by graph/planning/ (Plan 03 Step 4): re-exported via
``agent_utilities.graph.planning.reward``. This module remains the live
implementation.

Separates trajectory-level reward (goal achievement) from step-level
reward (local constraint satisfaction) to prevent penalising correct
intermediate steps in failed trajectories.  Derived from the Long-Horizon
Training research (Kim et al., ICML 2026), Section 3.1.

The key insight: in long-horizon tasks with sparse rewards, a failed
trajectory assigns negative advantage to *every* step — including
individually correct ones.  Decomposition isolates step quality from
outcome, enabling accurate credit assignment and stable distillation
into ``ExperienceNode`` primitives.

Reward decomposition formula::

    R_total(τ) = R_trajectory(τ) + α · Σ R_step(s_t, a_t)

Where:
    - R_trajectory: Binary signal for full-task completion
    - R_step: Local signal for constraint satisfaction at each step
    - α: Weighting factor (default 0.2) balancing local vs global feedback

Integrates with:
    - CONCEPT:AU-AHE.evaluation.backtest-harness (Experience Distillation): Per-step success feeds ExperienceNode
    - CONCEPT:AU-AHE.evaluation.backtest-harness (Heavy Thinking): Step rewards for each thinker trajectory
    - CONCEPT:AU-AHE.evaluation.backtest-harness (Horizon Curriculum): Subgoal checkpoint rewards
    - CONCEPT:AU-KG.query.object-graph-mapper (OGM): Persists reward records as KG nodes

See docs/overview.md §CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort
"""


import logging
import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Enumerations ─────────────────────────────────────────────────────


class StepOutcome(StrEnum):
    """Outcome classification for a single step within a trajectory.

    Attributes:
        CORRECT: Step satisfied all local constraints.
        PARTIAL: Step made progress but didn't fully satisfy constraints.
        INCORRECT: Step violated a constraint or made an error.
        NEUTRAL: Step had no measurable impact (e.g., observation-only).
    """

    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    NEUTRAL = "neutral"


class TrajectoryOutcome(StrEnum):
    """Outcome classification for a full trajectory.

    Attributes:
        SUCCESS: Task goal was achieved.
        PARTIAL_SUCCESS: Some subgoals reached but not the final goal.
        FAILURE: Task goal was not achieved.
        TIMEOUT: Task exceeded the horizon budget.
    """

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


# ── Data Models ──────────────────────────────────────────────────────


class StepReward(BaseModel):
    """Reward signal for a single step within a trajectory.

    CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort — Captures local constraint satisfaction at the
    step level, independent of whether the overall trajectory succeeded.

    Attributes:
        step_index: 0-based index of this step in the trajectory.
        action_description: Human-readable description of the action taken.
        outcome: Classification of step quality.
        reward: Numerical reward value for this step.
        constraint_satisfied: Whether local constraints were met.
        reasoning: Explanation of why this outcome was assigned.
    """

    step_index: int = Field(ge=0)
    action_description: str = ""
    outcome: StepOutcome = StepOutcome.NEUTRAL
    reward: float = Field(default=0.0, ge=-1.0, le=1.0)
    constraint_satisfied: bool = True
    reasoning: str = ""

    @classmethod
    def correct(
        cls, step_index: int, action: str = "", reasoning: str = ""
    ) -> StepReward:
        """Factory for a correct step outcome.

        Args:
            step_index: Position in the trajectory.
            action: Description of the action.
            reasoning: Why this was classified as correct.

        Returns:
            A ``StepReward`` with CORRECT outcome and reward +0.1.
        """
        return cls(
            step_index=step_index,
            action_description=action,
            outcome=StepOutcome.CORRECT,
            reward=0.1,
            constraint_satisfied=True,
            reasoning=reasoning,
        )

    @classmethod
    def incorrect(
        cls, step_index: int, action: str = "", reasoning: str = ""
    ) -> StepReward:
        """Factory for an incorrect step outcome.

        Args:
            step_index: Position in the trajectory.
            action: Description of the action.
            reasoning: Why this was classified as incorrect.

        Returns:
            A ``StepReward`` with INCORRECT outcome and reward -0.1.
        """
        return cls(
            step_index=step_index,
            action_description=action,
            outcome=StepOutcome.INCORRECT,
            reward=-0.1,
            constraint_satisfied=False,
            reasoning=reasoning,
        )


class TrajectoryReward(BaseModel):
    """Trajectory-level reward signal for goal achievement.

    CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort — Binary (or graded) signal for whether the
    overall task goal was achieved, independent of step-level quality.

    Attributes:
        outcome: Overall trajectory classification.
        reward: Numerical reward for goal achievement.
        subgoals_reached: Number of intermediate subgoals reached.
        subgoals_total: Total number of intermediate subgoals.
        completion_ratio: Fraction of subgoals completed.
    """

    outcome: TrajectoryOutcome = TrajectoryOutcome.FAILURE
    reward: float = Field(default=0.0, ge=-1.0, le=1.0)
    subgoals_reached: int = Field(default=0, ge=0)
    subgoals_total: int = Field(default=0, ge=0)
    completion_ratio: float = Field(default=0.0, ge=0.0, le=1.0)

    @classmethod
    def success(
        cls, subgoals_reached: int = 0, subgoals_total: int = 0
    ) -> TrajectoryReward:
        """Factory for a successful trajectory.

        Args:
            subgoals_reached: Number of checkpoints reached.
            subgoals_total: Total number of checkpoints.

        Returns:
            A ``TrajectoryReward`` with SUCCESS outcome and reward +1.0.
        """
        return cls(
            outcome=TrajectoryOutcome.SUCCESS,
            reward=1.0,
            subgoals_reached=subgoals_reached,
            subgoals_total=subgoals_total,
            completion_ratio=subgoals_reached / max(subgoals_total, 1),
        )

    @classmethod
    def failure(
        cls, subgoals_reached: int = 0, subgoals_total: int = 0
    ) -> TrajectoryReward:
        """Factory for a failed trajectory.

        Args:
            subgoals_reached: Number of checkpoints reached before failure.
            subgoals_total: Total number of checkpoints.

        Returns:
            A ``TrajectoryReward`` with FAILURE outcome and reward 0.0.
        """
        return cls(
            outcome=TrajectoryOutcome.FAILURE,
            reward=0.0,
            subgoals_reached=subgoals_reached,
            subgoals_total=subgoals_total,
            completion_ratio=subgoals_reached / max(subgoals_total, 1),
        )


class DecomposedRewardRecord(BaseModel):
    """Complete decomposed reward record for a single trajectory.

    CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort — Combines step-level and trajectory-level signals
    into a single record that can be persisted as a KG node and used
    for experience distillation.

    The total reward is computed as:
        R_total = R_trajectory + α · Σ R_step

    Attributes:
        id: Unique record identifier.
        trajectory_id: ID of the trajectory this record belongs to.
        trajectory_reward: Goal-level reward signal.
        step_rewards: Per-step reward signals.
        alpha: Weighting factor for step-level rewards.
        total_reward: Computed total reward.
        total_steps: Total number of steps in the trajectory.
        correct_steps: Number of steps classified as CORRECT.
        incorrect_steps: Number of steps classified as INCORRECT.
        step_accuracy: Fraction of correct steps.
        timestamp: ISO timestamp when this record was created.
    """

    id: str = Field(default_factory=lambda: f"drr_{uuid.uuid4().hex[:8]}")
    trajectory_id: str = ""
    trajectory_reward: TrajectoryReward = Field(default_factory=TrajectoryReward)
    step_rewards: list[StepReward] = Field(default_factory=list)
    alpha: float = Field(default=0.2, ge=0.0, le=1.0)
    total_reward: float = 0.0
    total_steps: int = 0
    correct_steps: int = 0
    incorrect_steps: int = 0
    step_accuracy: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    def model_post_init(self, __context: Any) -> None:
        """Compute derived fields from step and trajectory rewards."""
        if __context is not None:
            logger.debug(
                "DecomposedRewardRecord initialized with context: %s", __context
            )
        self._recompute()

    def _recompute(self) -> None:
        """Recalculate total_reward and derived statistics."""
        self.total_steps = len(self.step_rewards)
        self.correct_steps = sum(
            1 for s in self.step_rewards if s.outcome == StepOutcome.CORRECT
        )
        self.incorrect_steps = sum(
            1 for s in self.step_rewards if s.outcome == StepOutcome.INCORRECT
        )
        self.step_accuracy = self.correct_steps / max(self.total_steps, 1)

        step_reward_sum = sum(s.reward for s in self.step_rewards)
        self.total_reward = round(
            self.trajectory_reward.reward + self.alpha * step_reward_sum,
            4,
        )

    def add_step(self, step: StepReward) -> None:
        """Add a step reward and recompute totals.

        Args:
            step: The step reward to add.
        """
        self.step_rewards.append(step)
        self._recompute()


# ── Reward Decomposer ────────────────────────────────────────────────


class RewardDecomposer:
    """Decomposes trajectory outcomes into step + trajectory reward signals.

    CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort — Implements the decomposed reward framework from
    Long-Horizon Training research.  Designed to be called after each
    trajectory execution to produce a ``DecomposedRewardRecord`` that
    feeds into experience distillation (CONCEPT:AU-AHE.evaluation.backtest-harness).

    The decomposer addresses the credit assignment problem: when a
    trajectory fails, individual correct steps should not be penalized.
    By separating step-level rewards from the trajectory outcome, the
    distillation pipeline can learn from both successful and failed
    trajectories.

    Args:
        alpha: Weighting factor for step-level vs trajectory-level rewards.
            Higher alpha emphasises step quality; lower emphasises outcome.
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self._records: list[DecomposedRewardRecord] = []

        logger.info(
            "[CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort] RewardDecomposer initialized (alpha=%.2f)",
            self.alpha,
        )

    def decompose(
        self,
        trajectory_id: str,
        steps: list[dict[str, Any]],
        goal_achieved: bool,
        subgoals_reached: int = 0,
        subgoals_total: int = 0,
    ) -> DecomposedRewardRecord:
        """Decompose a trajectory into step + trajectory rewards.

        Args:
            trajectory_id: ID of the trajectory.
            steps: List of step dicts with at least ``action`` and
                ``success`` keys.  Optional ``reasoning`` key.
            goal_achieved: Whether the overall task goal was met.
            subgoals_reached: Number of intermediate checkpoints reached.
            subgoals_total: Total number of checkpoints.

        Returns:
            A fully computed ``DecomposedRewardRecord``.
        """
        # Build step rewards
        step_rewards = []
        for i, step_data in enumerate(steps):
            action = step_data.get("action", f"step_{i}")
            success = step_data.get("success", True)
            reasoning = step_data.get("reasoning", "")

            if success:
                sr = StepReward.correct(i, action, reasoning)
            else:
                sr = StepReward.incorrect(i, action, reasoning)
            step_rewards.append(sr)

        # Build trajectory reward
        if goal_achieved:
            traj_reward = TrajectoryReward.success(subgoals_reached, subgoals_total)
        else:
            traj_reward = TrajectoryReward.failure(subgoals_reached, subgoals_total)

        record = DecomposedRewardRecord(
            trajectory_id=trajectory_id,
            trajectory_reward=traj_reward,
            step_rewards=step_rewards,
            alpha=self.alpha,
        )

        self._records.append(record)

        logger.info(
            "[CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort] Decomposed trajectory %s: "
            "total_reward=%.3f, step_accuracy=%.1f%%, outcome=%s",
            trajectory_id,
            record.total_reward,
            record.step_accuracy * 100,
            traj_reward.outcome.value,
        )

        return record

    def get_distillation_insights(self) -> dict[str, Any]:
        """Extract actionable insights for experience distillation.

        Analyzes accumulated reward records to identify:
        1. Steps that are consistently correct across failed trajectories
           (should be reinforced, not penalized)
        2. Steps that are consistently incorrect across successful trajectories
           (indicate fragile success that may not generalize)
        3. Overall step accuracy trend

        Returns:
            Dict with ``correct_in_failures`` count, ``incorrect_in_successes``
            count, ``overall_step_accuracy``, and ``records_analyzed``.
        """
        correct_in_failures = 0
        incorrect_in_successes = 0
        total_steps = 0
        total_correct = 0

        for record in self._records:
            is_success = record.trajectory_reward.outcome == TrajectoryOutcome.SUCCESS
            for step in record.step_rewards:
                total_steps += 1
                if step.outcome == StepOutcome.CORRECT:
                    total_correct += 1
                    if not is_success:
                        correct_in_failures += 1
                elif step.outcome == StepOutcome.INCORRECT:
                    if is_success:
                        incorrect_in_successes += 1

        # Training-spine signals (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort): group-normalized advantage
        # spread + localized failure points — surfaced here so they ride the live
        # distillation-insights path consumed by the EvaluationEngine.
        advantages = self.batch_advantages()
        advantage_spread = (
            round(max(advantages) - min(advantages), 4) if advantages else 0.0
        )
        fp_indices = [fp for fp in self.failure_points() if fp is not None]
        mean_failure_point = (
            round(sum(fp_indices) / len(fp_indices), 2) if fp_indices else None
        )

        return {
            "correct_in_failures": correct_in_failures,
            "incorrect_in_successes": incorrect_in_successes,
            "overall_step_accuracy": round(total_correct / max(total_steps, 1), 3),
            "records_analyzed": len(self._records),
            "total_steps_analyzed": total_steps,
            "advantage_spread": advantage_spread,
            "localized_failures": len(fp_indices),
            "mean_failure_point": mean_failure_point,
            "insight": (
                f"{correct_in_failures} correct steps would be wrongly penalized "
                f"without decomposition. {incorrect_in_successes} incorrect steps "
                f"in successful trajectories indicate fragile success patterns."
            ),
        }

    def batch_advantages(self) -> list[float]:
        """Group-normalized advantage over accumulated trajectories (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

        Turns the per-trajectory ``total_reward``s into GRPO-style advantages —
        the reward signal the Wave-C policy trainers consume.
        """
        from .training_signals import batch_normalized_advantage

        return batch_normalized_advantage([r.total_reward for r in self._records])

    def step_advantages(
        self,
        record: DecomposedRewardRecord | None = None,
        *,
        step_entropies: list[float] | None = None,
    ) -> list[float]:
        """Per-step advantage within one trajectory (CONCEPT:AU-AHE.reward.this-is-read-back / AHE-3.1).

        Group-normalizes the step rewards of ``record`` (default: most recent) into
        per-step advantages — the agent-step credit ARPO writes back into the
        capability reward-EMA, instead of crediting only the final answer. When
        ``step_entropies`` is supplied, the advantages are reweighted by EP-GRPO
        entropy-progress weights so steps that *reduce* uncertainty (advance the
        solution) carry more credit (arXiv:2605.04960).
        """
        from .training_signals import (
            batch_normalized_advantage,
            entropy_progress_weights,
        )

        rec = record or (self._records[-1] if self._records else None)
        if rec is None or not rec.step_rewards:
            return []
        adv = batch_normalized_advantage([s.reward for s in rec.step_rewards])
        if step_entropies is not None:
            weights = entropy_progress_weights(step_entropies)
            n = min(len(adv), len(weights))
            adv = [round(adv[i] * weights[i], 6) for i in range(n)] + adv[n:]
        return adv

    def failure_points(self) -> list[int | None]:
        """First-divergence step index per accumulated record (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

        ``None`` when a trajectory has no INCORRECT step. Anchors error-attributed
        preference-pair construction for DPO-style training.
        """
        from .training_signals import failure_point

        return [
            failure_point([s.outcome == StepOutcome.INCORRECT for s in r.step_rewards])
            for r in self._records
        ]

    def clear(self) -> None:
        """Clear accumulated reward records."""
        self._records.clear()
