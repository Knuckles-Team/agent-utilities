#!/usr/bin/python
"""CONCEPT:AHE-3.9 — Horizon-Aware Task Curriculum.

Implements progressive horizon scheduling derived from the "Long-Horizon
Training" research (Kim et al., ICML 2026).  The core insight is that
RL-trained LLM agents suffer training instability as horizon length
increases, even when reasoning complexity remains constant.  Horizon
reduction via macro-action composition and subgoal curriculum staging
stabilises training and improves generalization to longer horizons.

Key primitives:

* **MacroAction**: Composes a sequence of atomic actions into a single
  higher-level primitive, reducing effective interaction steps.
* **HorizonCurriculum**: Schedules tasks from short-horizon (macro-action)
  variants to progressively longer horizons as the agent demonstrates
  proficiency.
* **SubgoalCheckpoint**: Intermediate milestone within a long-horizon task
  that enables step-level reward attribution.

Integrates with:
    - CONCEPT:ORCH-1.1 (HTN Planning): Macro-actions as plan step composites
    - CONCEPT:ORCH-1.4 (Swarm Presets): Curriculum stages as DAG phases
    - CONCEPT:AHE-3.7 (Heavy Thinking): Trajectory diversity via horizon variants
    - CONCEPT:AHE-3.10 (Reward Decomposition): Step-level checkpoint rewards

See docs/overview.md §CONCEPT:AHE-3.9.
"""

from __future__ import annotations

import logging
import uuid
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Enumerations ─────────────────────────────────────────────────────


class CurriculumStage(StrEnum):
    """Progression stages for horizon-aware curriculum scheduling.

    Attributes:
        MACRO: Short-horizon variant using composite macro-actions.
        INTERMEDIATE: Medium-horizon with partial macro expansion.
        FULL: Full atomic-action horizon (production execution).
    """

    MACRO = "macro"
    INTERMEDIATE = "intermediate"
    FULL = "full"


class PromotionPolicy(StrEnum):
    """Policies governing when to advance to a longer horizon stage.

    Attributes:
        THRESHOLD: Promote when success rate exceeds a fixed threshold.
        PLATEAU: Promote when performance improvement plateaus.
        ADAPTIVE: Use exponential moving average of success rate.
    """

    THRESHOLD = "threshold"
    PLATEAU = "plateau"
    ADAPTIVE = "adaptive"


# ── Data Models ──────────────────────────────────────────────────────


class MacroAction(BaseModel):
    """A composite action that bundles multiple atomic steps.

    CONCEPT:AHE-3.9 — Macro-actions reduce effective horizon length by
    combining sequential atomic actions into single execution primitives.

    Example: In a code-editing agent, the atomic actions ``open_file``,
    ``navigate_to_line``, ``insert_text`` can be composed into a single
    macro-action ``edit_file_at_line(path, line, text)``.

    Attributes:
        id: Unique identifier for this macro-action.
        name: Human-readable name.
        atomic_steps: Ordered list of atomic action descriptions.
        step_count: Number of atomic steps this macro replaces.
        preconditions: Conditions that must hold before execution.
        postconditions: Expected state after successful execution.
        success_rate: Historical success rate for this macro.
    """

    id: str = Field(default_factory=lambda: f"macro_{uuid.uuid4().hex[:8]}")
    name: str
    atomic_steps: list[str]
    step_count: int = 0
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        """Auto-derive step_count from atomic_steps if not set."""
        if __context is not None:
            logger.debug("MacroAction initialized with context: %s", __context)
        if self.step_count == 0:
            self.step_count = len(self.atomic_steps)


class SubgoalCheckpoint(BaseModel):
    """An intermediate milestone within a long-horizon task.

    CONCEPT:AHE-3.9 — Subgoal checkpoints decompose trajectory-level
    goals into step-level verifiable milestones, enabling credit
    assignment at intermediate points rather than only at trajectory end.

    Attributes:
        id: Unique identifier for this checkpoint.
        description: Human-readable milestone description.
        verification_fn: Name of the function that checks completion.
        order: Position in the subgoal sequence (0-indexed).
        reached: Whether this checkpoint has been reached.
        reached_at_step: Step number when checkpoint was reached.
        reward: Step-level reward for reaching this checkpoint.
    """

    id: str = Field(default_factory=lambda: f"sg_{uuid.uuid4().hex[:8]}")
    description: str
    verification_fn: str = ""
    order: int = 0
    reached: bool = False
    reached_at_step: int | None = None
    reward: float = Field(default=0.1, ge=0.0, le=1.0)

    def mark_reached(self, step: int) -> None:
        """Record that this checkpoint was reached at the given step.

        Args:
            step: The step number at which this checkpoint was reached.
        """
        self.reached = True
        self.reached_at_step = step


class HorizonStageConfig(BaseModel):
    """Configuration for a single curriculum stage.

    Attributes:
        stage: The curriculum stage level.
        max_horizon: Maximum number of interaction steps allowed.
        macro_actions: Available macro-actions for this stage.
        subgoals: Checkpoint milestones for this stage.
        promotion_threshold: Success rate needed to advance.
        min_episodes: Minimum episodes before promotion is considered.
    """

    stage: CurriculumStage
    max_horizon: int
    macro_actions: list[MacroAction] = Field(default_factory=list)
    subgoals: list[SubgoalCheckpoint] = Field(default_factory=list)
    promotion_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    min_episodes: int = Field(default=3, ge=1)


# ── Curriculum Engine ────────────────────────────────────────────────


class HorizonCurriculum:
    """Progressive horizon scheduling engine.

    CONCEPT:AHE-3.9 — Implements the key insight from Long-Horizon
    Training research: models trained on reduced horizons generalize
    better to longer horizons.  This engine stages task execution through
    progressively longer horizons:

    1. **MACRO stage**: Execute with composite macro-actions, minimising
       the effective number of interaction steps.
    2. **INTERMEDIATE stage**: Partially expand macros, allowing some
       atomic steps while keeping the horizon manageable.
    3. **FULL stage**: Execute with all atomic actions at production
       horizon length.

    Promotion between stages is governed by a configurable policy:
    ``THRESHOLD`` (fixed success rate), ``PLATEAU`` (improvement stall),
    or ``ADAPTIVE`` (exponential moving average).

    Args:
        stages: Ordered list of ``HorizonStageConfig`` (short → long).
        promotion_policy: Policy for advancing between stages.
        ema_alpha: Smoothing factor for adaptive promotion (0 < α ≤ 1).
    """

    def __init__(
        self,
        stages: list[HorizonStageConfig] | None = None,
        promotion_policy: PromotionPolicy = PromotionPolicy.THRESHOLD,
        ema_alpha: float = 0.3,
    ) -> None:
        self.stages = stages or self._default_stages()
        self.promotion_policy = promotion_policy
        self.ema_alpha = ema_alpha

        self._current_stage_idx: int = 0
        self._episode_results: list[list[bool]] = [[] for _ in self.stages]
        self._ema_success: list[float] = [0.0 for _ in self.stages]

        logger.info(
            "[CONCEPT:AHE-3.9] Horizon Curriculum initialized: %d stages, policy=%s",
            len(self.stages),
            self.promotion_policy.value,
        )

    @staticmethod
    def _default_stages() -> list[HorizonStageConfig]:
        """Create default 3-stage curriculum.

        Returns:
            A list of ``HorizonStageConfig`` with sensible defaults:
            MACRO (horizon=5), INTERMEDIATE (horizon=15), FULL (horizon=50).
        """
        return [
            HorizonStageConfig(
                stage=CurriculumStage.MACRO,
                max_horizon=5,
                promotion_threshold=0.7,
                min_episodes=3,
            ),
            HorizonStageConfig(
                stage=CurriculumStage.INTERMEDIATE,
                max_horizon=15,
                promotion_threshold=0.6,
                min_episodes=5,
            ),
            HorizonStageConfig(
                stage=CurriculumStage.FULL,
                max_horizon=50,
                promotion_threshold=0.5,
                min_episodes=5,
            ),
        ]

    @property
    def current_stage(self) -> HorizonStageConfig:
        """Get the current curriculum stage configuration.

        Returns:
            The active ``HorizonStageConfig``.
        """
        return self.stages[self._current_stage_idx]

    @property
    def current_stage_name(self) -> str:
        """Human-readable name of the current stage.

        Returns:
            The stage enum value as a string.
        """
        return self.current_stage.stage.value

    @property
    def is_final_stage(self) -> bool:
        """Whether the curriculum has reached its final (longest) stage.

        Returns:
            True if at the last stage.
        """
        return self._current_stage_idx >= len(self.stages) - 1

    def record_episode(self, success: bool) -> dict[str, Any]:
        """Record the outcome of an episode and check for promotion.

        Called after each task execution to update statistics and
        potentially advance to the next horizon stage.

        Args:
            success: Whether the episode completed successfully.

        Returns:
            Dict with ``stage``, ``promoted``, ``success_rate``,
            ``episodes_in_stage``, and ``effective_horizon``.
        """
        idx = self._current_stage_idx
        self._episode_results[idx].append(success)

        # Update EMA
        outcome = 1.0 if success else 0.0
        self._ema_success[idx] = (
            self.ema_alpha * outcome + (1 - self.ema_alpha) * self._ema_success[idx]
        )

        promoted = False
        if not self.is_final_stage:
            promoted = self._check_promotion()

        stage = self.current_stage
        episodes = self._episode_results[idx]
        success_rate = sum(1 for e in episodes if e) / max(len(episodes), 1)

        result = {
            "stage": stage.stage.value,
            "promoted": promoted,
            "success_rate": round(success_rate, 3),
            "ema_success": round(self._ema_success[idx], 3),
            "episodes_in_stage": len(episodes),
            "effective_horizon": stage.max_horizon,
        }

        if promoted:
            logger.info(
                "[CONCEPT:AHE-3.9] Promoted to stage %s (horizon=%d)",
                self.current_stage.stage.value,
                self.current_stage.max_horizon,
            )

        return result

    def _check_promotion(self) -> bool:
        """Check if promotion to the next stage is warranted.

        Uses the configured ``PromotionPolicy`` to determine readiness.

        Returns:
            True if the agent was promoted.
        """
        idx = self._current_stage_idx
        stage = self.stages[idx]
        episodes = self._episode_results[idx]

        if len(episodes) < stage.min_episodes:
            return False

        promoted = False

        if self.promotion_policy == PromotionPolicy.THRESHOLD:
            success_rate = sum(1 for e in episodes if e) / len(episodes)
            promoted = success_rate >= stage.promotion_threshold

        elif self.promotion_policy == PromotionPolicy.PLATEAU:
            if len(episodes) >= stage.min_episodes + 3:
                recent = episodes[-3:]
                older = episodes[-6:-3] if len(episodes) >= 6 else episodes[:3]
                recent_rate = sum(1 for e in recent if e) / max(len(recent), 1)
                older_rate = sum(1 for e in older if e) / max(len(older), 1)
                # Plateau: improvement < 5% and above minimum threshold
                promoted = (
                    abs(recent_rate - older_rate) < 0.05
                    and recent_rate >= stage.promotion_threshold * 0.8
                )

        elif self.promotion_policy == PromotionPolicy.ADAPTIVE:
            promoted = self._ema_success[idx] >= stage.promotion_threshold

        if promoted:
            self._current_stage_idx = min(idx + 1, len(self.stages) - 1)

        return promoted

    def get_macro_actions(self) -> list[MacroAction]:
        """Get available macro-actions for the current stage.

        In MACRO stage, all macros are available.  In INTERMEDIATE, only
        high-reliability macros (success_rate > 0.8) are available.
        In FULL stage, no macros are provided.

        Returns:
            List of ``MacroAction`` available at the current stage.
        """
        stage = self.current_stage.stage
        if stage == CurriculumStage.FULL:
            return []
        if stage == CurriculumStage.INTERMEDIATE:
            return [m for m in self.current_stage.macro_actions if m.success_rate > 0.8]
        return list(self.current_stage.macro_actions)

    def compute_horizon_reduction_ratio(self) -> float:
        """Calculate how much the effective horizon is reduced vs full.

        Returns:
            Ratio in [0.0, 1.0] where 0.0 = fully reduced, 1.0 = full horizon.
        """
        if not self.stages:
            return 1.0
        full_horizon = self.stages[-1].max_horizon
        current_horizon = self.current_stage.max_horizon
        if full_horizon == 0:
            return 1.0
        return round(current_horizon / full_horizon, 3)

    def reset(self) -> None:
        """Reset the curriculum to the first (shortest-horizon) stage.

        Clears all episode history and EMA values.
        """
        self._current_stage_idx = 0
        self._episode_results = [[] for _ in self.stages]
        self._ema_success = [0.0 for _ in self.stages]
        logger.info("[CONCEPT:AHE-3.9] Curriculum reset to stage MACRO")

    def to_dict(self) -> dict[str, Any]:
        """Serialize curriculum state for persistence or logging.

        Returns:
            Dict with full curriculum state including per-stage stats.
        """
        return {
            "current_stage": self.current_stage.stage.value,
            "current_stage_idx": self._current_stage_idx,
            "total_stages": len(self.stages),
            "promotion_policy": self.promotion_policy.value,
            "horizon_reduction_ratio": self.compute_horizon_reduction_ratio(),
            "stages": [
                {
                    "stage": s.stage.value,
                    "max_horizon": s.max_horizon,
                    "promotion_threshold": s.promotion_threshold,
                    "episodes": len(self._episode_results[i]),
                    "success_rate": round(
                        sum(1 for e in self._episode_results[i] if e)
                        / max(len(self._episode_results[i]), 1),
                        3,
                    ),
                    "ema_success": round(self._ema_success[i], 3),
                    "macro_action_count": len(s.macro_actions),
                    "subgoal_count": len(s.subgoals),
                }
                for i, s in enumerate(self.stages)
            ],
        }
