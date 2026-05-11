"""Evaluation Engine — Consolidated Evaluation Facade.

CONCEPT:AHE-3.1 — Evaluation Engine

Provides a single entry point for evaluation-related capabilities:
- Continuous evaluation pipeline (AHE-3.1 via ``ContinuousEvaluationEngine``)
- Decomposed reward signals (AHE-3.10 via ``RewardDecomposer``)
- Trace distillation (KG-2.4 via ``TraceDistiller``)

The facade enables a unified evaluation cycle:
    execute_trajectory() → decompose_rewards() → distill_traces()
    → extract_insights() → evolve_agents()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Consolidated evaluation engine.

    CONCEPT:AHE-3.1 — Evaluation Engine

    Combines continuous evaluation (AHE-3.1), reward decomposition
    (AHE-3.10), and trace distillation (KG-2.4) into a unified
    evaluation lifecycle.

    Usage::

        eval_engine = EvaluationEngine(kg_engine)

        # Decompose a trajectory
        record = eval_engine.decompose_trajectory(
            trajectory_id="traj-1",
            steps=[{"action": "search", "success": True}],
            goal_achieved=True,
        )

        # Get distillation insights
        insights = eval_engine.get_distillation_insights()

        # Start background trace distiller
        eval_engine.start_trace_distiller()

    Args:
        engine: The IntelligenceGraphEngine for KG access.
        alpha: Reward decomposition weighting factor (step vs trajectory).
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        alpha: float = 0.2,
    ) -> None:
        self._engine = engine
        self._alpha = alpha
        self._reward_decomposer: Any = None
        self._trace_distiller: Any = None
        self._continuous_eval: Any = None
        self._initialized = False

    def _lazy_init(self) -> None:
        """Lazily initialize sub-components."""
        if self._initialized:
            return
        self._initialized = True

        # Reward decomposition (AHE-3.10)
        try:
            from ..graph.reward_decomposition import RewardDecomposer

            self._reward_decomposer = RewardDecomposer(alpha=self._alpha)
        except Exception as e:
            logger.debug("RewardDecomposer not available: %s", e)

        # Trace distiller (KG-2.4)
        if self._engine:
            try:
                from ..knowledge_graph.adaptation.trace_distiller import TraceDistiller

                self._trace_distiller = TraceDistiller(self._engine)
            except Exception as e:
                logger.debug("TraceDistiller not available: %s", e)

    # --- Reward Decomposition API (AHE-3.10) ---

    def decompose_trajectory(
        self,
        trajectory_id: str,
        steps: list[dict[str, Any]],
        goal_achieved: bool,
        subgoals_reached: int = 0,
        subgoals_total: int = 0,
    ) -> Any:
        """Decompose a trajectory into step + trajectory rewards.

        Separates trajectory-level reward (goal achievement) from step-level
        reward (local constraint satisfaction) to prevent penalizing correct
        intermediate steps in failed trajectories.

        Args:
            trajectory_id: ID of the trajectory.
            steps: List of step dicts with ``action`` and ``success`` keys.
            goal_achieved: Whether the overall task goal was met.
            subgoals_reached: Intermediate checkpoints reached.
            subgoals_total: Total checkpoints.

        Returns:
            DecomposedRewardRecord with computed rewards.
        """
        self._lazy_init()
        if not self._reward_decomposer:
            raise RuntimeError("RewardDecomposer not available")
        return self._reward_decomposer.decompose(
            trajectory_id, steps, goal_achieved, subgoals_reached, subgoals_total
        )

    def get_distillation_insights(self) -> dict[str, Any]:
        """Extract actionable insights from accumulated reward records.

        Identifies:
        - Correct steps in failed trajectories (should be reinforced)
        - Incorrect steps in successful trajectories (fragile success)
        - Overall step accuracy trend

        Returns:
            Dict with insight counts and analysis.
        """
        self._lazy_init()
        if not self._reward_decomposer:
            return {}
        return self._reward_decomposer.get_distillation_insights()

    # --- Trace Distillation API (KG-2.4) ---

    def start_trace_distiller(self) -> None:
        """Start background trace distillation task."""
        self._lazy_init()
        if self._trace_distiller:
            self._trace_distiller.start()

    def stop_trace_distiller(self) -> None:
        """Stop background trace distillation task."""
        if self._trace_distiller:
            self._trace_distiller.stop()

    # --- Unified Evaluation ---

    def evaluate_and_decompose(
        self,
        trajectory_id: str,
        steps: list[dict[str, Any]],
        goal_achieved: bool,
    ) -> dict[str, Any]:
        """Run a combined evaluation and decomposition cycle.

        Args:
            trajectory_id: Trajectory identifier.
            steps: Step list with action/success keys.
            goal_achieved: Whether the goal was achieved.

        Returns:
            Dict with decomposed rewards and insights.
        """
        record = self.decompose_trajectory(
            trajectory_id=trajectory_id,
            steps=steps,
            goal_achieved=goal_achieved,
        )
        insights = self.get_distillation_insights()

        return {
            "trajectory_id": trajectory_id,
            "total_reward": record.total_reward,
            "step_accuracy": record.step_accuracy,
            "trajectory_outcome": record.trajectory_reward.outcome,
            "insights": insights,
        }
