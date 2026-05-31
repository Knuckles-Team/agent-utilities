"""Evaluation Engine — Synthesized Evaluation Facade.

CONCEPT:AHE-3.1 — Evaluation Engine

Provides a single entry point for evaluation-related capabilities:
- Continuous evaluation pipeline (AHE-3.1 via ``ContinuousEvaluationEngine``)
- Decomposed reward signals (AHE-3.10 via ``RewardDecomposer``)
- Trace distillation (KG-2.4 via ``TraceDistiller``)

The facade enables a unified evaluation cycle:
    execute_trajectory() → decompose_rewards() → distill_traces()
    → extract_insights() → evolve_agents()

BrowseComp-Plus Extensions (arXiv:2508.06600):
    - Adaptive reasoning effort (continuous 0.0–1.0 scaling)
    - Disentangled evaluation (retriever vs LLM vs tool-use)
    - Citation quality tracking (precision/recall/F1)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Synthesized evaluation engine.

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
        self._citation_tracker: Any = None
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

        # Citation tracking (AHE-3.1 — BrowseComp-Plus)
        try:
            from .citation_tracker import CitationTracker

            self._citation_tracker = CitationTracker()
        except Exception as e:
            logger.debug("CitationTracker not available: %s", e)

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
        reasoning_effort: float = 0.5,
    ) -> dict[str, Any]:
        """Run a combined evaluation and decomposition cycle.

        Args:
            trajectory_id: Trajectory identifier.
            steps: Step list with action/success keys.
            goal_achieved: Whether the goal was achieved.
            reasoning_effort: Continuous reasoning effort level [0.0, 1.0]
                (CONCEPT:AHE-3.1 — Adaptive Reasoning Effort).

        Returns:
            Dict with decomposed rewards, insights, and reasoning budget.
        """
        # Resolve reasoning budget (CONCEPT:AHE-3.1)
        budget_info: dict[str, Any] = {}
        try:
            from .reasoning_effort import get_budget

            budget = get_budget(reasoning_effort)
            budget_info = budget.model_dump()
        except Exception as e:
            logger.debug("ReasoningBudget not available: %s", e)

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
            "reasoning_budget": budget_info,
        }

    # --- Disentangled Evaluation (BrowseComp-Plus) ---

    def evaluate_disentangled(
        self,
        trajectory_id: str,
        steps: list[dict[str, Any]],
        goal_achieved: bool,
        retrieval_results: list[dict[str, Any]] | None = None,
        gold_doc_ids: set[str] | None = None,
        response_text: str = "",
    ) -> dict[str, Any]:
        """Run disentangled evaluation separating retriever from LLM quality.

        CONCEPT:AHE-3.1 — Disentangled Evaluation (BrowseComp-Plus)

        Returns separate scores for:
        - retriever_metrics: precision, recall, nDCG, MRR
        - reasoning_metrics: step_accuracy, goal_achievement
        - citation_metrics: citation precision/recall/F1

        Args:
            trajectory_id: Trajectory identifier.
            steps: Step list with action/success keys.
            goal_achieved: Whether the goal was achieved.
            retrieval_results: Raw retrieval results with _score fields.
            gold_doc_ids: Known correct document IDs for nDCG.
            response_text: Agent response text for citation extraction.

        Returns:
            Dict with disentangled metrics.
        """
        self._lazy_init()

        # 1. Retriever metrics
        retriever_metrics: dict[str, Any] = {}
        if retrieval_results is not None:
            try:
                from ..knowledge_graph.retrieval.retrieval_quality import (
                    RetrievalQualityGate,
                )

                if self._engine:
                    gate = RetrievalQualityGate(self._engine)
                    report = gate.assess_quality(retrieval_results)
                    retriever_metrics = {
                        "precision": report.context_precision,
                        "recall": report.context_recall,
                        "mrr": report.mean_reciprocal_rank,
                        "mean_relevance": report.mean_relevance_score,
                        "composite_quality": report.composite_quality,
                        "failure_modes": [
                            m.value for m in report.failure_modes_detected
                        ],
                    }
                    # nDCG if gold docs provided
                    if gold_doc_ids:
                        retriever_metrics["ndcg"] = gate.compute_ndcg(
                            retrieval_results, gold_doc_ids
                        )
            except Exception as e:
                logger.debug("Retriever metrics failed: %s", e)

        # 2. Reasoning metrics
        reasoning_metrics: dict[str, Any] = {
            "goal_achieved": goal_achieved,
        }
        try:
            record = self.decompose_trajectory(
                trajectory_id=trajectory_id,
                steps=steps,
                goal_achieved=goal_achieved,
            )
            reasoning_metrics.update(
                {
                    "total_reward": record.total_reward,
                    "step_accuracy": record.step_accuracy,
                    "trajectory_outcome": record.trajectory_reward.outcome,
                }
            )
        except Exception as e:
            logger.debug("Reasoning metrics failed: %s", e)

        # 3. Citation metrics
        citation_metrics: dict[str, Any] = {}
        if response_text and self._citation_tracker:
            try:
                citations = self._citation_tracker.extract_citations(response_text)
                retrieved_ids = {r.get("id", "") for r in (retrieval_results or [])} - {
                    ""
                }
                report = self._citation_tracker.evaluate_citations(
                    citations, retrieved_ids, gold_doc_ids or set()
                )
                citation_metrics = report.model_dump()
            except Exception as e:
                logger.debug("Citation metrics failed: %s", e)

        return {
            "trajectory_id": trajectory_id,
            "retriever_metrics": retriever_metrics,
            "reasoning_metrics": reasoning_metrics,
            "citation_metrics": citation_metrics,
        }
