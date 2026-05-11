#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AHE-3.3 — Agent-Interpretable Model Evolver.

Autoresearch loop from arXiv:2605.03808, adapted for the agent-utilities
harness. Manages evolutionary model discovery, Pareto frontier tracking,
and KG persistence. Actual model fitting is delegated to MCP tools.

See docs/overview.md §CONCEPT:AHE-3.3
"""


import hashlib
import logging
import time
import uuid
from typing import TYPE_CHECKING

from ..models.imodel import (
    DisplayComplexityBudget,
    DisplayStrategy,
    IModelCandidate,
    IModelNode,
    ParetoPoint,
)
from ..models.knowledge_graph import RegistryEdgeType, RegistryNodeType

if TYPE_CHECKING:
    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class ParetoFrontier:
    """Manages the Pareto frontier of accuracy vs interpretability.

    CONCEPT:AHE-3.3 — A model is Pareto-optimal if no other model
    dominates it on *both* axes (lower predictive_rank AND higher
    interpretability_score).

    Args:
        engine: Optional KG engine for persisting dominance edges.
    """

    def __init__(self, engine: IntelligenceGraphEngine | None = None) -> None:
        self._engine = engine
        self._points: list[ParetoPoint] = []

    @property
    def points(self) -> list[ParetoPoint]:
        return list(self._points)

    @property
    def size(self) -> int:
        return len(self._points)

    def add_model(self, candidate: IModelCandidate, model_id: str = "") -> bool:
        """Add a candidate and update the frontier. Returns True if non-dominated."""
        mid = model_id or f"imodel:{uuid.uuid4().hex[:8]}"
        point = ParetoPoint(
            model_id=mid,
            model_class_name=candidate.model_class_name,
            predictive_rank=candidate.predictive_rank,
            interpretability_score=candidate.interpretability_score,
            generation=candidate.generation,
        )
        if self.is_dominated(point):
            return False

        dominated = [e for e in self._points if self._dominates(point, e)]
        for d in dominated:
            self._points.remove(d)
            self._persist_dominance(point.model_id, d.model_id)
        self._points.append(point)
        return True

    def is_dominated(self, point: ParetoPoint) -> bool:
        """Check if a point is dominated by any existing frontier point."""
        return any(self._dominates(e, point) for e in self._points)

    def get_frontier(self) -> list[ParetoPoint]:
        return list(self._points)

    @staticmethod
    def _dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
        at_least = (
            a.predictive_rank <= b.predictive_rank
            and a.interpretability_score >= b.interpretability_score
        )
        strictly = (
            a.predictive_rank < b.predictive_rank
            or a.interpretability_score > b.interpretability_score
        )
        return at_least and strictly

    def _persist_dominance(self, dominator_id: str, dominated_id: str) -> None:
        if not self._engine:
            return
        try:
            from ..knowledge_graph.core.ogm import KGMapper

            ogm = KGMapper(self._engine)
            ogm.upsert_edge(
                dominator_id,
                dominated_id,
                RegistryEdgeType.PARETO_DOMINATES,
                {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            )
        except Exception as exc:
            logger.debug("Failed to persist dominance edge: %s", exc)


class IModelEvolver:
    """Orchestrates the evolutionary model discovery loop.

    CONCEPT:AHE-3.3 — Agent-Interpretable Model Evolver

    Manages: register → rank → frontier update → KG persist → advance gen.
    Actual model fitting delegated to external ``data-science-mcp``.

    Args:
        engine: KG engine for persistence.
        display_budget: Default display complexity budget.
    """

    def __init__(
        self,
        engine: IntelligenceGraphEngine | None = None,
        display_budget: DisplayComplexityBudget | None = None,
    ) -> None:
        self._engine = engine
        self._display_budget = display_budget or DisplayComplexityBudget()
        self._frontier = ParetoFrontier(engine)
        self._generation: int = 0
        self._candidates: list[IModelCandidate] = []

    @property
    def frontier(self) -> ParetoFrontier:
        return self._frontier

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def candidates(self) -> list[IModelCandidate]:
        return list(self._candidates)

    def register_candidate(
        self,
        model_class_name: str,
        source_code: str = "",
        str_output: str = "",
        rmse_scores: dict[str, float] | None = None,
        interpretability_score: float = 0.0,
        parent_id: str = "",
        display_strategy: DisplayStrategy = DisplayStrategy.ADAPTIVE,
    ) -> IModelCandidate:
        """Register a new model candidate for evaluation."""
        candidate = IModelCandidate(
            model_class_name=model_class_name,
            source_code=source_code,
            str_output=str_output,
            rmse_scores=rmse_scores or {},
            interpretability_score=interpretability_score,
            generation=self._generation,
            parent_id=parent_id,
            display_strategy=display_strategy,
        )
        self._candidates.append(candidate)
        return candidate

    def rank_models(
        self,
        candidates: list[IModelCandidate] | None = None,
    ) -> list[IModelCandidate]:
        """Rank models by aggregate RMSE. Assigns normalized rank (0=best)."""
        pool = candidates if candidates is not None else self._candidates
        if not pool:
            return []
        scored: list[tuple[float, IModelCandidate]] = []
        for c in pool:
            mean_rmse = (
                sum(c.rmse_scores.values()) / len(c.rmse_scores)
                if c.rmse_scores
                else float("inf")
            )
            scored.append((mean_rmse, c))
        scored.sort(key=lambda x: x[0])
        n = len(scored)
        for rank_idx, (_, cand) in enumerate(scored):
            cand.predictive_rank = rank_idx / max(n - 1, 1)
        return [c for _, c in scored]

    def evolve_round(
        self,
        candidates: list[IModelCandidate] | None = None,
    ) -> list[ParetoPoint]:
        """Run one evolution round: rank → frontier → persist → advance gen."""
        pool = candidates if candidates is not None else self._candidates
        if not pool:
            return []
        ranked = self.rank_models(pool)
        for candidate in ranked:
            model_id = f"imodel:{uuid.uuid4().hex[:8]}"
            on_frontier = self._frontier.add_model(candidate, model_id)
            if on_frontier:
                self._persist_model(candidate, model_id)
        self._generation += 1
        return self._frontier.get_frontier()

    def compute_reward_decomposition(
        self,
        candidate: IModelCandidate,
    ) -> dict[str, float]:
        """Decompose fitness into trajectory + step rewards (CONCEPT:AHE-3.1).

        trajectory_reward = 1.0 - predictive_rank (accuracy)
        step_reward = interpretability_score
        """
        alpha = 0.5
        trajectory_reward = 1.0 - candidate.predictive_rank
        step_reward = candidate.interpretability_score
        return {
            "trajectory_reward": trajectory_reward,
            "step_reward": step_reward,
            "total_reward": trajectory_reward + alpha * step_reward,
            "alpha": alpha,
        }

    def select_display_strategy(
        self,
        str_output: str,
        n_features: int = 1,
    ) -> DisplayStrategy:
        """Auto-select display strategy based on output characteristics."""
        lines = str_output.strip().split("\n")
        n_lines = len(lines)
        if n_lines == 1 and any(op in str_output for op in ["=", "+", "*"]):
            return DisplayStrategy.SYMBOLIC_EQUATION
        if n_lines <= 5 and n_features <= 5:
            return DisplayStrategy.LINEAR_COLLAPSE
        if any("|" in line or "\t" in line for line in lines):
            return DisplayStrategy.PIECEWISE_TABLE
        if any("coef" in line.lower() or "weight" in line.lower() for line in lines):
            return DisplayStrategy.COEFFICIENT_SUMMARY
        return DisplayStrategy.ADAPTIVE

    def _persist_model(self, candidate: IModelCandidate, model_id: str) -> None:
        if not self._engine:
            return
        try:
            from ..knowledge_graph.core.ogm import KGMapper

            ogm = KGMapper(self._engine)
            node = IModelNode(
                id=model_id,
                type=RegistryNodeType.IMODEL,
                name=f"IModel: {candidate.model_class_name}",
                description=(
                    f"Agent-interpretable model (gen={candidate.generation}, "
                    f"interp={candidate.interpretability_score:.3f}, "
                    f"rank={candidate.predictive_rank:.3f})"
                ),
                model_class_name=candidate.model_class_name,
                str_representation=candidate.str_output,
                source_code=candidate.source_code,
                interpretability_score=candidate.interpretability_score,
                predictive_rank=candidate.predictive_rank,
                pareto_optimal=True,
                display_strategy=candidate.display_strategy,
                generation=candidate.generation,
                accuracy_metrics=dict(candidate.rmse_scores),
                metadata={
                    "str_hash": hashlib.sha256(
                        candidate.str_output.encode()
                    ).hexdigest()[:16],
                    "concept": "AHE-3.15",
                    "paper": "arXiv:2605.03808",
                },
            )
            ogm.upsert(node)
            if candidate.parent_id:
                ogm.upsert_edge(
                    model_id,
                    candidate.parent_id,
                    RegistryEdgeType.EVOLVED_MODEL,
                    {
                        "generation": candidate.generation,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
        except Exception as exc:
            logger.warning("Failed to persist IModelNode: %s", exc)
