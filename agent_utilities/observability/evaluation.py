#!/usr/bin/python
"""Evaluation & Monitoring Framework — CONCEPT:AHE-3.1.

Multi-dimensional evaluation with LLM-as-Judge rubrics, trend monitoring,
quality alerting, and human calibration support.

Design-pattern source: Chapter 19 — Evaluation and Monitoring.

OWL: :EvaluationRecord rdfs:subClassOf :Observation
See docs/design-patterns-alignment.md §AU-020.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Dimension weight defaults (configurable)
DEFAULT_WEIGHTS: dict[str, float] = {
    "correctness": 0.35,
    "completeness": 0.25,
    "relevance": 0.25,
    "safety": 0.15,
}


class EvaluationDimension(BaseModel):
    """A single evaluation dimension with score and evidence."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=0.25, ge=0.0, le=1.0)
    rubric: str = ""
    evidence: str = ""


class EvaluationRubric(BaseModel):
    """Scoring rubric defining dimensions and their criteria."""

    id: str = "default"
    name: str = "Default Evaluation Rubric"
    dimensions: list[EvaluationDimension] = Field(
        default_factory=lambda: [
            EvaluationDimension(
                name="correctness",
                score=0.0,
                weight=0.35,
                rubric="Is the response factually accurate?",
            ),
            EvaluationDimension(
                name="completeness",
                score=0.0,
                weight=0.25,
                rubric="Does the response address all parts of the query?",
            ),
            EvaluationDimension(
                name="relevance",
                score=0.0,
                weight=0.25,
                rubric="Is the response directly relevant to the query?",
            ),
            EvaluationDimension(
                name="safety",
                score=0.0,
                weight=0.15,
                rubric="Is the response safe and appropriate?",
            ),
        ]
    )


class MultiDimensionalEvaluation(BaseModel):
    """Complete multi-dimensional evaluation result."""

    dimensions: list[EvaluationDimension] = Field(default_factory=list)
    composite_score: float = Field(default=0.0, ge=0.0, le=1.0)
    evaluator: str = "llm-judge"
    rubric_id: str = "default"
    timestamp: float = Field(default_factory=time.time)
    session_id: str = ""
    query_preview: str = ""
    response_preview: str = ""

    def compute_composite(self) -> float:
        """Compute weighted composite score from dimensions."""
        if not self.dimensions:
            return 0.0
        total_weight = sum(d.weight for d in self.dimensions) or 1.0
        self.composite_score = (
            sum(d.score * d.weight for d in self.dimensions) / total_weight
        )
        return self.composite_score


class QualityAlert(BaseModel):
    """An alert triggered when quality degrades below threshold."""

    dimension: str
    current_score: float
    threshold: float
    trend: list[float] = Field(default_factory=list)
    message: str = ""
    severity: str = "warning"  # warning, critical


class EvaluationMonitor:
    """Continuous quality monitoring with trend analysis and alerting.

    Parameters
    ----------
    alert_threshold : float
        Score below which a quality alert is triggered (default: 0.6).
    critical_threshold : float
        Score below which a critical alert is triggered (default: 0.3).
    kg_engine : optional
        If provided, evaluations are persisted to the KG.
    """

    def __init__(
        self,
        alert_threshold: float = 0.6,
        critical_threshold: float = 0.3,
        kg_engine: Any = None,
    ) -> None:
        self._alert_threshold = alert_threshold
        self._critical_threshold = critical_threshold
        self._engine = kg_engine
        self._history: list[MultiDimensionalEvaluation] = []
        self._dimension_history: dict[str, list[float]] = defaultdict(list)

    def evaluate(
        self,
        query: str,
        response: str,
        dimension_scores: dict[str, float] | None = None,
        rubric: EvaluationRubric | None = None,
        evaluator: str = "llm-judge",
        session_id: str = "",
    ) -> MultiDimensionalEvaluation:
        """Create a multi-dimensional evaluation.

        Parameters
        ----------
        query : str
            The original query.
        response : str
            The agent's response.
        dimension_scores : dict[str, float] | None
            Pre-computed dimension scores (e.g. from LLM-as-Judge).
        rubric : EvaluationRubric | None
            Scoring rubric (defaults to standard 4-dimension rubric).
        evaluator : str
            Who performed the evaluation.
        session_id : str
            Session identifier.
        """
        rubric = rubric or EvaluationRubric()
        scores = dimension_scores or {}

        dimensions = []
        for dim in rubric.dimensions:
            score = scores.get(dim.name, dim.score)
            dimensions.append(
                EvaluationDimension(
                    name=dim.name,
                    score=score,
                    weight=dim.weight,
                    rubric=dim.rubric,
                )
            )

        evaluation = MultiDimensionalEvaluation(
            dimensions=dimensions,
            evaluator=evaluator,
            rubric_id=rubric.id,
            session_id=session_id,
            query_preview=query[:200],
            response_preview=response[:200],
        )
        evaluation.compute_composite()

        self.record_evaluation(evaluation)
        return evaluation

    def record_evaluation(self, evaluation: MultiDimensionalEvaluation) -> None:
        """Record an evaluation for trend tracking."""
        self._history.append(evaluation)
        for dim in evaluation.dimensions:
            self._dimension_history[dim.name].append(dim.score)

    def get_trend(self, dimension: str, lookback: int = 50) -> list[float]:
        """Get score trend for a dimension over recent evaluations."""
        scores = self._dimension_history.get(dimension, [])
        return scores[-lookback:]

    def get_composite_trend(self, lookback: int = 50) -> list[float]:
        """Get composite score trend over recent evaluations."""
        return [e.composite_score for e in self._history[-lookback:]]

    def check_alerts(self) -> list[QualityAlert]:
        """Check for quality degradation alerts across all dimensions."""
        alerts: list[QualityAlert] = []
        for dim_name, scores in self._dimension_history.items():
            if len(scores) < 3:
                continue
            recent_avg = sum(scores[-5:]) / len(scores[-5:])
            if recent_avg < self._critical_threshold:
                alerts.append(
                    QualityAlert(
                        dimension=dim_name,
                        current_score=recent_avg,
                        threshold=self._critical_threshold,
                        trend=scores[-10:],
                        severity="critical",
                        message=f"CRITICAL: {dim_name} score ({recent_avg:.2f}) "
                        f"below critical threshold ({self._critical_threshold})",
                    )
                )
            elif recent_avg < self._alert_threshold:
                alerts.append(
                    QualityAlert(
                        dimension=dim_name,
                        current_score=recent_avg,
                        threshold=self._alert_threshold,
                        trend=scores[-10:],
                        severity="warning",
                        message=f"WARNING: {dim_name} score ({recent_avg:.2f}) "
                        f"below alert threshold ({self._alert_threshold})",
                    )
                )
        return alerts

    async def persist_to_kg(self, evaluation: MultiDimensionalEvaluation) -> None:
        """Persist evaluation record to the Knowledge Graph."""
        if self._engine is None:
            return
        try:
            from agent_utilities.models.knowledge_graph import (
                EvaluationRecordNode,
                RegistryNodeType,
            )

            node = EvaluationRecordNode(
                id=f"eval:{evaluation.session_id}:{evaluation.timestamp}",
                type=RegistryNodeType.EVALUATION_RECORD,
                name=f"Evaluation: {evaluation.session_id}",
                correctness_score=next(
                    (d.score for d in evaluation.dimensions if d.name == "correctness"),
                    0.0,
                ),
                completeness_score=next(
                    (
                        d.score
                        for d in evaluation.dimensions
                        if d.name == "completeness"
                    ),
                    0.0,
                ),
                relevance_score=next(
                    (d.score for d in evaluation.dimensions if d.name == "relevance"),
                    0.0,
                ),
                safety_score=next(
                    (d.score for d in evaluation.dimensions if d.name == "safety"), 1.0
                ),
                composite_score=evaluation.composite_score,
                evaluator=evaluation.evaluator,
                rubric_id=evaluation.rubric_id,
                session_id=evaluation.session_id,
            )
            if hasattr(self._engine, "upsert_node"):
                self._engine.upsert_node(node.model_dump())
            logger.info("Persisted evaluation to KG: %.2f", evaluation.composite_score)
        except Exception:
            logger.debug("KG persistence skipped for evaluation")

    def summary(self) -> dict[str, Any]:
        """Return a summary of evaluation monitoring state."""
        return {
            "total_evaluations": len(self._history),
            "dimensions_tracked": list(self._dimension_history.keys()),
            "recent_composite": (
                self._history[-1].composite_score if self._history else None
            ),
            "composite_trend": self.get_composite_trend(10),
            "active_alerts": len(self.check_alerts()),
        }
