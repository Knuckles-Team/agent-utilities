#!/usr/bin/python
"""Evaluation & Monitoring Framework — CONCEPT:AHE-3.1.

Multi-dimensional evaluation with LLM-as-Judge rubrics, trend monitoring,
quality alerting, and human calibration support.

Design-pattern source: Chapter 19 — Evaluation and Monitoring.

OWL: :EvaluationRecord rdfs:subClassOf :Observation
See docs/design-patterns-alignment.md §CONCEPT:AHE-3.1.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from enum import Enum
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


# ---------------------------------------------------------------------------
# EvalRunner — Multi-Strategy Scoring (CONCEPT:AHE-3.12)
# ---------------------------------------------------------------------------
# Ported from MATE's eval_runner.py. Provides three concrete scoring
# strategies that execute automatically against test cases:
#   1. Exact Match — normalized string comparison
#   2. Semantic Similarity — cosine similarity via embedding model
#   3. LLM-as-Judge — structured JSON prompt for consistent scoring
#
# OWL synergy: Results persist as EvaluationRecordNode. OWL reasoning
# can infer `degradedPerformance` across sessions — a capability MATE
# lacks because it has no knowledge graph.
# ---------------------------------------------------------------------------


class EvalStrategy(str, Enum):
    """Evaluation strategy for scoring agent responses.

    CONCEPT:AHE-3.12 — Multi-Strategy Evaluation

    Ported from MATE's eval_runner.py pattern with three concrete
    strategies plus a composite mode that combines all three.
    """

    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    LLM_JUDGE = "llm_judge"
    COMPOSITE = "composite"


class TestCase(BaseModel):
    """A single evaluation test case with expected output.

    CONCEPT:AHE-3.12 — Multi-Strategy Evaluation

    Mirrors MATE's test case schema but adds KG provenance fields
    for integration with the agent-utilities knowledge graph.
    """

    id: str = ""
    query: str
    expected_output: str
    agent_name: str = ""
    strategy: EvalStrategy = EvalStrategy.COMPOSITE
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Result of evaluating a single test case.

    CONCEPT:AHE-3.12 — Multi-Strategy Evaluation

    Contains per-strategy scores and a final composite score.
    """

    test_case_id: str = ""
    query: str = ""
    expected_output: str = ""
    actual_output: str = ""
    strategy: EvalStrategy = EvalStrategy.COMPOSITE
    exact_match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_judge_score: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_judge_reasoning: str = ""
    final_score: float = Field(default=0.0, ge=0.0, le=1.0)
    passed: bool = False
    timestamp: float = Field(default_factory=time.time)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRunner:
    """Multi-strategy evaluation runner.

    CONCEPT:AHE-3.12 — Multi-Strategy Evaluation

    Ported from MATE's ``EvalRunner`` with three scoring strategies:

    1. **Exact Match** — Normalized string comparison after lowercasing,
       stripping whitespace, and removing punctuation.

    2. **Semantic Similarity** — Cosine similarity between embeddings
       of expected and actual outputs.

    3. **LLM-as-Judge** — Structured JSON prompt that forces a
       consistent single-line output format with score and reasoning.
       Ported from MATE's ``llm_judge_eval`` prompt pattern.

    The runner integrates with the existing ``EvaluationMonitor`` for
    trend tracking and alerting, and persists results as
    ``EvaluationRecordNode`` in the Knowledge Graph.

    Parameters
    ----------
    monitor : EvaluationMonitor | None
        If provided, each eval result is also recorded in the monitor
        for trend tracking and alerting.
    pass_threshold : float
        Score at or above which a test case is considered passing.
    exact_weight : float
        Weight of exact match in composite scoring.
    semantic_weight : float
        Weight of semantic similarity in composite scoring.
    judge_weight : float
        Weight of LLM-as-Judge in composite scoring.
    """

    # Default LLM-as-Judge prompt — ported from MATE's structured judge
    # prompt that forces consistent single-line JSON output.
    LLM_JUDGE_PROMPT = (
        "You are an expert evaluator. Compare the actual output against "
        "the expected output for the given query.\n\n"
        "Query: {query}\n"
        "Expected Output: {expected}\n"
        "Actual Output: {actual}\n\n"
        "Rate the actual output on a scale of 0.0 to 1.0 where:\n"
        "- 1.0 = perfect match in meaning and completeness\n"
        "- 0.7 = mostly correct with minor omissions\n"
        "- 0.4 = partially correct but significant gaps\n"
        "- 0.0 = completely wrong or irrelevant\n\n"
        'Respond with ONLY a single JSON line: {{"score": <float>, "reasoning": "<brief explanation>"}}'
    )

    def __init__(
        self,
        monitor: EvaluationMonitor | None = None,
        pass_threshold: float = 0.7,
        exact_weight: float = 0.2,
        semantic_weight: float = 0.3,
        judge_weight: float = 0.5,
    ) -> None:
        self._monitor = monitor
        self._pass_threshold = pass_threshold
        self._exact_weight = exact_weight
        self._semantic_weight = semantic_weight
        self._judge_weight = judge_weight
        self._results: list[EvalResult] = []

    @property
    def results(self) -> list[EvalResult]:
        """All evaluation results collected so far."""
        return list(self._results)

    def run_eval(
        self,
        test_case: TestCase,
        actual_output: str,
        strategy: EvalStrategy | None = None,
    ) -> EvalResult:
        """Evaluate a single test case against actual output.

        Parameters
        ----------
        test_case : TestCase
            The test case with query and expected output.
        actual_output : str
            The agent's actual response.
        strategy : EvalStrategy | None
            Override the test case's default strategy.

        Returns
        -------
        EvalResult
            The evaluation result with per-strategy scores.
        """
        start = time.time()
        effective_strategy = strategy or test_case.strategy

        result = EvalResult(
            test_case_id=test_case.id,
            query=test_case.query,
            expected_output=test_case.expected_output,
            actual_output=actual_output,
            strategy=effective_strategy,
        )

        # Always compute exact match (it's free)
        result.exact_match_score = self._exact_match_eval(
            test_case.expected_output, actual_output
        )

        if effective_strategy == EvalStrategy.EXACT_MATCH:
            result.final_score = result.exact_match_score

        elif effective_strategy == EvalStrategy.SEMANTIC_SIMILARITY:
            result.semantic_similarity_score = self._semantic_similarity_eval(
                test_case.expected_output, actual_output
            )
            result.final_score = result.semantic_similarity_score

        elif effective_strategy == EvalStrategy.LLM_JUDGE:
            score, reasoning = self._llm_judge_eval(
                test_case.query, test_case.expected_output, actual_output
            )
            result.llm_judge_score = score
            result.llm_judge_reasoning = reasoning
            result.final_score = result.llm_judge_score

        elif effective_strategy == EvalStrategy.COMPOSITE:
            # Compute all strategies
            result.semantic_similarity_score = self._semantic_similarity_eval(
                test_case.expected_output, actual_output
            )
            score, reasoning = self._llm_judge_eval(
                test_case.query, test_case.expected_output, actual_output
            )
            result.llm_judge_score = score
            result.llm_judge_reasoning = reasoning
            # Weighted composite
            result.final_score = (
                self._exact_weight * result.exact_match_score
                + self._semantic_weight * result.semantic_similarity_score
                + self._judge_weight * result.llm_judge_score
            )

        result.passed = result.final_score >= self._pass_threshold
        result.duration_ms = (time.time() - start) * 1000
        result.timestamp = time.time()

        self._results.append(result)

        # Feed into EvaluationMonitor for trend tracking
        if self._monitor:
            self._monitor.evaluate(
                query=test_case.query,
                response=actual_output,
                dimension_scores={
                    "correctness": result.final_score,
                    "completeness": result.semantic_similarity_score,
                    "relevance": result.exact_match_score,
                    "safety": 1.0,  # not evaluated by EvalRunner
                },
                evaluator="eval_runner",
                session_id=test_case.id,
            )

        return result

    def run_batch(
        self,
        test_cases: list[TestCase],
        actual_outputs: list[str],
        strategy: EvalStrategy | None = None,
    ) -> list[EvalResult]:
        """Run evaluation on a batch of test cases.

        Parameters
        ----------
        test_cases : list[TestCase]
            List of test cases.
        actual_outputs : list[str]
            Corresponding actual outputs (must match length).
        strategy : EvalStrategy | None
            Override strategy for all cases.

        Returns
        -------
        list[EvalResult]
            Evaluation results for each test case.
        """
        if len(test_cases) != len(actual_outputs):
            raise ValueError(
                f"Mismatch: {len(test_cases)} test cases vs "
                f"{len(actual_outputs)} actual outputs"
            )
        return [
            self.run_eval(tc, output, strategy)
            for tc, output in zip(test_cases, actual_outputs)
        ]

    def summary(self) -> dict[str, Any]:
        """Return aggregate statistics for all evaluations run so far."""
        if not self._results:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
            }
        passed = sum(1 for r in self._results if r.passed)
        return {
            "total": len(self._results),
            "passed": passed,
            "failed": len(self._results) - passed,
            "pass_rate": passed / len(self._results),
            "avg_score": sum(r.final_score for r in self._results) / len(self._results),
            "avg_duration_ms": sum(r.duration_ms for r in self._results)
            / len(self._results),
        }

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for exact matching."""
        import re as _re

        text = text.lower().strip()
        text = _re.sub(r"[^\w\s]", "", text)
        text = _re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _exact_match_eval(expected: str, actual: str) -> float:
        """Exact match with normalization.

        Returns 1.0 for exact match, or a partial score based on
        token overlap (Jaccard similarity) for near-matches.
        """
        norm_expected = EvalRunner._normalize(expected)
        norm_actual = EvalRunner._normalize(actual)

        if norm_expected == norm_actual:
            return 1.0

        # Jaccard similarity for partial credit
        expected_tokens = set(norm_expected.split())
        actual_tokens = set(norm_actual.split())
        if not expected_tokens and not actual_tokens:
            return 1.0
        if not expected_tokens or not actual_tokens:
            return 0.0
        intersection = expected_tokens & actual_tokens
        union = expected_tokens | actual_tokens
        return len(intersection) / len(union)

    @staticmethod
    def _semantic_similarity_eval(expected: str, actual: str) -> float:
        """Semantic similarity via cosine distance of embeddings.

        Falls back to token overlap if no embedding model is available.
        """
        try:
            from agent_utilities.core.embedding_utilities import (
                create_embedding_model,
            )

            model = create_embedding_model()
            if model is not None:
                expected_emb = model.embed(expected)
                actual_emb = model.embed(actual)
                # Cosine similarity
                dot = sum(a * b for a, b in zip(expected_emb, actual_emb))
                norm_e = sum(a * a for a in expected_emb) ** 0.5
                norm_a = sum(a * a for a in actual_emb) ** 0.5
                if norm_e > 0 and norm_a > 0:
                    return max(0.0, min(1.0, dot / (norm_e * norm_a)))
        except Exception:
            pass

        # Fallback: token overlap (same as exact match partial)
        return EvalRunner._exact_match_eval(expected, actual)

    @staticmethod
    def _llm_judge_eval(query: str, expected: str, actual: str) -> tuple[float, str]:
        """LLM-as-Judge evaluation with structured JSON output.

        Ported from MATE's ``llm_judge_eval`` — uses a structured prompt
        that forces a single-line JSON response with score and reasoning.

        Falls back to semantic similarity if no LLM is available.
        """
        try:
            import json as _json

            from agent_utilities.core.model_factory import create_model

            model = create_model()
            prompt = EvalRunner.LLM_JUDGE_PROMPT.format(
                query=query, expected=expected, actual=actual
            )
            # Use synchronous run_sync for compatibility
            result = model.run_sync(prompt)
            response_text = result.output if hasattr(result, "output") else str(result)

            # Parse JSON from response
            # Handle potential markdown code blocks
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = _json.loads(clean)
            score = float(parsed.get("score", 0.0))
            reasoning = str(parsed.get("reasoning", ""))
            return (max(0.0, min(1.0, score)), reasoning)

        except Exception as exc:
            logger.debug("LLM judge fallback (no model available): %s", exc)
            # Fallback: use semantic similarity as proxy
            fallback_score = EvalRunner._semantic_similarity_eval(expected, actual)
            return (
                fallback_score,
                f"LLM judge unavailable, using semantic fallback: {exc}",
            )
