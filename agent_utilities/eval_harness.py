#!/usr/bin/python
"""Evaluation Harness Module.

Provides a pluggable evaluation framework for scoring agent outputs.
Scorers implement the :class:`EvalScorer` protocol and are aggregated
by :class:`EvalHarness` which persists results to the Knowledge Graph
as ``EvalNode`` instances.

Concept: eval-tracing
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """Result of a single evaluation scorer."""

    score: float = Field(ge=0.0, le=1.0, description="Normalised score 0-1")
    passed: bool = Field(description="Whether the evaluation passed")
    evaluator: str = Field(description="Name of the scorer that produced this result")
    metrics: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


class AggregatedEvalResult(BaseModel):
    """Aggregated result from running multiple scorers."""

    overall_score: float = Field(ge=0.0, le=1.0)
    all_passed: bool
    results: list[EvalResult]
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


# ---------------------------------------------------------------------------
# Scorer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EvalScorer(Protocol):
    """Protocol for evaluation scorers."""

    name: str

    def score(
        self, input_text: str, output_text: str, context: dict[str, Any] | None = None
    ) -> EvalResult:
        """Score an input/output pair and return an EvalResult."""
        ...


# ---------------------------------------------------------------------------
# Built-in scorers
# ---------------------------------------------------------------------------


@dataclass
class LengthScorer:
    """Rejects outputs shorter than a minimum length.

    Concept: eval-tracing
    """

    name: str = "length"
    min_length: int = 10
    max_length: int = 100_000

    def score(
        self, input_text: str, output_text: str, context: dict[str, Any] | None = None
    ) -> EvalResult:
        length = len(output_text)
        in_range = self.min_length <= length <= self.max_length
        return EvalResult(
            score=1.0 if in_range else 0.0,
            passed=in_range,
            evaluator=self.name,
            metrics={"length": length, "min": self.min_length, "max": self.max_length},
            reason=""
            if in_range
            else f"Output length {length} outside [{self.min_length}, {self.max_length}]",
        )


@dataclass
class RegexScorer:
    """Validates that output matches an expected regex pattern.

    Concept: eval-tracing
    """

    name: str = "regex"
    pattern: str = ".*"
    must_match: bool = True

    def score(
        self, input_text: str, output_text: str, context: dict[str, Any] | None = None
    ) -> EvalResult:
        matched = bool(re.search(self.pattern, output_text, re.DOTALL))
        passed = matched if self.must_match else not matched
        return EvalResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            evaluator=self.name,
            metrics={"pattern": self.pattern, "matched": matched},
            reason=""
            if passed
            else f"Pattern {'not found' if self.must_match else 'unexpectedly found'}: {self.pattern}",
        )


@dataclass
class JsonSchemaScorer:
    """Validates that output conforms to a JSON schema.

    Concept: eval-tracing
    """

    name: str = "json_schema"
    required_keys: list[str] = field(default_factory=list)

    def score(
        self, input_text: str, output_text: str, context: dict[str, Any] | None = None
    ) -> EvalResult:
        try:
            data = json.loads(output_text)
        except (json.JSONDecodeError, TypeError):
            return EvalResult(
                score=0.0,
                passed=False,
                evaluator=self.name,
                metrics={"valid_json": False},
                reason="Output is not valid JSON",
            )

        if not isinstance(data, dict):
            return EvalResult(
                score=0.5,
                passed=not self.required_keys,
                evaluator=self.name,
                metrics={"valid_json": True, "is_object": False},
                reason="Output is valid JSON but not an object",
            )

        missing = [k for k in self.required_keys if k not in data]
        passed = len(missing) == 0
        score = 1.0 - (len(missing) / max(len(self.required_keys), 1))

        return EvalResult(
            score=max(0.0, score),
            passed=passed,
            evaluator=self.name,
            metrics={"valid_json": True, "missing_keys": missing},
            reason="" if passed else f"Missing required keys: {missing}",
        )


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------


@dataclass
class EvalHarness:
    """Runs a list of scorers and aggregates results.

    Optionally persists results to the Knowledge Graph as ``EvalNode`` instances.

    Concept: eval-tracing
    """

    scorers: list[EvalScorer] = field(default_factory=list)

    def register(self, scorer: EvalScorer) -> None:
        """Register a scorer to be run on evaluate()."""
        self.scorers.append(scorer)

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        context: dict[str, Any] | None = None,
    ) -> AggregatedEvalResult:
        """Run all registered scorers and return aggregated results."""
        results: list[EvalResult] = []
        for scorer in self.scorers:
            try:
                result = scorer.score(input_text, output_text, context)
                results.append(result)
            except Exception as exc:
                logger.warning("Scorer %s failed: %s", scorer.name, exc)
                results.append(
                    EvalResult(
                        score=0.0,
                        passed=False,
                        evaluator=scorer.name,
                        reason=f"Scorer error: {exc}",
                    )
                )

        overall = sum(r.score for r in results) / max(len(results), 1)
        all_passed = all(r.passed for r in results)

        return AggregatedEvalResult(
            overall_score=overall,
            all_passed=all_passed,
            results=results,
        )

    def persist_to_graph(
        self, graph: Any, aggregated: AggregatedEvalResult, run_id: str = ""
    ) -> str:
        """Persist evaluation results as an EvalNode in the KG.

        Args:
            graph: A NetworkX graph instance.
            aggregated: The aggregated evaluation result.
            run_id: Optional run identifier for traceability.

        Returns:
            The node ID of the created EvalNode.
        """
        import hashlib

        node_id = f"eval_{hashlib.sha256(f'{run_id}:{aggregated.timestamp}'.encode()).hexdigest()[:12]}"
        graph.add_node(
            node_id,
            type="eval",
            overall_score=aggregated.overall_score,
            all_passed=aggregated.all_passed,
            scorer_count=len(aggregated.results),
            results=[r.model_dump() for r in aggregated.results],
            run_id=run_id,
            timestamp=aggregated.timestamp,
            importance_score=0.5,
        )
        logger.info(
            "EvalNode %s: score=%.2f, passed=%s (%d scorers)",
            node_id,
            aggregated.overall_score,
            aggregated.all_passed,
            len(aggregated.results),
        )
        return node_id
