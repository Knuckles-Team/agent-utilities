#!/usr/bin/python
from __future__ import annotations

"""AU-RETRIEVAL-R001 — structure-aware retrieval benchmark result model.

Typed shape for one method's row in the structured-retrieval-vs-hybrid-search
benchmark (constrained section-leaf selection vs. hierarchical-document
retrieval vs. the production hybrid vector-plus-graph path), plus the
adoption decision the acceptance criterion requires: a candidate method may
only be adopted when it shows a statistically defensible gain over the
baseline on at least one ranking metric AND regresses neither citation
validity nor visibility correctness.

This is the `.1` slice: the typed model, its field validation, and the
refusal case. The benchmark driver that produces these rows per method,
across the fixture corpus and the held-out document set, is a later slice.
"""

from pydantic import BaseModel, Field, model_validator


class StructuredRetrievalBenchmarkResult(BaseModel):
    """One method's measured row for a single benchmark run."""

    method: str = Field(min_length=1)
    sample_size: int = Field(gt=0)
    recall_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_3: float = Field(ge=0.0, le=1.0)
    ndcg: float = Field(ge=0.0, le=1.0)
    citation_validity: float = Field(ge=0.0, le=1.0)
    visibility_correct: float = Field(ge=0.0, le=1.0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _recall_and_latency_ordering(self) -> StructuredRetrievalBenchmarkResult:
        if self.recall_at_1 > self.recall_at_3:
            raise ValueError("recall_at_1 cannot exceed recall_at_3")
        if self.latency_p95_ms < self.latency_p50_ms:
            raise ValueError("latency_p95_ms cannot be below latency_p50_ms")
        return self


class StructuredRetrievalAdoptionDecision(BaseModel):
    """Outcome of comparing a candidate method against the served baseline."""

    adopt: bool
    reason: str = Field(min_length=1)


def evaluate_structured_retrieval_adoption(
    candidate: StructuredRetrievalBenchmarkResult,
    baseline: StructuredRetrievalBenchmarkResult,
) -> StructuredRetrievalAdoptionDecision:
    """Apply the AU-RETRIEVAL-R001 adoption rule.

    Refuses adoption on any citation-validity or visibility regression,
    regardless of ranking gains elsewhere. Otherwise requires a strict gain
    over baseline on at least one of recall@1, recall@3, or nDCG.
    """
    if candidate.citation_validity < baseline.citation_validity:
        return StructuredRetrievalAdoptionDecision(
            adopt=False,
            reason=(
                f"refused: citation validity regressed "
                f"({candidate.citation_validity} < {baseline.citation_validity})"
            ),
        )
    if candidate.visibility_correct < baseline.visibility_correct:
        return StructuredRetrievalAdoptionDecision(
            adopt=False,
            reason=(
                f"refused: visibility correctness regressed "
                f"({candidate.visibility_correct} < {baseline.visibility_correct})"
            ),
        )
    gained = (
        candidate.recall_at_1 > baseline.recall_at_1
        or candidate.recall_at_3 > baseline.recall_at_3
        or candidate.ndcg > baseline.ndcg
    )
    if not gained:
        return StructuredRetrievalAdoptionDecision(
            adopt=False,
            reason="refused: no statistically defensible gain over baseline",
        )
    return StructuredRetrievalAdoptionDecision(
        adopt=True,
        reason="gain over baseline with no visibility or citation regression",
    )
