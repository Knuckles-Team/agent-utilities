#!/usr/bin/python
"""Tests for AU-RETRIEVAL-R001's structured-retrieval benchmark result model.

Covers field validation on the typed row and the refusal case in the
adoption decision: a candidate that regresses citation validity must never
be adopted, even when its ranking metrics improve on the baseline.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_utilities.knowledge_graph.retrieval.structured_retrieval_benchmark import (
    StructuredRetrievalBenchmarkResult,
    evaluate_structured_retrieval_adoption,
)


def _row(**overrides: object) -> StructuredRetrievalBenchmarkResult:
    base: dict[str, object] = {
        "method": "constrained_section_leaf",
        "sample_size": 50,
        "recall_at_1": 0.70,
        "recall_at_3": 0.85,
        "ndcg": 0.80,
        "citation_validity": 0.99,
        "visibility_correct": 1.0,
        "latency_p50_ms": 12.0,
        "latency_p95_ms": 40.0,
    }
    base.update(overrides)
    return StructuredRetrievalBenchmarkResult(**base)


@pytest.mark.spec(
    "AU-INTEGRATION-R014.1",
    "AU-INTEGRATION-R018.1",
    "AU-FREEZE-R003",
    "AU-RETRIEVAL-R001",
)
def test_valid_row_round_trips() -> None:
    row = _row()
    assert row.method == "constrained_section_leaf"
    assert row.recall_at_1 <= row.recall_at_3


@pytest.mark.spec(
    "AU-INTEGRATION-R014.1",
    "AU-INTEGRATION-R018.1",
    "AU-FREEZE-R003",
    "AU-RETRIEVAL-R001",
)
def test_recall_at_1_above_recall_at_3_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _row(recall_at_1=0.9, recall_at_3=0.5)


@pytest.mark.spec(
    "AU-INTEGRATION-R014.1",
    "AU-INTEGRATION-R018.1",
    "AU-FREEZE-R003",
    "AU-RETRIEVAL-R001",
)
def test_latency_p95_below_p50_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _row(latency_p50_ms=40.0, latency_p95_ms=10.0)


def test_metric_out_of_unit_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _row(ndcg=1.2)


def test_adoption_accepts_gain_with_no_regression() -> None:
    baseline = _row(
        method="hybrid_vector_graph", recall_at_1=0.60, recall_at_3=0.75, ndcg=0.70
    )
    candidate = _row(recall_at_1=0.70, recall_at_3=0.85, ndcg=0.80)

    decision = evaluate_structured_retrieval_adoption(candidate, baseline)

    assert decision.adopt is True


def test_adoption_refuses_on_citation_validity_regression() -> None:
    """Refusal case: ranking gains never override a citation-validity regression."""
    baseline = _row(
        method="hybrid_vector_graph",
        recall_at_1=0.50,
        recall_at_3=0.65,
        ndcg=0.60,
        citation_validity=0.99,
    )
    candidate = _row(
        recall_at_1=0.90,
        recall_at_3=0.95,
        ndcg=0.95,
        citation_validity=0.80,
    )

    decision = evaluate_structured_retrieval_adoption(candidate, baseline)

    assert decision.adopt is False
    assert "citation validity regressed" in decision.reason


def test_adoption_refuses_on_visibility_regression() -> None:
    baseline = _row(method="hybrid_vector_graph", visibility_correct=1.0)
    candidate = _row(
        recall_at_1=0.90, recall_at_3=0.95, ndcg=0.95, visibility_correct=0.90
    )

    decision = evaluate_structured_retrieval_adoption(candidate, baseline)

    assert decision.adopt is False
    assert "visibility" in decision.reason


def test_adoption_refuses_without_a_defensible_gain() -> None:
    baseline = _row(
        method="hybrid_vector_graph", recall_at_1=0.70, recall_at_3=0.85, ndcg=0.80
    )
    candidate = _row(recall_at_1=0.70, recall_at_3=0.85, ndcg=0.80)

    decision = evaluate_structured_retrieval_adoption(candidate, baseline)

    assert decision.adopt is False
    assert "no statistically defensible gain" in decision.reason
