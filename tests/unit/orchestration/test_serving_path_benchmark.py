"""Tests for AU-INTEGRATION-R014.1's serving-path benchmark-comparison model.

Covers field validation on the typed row and the refusal case the
requirement names directly: a candidate authorized by a single shared token
is never integrated, even when it beats the existing path on throughput and
latency.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_utilities.orchestration.serving_path_benchmark import (
    ServingPathBenchmarkRow,
    evaluate_serving_path_candidate,
)


def _row(**overrides: object) -> ServingPathBenchmarkRow:
    base: dict[str, object] = {
        "name": "existing_typed_query_path",
        "authorization_mode": "per_tenant_purpose_identity",
        "tenant_id": "tenant-a",
        "purpose": "retrieval",
        "throughput_rps": 100.0,
        "latency_p50_ms": 10.0,
        "latency_p95_ms": 25.0,
    }
    base.update(overrides)
    return ServingPathBenchmarkRow(**base)


def test_valid_row_round_trips() -> None:
    row = _row()
    assert row.authorization_mode == "per_tenant_purpose_identity"


def test_latency_p95_below_p50_is_rejected() -> None:
    with pytest.raises(ValueError):
        _row(latency_p50_ms=25.0, latency_p95_ms=10.0)


def test_unknown_authorization_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _row(authorization_mode="anonymous")


def test_integration_accepts_gain_with_verified_identity() -> None:
    existing = _row(
        name="existing_typed_query_path", throughput_rps=100.0, latency_p95_ms=25.0
    )
    candidate = _row(
        name="kv_reuse_candidate", throughput_rps=150.0, latency_p95_ms=15.0
    )

    decision = evaluate_serving_path_candidate(candidate, existing)

    assert decision.integrate is True


def test_integration_refuses_shared_token_candidate() -> None:
    """Refusal case: a shared-token candidate never qualifies, gain or not."""
    existing = _row(
        name="existing_typed_query_path", throughput_rps=100.0, latency_p95_ms=25.0
    )
    candidate = _row(
        name="gpu_fairness_candidate",
        authorization_mode="shared_token",
        throughput_rps=500.0,
        latency_p50_ms=2.0,
        latency_p95_ms=5.0,
    )

    decision = evaluate_serving_path_candidate(candidate, existing)

    assert decision.integrate is False
    assert "shared token" in decision.reason


def test_integration_refuses_without_a_gain() -> None:
    existing = _row(
        name="existing_typed_query_path", throughput_rps=100.0, latency_p95_ms=25.0
    )
    candidate = _row(
        name="predicate_grouping_candidate", throughput_rps=100.0, latency_p95_ms=25.0
    )

    decision = evaluate_serving_path_candidate(candidate, existing)

    assert decision.integrate is False
    assert "no throughput or latency gain" in decision.reason
