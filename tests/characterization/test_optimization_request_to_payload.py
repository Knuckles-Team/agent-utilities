"""Characterization tests for OptimizationRequest.to_payload (CCN 32),
agent_utilities/harness/optimization_backend.py.

tests/unit/test_optimization_backend.py already covers the broad contract
(opaque references, all modality families, budget bounds, all optimizer
families, avatar success/failure). This file targets the finer per-row
branches inside to_payload that a decomposition could silently break: input
dedup/fallback, missing-response/failed-row field omission, and
feedback/trace-ref triggers -- plus re-pins the raise ordering.

Pins OBSERVED behaviour of the unmodified function before a
complexity-reduction refactor. Must stay byte-identical across the refactor
commit.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.optimization_backend import (
    OptimizationCapabilityUnavailable,
    OptimizationDataUnavailable,
    OptimizationRequest,
)


def _request(data, *, optimizer="bootstrap_few_shot", target="skill", objective="obj"):
    return OptimizationRequest(
        target=target, objective=objective, data=data, optimizer=optimizer
    )


def test_no_rows_raises_data_unavailable():
    with pytest.raises(OptimizationDataUnavailable):
        _request({}).to_payload()


def test_unknown_optimizer_raises_capability_unavailable():
    with pytest.raises(OptimizationCapabilityUnavailable):
        _request(
            {"examples": [{"task": "t", "response": "r"}]}, optimizer="not-a-real-one"
        ).to_payload()


def test_avatar_without_tool_refs_raises_value_error():
    with pytest.raises(ValueError, match="governed tool reference"):
        _request(
            {"examples": [{"task": "t", "response": "r"}]}, optimizer="avatar"
        ).to_payload()


def test_avatar_with_tools_but_no_contrastive_pair_raises_value_error():
    data = {
        "tool_refs": ["search"],
        "examples": [
            {"task": "t", "response": "r", "source": "kg_trace", "success": True},
        ],
    }
    with pytest.raises(ValueError, match="positive and negative governed traces"):
        _request(data, optimizer="avatar").to_payload()


def test_avatar_with_valid_contrastive_pair_succeeds_with_react_module():
    data = {
        "tool_refs": ["search"],
        "examples": [
            {
                "task": "t1",
                "response": "r1",
                "source": "kg_trace",
                "success": True,
            },
            {
                "task": "t2",
                "response": "r2",
                "source": "kg_trace",
                "success": False,
                "failure_reason": "bad output",
            },
        ],
    }
    payload = _request(data, optimizer="avatar").to_payload()
    assert payload["program"]["module"] == "react"
    assert payload["budget"]["max_model_calls"] == 8


def test_default_optimizer_uses_predict_module_and_zero_model_calls():
    payload = _request({"examples": [{"task": "t", "response": "r"}]}).to_payload()
    assert payload["program"]["module"] == "predict"
    assert payload["budget"]["max_model_calls"] == 0
    assert payload["budget"]["max_training_steps"] == 0


def test_context_and_task_produce_distinct_input_refs_even_with_equal_values():
    # OBSERVED: the dedup check (`if reference not in inputs`) can never fire in
    # this loop because the opaque reference is derived from
    # _canonical({field_name: value}), and field_name ("context" vs "task")
    # always differs -- so equal context/task text still yields two distinct
    # references. Pinning the actual behaviour, not the apparent intent.
    payload = _request(
        {"examples": [{"context": "same", "task": "same", "response": "r"}]}
    ).to_payload()
    example = payload["corpus"]["examples"][0]
    assert len(example["input_refs"]) == 2
    assert len(set(example["input_refs"])) == 2


def test_missing_context_and_task_falls_back_to_row_seed_reference():
    payload = _request({"examples": [{"response": "r"}]}).to_payload()
    example = payload["corpus"]["examples"][0]
    assert len(example["input_refs"]) == 1


def test_missing_response_gives_none_expected_output_ref():
    payload = _request({"examples": [{"task": "t"}]}).to_payload()
    example = payload["corpus"]["examples"][0]
    assert example["expected_output_ref"] is None


def test_failed_row_gives_none_observed_output_ref_and_failure_outcome():
    payload = _request(
        {"examples": [{"task": "t", "response": "r", "success": False}]}
    ).to_payload()
    example = payload["corpus"]["examples"][0]
    assert example["observed_output_ref"] is None
    assert example["outcome"] == "failure"


def test_failure_reason_sets_feedback_ref():
    with_reason = _request(
        {
            "examples": [
                {"task": "t", "response": "r", "success": False, "failure_reason": "x"}
            ]
        }
    ).to_payload()
    without_reason = _request(
        {"examples": [{"task": "t", "response": "r", "success": False}]}
    ).to_payload()
    assert with_reason["corpus"]["examples"][0]["feedback_ref"] is not None
    assert without_reason["corpus"]["examples"][0]["feedback_ref"] is None


def test_kg_trace_source_sets_trace_ref_even_without_explicit_trace_ref():
    payload = _request(
        {"examples": [{"task": "t", "response": "r", "source": "kg_trace"}]}
    ).to_payload()
    example = payload["corpus"]["examples"][0]
    assert example["trace_ref"] is not None


def test_no_trace_source_or_ref_gives_none_trace_ref():
    payload = _request({"examples": [{"task": "t", "response": "r"}]}).to_payload()
    example = payload["corpus"]["examples"][0]
    assert example["trace_ref"] is None


def test_baseline_aggregate_score_is_mean_of_example_scores():
    payload = _request(
        {
            "examples": [
                {"task": "t1", "response": "r1", "score": 1.0},
                {"task": "t2", "response": "r2", "score": 0.0},
            ]
        }
    ).to_payload()
    assert payload["baseline"]["aggregate_score"] == pytest.approx(0.5)


def test_modality_scores_are_averaged_per_modality():
    payload = _request(
        {
            "examples": [
                {"task": "t1", "response": "r1", "score": 1.0, "modality": "text"},
                {"task": "t2", "response": "r2", "score": 0.5, "modality": "text"},
            ]
        }
    ).to_payload()
    assert payload["baseline"]["modality_scores"]["text"] == pytest.approx(0.75)
