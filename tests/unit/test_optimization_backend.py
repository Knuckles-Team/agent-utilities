from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.harness.optimization_backend import (
    _MAX_NATIVE_DEMONSTRATIONS,
    _OPTIMIZER_EXECUTIONS,
    OptimizationRequest,
    _demonstration_budget,
    is_opaque_program_reference,
    opaque_program_reference,
    try_native_optimization,
)
from agent_utilities.knowledge_graph.core.graph_compute import (
    _PROGRAM_RESULT_SCHEMA,
    GraphComputeEngine,
)

_MODALITIES = [
    "text",
    "document",
    "image",
    "audio",
    "video",
    "graph",
    "table",
    "time_series",
    "vector",
    "spatial",
    "tensor",
    "code",
    "trace",
    "binary",
]
_OPTIMIZER_CASES = [
    ("labeled_few_shot", "native_kernel", 0, 0),
    ("bootstrap_few_shot", "native_kernel", 0, 0),
    ("bootstrap_few_shot_with_random_search", "native_kernel", 0, 0),
    ("avatar", "model_transport_plan", 8, 0),
    ("copro", "model_transport_plan", 8, 0),
    ("mipro_v2", "model_transport_plan", 8, 0),
    ("simba", "model_transport_plan", 8, 0),
    ("gepa", "model_transport_plan", 8, 0),
    ("better_together", "composite_plan", 8, 32),
    ("bootstrap_finetune", "trainer_plan", 0, 32),
    ("knn_few_shot", "graph_kernel_plan", 0, 0),
    ("ensemble", "native_kernel", 0, 0),
    ("infer_rules", "model_transport_plan", 8, 0),
]


class _NativeEngine:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.request: dict[str, Any] | None = None

    def optimize_program(self, request: dict[str, Any]) -> Any:
        self.request = request
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _request() -> OptimizationRequest:
    return OptimizationRequest(
        target="skill",
        objective="skill invocation reliability",
        data={"examples": [{"task": "synthetic-task", "response": "route-a"}]},
    )


def test_native_request_is_opaque_and_uses_exact_program_contract() -> None:
    payload = _request().to_payload()

    assert set(payload) == {
        "schema_version",
        "request_ref",
        "program",
        "corpus",
        "optimizer",
        "budget",
        "promotion",
        "baseline",
        "optimizer_artifacts",
        "candidate_evaluations",
    }
    assert payload["optimizer"] == "bootstrap_few_shot"
    assert payload["budget"]["max_model_calls"] == 0
    assert payload["corpus"]["privacy"] == {
        "scanner_ref": payload["corpus"]["privacy"]["scanner_ref"],
        "policy_version_ref": payload["corpus"]["privacy"]["policy_version_ref"],
        "raw_pii_persisted": False,
        "local_identifiers_persisted": False,
    }
    assert "synthetic-task" not in repr(payload)
    assert "route-a" not in repr(payload)
    address = payload["corpus"]["examples"][0]["evidence"][0]["locus"]["address"]
    assert address == {"kind": "character_range", "start": 0, "end": 1}


def test_native_request_preserves_governed_example_reference() -> None:
    example_ref = "eg:example:" + "b" * 64
    payload = OptimizationRequest(
        target="skill",
        objective="skill invocation reliability",
        data={
            "examples": [
                {
                    "example_ref": example_ref,
                    "task": "ephemeral task",
                    "response": "ephemeral response",
                }
            ]
        },
    ).to_payload()

    assert payload["corpus"]["examples"][0]["example_ref"] == example_ref
    assert "ephemeral task" not in repr(payload)
    assert "ephemeral response" not in repr(payload)


def test_native_request_projects_all_program_modalities() -> None:
    payload = OptimizationRequest(
        target="cross-modal-program",
        objective="modality-complete evaluation",
        data={
            "examples": [
                {
                    "task": f"ephemeral-{index}",
                    "response": f"ephemeral-result-{index}",
                    "modality": modality,
                }
                for index, modality in enumerate(_MODALITIES)
            ]
        },
    ).to_payload()

    examples = payload["corpus"]["examples"]
    assert {item["evidence"][0]["modality"] for item in examples} == set(_MODALITIES)
    assert set(payload["baseline"]["modality_scores"]) == set(_MODALITIES)
    assert payload["budget"]["max_demonstrations"] == len(_MODALITIES)
    assert {item["evidence"][0]["locus"]["address"]["kind"] for item in examples} == {
        "character_range",
        "image_region",
        "audio_range",
        "frame_range",
        "row_version",
        "table_cell_range",
        "metric_window",
        "point",
        "code_symbol",
        "trace_span",
    }
    assert "ephemeral-" not in repr(payload)


def test_modality_cover_budget_is_bounded_and_fails_closed() -> None:
    assert _MAX_NATIVE_DEMONSTRATIONS == 256
    assert _demonstration_budget(1, 14) == 1
    assert _demonstration_budget(14, 14) == 14
    with pytest.raises(ValueError, match="native demonstration bound"):
        _demonstration_budget(
            _MAX_NATIVE_DEMONSTRATIONS + 1,
            _MAX_NATIVE_DEMONSTRATIONS + 1,
        )


@pytest.mark.parametrize(
    ("optimizer", "execution", "model_calls", "training_steps"),
    _OPTIMIZER_CASES,
)
def test_native_request_exposes_every_optimizer_family(
    optimizer: str, execution: str, model_calls: int, training_steps: int
) -> None:
    examples = [
        {
            "task": f"ephemeral-{index}",
            "response": f"ephemeral-result-{index}",
            "modality": modality,
            **(
                {
                    "success": index % 2 == 0,
                    "source": "kg_trace",
                    "failure_reason": "bounded-negative" if index % 2 else "",
                }
                if optimizer == "avatar"
                else {}
            ),
        }
        for index, modality in enumerate(_MODALITIES)
    ]
    data: dict[str, Any] = {"examples": examples}
    if optimizer == "avatar":
        data["tool_refs"] = ["governed-tool"]
    request = OptimizationRequest(
        target="program",
        objective="governed improvement",
        optimizer=optimizer,
        data=data,
    ).to_payload()

    assert request["optimizer"] == optimizer
    assert request["budget"]["max_demonstrations"] == len(_MODALITIES)
    assert request["budget"]["max_model_calls"] == model_calls
    assert request["budget"]["max_training_steps"] == training_steps
    assert set(request["baseline"]["modality_scores"]) == set(_MODALITIES)
    assert _OPTIMIZER_EXECUTIONS[optimizer] == execution
    assert request["promotion"] == {
        "min_score_delta": 0.0,
        "max_modality_regression": 0.0,
        "require_all_observed_modalities": True,
        "min_evidence_refs": 1,
    }
    assert "ephemeral-" not in repr(request)


def test_avatar_projects_only_governed_tool_and_contrastive_trace_references() -> None:
    payload = OptimizationRequest(
        target="tool-agent",
        objective="improve tool selection",
        optimizer="avatar",
        data={
            "tool_refs": ["private-tool-description"],
            "examples": [
                {
                    "task": "ephemeral-positive",
                    "response": "ephemeral-positive-output",
                    "success": True,
                    "source": "kg_trace",
                },
                {
                    "task": "ephemeral-negative",
                    "response": "ephemeral-negative-output",
                    "success": False,
                    "source": "kg_trace",
                    "failure_reason": "ephemeral-failure",
                },
            ],
        },
    ).to_payload()

    assert payload["program"]["module"] == "react"
    assert len(payload["program"]["tool_refs"]) == 1
    assert is_opaque_program_reference(
        payload["program"]["tool_refs"][0], namespace="tool"
    )
    assert {example["outcome"] for example in payload["corpus"]["examples"]} == {
        "success",
        "failure",
    }
    assert all(
        example["trace_ref"] is not None
        for example in payload["corpus"]["examples"]
    )
    assert "private-tool-description" not in repr(payload)
    assert "ephemeral-" not in repr(payload)


@pytest.mark.parametrize(
    "data",
    [
        {"examples": [{"task": "positive", "response": "output"}]},
        {
            "tool_refs": ["tool"],
            "examples": [
                {
                    "task": "positive",
                    "response": "output",
                    "source": "kg_trace",
                }
            ],
        },
    ],
)
def test_avatar_fails_closed_without_tools_or_contrastive_traces(
    data: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="Avatar optimization requires"):
        OptimizationRequest(
            target="tool-agent",
            objective="improve tool selection",
            optimizer="avatar",
            data=data,
        ).to_payload()


def test_opaque_reference_cannot_cross_namespaces() -> None:
    foreign = "eg:trace:" + "c" * 64

    example_ref = opaque_program_reference("example", foreign)

    assert example_ref != foreign
    assert is_opaque_program_reference(example_ref, namespace="example")
    assert not is_opaque_program_reference(foreign, namespace="example")


def test_native_invocation_is_the_only_backend() -> None:
    engine = _NativeEngine(
        {"status": "proposed", "result": {"rows": [{"kind": "program_candidate"}]}}
    )

    result = try_native_optimization(engine, _request())

    assert result.disposition == "completed"
    assert engine.request is not None
    assert try_native_optimization(object(), _request()).disposition == "unavailable"


def test_operational_graph_facade_projects_to_its_compute_authority() -> None:
    native = _NativeEngine({"status": "proposed", "result": {"rows": []}})

    facade = type("Facade", (), {"graph_compute": native})()

    result = try_native_optimization(facade, _request())

    assert result.disposition == "completed"
    assert native.request is not None


def test_native_failure_is_content_safe() -> None:
    result = try_native_optimization(
        _NativeEngine(RuntimeError("sensitive local diagnostic")), _request()
    )

    assert result.disposition == "error"
    assert result.error_code == "native_execution_failed"
    assert "sensitive" not in repr(result)


def test_removed_response_alias_fails_closed() -> None:
    result = try_native_optimization(
        _NativeEngine({"status": "completed", "result": {}}), _request()
    )

    assert result.disposition == "error"
    assert result.error_code == "native_invalid_response"


def _candidate_job(
    *,
    optimizer: str = "bootstrap_few_shot",
    execution: str = "native_kernel",
    modalities: list[str] | None = None,
) -> dict[str, Any]:
    reference = "eg:test:" + "a" * 64
    schema = [
        {"name": name, "logical_type": logical_type, "nullable": nullable}
        for name, logical_type, nullable in _PROGRAM_RESULT_SCHEMA
    ]
    row = {
        "id": reference,
        "kind": "program_candidate",
        "confidence": 1.0,
        "evidence_refs": [reference],
        "source_refs": [reference],
        "proof_ids": [],
        "contradiction_ids": [],
        "program_ref": reference,
        "optimizer": optimizer,
        "execution": execution,
        "candidate_role": "proposal",
        "demonstration_refs": [reference],
        "artifact_refs": [reference] if optimizer == "avatar" else [],
        "composition_refs": [],
        "instruction_ref": None if optimizer == "avatar" else reference,
        "tool_policy_ref": reference if optimizer == "avatar" else None,
        "model_profile_ref": None,
        "modalities": modalities or ["text"],
        "plan_ref": None,
        "plan_step_kinds": [],
        "plan_executors": [],
        "plan_input_refs": [],
        "plan_output_refs": [],
        "plan_depends_on": [],
        "max_operations": None,
        "selected": True,
    }
    return {
        "state": {"Succeeded": {"result_ref": reference, "checkpoint": {}}},
        "output": {"schema": schema, "rows": [row]},
    }


def test_native_result_validator_enforces_frozen_typed_schema() -> None:
    job = _candidate_job()

    result = GraphComputeEngine.program_optimization_result(job)

    assert result["rows"][0]["selected"] is True


@pytest.mark.parametrize(
    ("optimizer", "execution"),
    [
        (optimizer, execution)
        for optimizer, execution, _calls, _steps in _OPTIMIZER_CASES
    ],
)
def test_native_result_validator_accepts_all_family_modality_lineage(
    optimizer: str, execution: str
) -> None:
    job = _candidate_job(
        optimizer=optimizer,
        execution=execution,
        modalities=_MODALITIES,
    )

    result = GraphComputeEngine.program_optimization_result(job)

    assert result["rows"][0]["optimizer"] == optimizer
    assert result["rows"][0]["execution"] == execution
    assert result["rows"][0]["modalities"] == _MODALITIES


def test_native_result_validator_rejects_non_governed_references() -> None:
    job = _candidate_job()
    job["output"]["rows"][0]["source_refs"] = ["raw-local-source"]

    with pytest.raises(RuntimeError, match="reference list"):
        GraphComputeEngine.program_optimization_result(job)
