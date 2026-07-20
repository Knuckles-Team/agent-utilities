"""Provider-free request boundary for native epistemic-graph optimization.

CONCEPT:AU-AHE.optimization.optimizable-target-registry — every target is
compiled through the engine-owned ``ProgramOptimize`` analytics job. Durable
contracts contain only opaque references, numeric evidence coordinates, bounded
schema identifiers, and policy metadata; raw content and local identifiers remain
ephemeral in the caller.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_utilities.security.persistence_privacy import persistence_reference

NativeDisposition = Literal["completed", "unavailable", "error"]
_OPAQUE_REF = re.compile(
    r"^eg:[a-z0-9_-]{1,32}(?::[a-z0-9_-]{1,32}){0,3}:[0-9a-f]{16,128}$"
)
_PROGRAM_MODALITIES = frozenset(
    {
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
    }
)
# Exact ``eg-program`` contract bound (`MAX_DEMONSTRATIONS` in the native
# optimizer).  This is a wire limit, not an operator tunable.
_MAX_NATIVE_DEMONSTRATIONS = 256
_DEFAULT_MAX_DEMONSTRATIONS = 8
_OPTIMIZER_EXECUTIONS = {
    "labeled_few_shot": "native_kernel",
    "bootstrap_few_shot": "native_kernel",
    "bootstrap_few_shot_with_random_search": "native_kernel",
    "avatar": "model_transport_plan",
    "copro": "model_transport_plan",
    "mipro_v2": "model_transport_plan",
    "simba": "model_transport_plan",
    "gepa": "model_transport_plan",
    "better_together": "composite_plan",
    "bootstrap_finetune": "trainer_plan",
    "knn_few_shot": "graph_kernel_plan",
    "ensemble": "native_kernel",
    "infer_rules": "model_transport_plan",
}


def _tool_references(data: Mapping[str, Any]) -> list[str]:
    """Return unique opaque tool references without retaining tool metadata."""

    supplied = data.get("tool_refs", data.get("tools", ()))
    if isinstance(supplied, str):
        values = [supplied]
    elif isinstance(supplied, Sequence) and not isinstance(supplied, bytes):
        values = list(supplied)
    else:
        raise ValueError("program tool references must be a string or sequence")
    references: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("program tool reference is invalid")
        reference = _opaque("tool", value.strip())
        if reference not in references:
            references.append(reference)
    return references


def _demonstration_budget(example_count: int, modality_count: int) -> int:
    """Return a bounded budget that can cover every observed modality.

    The native covering selector needs no more than one distinct example per
    observed modality and can never select more examples than the corpus owns.
    Refuse a future contract expansion that cannot fit the engine's fixed wire
    bound instead of emitting a request that can never satisfy promotion.
    """

    required = min(example_count, modality_count)
    if required > _MAX_NATIVE_DEMONSTRATIONS:
        raise ValueError(
            "program modality coverage exceeds the native demonstration bound"
        )
    return min(
        example_count,
        max(_DEFAULT_MAX_DEMONSTRATIONS, required),
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _opaque(namespace: str, value: Any) -> str:
    rendered = str(value or "")
    safe_namespace = re.sub(r"[^a-z0-9_-]+", "_", namespace.casefold()).strip("_")
    if not safe_namespace or len(safe_namespace) > 32:
        raise ValueError("program reference namespace is invalid")
    if _OPAQUE_REF.fullmatch(rendered) and rendered.startswith(f"eg:{safe_namespace}:"):
        return rendered
    reference = persistence_reference(
        safe_namespace, rendered or safe_namespace, namespace="eg-program"
    )
    digest = reference.rsplit("_", 1)[-1]
    return f"eg:{safe_namespace}:{digest}"


def opaque_program_reference(namespace: str, value: Any) -> str:
    """Return a validated, non-reversible reference for program metadata."""
    return _opaque(namespace, value)


def is_opaque_program_reference(value: Any, *, namespace: str | None = None) -> bool:
    """Return whether *value* is a bounded native program reference.

    ``namespace`` additionally pins the first reference segment. This is used on
    governed resolver boundaries so a valid reference from an unrelated namespace
    cannot be substituted for an example or proposal reference.
    """
    rendered = str(value or "")
    if not _OPAQUE_REF.fullmatch(rendered):
        return False
    return namespace is None or rendered.startswith(f"eg:{namespace}:")


def _policy(seed: str) -> dict[str, Any]:
    return {
        "tenant_ref": _opaque("tenant", seed),
        "access_policy_ref": _opaque("policy", seed),
        "classification": "internal",
        "retention_policy_ref": _opaque("retention", seed),
        "deletion_policy_ref": _opaque("deletion", seed),
        "legal_hold_ref": None,
        "purpose_refs": [_opaque("purpose", "program-optimization")],
    }


def _training_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    supplied = data.get("trainset") or data.get("examples")
    rows: list[Any]
    if isinstance(supplied, Sequence) and not isinstance(supplied, str | bytes):
        rows = list(supplied)
    elif isinstance(data.get("documents"), Sequence) and not isinstance(
        data.get("documents"), str | bytes
    ):
        rows = [
            {"context": "", "task": "extract governed facts", "response": value}
            for value in data["documents"]
        ]
    elif isinstance(data.get("labeled_pairs"), Sequence) and not isinstance(
        data.get("labeled_pairs"), str | bytes
    ):
        rows = [
            {
                "context": pair[0],
                "task": pair[1],
                "response": "relevant" if bool(pair[2]) else "not_relevant",
            }
            for pair in data["labeled_pairs"]
            if isinstance(pair, Sequence)
            and not isinstance(pair, str | bytes)
            and len(pair) >= 3
        ]
    elif isinstance(data.get("traces"), Sequence) and not isinstance(
        data.get("traces"), str | bytes
    ):
        rows = list(data["traces"])
    else:
        rows = []

    normalized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            normalized.append(dict(row))
        else:
            normalized.append({"context": "", "task": str(row), "response": str(row)})
    return normalized


def _score(row: Mapping[str, Any]) -> float:
    value = row.get("reward", row.get("score", 1.0))
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.5


def _row_modalities(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the governed modality set for one ephemeral training row."""

    supplied = row.get("modalities", row.get("modality", "text"))
    if isinstance(supplied, str):
        values = [supplied]
    elif isinstance(supplied, Sequence) and not isinstance(supplied, bytes):
        values = [str(value) for value in supplied]
    else:
        raise ValueError("program example modalities must be a string or sequence")
    normalized = tuple(dict.fromkeys(value.strip().casefold() for value in values))
    if not normalized or any(value not in _PROGRAM_MODALITIES for value in normalized):
        raise ValueError("program example contains an unsupported modality")
    return normalized


def _evidence_address(modality: str, seed: str) -> dict[str, Any]:
    """Build a valid numeric/opaque locus for every native program modality."""

    if modality in {"text", "document"}:
        return {"kind": "character_range", "start": 0, "end": 1}
    if modality == "image":
        return {
            "kind": "image_region",
            "x": 0.0,
            "y": 0.0,
            "width": 1.0,
            "height": 1.0,
        }
    if modality == "audio":
        return {"kind": "audio_range", "start_ms": 0, "end_ms": 1}
    if modality == "video":
        return {"kind": "frame_range", "start_frame": 0, "end_frame": 0}
    if modality == "time_series":
        return {"kind": "metric_window", "start_ms": 0, "end_ms": 1}
    if modality in {"graph", "vector", "binary"}:
        return {
            "kind": "row_version",
            "row_ref": _opaque("row", seed),
            "version": 1,
        }
    if modality in {"table", "tensor"}:
        return {
            "kind": "table_cell_range",
            "row_start": 0,
            "row_end": 0,
            "col_start": 0,
            "col_end": 0,
        }
    if modality == "spatial":
        return {"kind": "point", "x": 0.0, "y": 0.0}
    if modality == "code":
        return {
            "kind": "code_symbol",
            "revision_ref": _opaque("revision", seed),
            "symbol_ref": _opaque("symbol", seed),
            "start_line": 1,
            "end_line": 1,
        }
    if modality == "trace":
        return {
            "kind": "trace_span",
            "trace_ref": _opaque("trace", seed),
            "span_ref": _opaque("span", seed),
        }
    raise ValueError("program example contains an unsupported modality")


@dataclass(frozen=True)
class OptimizationRequest:
    """High-level optimization input converted to the exact eg-program v1 map."""

    target: str
    objective: str
    data: Mapping[str, Any] = field(default_factory=dict)
    optimizer: str = "bootstrap_few_shot"
    schema_version: int = 1

    def to_payload(self) -> dict[str, Any]:
        rows = _training_rows(self.data)
        if not rows:
            raise ValueError("native program optimization requires training examples")
        optimizer = str(self.optimizer).strip().casefold()
        execution = _OPTIMIZER_EXECUTIONS.get(optimizer)
        if execution is None:
            raise ValueError("native program optimizer is unsupported")
        tool_refs = _tool_references(self.data)
        if optimizer == "avatar" and not tool_refs:
            raise ValueError("Avatar optimization requires a governed tool reference")

        seed = _canonical(
            {
                "target": self.target,
                "objective": self.objective,
                "artifact": self.data.get("artifact", {}),
                "tool_refs": tool_refs,
            }
        )
        policy = _policy(seed)
        program_ref = _opaque("program", seed)
        examples: list[dict[str, Any]] = []
        modality_scores: dict[str, list[float]] = {}
        for index, row in enumerate(rows):
            row_seed = _canonical({"index": index, "row": row})
            success = bool(row.get("success", True))
            context = row.get("context", "")
            task = row.get("task", row.get("task_text", ""))
            response = row.get(
                "response", row.get("expected_output", row.get("primitive_used", ""))
            )
            inputs = []
            for field_name, value in (("context", context), ("task", task)):
                if value not in (None, ""):
                    reference = _opaque("content", _canonical({field_name: value}))
                    if reference not in inputs:
                        inputs.append(reference)
            if not inputs:
                inputs.append(_opaque("content", row_seed))
            observed = _opaque(
                "content",
                _canonical({"response": response, "row": row_seed}),
            )
            expected = (
                _opaque("content", _canonical({"expected": response}))
                if response not in (None, "")
                else None
            )
            artifact_ref = _opaque("artifact", row_seed)
            score = _score(row)
            modalities = _row_modalities(row)
            evidence = []
            for modality in modalities:
                modality_seed = _canonical({"row": row_seed, "modality": modality})
                modality_scores.setdefault(modality, []).append(score)
                evidence.append(
                    {
                        "modality": modality,
                        "locus": {
                            "id": _opaque("locus", modality_seed),
                            "subject": {"kind": "artifact", "id": artifact_ref},
                            "address": _evidence_address(modality, modality_seed),
                            "policy_ref": policy["access_policy_ref"],
                            "derivation_ref": _opaque("derivation", modality_seed),
                        },
                        "policy": policy,
                    }
                )
            examples.append(
                {
                    "example_ref": _opaque(
                        "example", row.get("example_ref") or row_seed
                    ),
                    "input_refs": inputs,
                    "expected_output_ref": expected,
                    "observed_output_ref": observed if success else None,
                    "feedback_ref": (
                        _opaque("feedback", row_seed)
                        if row.get("failure_reason")
                        else None
                    ),
                    "trace_ref": (
                        _opaque("trace", row_seed)
                        if row.get("source") == "kg_trace" or row.get("trace_ref")
                        else None
                    ),
                    "split": "train",
                    "outcome": "success" if success else "failure",
                    "score": score,
                    "weight": 1.0,
                    "evidence": evidence,
                }
            )

        if optimizer == "avatar":
            positive = any(
                example["outcome"] == "success" and example["trace_ref"] is not None
                for example in examples
            )
            negative = any(
                example["outcome"] == "failure"
                and example["trace_ref"] is not None
                and example["feedback_ref"] is not None
                for example in examples
            )
            if not positive or not negative:
                raise ValueError(
                    "Avatar optimization requires positive and negative governed traces"
                )

        baseline_score = sum(example["score"] for example in examples) / len(examples)
        request_seed = _canonical(
            {
                "program_ref": program_ref,
                "examples": [item["example_ref"] for item in examples],
                "optimizer": optimizer,
            }
        )
        return {
            "schema_version": self.schema_version,
            "request_ref": _opaque("optimization", request_seed),
            "program": {
                "schema_version": self.schema_version,
                "program_ref": program_ref,
                "revision": 1,
                "parent_ref": None,
                "signature": {
                    "signature_ref": _opaque("signature", seed),
                    "instruction_ref": _opaque("instruction", seed),
                    "fields": [
                        {
                            "name": "context",
                            "role": "input",
                            "schema_ref": _opaque("schema", "text-context-v1"),
                            "description_ref": None,
                            "required": False,
                        },
                        {
                            "name": "task",
                            "role": "input",
                            "schema_ref": _opaque("schema", "text-task-v1"),
                            "description_ref": None,
                            "required": True,
                        },
                        {
                            "name": "response",
                            "role": "output",
                            "schema_ref": _opaque("schema", "text-response-v1"),
                            "description_ref": None,
                            "required": True,
                        },
                    ],
                },
                "module": "react" if optimizer == "avatar" else "predict",
                "adapter": "chat",
                "tool_refs": tool_refs,
                "policy": policy,
            },
            "corpus": {
                "corpus_ref": _opaque("corpus", request_seed),
                "snapshot_version": 1,
                "privacy": {
                    "scanner_ref": _opaque("scanner", "persistence-privacy-v1"),
                    "policy_version_ref": _opaque(
                        "privacy_policy", "persistence-privacy-v1"
                    ),
                    "raw_pii_persisted": False,
                    "local_identifiers_persisted": False,
                },
                "examples": examples,
            },
            "optimizer": optimizer,
            "budget": {
                "max_candidates": 8,
                "max_demonstrations": _demonstration_budget(
                    len(examples), len(modality_scores)
                ),
                "max_model_calls": (
                    8 if execution in {"model_transport_plan", "composite_plan"} else 0
                ),
                "max_evaluator_calls": 0,
                "max_training_steps": (
                    32 if execution in {"trainer_plan", "composite_plan"} else 0
                ),
                "seed": 0,
            },
            "promotion": {
                "min_score_delta": 0.0,
                "max_modality_regression": 0.0,
                "require_all_observed_modalities": True,
                "min_evidence_refs": 1,
            },
            "baseline": {
                "subject_ref": program_ref,
                "aggregate_score": baseline_score,
                "modality_scores": {
                    modality: sum(scores) / len(scores)
                    for modality, scores in sorted(modality_scores.items())
                },
                "evidence_refs": [_opaque("evaluation", request_seed)],
            },
            "optimizer_artifacts": [],
            "candidate_evaluations": [],
        }


@dataclass(frozen=True)
class NativeOptimizationAttempt:
    """Content-safe normalized result from invoking the native engine."""

    disposition: NativeDisposition
    payload: dict[str, Any] = field(default_factory=dict)
    error_code: str = ""


def _program_authority(engine: Any) -> Any:
    """Project an operational graph facade onto its single compute authority."""
    if engine is None:
        try:
            from agent_utilities.knowledge_graph.core.graph_compute import (
                GraphComputeEngine,
            )

            return GraphComputeEngine.get_active()
        except Exception:  # noqa: BLE001 - a cold process has no active authority
            return None
    if callable(getattr(engine, "optimize_program", None)):
        return engine
    compute = getattr(engine, "graph_compute", None)
    if callable(getattr(compute, "optimize_program", None)):
        return compute
    return None


def try_native_optimization(
    engine: Any, request: OptimizationRequest
) -> NativeOptimizationAttempt:
    """Invoke the sole native backend and fail closed on protocol errors."""
    method = getattr(_program_authority(engine), "optimize_program", None)
    if not callable(method):
        return NativeOptimizationAttempt(disposition="unavailable")
    try:
        raw = method(request.to_payload())
    except Exception:  # noqa: BLE001 - durable reports receive only bounded codes
        return NativeOptimizationAttempt(
            disposition="error", error_code="native_execution_failed"
        )
    if inspect.isawaitable(raw):
        close = getattr(raw, "close", None)
        if callable(close):
            close()
        return NativeOptimizationAttempt(
            disposition="error", error_code="native_async_result_on_sync_boundary"
        )
    if not isinstance(raw, Mapping) or set(raw) != {"status", "result"}:
        return NativeOptimizationAttempt(
            disposition="error", error_code="native_invalid_response"
        )
    if raw.get("status") != "proposed" or not isinstance(raw.get("result"), Mapping):
        return NativeOptimizationAttempt(
            disposition="error", error_code="native_invalid_response"
        )
    return NativeOptimizationAttempt(disposition="completed", payload=dict(raw))


__all__ = [
    "is_opaque_program_reference",
    "NativeOptimizationAttempt",
    "OptimizationRequest",
    "opaque_program_reference",
    "try_native_optimization",
]
