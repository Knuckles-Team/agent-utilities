#!/usr/bin/env python3
"""Bind all local exact-artifact campaign evidence to one release identity."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Final

from scripts.release.exact_local_gates_manifest import (
    ManifestError,
    _write_new_private,
    generate_manifest,
    validate_manifest,
)
from scripts.release.promote_local_release import (
    ReleaseError,
    _assert_path_free_evidence,
    _external_json,
    _validate_signature_shape,
    verify_evidence_file,
)

SCHEMA_VERSION: Final = 1
SIGNER_ENV: Final = "EXACT_ARTIFACT_CLOSURE_SIGNER_COMMAND"
VERIFIER_ENV: Final = "EXACT_ARTIFACT_CLOSURE_VERIFIER_COMMAND"
HEX_64 = re.compile(r"^[a-f0-9]{64}$")
SHA256 = re.compile(r"^sha256:[a-f0-9]{64}$")
RELEASE_ID = re.compile(r"^release-[a-z0-9][a-z0-9.-]{2,63}$")
OPAQUE_SIGNER = re.compile(r"^signer:[a-z0-9][a-z0-9_.:-]{2,63}$")
MAX_EVIDENCE_BYTES: Final = 16 * 1024 * 1024
GATES: Final = (
    "G-01",
    "G-02",
    "G-04",
    "G-05",
    "G-08",
    "G-09",
    "G-14",
    "G-15",
    "G-17",
    "G-26",
    "G-30",
    "G-32",
    "G-34",
    "G-35",
    "G-37",
)
COMMIT_PHASES: Final = (
    "before_rows",
    "after_rows_before_metadata",
    "before_commit",
    "after_commit_before_ack",
)
MUTATION_DOMAINS: Final = (
    "graph_rows",
    "graph_snapshot",
    "rdf_dataset",
    "sql_catalog",
    "blob_store",
    "kv_store",
    "time_series",
    "analytics_job",
    "broker",
    "cross_modal",
    "multi_graph",
    "lifecycle",
    "control_plane",
)
MODALITIES: Final = ("document", "image", "audio", "video")
WIRE_PROTOCOLS: Final = (
    "native_rpc",
    "postgresql",
    "mysql",
    "mssql",
    "sqlite",
    "bolt",
    "redis",
    "amqp",
    "mqtt",
    "stomp",
)
WIRE_FEATURES: Final = {
    "native_rpc": "server",
    "postgresql": "pgwire",
    "mysql": "mysql-wire",
    "mssql": "mssql-wire",
    "sqlite": "sqlite-wire",
    "bolt": "bolt-wire",
    "redis": "redis-wire",
    "amqp": "amqp-wire",
    "mqtt": "mqtt-wire",
    "stomp": "stomp-wire",
}
DATA_PATHS: Final = (
    "graph",
    "property",
    "union",
    "semantic",
    "topology",
    "rdf",
    "time_series",
    "vector",
    "blob",
    "job",
    "sql",
    "cache",
    "kv",
    "broker",
)
KNOWLEDGE_FAMILIES: Final = (
    "graph",
    "sql",
    "rdf",
    "vector",
    "time_series",
    "job",
    "cross_modal",
)
KNOWLEDGE_REQUIREMENTS: Final = (
    "arrow_parity",
    "pushdown",
    "bounded_streaming",
    "paging_resume",
    "cancellation",
    "backpressure",
    "snapshot_correctness",
)
REASONING_CASES: Final = (
    "projection_lag",
    "restart",
    "contradiction",
    "retraction",
    "valid_transaction_time_change",
    "causal_recomputation",
    "assumptions",
    "counterexamples",
    "repair",
)
G08_CASES: Final = (
    "fairness_scoped_claim",
    "renewable_lease",
    "checkpoint_fencing",
    "retry_schedule",
    "dependency_release",
    "dead_letter",
    "stale_worker_rejection",
    "idempotent_terminal_commit",
)
G09_CASES: Final = (
    "atomic_inbox_workitem_commit",
    "crash_window_replay_idempotent",
)
G35_CASES: Final = (
    "referenced_signing_bootstrap",
    "closed_world_function_denial",
    "mcp_identity_denial",
    "ontology_permission_denial",
    "action_capability_denial",
    "governed_constructor_denial",
    "delegation_context_preserved",
    "invalid_authority_and_policy_rejected",
)
G37_COVERAGE: Final = {
    "cold_start",
    "routing",
    "ingest",
    "query",
    "job",
    "modality",
    "memory",
    "hot_path_scenarios",
}


class ClosureError(RuntimeError):
    """A stable, privacy-safe closure failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise ClosureError(code)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not 0 < metadata.st_size <= MAX_EVIDENCE_BYTES
        ):
            _fail("evidence_file_invalid")
        raw = bytearray()
        while chunk := os.read(
            descriptor, min(64 * 1024, MAX_EVIDENCE_BYTES - len(raw) + 1)
        ):
            raw.extend(chunk)
            if len(raw) > MAX_EVIDENCE_BYTES:
                _fail("evidence_file_invalid")
    except ClosureError:
        raise
    except OSError:
        _fail("evidence_file_invalid")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        value = json.loads(
            raw,
            object_pairs_hook=lambda pairs: _object_without_duplicates(pairs),
        )
    except (UnicodeError, json.JSONDecodeError):
        _fail("evidence_json_invalid")
    if not isinstance(value, dict):
        _fail("evidence_json_invalid")
    return value, hashlib.sha256(raw).hexdigest()


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _fail("evidence_json_duplicate_key")
        value[key] = item
    return value


def _raw_digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or HEX_64.fullmatch(value) is None:
        _fail(code)
    return value


def _prefixed_digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        _fail(code)
    return value


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_fault_restart(
    value: dict[str, Any], engine_sha256: str
) -> dict[str, int]:
    if set(value) != {
        "binary",
        "certification",
        "commit_phases",
        "g05",
        "matrix",
        "mutation_domains",
        "schema_version",
        "summary",
    }:
        _fail("fault_restart_schema_invalid")
    binary = value.get("binary")
    matrix = value.get("matrix")
    summary = value.get("summary")
    g05 = value.get("g05")
    if (
        value.get("schema_version") != 1
        or value.get("certification") != "epistemic-graph-exact-fault-restart"
        or binary != {"sha256": engine_sha256}
        or value.get("commit_phases") != list(COMMIT_PHASES)
        or value.get("mutation_domains") != list(MUTATION_DOMAINS)
        or not isinstance(matrix, list)
        or len(matrix) != 60
        or summary != {"matrix_cases": 60, "passed": 60, "status": "pass"}
        or not isinstance(g05, dict)
        or set(g05) != {"spatial_restart_lazy_open", "tenant_qualified_time_series"}
    ):
        _fail("fault_restart_not_passing")
    family_phases: set[tuple[str, str]] = set()
    families: set[str] = set()
    for row in matrix:
        if not isinstance(row, dict) or set(row) != {
            "domain",
            "expected",
            "family",
            "observed",
            "passed",
            "phase",
            "recovery",
        }:
            _fail("fault_restart_matrix_invalid")
        family = str(row.get("family") or "")
        phase = str(row.get("phase") or "")
        domain = str(row.get("domain") or "")
        if (
            not family
            or phase not in COMMIT_PHASES
            or domain not in value.get("mutation_domains", [])
            or row.get("passed") is not True
            or row.get("observed") not in {"no_effect", "complete_effect"}
            or row.get("expected")
            not in {"no_effect", "complete_effect", "complete_after_exact_retry"}
            or row.get("recovery")
            not in {"authoritative_restart_read", "exact_request_replay"}
            or (family, phase) in family_phases
        ):
            _fail("fault_restart_matrix_invalid")
        family_phases.add((family, phase))
        families.add(family)
    if len(families) != 15 or any(
        sum(item[0] == family for item in family_phases) != 4 for family in families
    ):
        _fail("fault_restart_matrix_invalid")
    spatial = g05["spatial_restart_lazy_open"]
    timeseries = g05["tenant_qualified_time_series"]
    if (
        not isinstance(spatial, dict)
        or set(spatial)
        != {
            "expected_hits",
            "final_result_digest",
            "lazy_open_cycles",
            "partial_state_observed",
            "passed",
        }
        or spatial.get("passed") is not True
        or spatial.get("partial_state_observed") is not True
        or spatial.get("lazy_open_cycles") != 2
        or not _positive_int(spatial.get("expected_hits"))
        or HEX_64.fullmatch(str(spatial.get("final_result_digest") or "")) is None
        or not isinstance(timeseries, dict)
        or set(timeseries)
        != {
            "identical_local_series_ids",
            "isolated_result_digests",
            "passed",
            "restart_cycles",
        }
        or timeseries.get("passed") is not True
        or timeseries.get("identical_local_series_ids") != 2
        or timeseries.get("restart_cycles") != 3
        or not isinstance(timeseries.get("isolated_result_digests"), list)
        or len(set(timeseries["isolated_result_digests"])) != 2
        or any(
            HEX_64.fullmatch(str(item)) is None
            for item in timeseries["isolated_result_digests"]
        )
    ):
        _fail("fault_restart_g05_invalid")
    return {"matrix_cases": 60, "mutation_families": 15}


def _all_passed(values: Any) -> bool:
    return bool(
        isinstance(values, dict)
        and values
        and all(
            isinstance(item, dict) and item.get("passed") is True
            for item in values.values()
        )
    )


def _validate_performance(value: dict[str, Any], engine_sha256: str) -> dict[str, int]:
    expected_keys = {
        "schema_version",
        "gate",
        "status",
        "started_at_utc",
        "completed_at_utc",
        "exact_artifact",
        "client_artifact",
        "authority",
        "deployment_profile",
        "dataset",
        "threshold_manifest_sha256",
        "scenario_contract",
        "scenario_evidence_binding",
        "hardware_class",
        "coverage",
        "measurements",
        "metric_results",
        "complexity_results",
        "scenario_family_evidence",
        "hot_path_row_evidence",
        "failures",
    }
    if set(value) != expected_keys:
        _fail("performance_schema_invalid")
    artifact = value.get("exact_artifact")
    contract = value.get("scenario_contract")
    families = value.get("scenario_family_evidence")
    rows = value.get("hot_path_row_evidence")
    coverage = value.get("coverage")
    if (
        value.get("schema_version") != "1"
        or value.get("gate") != "G-37"
        or value.get("status") != "pass"
        or value.get("failures") != []
        or not isinstance(artifact, dict)
        or artifact.get("component") != "epistemic-graph-server"
        or artifact.get("sha256") != engine_sha256
        or artifact.get("staged_copy_verified") is not True
        or not isinstance(contract, dict)
        or contract.get("scenario_families") != 30
        or contract.get("ledger_rows") != 54
        or not isinstance(families, dict)
        or len(families) != 30
        or not isinstance(rows, dict)
        or len(rows) != 54
        or set(rows) != {f"G37-HP-{ordinal:03d}" for ordinal in range(1, 55)}
        or not _all_passed(value.get("metric_results"))
        or not _all_passed(value.get("complexity_results"))
        or not isinstance(coverage, dict)
        or set(coverage) != G37_COVERAGE
    ):
        _fail("performance_not_passing")
    if any(
        not isinstance(item, dict)
        or item.get("passed") is not True
        or not _all_passed(item.get("threshold_results"))
        for item in rows.values()
    ):
        _fail("performance_row_evidence_invalid")
    modality = coverage.get("modality")
    if (
        not isinstance(modality, dict)
        or modality.get("component_probes") != list(MODALITIES)
        or modality.get("results_verified") is not True
        or any(
            not isinstance(coverage[name], dict) or not coverage[name]
            for name in G37_COVERAGE - {"modality"}
        )
        or coverage.get("hot_path_scenarios")
        != {
            "scenario_families": 30,
            "ledger_rows": 54,
            "raw_results_validated": True,
            "exact_binary_subcommands": 30,
        }
    ):
        _fail("performance_coverage_invalid")
    return {"scenario_families": 30, "ledger_rows": 54}


def _validate_multimodal(
    value: dict[str, Any], engine_sha256: str, performance_digest: str
) -> dict[str, int]:
    if set(value) != {
        "binary",
        "certification",
        "exact_behavior_dimensions",
        "fault_matrix",
        "matrix",
        "modalities",
        "performance",
        "schema_version",
        "summary",
    }:
        _fail("multimodal_schema_invalid")
    summary = value.get("summary")
    matrix = value.get("matrix")
    faults = value.get("fault_matrix")
    performance = value.get("performance")
    if (
        value.get("schema_version") != 1
        or value.get("certification") != "epistemic-graph-exact-multimodal"
        or value.get("binary")
        != {"sha256": engine_sha256, "sealed_copy_verified": True}
        or value.get("modalities") != list(MODALITIES)
        or not isinstance(value.get("exact_behavior_dimensions"), list)
        or len(value["exact_behavior_dimensions"]) != 12
        or not isinstance(matrix, list)
        or len(matrix) != 4
        or not isinstance(faults, list)
        or len(faults) != 16
        or summary
        != {
            "dimensions_per_modality": 12,
            "fault_cases": 16,
            "modalities": 4,
            "passed": 4,
            "status": "pass",
        }
        or not isinstance(performance, dict)
        or performance.get("gate") != "G-37"
        or performance.get("report_sha256") != performance_digest
        or performance.get("same_artifact_verified") is not True
        or performance.get("status") != "pass"
    ):
        _fail("multimodal_not_passing")
    if any(
        not isinstance(row, dict)
        or row.get("modality") not in MODALITIES
        or row.get("component_tck_pass") != 12
        or row.get("component_tck_not_applicable") != 0
        or not isinstance(row.get("dimensions"), dict)
        or set(row["dimensions"]) != set(value["exact_behavior_dimensions"])
        or not all(item is True for item in row["dimensions"].values())
        for row in matrix
    ) or {row["modality"] for row in matrix} != set(MODALITIES):
        _fail("multimodal_matrix_invalid")
    fault_pairs: set[tuple[str, str]] = set()
    for row in faults:
        if (
            not isinstance(row, dict)
            or set(row) != {"expected", "modality", "observed", "passed", "phase"}
            or row.get("modality") not in MODALITIES
            or row.get("phase") not in COMMIT_PHASES
            or row.get("passed") is not True
            or row.get("expected") != row.get("observed")
        ):
            _fail("multimodal_fault_matrix_invalid")
        fault_pairs.add((row["modality"], row["phase"]))
    if len(fault_pairs) != 16:
        _fail("multimodal_fault_matrix_invalid")
    return {"modalities": 4, "behavior_dimensions": 12, "fault_cases": 16}


def _validate_protocol_authorization(
    value: dict[str, Any], engine_sha256: str
) -> dict[str, int]:
    if set(value) != {
        "binary",
        "certification",
        "cross_tenant",
        "data_path_matrix",
        "data_paths",
        "schema_version",
        "summary",
        "wire_matrix",
        "wire_protocols",
    }:
        _fail("protocol_authorization_schema_invalid")
    data_matrix = value.get("data_path_matrix")
    wire_matrix = value.get("wire_matrix")
    if (
        value.get("schema_version") != 1
        or value.get("certification") != "epistemic-graph-exact-protocol-authorization"
        or value.get("binary") != {"sha256": engine_sha256}
        or value.get("data_paths") != list(DATA_PATHS)
        or value.get("wire_protocols") != list(WIRE_PROTOCOLS)
        or value.get("summary")
        != {
            "data_path_cases": len(DATA_PATHS),
            "protocol_cases": len(WIRE_PROTOCOLS),
            "status": "pass",
        }
        or value.get("cross_tenant")
        != {
            "graph_row_hidden": True,
            "kv_namespace_hidden": True,
            "time_series_namespace_hidden": True,
        }
        or not isinstance(data_matrix, list)
        or len(data_matrix) != len(DATA_PATHS)
        or not isinstance(wire_matrix, list)
        or len(wire_matrix) != len(WIRE_PROTOCOLS)
    ):
        _fail("protocol_authorization_not_passing")
    if any(
        row
        != {
            "denial_observed": True,
            "path": path,
            "tenant_relation": "same",
        }
        for row, path in zip(data_matrix, DATA_PATHS, strict=True)
    ):
        _fail("protocol_authorization_data_matrix_invalid")
    if any(
        row
        != {
            "auth_denial_observed": True,
            "feature": WIRE_FEATURES[protocol],
            "protocol": protocol,
        }
        for row, protocol in zip(wire_matrix, WIRE_PROTOCOLS, strict=True)
    ):
        _fail("protocol_authorization_wire_matrix_invalid")
    return {
        "data_path_cases": len(DATA_PATHS),
        "protocol_cases": len(WIRE_PROTOCOLS),
    }


def _validate_knowledge_batch(
    value: dict[str, Any], engine_sha256: str
) -> dict[str, int]:
    if set(value) != {
        "binary",
        "certification",
        "families",
        "requirements",
        "schema_version",
        "snapshot_matrix",
        "summary",
    }:
        _fail("knowledge_batch_schema_invalid")
    families = value.get("families")
    snapshots = value.get("snapshot_matrix")
    if (
        value.get("schema_version") != 1
        or value.get("certification") != "epistemic-graph-exact-knowledge-batch"
        or value.get("binary") != {"sha256": engine_sha256}
        or value.get("requirements") != list(KNOWLEDGE_REQUIREMENTS)
        or value.get("summary")
        != {
            "families": len(KNOWLEDGE_FAMILIES),
            "requirements": len(KNOWLEDGE_REQUIREMENTS),
            "snapshot_cases": len(KNOWLEDGE_FAMILIES),
            "status": "pass",
        }
        or not isinstance(families, list)
        or len(families) != len(KNOWLEDGE_FAMILIES)
        or not isinstance(snapshots, list)
        or len(snapshots) != len(KNOWLEDGE_FAMILIES)
    ):
        _fail("knowledge_batch_not_passing")
    for row, family in zip(families, KNOWLEDGE_FAMILIES, strict=True):
        payload_digests = (
            row.get("page_payload_sha256") if isinstance(row, dict) else None
        )
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "arrow_schema_sha256",
                "backpressure_seconds",
                "cancel_resume",
                "family",
                "page_payload_sha256",
                "pages",
                "rows",
                "tamper_denied",
            }
            or row.get("family") != family
            or HEX_64.fullmatch(str(row.get("arrow_schema_sha256") or "")) is None
            or row.get("backpressure_seconds") != 0.02
            or row.get("cancel_resume") is not True
            or row.get("tamper_denied") is not True
            or not isinstance(payload_digests, list)
            or not payload_digests
            or any(HEX_64.fullmatch(str(item)) is None for item in payload_digests)
            or row.get("pages") != len(payload_digests)
            or not _positive_int(row.get("pages"))
            or not isinstance(row.get("rows"), int)
            or isinstance(row.get("rows"), bool)
            or row["rows"] < 3
        ):
            _fail("knowledge_batch_family_invalid")
    for row, family in zip(snapshots, KNOWLEDGE_FAMILIES, strict=True):
        expected_outcome = (
            "immutable_source_resumed" if family == "job" else "changed_snapshot_denied"
        )
        if row != {"family": family, "outcome": expected_outcome, "passed": True}:
            _fail("knowledge_batch_snapshot_invalid")
    if len({row["arrow_schema_sha256"] for row in families}) != 1:
        _fail("knowledge_batch_schema_parity_invalid")
    return {
        "families": len(KNOWLEDGE_FAMILIES),
        "requirements": len(KNOWLEDGE_REQUIREMENTS),
        "snapshot_cases": len(KNOWLEDGE_FAMILIES),
    }


def _validate_reasoning_repair(
    value: dict[str, Any], engine_sha256: str
) -> dict[str, int]:
    if set(value) != {
        "binary",
        "cases",
        "certification",
        "matrix",
        "schema_version",
        "summary",
    }:
        _fail("reasoning_repair_schema_invalid")
    matrix = value.get("matrix")
    summary = value.get("summary")
    if (
        value.get("schema_version") != 1
        or value.get("certification") != "epistemic-graph-exact-reasoning-repair"
        or value.get("binary") != {"sha256": engine_sha256}
        or value.get("cases") != list(REASONING_CASES)
        or not isinstance(matrix, dict)
        or tuple(matrix) != REASONING_CASES
        or not isinstance(summary, dict)
        or set(summary) != {"cases", "initial_projection_polls", "passed", "status"}
        or summary.get("cases") != len(REASONING_CASES)
        or summary.get("passed") != len(REASONING_CASES)
        or summary.get("status") != "pass"
        or not _positive_int(summary.get("initial_projection_polls"))
    ):
        _fail("reasoning_repair_not_passing")

    def digest_field(item: Any, field: str) -> bool:
        return (
            isinstance(item, dict)
            and HEX_64.fullmatch(str(item.get(field) or "")) is not None
        )

    projection = matrix["projection_lag"]
    restart = matrix["restart"]
    contradiction = matrix["contradiction"]
    retraction = matrix["retraction"]
    temporal = matrix["valid_transaction_time_change"]
    causal = matrix["causal_recomputation"]
    assumptions = matrix["assumptions"]
    counterexamples = matrix["counterexamples"]
    repair = matrix["repair"]
    if (
        not isinstance(projection, dict)
        or set(projection)
        != {"converged", "immediate_prior_watermark", "polls", "watermark_advanced"}
        or projection.get("converged") is not True
        or not isinstance(projection.get("immediate_prior_watermark"), bool)
        or not _positive_int(projection.get("polls"))
        or projection.get("watermark_advanced") is not True
        or not digest_field(restart, "projection_sha256")
        or set(restart) != {"durable_projection_equal", "polls", "projection_sha256"}
        or restart.get("durable_projection_equal") is not True
        or not _positive_int(restart.get("polls"))
        or not digest_field(contradiction, "classification_sha256")
        or set(contradiction) != {"classification_sha256", "undecided"}
        or contradiction.get("undecided") != 2
        or not digest_field(retraction, "result_sha256")
        or set(retraction)
        != {"belief_flipped", "durable_replay_equal", "result_sha256"}
        or retraction.get("belief_flipped") is not True
        or retraction.get("durable_replay_equal") is not True
        or not digest_field(temporal, "result_sha256")
        or set(temporal) != {"changed", "claim_flipped", "result_sha256"}
        or not _positive_int(temporal.get("changed"))
        or temporal.get("claim_flipped") is not True
        or not digest_field(causal, "result_sha256")
        or causal.get("deterministic") is not True
        or causal.get("variables") != 3
        or not digest_field(assumptions, "result_sha256")
        or assumptions.get("calibrated_intervals") != 3
        or assumptions.get("observe_intervene_distinct") is not True
        or not isinstance(counterexamples, dict)
        or set(counterexamples)
        != {"causal", "minimal_flip_cardinality", "status_sha256"}
        or counterexamples.get("minimal_flip_cardinality") != 1
        or HEX_64.fullmatch(str(counterexamples.get("status_sha256") or "")) is None
        or not isinstance(counterexamples.get("causal"), dict)
        or counterexamples["causal"].get("deterministic") is not True
        or counterexamples["causal"].get("variables") != 3
        or not digest_field(counterexamples["causal"], "result_sha256")
        or not isinstance(repair, dict)
        or set(repair)
        != {
            "dependencies",
            "fence_epoch_positive",
            "fresh",
            "polls",
            "post_retraction_projection_polls",
            "post_retraction_projection_sha256",
            "stale_fence_denied",
        }
        or repair.get("dependencies") != 1
        or repair.get("fence_epoch_positive") is not True
        or repair.get("fresh") is not True
        or not _positive_int(repair.get("polls"))
        or not _positive_int(repair.get("post_retraction_projection_polls"))
        or HEX_64.fullmatch(str(repair.get("post_retraction_projection_sha256") or ""))
        is None
        or repair.get("stale_fence_denied") is not True
    ):
        _fail("reasoning_repair_matrix_invalid")
    return {"cases": len(REASONING_CASES)}


def _decode_urlsafe(value: str, code: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError):
        _fail(code)


def _verify_exact_local_signature(value: dict[str, Any]) -> None:
    signature = value.get("signature")
    if (
        not isinstance(signature, dict)
        or set(signature) != {"algorithm", "public_key", "signature", "signer_id"}
        or signature.get("algorithm") != "ed25519"
        or not isinstance(signature.get("public_key"), str)
        or not isinstance(signature.get("signature"), str)
        or not isinstance(signature.get("signer_id"), str)
        or OPAQUE_SIGNER.fullmatch(signature["signer_id"]) is None
    ):
        _fail("exact_local_signature_invalid")
    public_key = _decode_urlsafe(
        signature["public_key"], "exact_local_signature_invalid"
    )
    signed = _decode_urlsafe(signature["signature"], "exact_local_signature_invalid")
    if len(public_key) != 32 or len(signed) != 64:
        _fail("exact_local_signature_invalid")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signed,
            _canonical(
                {key: item for key, item in value.items() if key != "signature"}
            ),
        )
    except (ImportError, ValueError):
        _fail("exact_local_signature_invalid")
    except Exception:
        _fail("exact_local_signature_invalid")


def _valid_pytest_summary(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "collected",
        "passed",
        "required",
        "required_cases",
        "scenario_evidence",
        "skipped",
    }:
        return False
    required_cases = value.get("required_cases")
    scenario_evidence = value.get("scenario_evidence")
    return bool(
        _positive_int(value.get("collected"))
        and value.get("passed") == value.get("collected")
        and _positive_int(value.get("required"))
        and value.get("skipped") == 0
        and isinstance(required_cases, list)
        and required_cases == sorted(set(required_cases))
        and len(required_cases) == value.get("required")
        and scenario_evidence == {case: "pass" for case in required_cases}
    )


def _validate_exact_local(
    value: dict[str, Any], manifest: dict[str, Any], manifest_digest: str
) -> dict[str, dict[str, int]]:
    if set(value) != {
        "artifacts",
        "certification",
        "gates",
        "release_id",
        "schema_version",
        "signature",
        "status",
    }:
        _fail("exact_local_schema_invalid")
    _verify_exact_local_signature(value)
    artifacts = value.get("artifacts")
    gates = value.get("gates")
    if (
        value.get("schema_version") != 1
        or value.get("certification") != "agent-utilities-exact-local-gates"
        or value.get("release_id") != manifest["release_id"]
        or value.get("status") != "pass"
        or not isinstance(artifacts, dict)
        or set(artifacts)
        != {
            "agent_utilities",
            "epistemic_graph",
            "graphos",
            "harness",
            "promotion_evidence",
            "release_manifest",
            "release_python",
            "release_spec",
            "test_catalog",
        }
        or not isinstance(gates, dict)
        or set(gates) != {"g08", "g09", "g26", "g30", "g32", "g34", "g35"}
        or any(
            not isinstance(item, dict) or item.get("status") != "pass"
            for item in gates.values()
        )
    ):
        _fail("exact_local_not_passing")
    expected = {
        "epistemic_graph": manifest["engine_sha256"],
        "graphos": manifest["graphos_sha256"],
        "harness": manifest["harness_sha256"],
        "promotion_evidence": manifest["promotion_evidence_sha256"],
        "release_manifest": manifest_digest,
        "release_python": manifest["release_python_sha256"],
        "release_spec": manifest["release_spec_sha256"],
        "test_catalog": manifest["test_catalog_sha256"],
    }
    for name, digest in expected.items():
        if artifacts.get(name) != {"sha256": digest}:
            _fail("exact_local_artifact_binding_invalid")
    agent = artifacts.get("agent_utilities")
    if (
        not isinstance(agent, dict)
        or set(agent)
        != {"closure_sha256", "distribution_count", "files", "sha256", "version"}
        or agent.get("sha256") != manifest["agent_utilities_sha256"]
        or agent.get("closure_sha256") != manifest["distribution_closure_sha256"]
        or not _positive_int(agent.get("distribution_count"))
        or not _positive_int(agent.get("files"))
    ):
        _fail("exact_local_artifact_binding_invalid")
    g08 = gates["g08"]
    g09 = gates["g09"]
    g26 = gates["g26"]
    g30 = gates["g30"]
    g32 = gates["g32"]
    g34 = gates["g34"]
    g35 = gates["g35"]
    if (
        g08 != {"case_count": 8, "cases": list(G08_CASES), "status": "pass"}
        or g09 != {"case_count": 2, "cases": list(G09_CASES), "status": "pass"}
        or g26.get("initial_tool_count") != 11
        or g26.get("injection_denied") is not True
        or g26.get("poisoned_feedback_denied") is not True
        or g30.get("family_count") != 13
        or g30.get("modality_count") != 14
        or g30.get("runtime_dependency_duplication") is not False
        or any(
            g32.get(field) is not True
            for field in (
                "atomic_concurrent_activation",
                "content_addressed_generation",
                "current_generation_doctor",
                "distribution_ownership",
                "path_free_diagnostics",
                "special_file_rejected",
                "stale_generation_rejected",
                "tamper_rejected",
            )
        )
        or not isinstance(g34.get("scenarios"), list)
        or len(g34["scenarios"]) != 6
        or not isinstance(g35, dict)
        or set(g35) != {"case_count", "cases", "source_cases", "status"}
        or g35.get("case_count") != 8
        or g35.get("cases") != list(G35_CASES)
        or g35.get("status") != "pass"
        or not _valid_pytest_summary(g35.get("source_cases"))
    ):
        _fail("exact_local_gate_detail_invalid")
    return {
        "exactLocal": {
            "gates": 7,
            "optimizer_families": 13,
            "optimizer_modalities": 14,
        },
        "workItemAgentBus": {"work_item_cases": 8, "agent_bus_cases": 2},
        "permissionGovernance": {"cases": 8},
    }


def _sign_closure(unsigned: dict[str, Any]) -> dict[str, Any]:
    _assert_path_free_evidence(unsigned)
    subject_digest = "sha256:" + hashlib.sha256(_canonical(unsigned)).hexdigest()
    try:
        signature = _validate_signature_shape(
            _external_json(SIGNER_ENV, _canonical(unsigned))
        )
    except ReleaseError as exc:
        raise ClosureError("closure_signing_failed") from exc
    if signature["subjectDigest"] != subject_digest:
        _fail("closure_signature_invalid")
    signed = {**unsigned, "signature": signature}
    try:
        response = _external_json(VERIFIER_ENV, _canonical(signed))
    except ReleaseError as exc:
        raise ClosureError("closure_verification_failed") from exc
    if response != {
        "verified": True,
        "subjectDigest": subject_digest,
        "keyId": signature["keyId"],
    }:
        _fail("closure_verification_failed")
    return signed


def bind_closure(
    *,
    release_id: str,
    spec_path: Path,
    promotion_evidence_path: Path,
    source_root: Path,
    campaign_manifest_path: Path,
    fault_restart_path: Path,
    protocol_authorization_path: Path,
    performance_path: Path,
    multimodal_path: Path,
    knowledge_batch_path: Path,
    reasoning_repair_path: Path,
    exact_local_path: Path,
) -> dict[str, Any]:
    """Validate and cross-bind every current live exact-artifact campaign."""

    if RELEASE_ID.fullmatch(release_id) is None:
        _fail("release_id_invalid")
    try:
        verified_promotion = verify_evidence_file(
            spec_path=spec_path,
            release_id=release_id,
            evidence_path=promotion_evidence_path,
        )
        expected_manifest = generate_manifest(
            release_id=release_id,
            spec_path=spec_path,
            promotion_evidence_path=promotion_evidence_path,
            source_root=source_root,
        )
    except (ReleaseError, ManifestError) as exc:
        raise ClosureError("promotion_binding_invalid") from exc
    manifest, manifest_digest = _read_json(campaign_manifest_path)
    validate_manifest(manifest)
    if manifest != expected_manifest:
        _fail("campaign_manifest_binding_invalid")
    fault, fault_digest = _read_json(fault_restart_path)
    protocol, protocol_digest = _read_json(protocol_authorization_path)
    performance, performance_digest = _read_json(performance_path)
    multimodal, multimodal_digest = _read_json(multimodal_path)
    knowledge, knowledge_digest = _read_json(knowledge_batch_path)
    reasoning, reasoning_digest = _read_json(reasoning_repair_path)
    exact_local, exact_local_digest = _read_json(exact_local_path)
    fault_summary = _validate_fault_restart(fault, manifest["engine_sha256"])
    protocol_summary = _validate_protocol_authorization(
        protocol, manifest["engine_sha256"]
    )
    performance_summary = _validate_performance(performance, manifest["engine_sha256"])
    multimodal_summary = _validate_multimodal(
        multimodal, manifest["engine_sha256"], performance_digest
    )
    knowledge_summary = _validate_knowledge_batch(knowledge, manifest["engine_sha256"])
    reasoning_summary = _validate_reasoning_repair(reasoning, manifest["engine_sha256"])
    exact_local_summary = _validate_exact_local(exact_local, manifest, manifest_digest)
    promotion_digest = manifest["promotion_evidence_sha256"]
    if (
        promotion_digest != manifest["promotion_evidence_sha256"]
        or verified_promotion.get("specDigest")
        != f"sha256:{manifest['release_spec_sha256']}"
    ):
        _fail("promotion_binding_invalid")
    unsigned = {
        "apiVersion": "graphos.io/v1",
        "kind": "ExactArtifactClosureEvidence",
        "schemaVersion": SCHEMA_VERSION,
        "releaseId": release_id,
        "status": "passed",
        "privacySafe": True,
        "release": {
            "promotionEvidenceSha256": f"sha256:{promotion_digest}",
            "releaseSpecSha256": f"sha256:{manifest['release_spec_sha256']}",
            "campaignManifestSha256": f"sha256:{manifest_digest}",
            "agentUtilitiesSha256": f"sha256:{manifest['agent_utilities_sha256']}",
            "distributionClosureSha256": f"sha256:{manifest['distribution_closure_sha256']}",
            "releasePythonSha256": f"sha256:{manifest['release_python_sha256']}",
            "graphosSha256": f"sha256:{manifest['graphos_sha256']}",
            "engineSha256": f"sha256:{manifest['engine_sha256']}",
            "harnessSha256": f"sha256:{manifest['harness_sha256']}",
            "testCatalogSha256": f"sha256:{manifest['test_catalog_sha256']}",
        },
        "campaigns": {
            "faultRestart": {
                "evidenceSha256": f"sha256:{fault_digest}",
                **fault_summary,
            },
            "protocolAuthorization": {
                "evidenceSha256": f"sha256:{protocol_digest}",
                **protocol_summary,
            },
            "workItemAgentBus": {
                "evidenceSha256": f"sha256:{exact_local_digest}",
                **exact_local_summary["workItemAgentBus"],
            },
            "performance": {
                "evidenceSha256": f"sha256:{performance_digest}",
                **performance_summary,
            },
            "multimodal": {
                "evidenceSha256": f"sha256:{multimodal_digest}",
                "performanceEvidenceSha256": f"sha256:{performance_digest}",
                **multimodal_summary,
            },
            "knowledgeBatch": {
                "evidenceSha256": f"sha256:{knowledge_digest}",
                **knowledge_summary,
            },
            "reasoningRepair": {
                "evidenceSha256": f"sha256:{reasoning_digest}",
                **reasoning_summary,
            },
            "exactLocal": {
                "evidenceSha256": f"sha256:{exact_local_digest}",
                "campaignManifestSha256": f"sha256:{manifest_digest}",
                **exact_local_summary["exactLocal"],
            },
            "permissionGovernance": {
                "evidenceSha256": f"sha256:{exact_local_digest}",
                **exact_local_summary["permissionGovernance"],
            },
        },
        "gates": {gate: "passed" for gate in GATES},
    }
    return _sign_closure(unsigned)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bind-exact-local-release-evidence",
        description="Validate and sign the complete local exact-artifact closure.",
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--promotion-evidence", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument("--fault-restart-evidence", required=True, type=Path)
    parser.add_argument("--protocol-authorization-evidence", required=True, type=Path)
    parser.add_argument("--performance-evidence", required=True, type=Path)
    parser.add_argument("--multimodal-evidence", required=True, type=Path)
    parser.add_argument("--knowledge-batch-evidence", required=True, type=Path)
    parser.add_argument("--reasoning-repair-evidence", required=True, type=Path)
    parser.add_argument("--exact-local-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = bind_closure(
            release_id=args.release_id,
            spec_path=args.spec,
            promotion_evidence_path=args.promotion_evidence,
            source_root=args.source_root,
            campaign_manifest_path=args.campaign_manifest,
            fault_restart_path=args.fault_restart_evidence,
            protocol_authorization_path=args.protocol_authorization_evidence,
            performance_path=args.performance_evidence,
            multimodal_path=args.multimodal_evidence,
            knowledge_batch_path=args.knowledge_batch_evidence,
            reasoning_repair_path=args.reasoning_repair_evidence,
            exact_local_path=args.exact_local_evidence,
        )
        _write_new_private(args.output, evidence)
    except ClosureError as exc:
        print(f"closure_status=rejected error_code={exc.code}", file=sys.stderr)
        return 1
    except (ManifestError, ReleaseError):
        print(
            "closure_status=rejected error_code=input-verification-failed",
            file=sys.stderr,
        )
        return 1
    except Exception:
        print("closure_status=rejected error_code=internal-error", file=sys.stderr)
        return 1
    print("closure_status=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
