"""Focused contracts for exact-artifact manifest generation and closure binding."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from jsonschema import Draft202012Validator

from scripts.release import exact_artifact_closure as closure
from scripts.release import exact_local_gates_manifest as generator

ROOT = Path(__file__).resolve().parents[3]


def _promotion(digest: str = "a") -> dict[str, Any]:
    return {
        "apiVersion": "graphos.io/v2",
        "kind": "ExactLocalReleaseEvidence",
        "releaseId": "release-test-v1",
        "status": "promoted",
        "errorCode": None,
        "privacySafe": True,
        "specDigest": f"sha256:{'1' * 64}",
        "nativeArtifacts": {
            "epistemic-graph-server": f"sha256:{digest * 64}",
            "epistemic-graph-numeric": f"sha256:{'9' * 64}",
        },
        "certificationArtifacts": {
            "agentUtilitiesSha256": "2" * 64,
            "agentUtilitiesFileCount": 100,
            "distributionClosureSha256": "3" * 64,
            "releasePythonSha256": "4" * 64,
            "graphosSha256": "5" * 64,
            "engineSha256": digest * 64,
        },
    }


def _manifest() -> dict[str, Any]:
    return generator.manifest_from_verified_promotion(
        _promotion(),
        release_id="release-test-v1",
        promotion_evidence_sha256="6" * 64,
        bindings={"harness_sha256": "7" * 64, "test_catalog_sha256": "8" * 64},
    )


def _schema(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "deploy" / "release" / name).read_text(encoding="utf-8"))


def test_generator_output_matches_strict_schema_and_current_catalog() -> None:
    manifest = _manifest()
    Draft202012Validator(_schema("exact-local-gates-manifest.schema.json")).validate(
        manifest
    )
    bindings = generator.source_bindings(ROOT)
    assert bindings["harness_sha256"] == generator._sha256_file(
        ROOT / "scripts" / "certification" / "exact_local_gates.py",
        limit=4 * 1024 * 1024,
    )
    assert set(bindings) == {"harness_sha256", "test_catalog_sha256"}


def test_generator_rejects_engine_and_manifest_drift() -> None:
    evidence = _promotion()
    evidence["nativeArtifacts"]["epistemic-graph-server"] = f"sha256:{'0' * 64}"
    with pytest.raises(
        generator.ManifestError, match="promotion_engine_binding_invalid"
    ):
        generator.manifest_from_verified_promotion(
            evidence,
            release_id="release-test-v1",
            promotion_evidence_sha256="6" * 64,
            bindings={"harness_sha256": "7" * 64, "test_catalog_sha256": "8" * 64},
        )
    with pytest.raises(generator.ManifestError, match="manifest_schema_invalid"):
        generator.validate_manifest({**_manifest(), "legacy": True})


def _fault(engine: str) -> dict[str, Any]:
    matrix = []
    for index in range(15):
        family = f"family-{index:02d}"
        domain = closure.MUTATION_DOMAINS[index % len(closure.MUTATION_DOMAINS)]
        for phase in closure.COMMIT_PHASES:
            expected = (
                "complete_effect" if phase == "after_commit_before_ack" else "no_effect"
            )
            matrix.append(
                {
                    "domain": domain,
                    "expected": expected,
                    "family": family,
                    "observed": expected,
                    "passed": True,
                    "phase": phase,
                    "recovery": "authoritative_restart_read",
                }
            )
    return {
        "binary": {"sha256": engine},
        "certification": "epistemic-graph-exact-fault-restart",
        "commit_phases": list(closure.COMMIT_PHASES),
        "g05": {
            "spatial_restart_lazy_open": {
                "expected_hits": 96,
                "final_result_digest": "a" * 64,
                "lazy_open_cycles": 2,
                "partial_state_observed": True,
                "passed": True,
            },
            "tenant_qualified_time_series": {
                "identical_local_series_ids": 2,
                "isolated_result_digests": ["b" * 64, "c" * 64],
                "passed": True,
                "restart_cycles": 3,
            },
        },
        "matrix": matrix,
        "mutation_domains": list(closure.MUTATION_DOMAINS),
        "schema_version": 1,
        "summary": {"matrix_cases": 60, "passed": 60, "status": "pass"},
    }


def _performance(engine: str) -> dict[str, Any]:
    passed = {"metric": {"passed": True}}
    coverage = {name: {"verified": True} for name in closure.G37_COVERAGE}
    coverage["modality"] = {
        "component_probes": list(closure.MODALITIES),
        "ingests_by_modality": {name: 1 for name in closure.MODALITIES},
        "native_query_samples_by_modality": {name: 1 for name in closure.MODALITIES},
        "index_growth_ratio_by_modality": {name: 1.0 for name in closure.MODALITIES},
        "results_verified": True,
    }
    coverage["hot_path_scenarios"] = {
        "scenario_families": 30,
        "ledger_rows": 54,
        "raw_results_validated": True,
        "exact_binary_subcommands": 30,
    }
    rows = {
        f"G37-HP-{index:03d}": {
            "passed": True,
            "threshold_results": passed,
        }
        for index in range(1, 55)
    }
    return {
        "schema_version": "1",
        "gate": "G-37",
        "status": "pass",
        "started_at_utc": "2026-01-01T00:00:00Z",
        "completed_at_utc": "2026-01-01T00:00:01Z",
        "exact_artifact": {
            "component": "epistemic-graph-server",
            "sha256": engine,
            "staged_copy_verified": True,
        },
        "client_artifact": {},
        "authority": {},
        "deployment_profile": {},
        "dataset": {},
        "threshold_manifest_sha256": "a" * 64,
        "scenario_contract": {"scenario_families": 30, "ledger_rows": 54},
        "scenario_evidence_binding": {},
        "hardware_class": {},
        "coverage": coverage,
        "measurements": {},
        "metric_results": passed,
        "complexity_results": passed,
        "scenario_family_evidence": {f"scenario-{index}": {} for index in range(30)},
        "hot_path_row_evidence": rows,
        "failures": [],
    }


def _multimodal(engine: str, performance_digest: str) -> dict[str, Any]:
    dimensions = [f"dimension-{index}" for index in range(12)]
    matrix = [
        {
            "modality": modality,
            "component_tck_pass": 12,
            "component_tck_not_applicable": 0,
            "dimensions": {name: True for name in dimensions},
        }
        for modality in closure.MODALITIES
    ]
    faults = [
        {
            "expected": "complete_effect"
            if phase == "after_commit_before_ack"
            else "no_effect",
            "modality": modality,
            "observed": "complete_effect"
            if phase == "after_commit_before_ack"
            else "no_effect",
            "passed": True,
            "phase": phase,
        }
        for modality in closure.MODALITIES
        for phase in closure.COMMIT_PHASES
    ]
    return {
        "binary": {"sha256": engine, "sealed_copy_verified": True},
        "certification": "epistemic-graph-exact-multimodal",
        "exact_behavior_dimensions": dimensions,
        "fault_matrix": faults,
        "matrix": matrix,
        "modalities": list(closure.MODALITIES),
        "performance": {
            "gate": "G-37",
            "report_sha256": performance_digest,
            "same_artifact_verified": True,
            "status": "pass",
        },
        "schema_version": 1,
        "summary": {
            "dimensions_per_modality": 12,
            "fault_cases": 16,
            "modalities": 4,
            "passed": 4,
            "status": "pass",
        },
    }


def _protocol_authorization(engine: str) -> dict[str, Any]:
    return {
        "binary": {"sha256": engine},
        "certification": "epistemic-graph-exact-protocol-authorization",
        "cross_tenant": {
            "graph_row_hidden": True,
            "kv_namespace_hidden": True,
            "time_series_namespace_hidden": True,
        },
        "data_path_matrix": [
            {"denial_observed": True, "path": path, "tenant_relation": "same"}
            for path in closure.DATA_PATHS
        ],
        "data_paths": list(closure.DATA_PATHS),
        "schema_version": 1,
        "summary": {
            "data_path_cases": 14,
            "protocol_cases": 10,
            "status": "pass",
        },
        "wire_matrix": [
            {
                "auth_denial_observed": True,
                "feature": closure.WIRE_FEATURES[protocol],
                "protocol": protocol,
            }
            for protocol in closure.WIRE_PROTOCOLS
        ],
        "wire_protocols": list(closure.WIRE_PROTOCOLS),
    }


def _knowledge_batch(engine: str) -> dict[str, Any]:
    digest = "a" * 64
    return {
        "binary": {"sha256": engine},
        "certification": "epistemic-graph-exact-knowledge-batch",
        "families": [
            {
                "arrow_schema_sha256": digest,
                "backpressure_seconds": 0.02,
                "cancel_resume": True,
                "family": family,
                "page_payload_sha256": [digest],
                "pages": 1,
                "rows": 3,
                "tamper_denied": True,
            }
            for family in closure.KNOWLEDGE_FAMILIES
        ],
        "requirements": list(closure.KNOWLEDGE_REQUIREMENTS),
        "schema_version": 1,
        "snapshot_matrix": [
            {
                "family": family,
                "outcome": (
                    "immutable_source_resumed"
                    if family == "job"
                    else "changed_snapshot_denied"
                ),
                "passed": True,
            }
            for family in closure.KNOWLEDGE_FAMILIES
        ],
        "summary": {
            "families": 7,
            "requirements": 7,
            "snapshot_cases": 7,
            "status": "pass",
        },
    }


def _reasoning_repair(engine: str) -> dict[str, Any]:
    digest = "a" * 64
    matrix = {
        "projection_lag": {
            "converged": True,
            "immediate_prior_watermark": True,
            "polls": 1,
            "watermark_advanced": True,
        },
        "restart": {
            "durable_projection_equal": True,
            "polls": 1,
            "projection_sha256": digest,
        },
        "contradiction": {"classification_sha256": digest, "undecided": 2},
        "retraction": {
            "belief_flipped": True,
            "durable_replay_equal": True,
            "result_sha256": digest,
        },
        "valid_transaction_time_change": {
            "changed": 1,
            "claim_flipped": True,
            "result_sha256": digest,
        },
        "causal_recomputation": {
            "deterministic": True,
            "result_sha256": digest,
            "variables": 3,
        },
        "assumptions": {
            "calibrated_intervals": 3,
            "observe_intervene_distinct": True,
            "result_sha256": digest,
        },
        "counterexamples": {
            "causal": {
                "deterministic": True,
                "result_sha256": digest,
                "variables": 3,
            },
            "minimal_flip_cardinality": 1,
            "status_sha256": digest,
        },
        "repair": {
            "dependencies": 1,
            "fence_epoch_positive": True,
            "fresh": True,
            "polls": 1,
            "post_retraction_projection_polls": 1,
            "post_retraction_projection_sha256": digest,
            "stale_fence_denied": True,
        },
    }
    return {
        "binary": {"sha256": engine},
        "cases": list(closure.REASONING_CASES),
        "certification": "epistemic-graph-exact-reasoning-repair",
        "matrix": matrix,
        "schema_version": 1,
        "summary": {
            "cases": 9,
            "initial_projection_polls": 1,
            "passed": 9,
            "status": "pass",
        },
    }


def _exact_local(manifest: dict[str, Any], manifest_digest: str) -> dict[str, Any]:
    g32 = {
        name: True
        for name in (
            "atomic_concurrent_activation",
            "content_addressed_generation",
            "current_generation_doctor",
            "distribution_ownership",
            "path_free_diagnostics",
            "special_file_rejected",
            "stale_generation_rejected",
            "tamper_rejected",
        )
    }
    unsigned = {
        "artifacts": {
            "agent_utilities": {
                "closure_sha256": manifest["distribution_closure_sha256"],
                "distribution_count": 10,
                "files": 100,
                "sha256": manifest["agent_utilities_sha256"],
                "version": "1.0.0",
            },
            "epistemic_graph": {"sha256": manifest["engine_sha256"]},
            "graphos": {"sha256": manifest["graphos_sha256"]},
            "harness": {"sha256": manifest["harness_sha256"]},
            "promotion_evidence": {"sha256": manifest["promotion_evidence_sha256"]},
            "release_manifest": {"sha256": manifest_digest},
            "release_python": {"sha256": manifest["release_python_sha256"]},
            "release_spec": {"sha256": manifest["release_spec_sha256"]},
            "test_catalog": {"sha256": manifest["test_catalog_sha256"]},
        },
        "certification": "agent-utilities-exact-local-gates",
        "gates": {
            "g08": {
                "case_count": 8,
                "cases": list(closure.G08_CASES),
                "status": "pass",
            },
            "g09": {
                "case_count": 2,
                "cases": list(closure.G09_CASES),
                "status": "pass",
            },
            "g26": {
                "initial_tool_count": 11,
                "injection_denied": True,
                "poisoned_feedback_denied": True,
                "status": "pass",
            },
            "g30": {
                "family_count": 13,
                "modality_count": 14,
                "runtime_dependency_duplication": False,
                "status": "pass",
            },
            "g32": {**g32, "status": "pass"},
            "g34": {"scenarios": [f"case-{i}" for i in range(6)], "status": "pass"},
            "g35": {
                "case_count": 8,
                "cases": list(closure.G35_CASES),
                "source_cases": {
                    "collected": 2,
                    "passed": 2,
                    "required": 2,
                    "required_cases": ["case_a", "case_b"],
                    "scenario_evidence": {"case_a": "pass", "case_b": "pass"},
                    "skipped": 0,
                },
                "status": "pass",
            },
        },
        "release_id": manifest["release_id"],
        "schema_version": 1,
        "status": "pass",
    }
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    signature = private.sign(closure._canonical(unsigned))
    return {
        **unsigned,
        "signature": {
            "algorithm": "ed25519",
            "public_key": base64.urlsafe_b64encode(public).decode().rstrip("="),
            "signature": base64.urlsafe_b64encode(signature).decode().rstrip("="),
            "signer_id": "signer:test",
        },
    }


def test_campaign_validators_reject_cross_artifact_and_digest_drift() -> None:
    manifest = _manifest()
    assert (
        closure._validate_fault_restart(
            _fault(manifest["engine_sha256"]), manifest["engine_sha256"]
        )["matrix_cases"]
        == 60
    )
    performance = _performance(manifest["engine_sha256"])
    assert (
        closure._validate_performance(performance, manifest["engine_sha256"])[
            "ledger_rows"
        ]
        == 54
    )
    performance_digest = hashlib.sha256(json.dumps(performance).encode()).hexdigest()
    multimodal = _multimodal(manifest["engine_sha256"], performance_digest)
    assert (
        closure._validate_multimodal(
            multimodal, manifest["engine_sha256"], performance_digest
        )["modalities"]
        == 4
    )
    multimodal["performance"]["report_sha256"] = "0" * 64
    with pytest.raises(closure.ClosureError, match="multimodal_not_passing"):
        closure._validate_multimodal(
            multimodal, manifest["engine_sha256"], performance_digest
        )


def test_protocol_knowledge_and_reasoning_campaigns_bind_the_exact_engine() -> None:
    engine = _manifest()["engine_sha256"]
    assert closure._validate_protocol_authorization(
        _protocol_authorization(engine), engine
    ) == {"data_path_cases": 14, "protocol_cases": 10}
    assert closure._validate_knowledge_batch(_knowledge_batch(engine), engine) == {
        "families": 7,
        "requirements": 7,
        "snapshot_cases": 7,
    }
    assert closure._validate_reasoning_repair(_reasoning_repair(engine), engine) == {
        "cases": 9
    }
    drift = _knowledge_batch(engine)
    drift["binary"]["sha256"] = "0" * 64
    with pytest.raises(closure.ClosureError, match="knowledge_batch_not_passing"):
        closure._validate_knowledge_batch(drift, engine)


def test_exact_local_signature_and_manifest_binding_are_required() -> None:
    manifest = _manifest()
    manifest_digest = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    evidence = _exact_local(manifest, manifest_digest)
    assert closure._validate_exact_local(evidence, manifest, manifest_digest) == {
        "exactLocal": {
            "gates": 7,
            "optimizer_families": 13,
            "optimizer_modalities": 14,
        },
        "workItemAgentBus": {"work_item_cases": 8, "agent_bus_cases": 2},
        "permissionGovernance": {"cases": 8},
    }
    evidence["artifacts"]["graphos"]["sha256"] = "0" * 64
    with pytest.raises(closure.ClosureError, match="exact_local_signature_invalid"):
        closure._validate_exact_local(evidence, manifest, manifest_digest)


def test_signed_closure_is_schema_valid_and_uses_external_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_id = f"key:{'a' * 64}"

    def external(environment: str, payload: bytes) -> dict[str, Any]:
        value = json.loads(payload)
        if environment == closure.SIGNER_ENV:
            return {
                "algorithm": "ed25519",
                "keyId": key_id,
                "signature": "A" * 43,
                "subjectDigest": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
        assert environment == closure.VERIFIER_ENV
        return {
            "verified": True,
            "subjectDigest": value["signature"]["subjectDigest"],
            "keyId": key_id,
        }

    monkeypatch.setattr(closure, "_external_json", external)
    unsigned = {
        "apiVersion": "graphos.io/v1",
        "kind": "ExactArtifactClosureEvidence",
        "schemaVersion": 1,
        "releaseId": "release-test-v1",
        "status": "passed",
        "privacySafe": True,
        "release": {
            name: f"sha256:{'a' * 64}"
            for name in (
                "promotionEvidenceSha256",
                "releaseSpecSha256",
                "campaignManifestSha256",
                "agentUtilitiesSha256",
                "distributionClosureSha256",
                "releasePythonSha256",
                "graphosSha256",
                "engineSha256",
                "harnessSha256",
                "testCatalogSha256",
            )
        },
        "campaigns": {
            "faultRestart": {
                "evidenceSha256": f"sha256:{'a' * 64}",
                "matrix_cases": 60,
                "mutation_families": 15,
            },
            "protocolAuthorization": {
                "evidenceSha256": f"sha256:{'a' * 64}",
                "data_path_cases": 14,
                "protocol_cases": 10,
            },
            "workItemAgentBus": {
                "evidenceSha256": f"sha256:{'a' * 64}",
                "work_item_cases": 8,
                "agent_bus_cases": 2,
            },
            "performance": {
                "evidenceSha256": f"sha256:{'a' * 64}",
                "scenario_families": 30,
                "ledger_rows": 54,
            },
            "multimodal": {
                "evidenceSha256": f"sha256:{'a' * 64}",
                "performanceEvidenceSha256": f"sha256:{'a' * 64}",
                "modalities": 4,
                "behavior_dimensions": 12,
                "fault_cases": 16,
            },
            "knowledgeBatch": {
                "evidenceSha256": f"sha256:{'a' * 64}",
                "families": 7,
                "requirements": 7,
                "snapshot_cases": 7,
            },
            "reasoningRepair": {
                "evidenceSha256": f"sha256:{'a' * 64}",
                "cases": 9,
            },
            "exactLocal": {
                "evidenceSha256": f"sha256:{'a' * 64}",
                "campaignManifestSha256": f"sha256:{'a' * 64}",
                "gates": 7,
                "optimizer_families": 13,
                "optimizer_modalities": 14,
            },
            "permissionGovernance": {
                "evidenceSha256": f"sha256:{'a' * 64}",
                "cases": 8,
            },
        },
        "gates": {gate: "passed" for gate in closure.GATES},
    }
    signed = closure._sign_closure(unsigned)
    Draft202012Validator(
        _schema("exact-artifact-closure-evidence.schema.json")
    ).validate(signed)
    assert signed["gates"]["G-01"] == "passed"
    assert signed["campaigns"]["protocolAuthorization"]["evidenceSha256"] == (
        f"sha256:{'a' * 64}"
    )
    assert signed["signature"]["keyId"] == key_id
