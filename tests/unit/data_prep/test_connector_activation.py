"""Focused NE-113 activation-binding fixtures.

These fixtures exercise the adapter's bounded pre-commit contract.  They do
not contact a graph engine; the native call is replaced only at the existing
``ingest_envelope`` seam for the atomic-denial and redaction cases.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_utilities.data_prep import (
    ActivationAdmissionAdapter,
    ActivationArtifact,
    ActivationBinding,
    ActivationState,
    activation_binding_digest,
    begin_activation_rotation,
    bind_activation,
    commit_activation_rotation,
    initial_activation,
    rollback_activation,
)
from agent_utilities.knowledge_graph.ingestion.change_envelope import ChangeEnvelope


_DIGESTS = {
    "mapping": "sha256:" + "1" * 64,
    "shacl": "sha256:" + "2" * 64,
    "icv": "sha256:" + "3" * 64,
}


def _binding(**overrides: object) -> ActivationBinding:
    values: dict[str, object] = {
        "activation_version": "connector-activation.v1",
        "connector": "connector:gitlab-api",
        "connector_version": "1",
        "tenant": "tenant:alpha",
        "target_graph": "graph:alpha",
        "binding_ref": "activation:gitlab:v1",
        "mapping": ActivationArtifact(
            kind="mapping", ref="mapping:gitlab:v1", digest=_DIGESTS["mapping"]
        ),
        "shacl": ActivationArtifact(
            kind="shacl", ref="shape:gitlab:v1", digest=_DIGESTS["shacl"]
        ),
        "icv": ActivationArtifact(
            kind="icv", ref="icv:gitlab:v1", digest=_DIGESTS["icv"]
        ),
    }
    values.update(overrides)
    values["binding_digest"] = activation_binding_digest(values)
    return ActivationBinding.model_validate(values)


def _envelope(binding: ActivationBinding) -> ChangeEnvelope:
    artifacts = {
        kind: getattr(binding, kind).model_dump(mode="json")
        for kind in ("mapping", "shacl", "icv")
    }
    return ChangeEnvelope(
        connector=binding.connector,
        tenant=binding.tenant,
        schema_version=binding.connector_version,
        source_object_id="issue:1",
        source_version="version:1",
        typed_payload={"id": "issue:1", "type": "Issue", "title": "bounded"},
        provenance={
            "connector_preparation": {
                "contract_version": "connector-prep.v1",
                "artifacts": artifacts,
            }
        },
    )


def _session(binding: ActivationBinding) -> SimpleNamespace:
    return SimpleNamespace(tenant=binding.tenant, graph=binding.target_graph)


def test_binding_digest_is_deterministic_and_covers_governance_artifacts() -> None:
    binding = _binding()
    assert activation_binding_digest(binding) == binding.binding_digest

    changed = _binding(
        mapping=ActivationArtifact(
            kind="mapping", ref="mapping:gitlab:v2", digest=_DIGESTS["mapping"]
        )
    )
    assert changed.binding_digest != binding.binding_digest


def test_rotation_commit_and_rollback_are_pure_versioned_transitions() -> None:
    active = _binding()
    candidate = _binding(
        binding_ref="activation:gitlab:v2",
        mapping=ActivationArtifact(
            kind="mapping", ref="mapping:gitlab:v2", digest=_DIGESTS["mapping"]
        ),
    )
    initial = initial_activation(active)
    rotating = begin_activation_rotation(initial, candidate, "rotation:2026-01")
    assert rotating.status == "rotating"
    assert rotating.active == active
    assert rotating.pending == candidate

    committed = commit_activation_rotation(rotating)
    assert committed.status == "active"
    assert committed.active == candidate
    assert committed.previous == active
    assert committed.generation == 2

    rolled_back = rollback_activation(committed)
    assert rolled_back.status == "rolled_back"
    assert rolled_back.active == active
    assert rolled_back.previous == candidate
    assert rolled_back.generation == 3


def test_validation_rejects_cross_tenant_and_wrong_graph_without_raw_values() -> None:
    binding = _binding()
    state = initial_activation(binding)
    bound, bound_report = bind_activation(_envelope(binding), binding)
    assert bound is not None
    assert bound_report.outcome == "accepted"

    wrong_tenant = replace(bound, tenant="tenant:other")
    report = ActivationAdmissionAdapter().validate(
        wrong_tenant,
        state,
        session=_session(binding),
    )
    assert report.outcome == "rejected"
    assert "tenant:other" not in report.model_dump_json()
    assert {finding.code for finding in report.findings} >= {"tenant_mismatch"}

    report = ActivationAdmissionAdapter().validate(
        bound,
        state,
        session=SimpleNamespace(tenant=binding.tenant, graph="graph:other"),
    )
    assert {finding.code for finding in report.findings} >= {"session_graph_mismatch"}
    assert "graph:other" not in report.model_dump_json()


def test_digest_substitution_is_rejected_with_a_stable_bounded_code() -> None:
    binding = _binding()
    state = initial_activation(binding)
    bound, _ = bind_activation(_envelope(binding), binding)
    assert bound is not None
    provenance = dict(bound.provenance)
    preparation = dict(provenance["connector_preparation"])
    artifacts = dict(preparation["artifacts"])
    mapping = dict(artifacts["mapping"])
    mapping["digest"] = "sha256:" + "f" * 64
    artifacts["mapping"] = mapping
    preparation["artifacts"] = artifacts
    provenance["connector_preparation"] = preparation
    substituted = replace(bound, provenance=provenance)

    report = ActivationAdmissionAdapter().validate(
        substituted,
        state,
        session=_session(binding),
    )
    assert report.outcome == "rejected"
    assert {finding.code for finding in report.findings} >= {"mapping_digest_mismatch"}
    assert "sha256:" + "f" * 64 not in report.model_dump_json()


def test_rejected_preflight_is_atomic_and_native_reason_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    binding = _binding()
    state = initial_activation(binding)
    bound, _ = bind_activation(_envelope(binding), binding)
    assert bound is not None
    denied = replace(bound, tenant="tenant:other")
    calls: list[object] = []

    monkeypatch.setattr(
        "agent_utilities.data_prep.connector_activation._verified_write_session",
        lambda: _session(binding),
    )
    monkeypatch.setattr(
        "agent_utilities.data_prep.connector_activation._native_ingest",
        lambda engine, envelope: calls.append(envelope)
        or {"status": "rejected", "reason": "raw-provider-secret"},
    )
    report = ActivationAdmissionAdapter().admit(object(), denied, state)
    assert report.outcome == "rejected"
    assert not calls
    assert "raw-provider-secret" not in report.model_dump_json()

    monkeypatch.setattr(
        "agent_utilities.data_prep.connector_activation._native_ingest",
        lambda engine, envelope: {
            "status": "rejected",
            "reason": "raw-provider-secret",
        },
    )
    report = ActivationAdmissionAdapter().admit(object(), bound, state)
    assert report.outcome == "rejected"
    assert report.findings[0].code == "engine_rejected"
    assert "raw-provider-secret" not in report.model_dump_json()


def test_rotating_state_denies_admission_until_commit() -> None:
    active = _binding()
    candidate = _binding(binding_ref="activation:gitlab:v2")
    state = begin_activation_rotation(
        initial_activation(active), candidate, "rotation:2026-02"
    )
    bound, _ = bind_activation(_envelope(active), active)
    assert bound is not None
    report = ActivationAdmissionAdapter().validate(
        bound,
        state,
        session=_session(active),
    )
    assert report.findings[0].code == "activation_rotating"

