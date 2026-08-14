from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent_utilities.protocols.epistemic_operations import (
    CATALOG_SHA256,
    ClaimWorkItemResult,
    EvidenceBundle,
    OperationResult,
    PlacementRoute,
    RequestContext,
    ResourceReservationRequest,
    load_catalog,
    load_schema,
)


def _context_payload() -> dict[str, Any]:
    return {
        "schema_version": "2",
        "request_id": "request:example",
        "subject_id": "subject:opaque",
        "tenant_id": "tenant:example",
        "agent_id": "agent:example",
        "scopes": ["graph:read"],
        "audience": "graph-service",
        "authentication_method": "workload_identity",
        "policy_version": "policy:current",
        "graph": "tenant:example:graph",
        "placement_epoch": 3,
        "trace_id": "trace:example",
        "issued_at_ms": 1000,
        "expires_at_ms": 2000,
    }


def test_catalog_contains_exact_current_schema_set() -> None:
    catalog = load_catalog()
    assert catalog["protocol"] == "epistemic-operations"
    assert catalog["version"] == "1"
    assert catalog["compatibility_policy"] == "current-only"
    assert len(CATALOG_SHA256) == 64
    assert [entry["name"] for entry in catalog["schemas"]] == [
        "request_context",
        "mutation_batch",
        "change_envelope",
        "work_item",
        "artifact",
        "knowledge_batch",
        "analytics_job",
        "trace_outcome",
        "placement_route",
        "claim_work_item",
        "evidence_bundle",
        "operation_result",
        "resource_reservation",
        "resource_reservation_status",
        "resource_host_update",
        # RMDD-28 added the development_lane schema, deliberately growing the
        # catalog 15 -> 16 (the handoff records the bump explicitly). This is an
        # exact-set assertion, so a legitimate addition has to be reflected here
        # rather than the assertion being loosened -- the exactness is the point.
        "development_lane",
    ]
    assert [entry["version"] for entry in catalog["schemas"]] == [
        "2",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        # development_lane (RMDD-28), v1
        "1",
    ]


def test_request_context_rejects_unknown_fields() -> None:
    payload = _context_payload()
    payload["endpoint"] = "https://example.invalid"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RequestContext.model_validate(payload)


def test_request_context_accepts_opaque_identity_only_shape() -> None:
    context = RequestContext.model_validate(_context_payload())
    assert context.subject_id == "subject:opaque"
    assert context.model_dump()["schema_version"] == "2"


def test_request_context_rejects_type_coercion() -> None:
    payload = _context_payload()
    payload["issued_at_ms"] = "1000"
    with pytest.raises(ValidationError, match="valid integer"):
        RequestContext.model_validate(payload)


def _resource_request_payload(*, profile_version: str = "1") -> dict[str, Any]:
    return {
        "schema_version": "1",
        "tenant_ref": "tenant:opaque",
        "work_item_id": "work:opaque",
        "owner_id": "owner:opaque",
        "fence": "1",
        "lease_epoch": 1,
        "fencing_token": 1,
        "attempt": 1,
        "reservation_id": "reservation:opaque",
        "input_fingerprint": "v1:" + "0" * 64,
        "profile_name": "rust-build",
        "profile_version": profile_version,
        "host_ref": "host:opaque",
        "requirement": {
            "cpu_weight": 1,
            "memory_mib": 1,
            "disk_mib": 1,
            "process_slots": 1,
        },
        "target_kind": "local",
        "target_alias": None,
        "repository_id": "repo:opaque",
        "branch": "main",
        "concurrency_key": "rust-build",
        "concurrency_limit": None,
        "repository_exclusive": False,
        "branch_exclusive": False,
        "required_labels": [],
        "anti_affinity": [],
        "fairness_group": "default",
        "fairness_cost": 1,
        "disk_low_watermark_mib": None,
        "disk_high_watermark_mib": None,
        "disk_policy_key": "rust-v1",
        "reserved_at_ms": 1,
        "expires_at_ms": 2,
        "idempotency_key": "invocation:opaque",
        "now_ms": 1,
        "expected_host_revision": None,
        "expected_lifecycle_revision": None,
    }


def test_resource_projection_rejects_noncanonical_profile_version_and_duplicate_labels() -> None:
    with pytest.raises(ValidationError, match="profile_version"):
        ResourceReservationRequest.model_validate(
            _resource_request_payload(profile_version="01")
        )
    duplicate = _resource_request_payload()
    duplicate["required_labels"] = ["linux", "linux"]
    with pytest.raises(ValidationError, match="unique"):
        ResourceReservationRequest.model_validate(duplicate)


def test_schema_loader_is_catalog_bounded() -> None:
    schema = load_schema("mutation_batch")
    assert schema["$id"] == "urn:epistemic-operations:v1:mutation-batch"
    with pytest.raises(KeyError, match="unknown Epistemic Operations schema"):
        load_schema("unregistered")


def test_work_item_and_artifact_identity_contracts_are_current_only() -> None:
    work_item = load_schema("work_item")
    # W2.5 statechart migration: WorkItem.state is a validated state-id String,
    # NOT a closed wire enum. The 8-value lifecycle vocabulary is enforced at
    # runtime by the native WorkItem state machine (WORK_ITEM_DEF in the engine;
    # WorkItemStatus in agent_utilities.orchestration.work_item), so the shared
    # schema binds a non-empty string here and the generated Rust field is
    # `pub state: String` (no WorkItemState enum). See
    # reports/w2_5-statechart-migration-design.md section 2.5.
    state = work_item["properties"]["state"]
    assert state["type"] == "string"
    assert state["minLength"] == 1
    assert "enum" not in state
    artifact = load_schema("artifact")
    for field in (
        "occurrence_ids",
        "rendition_ids",
        "segment_ids",
        "feature_ids",
        "derivation_ids",
    ):
        assert field in artifact["required"]


def test_live_operation_projections_reject_ad_hoc_fields() -> None:
    route = {
        "schema_version": "1",
        "route_id": "route:opaque",
        "tenant_ref": "tenant:opaque",
        "partition_ref": "partition:opaque",
        "authoritative": True,
        "placed": True,
        "group": 3,
        "epoch": 7,
        "fencing_token": 3,
        "stale": False,
        "leader_ref": None,
    }
    assert PlacementRoute.model_validate(route).epoch == 7
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PlacementRoute.model_validate({**route, "endpoint": "https://example.invalid"})


def test_claim_evidence_and_operation_result_are_strict() -> None:
    claim = ClaimWorkItemResult(
        schema_version="1",
        claimed=False,
        reason="empty",
        work_item_id=None,
        kind=None,
        payload_ref=None,
        lease_holder_ref=None,
        lease_epoch=None,
        fencing_token=None,
        lease_expires_at_ms=None,
        attempt=None,
        max_attempts=None,
        tenant_in_flight=0,
        changed_work_item_ids=[],
    )
    assert claim.reason == "empty"
    evidence = EvidenceBundle(
        schema_version="1",
        bundle_id="bundle:opaque",
        resolved=False,
        answer_ref=None,
        claims=[],
        policy_exclusions=[],
        next_action_refs=[],
    )
    assert evidence.claims == []
    failure = OperationResult.model_validate(
        {
            "schema_version": "1",
            "operation_id": "operation:opaque",
            "status": "failed",
            "result_kind": None,
            "result_ref": None,
            "error": {
                "code": "operation_failed",
                "retryable": False,
                "correlation_id": "correlation:opaque",
                "detail_ref": None,
            },
            "redirect": None,
        }
    )
    assert failure.error is not None
