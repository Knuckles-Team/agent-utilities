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


def test_schema_loader_is_catalog_bounded() -> None:
    schema = load_schema("mutation_batch")
    assert schema["$id"] == "urn:epistemic-operations:v1:mutation-batch"
    with pytest.raises(KeyError, match="unknown Epistemic Operations schema"):
        load_schema("unregistered")


def test_work_item_and_artifact_identity_contracts_are_current_only() -> None:
    work_item = load_schema("work_item")
    assert work_item["properties"]["state"]["enum"] == [
        "submitted",
        "ready",
        "leased",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "dead_letter",
    ]
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
