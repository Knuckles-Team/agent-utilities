"""RF-021 transport-neutral MCP catalog authority contract tests."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from agent_utilities.mcp.catalog_reconciliation import (
    CatalogContractError,
    CatalogIdentity,
    CatalogRefreshError,
    CatalogRefreshRequest,
    CatalogRefreshResult,
    CatalogSessionResumeError,
    CatalogSessionResumeRequest,
    CatalogSessionResumeResult,
    ChildCatalogCandidate,
    McpCatalogReconciler,
)

_SCOPE = "a" * 64
_CONFIG = "config:1"


def _child(*, generation: int = 1, description: str = "one") -> ChildCatalogCandidate:
    return ChildCatalogCandidate.build(
        server_name="alpha-mcp",
        child_connection_generation=generation,
        tools=[
            {
                "name": "a__read",
                "originalName": "read",
                "description": description,
                "inputSchema": {"type": "object"},
                "annotations": {"readOnlyHint": True},
            }
        ],
        resources=[{"name": "doc", "uri": "mcp+alpha://doc"}],
        resource_templates=[
            {
                "name": "item",
                "uriTemplate": "mcp+alpha://items/{id}",
                "arguments": {"id": {"type": "string"}},
            }
        ],
        prompts=[{"name": "a__review", "originalName": "review"}],
    )


def _authority() -> McpCatalogReconciler:
    return McpCatalogReconciler(release_id="2.5.0", served_instance_id="graph-os:test")


def _published():
    authority = _authority()
    candidate = authority.candidate(
        authorization_scope_digest=_SCOPE,
        config_revision=_CONFIG,
        children=[_child()],
    )
    snapshot, changed = authority.publish(candidate, affected_session_ids=["session:1"])
    return authority, snapshot, changed


def _resume_request(
    identity: CatalogIdentity, **updates
) -> CatalogSessionResumeRequest:
    payload = {
        "session_id": "session:1",
        "previous_served_instance_id": identity.served_instance_id,
        "release_id": identity.release_id,
        "config_revision": identity.config_revision,
        "catalog_generation": identity.catalog_generation,
        "snapshot_digest": identity.snapshot_digest,
        "child_connection_generation": identity.child_connection_generation,
        "authorization_scope_digest": identity.authorization_scope_digest,
        "resume_token_digest": "b" * 64,
        "deadline_ms": 1_000,
    }
    payload.update(updates)
    return CatalogSessionResumeRequest.model_validate(payload)


def test_wire_contracts_have_exact_fields_and_reject_extras() -> None:
    assert set(CatalogRefreshRequest.model_fields) == {
        "request_id",
        "expected_config_revision",
        "expected_catalog_generation",
        "expected_snapshot_digest",
        "deadline_ms",
    }
    assert set(CatalogRefreshResult.model_fields) == {
        "request_id",
        "served_instance_id",
        "release_id",
        "config_revision",
        "catalog_generation",
        "snapshot_digest",
        "changed",
        "pending_list_change_generation",
        "reingestion_state",
        "reconciliation_receipt_digest",
    }
    assert set(CatalogRefreshError.model_fields) == {
        "request_id",
        "code",
        "retryable",
        "catalog_generation",
        "snapshot_digest",
        "details_digest",
    }
    assert set(CatalogSessionResumeRequest.model_fields) == {
        "session_id",
        "previous_served_instance_id",
        "release_id",
        "config_revision",
        "catalog_generation",
        "snapshot_digest",
        "child_connection_generation",
        "authorization_scope_digest",
        "resume_token_digest",
        "deadline_ms",
    }
    assert set(CatalogSessionResumeResult.model_fields) == {
        "session_id",
        "served_instance_id",
        "release_id",
        "config_revision",
        "catalog_generation",
        "snapshot_digest",
        "child_connection_generation",
        "authorization_scope_digest",
        "resume_state",
    }
    assert set(CatalogSessionResumeError.model_fields) == {
        "session_id",
        "code",
        "retryable",
        "required_action",
        "catalog_generation",
        "snapshot_digest",
    }
    with pytest.raises(ValidationError):
        CatalogRefreshRequest.model_validate(
            {
                "request_id": "request:1",
                "expected_config_revision": _CONFIG,
                "expected_catalog_generation": 0,
                "expected_snapshot_digest": "0" * 64,
                "deadline_ms": 10,
                "server_name": "duplicate-child-selector",
            }
        )


def test_four_family_snapshot_is_immutable_and_content_addressed() -> None:
    authority, snapshot, changed = _published()
    assert changed is True
    assert snapshot.identity.catalog_generation == 1
    assert snapshot.children[0].server_name == "alpha-mcp"
    assert [
        len(getattr(snapshot.children[0], family))
        for family in (
            "tools",
            "resources",
            "resource_templates",
            "prompts",
        )
    ] == [1, 1, 1, 1]

    projected = snapshot.children[0].tools[0].as_dict()
    projected["description"] = "mutated copy"
    assert snapshot.children[0].tools[0].as_dict()["description"] == "one"
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.children[0].server_name = "changed"  # type: ignore[misc]

    same = authority.candidate(
        authorization_scope_digest=_SCOPE,
        config_revision=_CONFIG,
        children=[_child()],
    )
    same_snapshot, same_changed = authority.publish(
        same, affected_session_ids=["session:1"]
    )
    assert same_changed is False
    assert same_snapshot.identity == snapshot.identity


def test_changed_content_advances_generation_once() -> None:
    authority, first, _ = _published()
    second = authority.candidate(
        authorization_scope_digest=_SCOPE,
        config_revision=_CONFIG,
        children=[_child(description="two")],
    )
    published, changed = authority.publish(second, affected_session_ids=["session:1"])
    assert changed is True
    assert (
        published.identity.catalog_generation == first.identity.catalog_generation + 1
    )
    assert published.identity.snapshot_digest != first.identity.snapshot_digest


def test_pending_notification_is_monotonic_and_acknowledges_exact_watermark() -> None:
    authority, first, _ = _published()
    assert authority.pending_generation("session:1") == 1
    second = authority.candidate(
        authorization_scope_digest=_SCOPE,
        config_revision=_CONFIG,
        children=[_child(description="two")],
    )
    published, _ = authority.publish(second, affected_session_ids=["session:1"])
    authority.acknowledge_pending("session:1", first.identity.catalog_generation)
    assert (
        authority.pending_generation("session:1")
        == published.identity.catalog_generation
    )
    authority.acknowledge_pending("session:1", published.identity.catalog_generation)
    assert authority.pending_generation("session:1") is None


def test_scope_partitions_do_not_grant_cross_scope_dispatch() -> None:
    authority, snapshot, _ = _published()
    resolved_child, resolved_tool = authority.resolve_tool(
        public_name="a__read",
        expected_generation=snapshot.identity.catalog_generation,
        expected_snapshot_digest=snapshot.identity.snapshot_digest,
        authorization_scope_digest=_SCOPE,
        config_revision=_CONFIG,
    )
    assert resolved_child.server_name == "alpha-mcp"
    assert resolved_tool.key == "a__read"

    other_scope = "c" * 64
    empty = authority.current(other_scope, config_revision=_CONFIG)
    with pytest.raises(CatalogContractError, match="not visible"):
        authority.resolve_tool(
            public_name="a__read",
            expected_generation=empty.identity.catalog_generation,
            expected_snapshot_digest=empty.identity.snapshot_digest,
            authorization_scope_digest=other_scope,
            config_revision=_CONFIG,
        )


def test_refresh_optimistic_identity_fails_closed() -> None:
    authority, snapshot, _ = _published()
    request = CatalogRefreshRequest(
        request_id="request:1",
        expected_config_revision=_CONFIG,
        expected_catalog_generation=snapshot.identity.catalog_generation,
        expected_snapshot_digest="0" * 64,
        deadline_ms=1_000,
    )
    with pytest.raises(CatalogContractError) as caught:
        authority.validate_refresh(request, snapshot)
    assert caught.value.code == "catalog-digest-mismatch"


def test_session_resume_requires_bound_token_and_exact_identity() -> None:
    authority, snapshot, _ = _published()
    authority.bind_session("session:1", snapshot.identity, resume_token_digest="b" * 64)
    resumed = authority.resume_session(
        _resume_request(snapshot.identity),
        authorization_scope_digest=_SCOPE,
        config_revision=_CONFIG,
        cohort_homogeneous=True,
    )
    assert resumed.resume_state == "resumed"

    with pytest.raises(CatalogContractError) as caught:
        authority.resume_session(
            _resume_request(snapshot.identity, resume_token_digest="d" * 64),
            authorization_scope_digest=_SCOPE,
            config_revision=_CONFIG,
            cohort_homogeneous=True,
        )
    assert caught.value.code == "session-resume-token-invalid"


def test_session_resume_requires_relist_after_child_generation_change() -> None:
    authority, first, _ = _published()
    authority.bind_session("session:1", first.identity, resume_token_digest="b" * 64)
    second = authority.candidate(
        authorization_scope_digest=_SCOPE,
        config_revision=_CONFIG,
        children=[_child(generation=2)],
    )
    current, _ = authority.publish(second, affected_session_ids=["session:1"])
    request = _resume_request(
        current.identity,
        child_connection_generation=first.identity.child_connection_generation,
    )
    result = authority.resume_session(
        request,
        authorization_scope_digest=_SCOPE,
        config_revision=_CONFIG,
        cohort_homogeneous=True,
    )
    assert result.resume_state == "relist-required"


def test_replica_cohort_divergence_fails_closed() -> None:
    _authority_instance, snapshot, _ = _published()
    peer = snapshot.identity.model_copy(update={"snapshot_digest": "f" * 64})
    with pytest.raises(CatalogContractError) as caught:
        McpCatalogReconciler.assert_homogeneous(snapshot.identity, [peer])
    assert caught.value.code == "replica-generation-divergent"
