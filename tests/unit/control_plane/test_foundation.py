"""Focused NE-083 identity, scope, lifecycle, and release-contract fixtures."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from agent_utilities.control_plane.foundation import (
    FoundationControlPlane,
    FoundationLifecycleError,
    FoundationRepositoryError,
    GatewayConfigVersion,
    GatewayFeatureVersion,
    GatewayRouteVersion,
    GatewayUpstreamVersion,
    GatewayVersionRef,
    InMemoryFoundationRepository,
    LifecycleState,
    OrganizationIdentity,
    PrincipalIdentity,
    RetentionPolicy,
    TenantIdentity,
    TenantScope,
    TombstoneRecord,
    config_id_for,
    feature_id_for,
    organization_id_for,
    principal_id_for,
    record_digest_for,
    route_id_for,
    tenant_id_for,
    tombstone_id,
    upstream_id_for,
)


_ZERO = "sha256:" + "0" * 64


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finalize(model_type, digest_field: str, **fields):
    seed = model_type.model_construct(**fields, **{digest_field: _ZERO})
    payload = seed.model_dump()
    payload[digest_field] = record_digest_for(seed)
    return model_type.model_validate(payload)


def _lifecycle() -> LifecycleState:
    return LifecycleState(
        lifecycle_version="lifecycle.v1",
        status="active",
        retention_policy_ref="retention:default",
    )


def _organization() -> OrganizationIdentity:
    slug = "acme"
    return _finalize(
        OrganizationIdentity,
        "identity_digest",
        identity_version="organization-identity.v1",
        organization_id=organization_id_for(slug),
        organization_slug=slug,
        revision=1,
        lifecycle=_lifecycle(),
    )


def _tenant(organization: OrganizationIdentity, slug: str = "prod") -> TenantIdentity:
    return _finalize(
        TenantIdentity,
        "identity_digest",
        identity_version="tenant-identity.v1",
        tenant_id=tenant_id_for(organization.organization_id, slug),
        organization_id=organization.organization_id,
        tenant_slug=slug,
        revision=1,
        lifecycle=_lifecycle(),
    )


def _principal(tenant: TenantIdentity, subject: str = "idp:user-1") -> PrincipalIdentity:
    return _finalize(
        PrincipalIdentity,
        "identity_digest",
        identity_version="principal-identity.v1",
        principal_id=principal_id_for(tenant.tenant_id, subject),
        organization_id=tenant.organization_id,
        tenant_id=tenant.tenant_id,
        principal_kind="user",
        subject_ref=subject,
        revision=1,
        lifecycle=_lifecycle(),
    )


def _retention_policy(tenant: TenantIdentity) -> RetentionPolicy:
    return _finalize(
        RetentionPolicy,
        "policy_digest",
        retention_version="retention-policy.v1",
        retention_policy_id="retention:default",
        organization_id=tenant.organization_id,
        tenant_id=tenant.tenant_id,
        retention_days=30,
        legal_hold=False,
        revision=1,
    )


def _scope(organization: OrganizationIdentity, tenant: TenantIdentity, principal: PrincipalIdentity) -> TenantScope:
    return TenantScope(
        scope_version="tenant-scope.v1",
        organization_id=organization.organization_id,
        tenant_id=tenant.tenant_id,
        principal_id=principal.principal_id,
        grant_digests=(),
        scope_digest=_digest(
            {
                "organization_id": organization.organization_id,
                "tenant_id": tenant.tenant_id,
                "principal_id": principal.principal_id,
                "client_id": None,
                "grant_digests": (),
            }
        ),
    )


def _upstream(tenant: TenantIdentity, version: int, name: str = "api") -> GatewayUpstreamVersion:
    return _finalize(
        GatewayUpstreamVersion,
        "record_digest",
        record_version="gateway-upstream.v1",
        upstream_id=upstream_id_for(tenant.tenant_id, name),
        organization_id=tenant.organization_id,
        tenant_id=tenant.tenant_id,
        upstream_name=name,
        version=version,
        endpoint_ref=f"endpoint:{name}:v{version}",
        tls_key_ref="keyref:gateway-tls",
        ca_ref="artifact:gateway-ca",
        lifecycle=_lifecycle(),
    )


def _version_ref(
    tenant: TenantIdentity,
    target_kind: str,
    target_id: str,
    target_version: int,
    target_digest: str,
) -> GatewayVersionRef:
    fields = {
        "ref_version": "gateway-version-ref.v1",
        "organization_id": tenant.organization_id,
        "tenant_id": tenant.tenant_id,
        "target_kind": target_kind,
        "target_id": target_id,
        "target_version": target_version,
        "target_digest": target_digest,
    }
    return _finalize(GatewayVersionRef, "ref_digest", **fields)


def _fixture() -> tuple[
    InMemoryFoundationRepository,
    FoundationControlPlane,
    OrganizationIdentity,
    TenantIdentity,
    PrincipalIdentity,
    TenantScope,
    GatewayUpstreamVersion,
    GatewayUpstreamVersion,
]:
    repository = InMemoryFoundationRepository()
    plane = FoundationControlPlane(repository)
    organization = _organization()
    tenant = _tenant(organization)
    principal = _principal(tenant)
    scope = _scope(organization, tenant, principal)
    plane.register_organization(organization)
    plane.register_tenant(tenant)
    plane.register_principal(scope, principal)
    plane.register_retention_policy(scope, _retention_policy(tenant))
    first = _upstream(tenant, 1)
    second = _upstream(tenant, 2)
    plane.register_upstream(scope, first)
    plane.register_upstream(scope, second)
    return repository, plane, organization, tenant, principal, scope, first, second


def test_scope_and_opaque_references_fail_closed() -> None:
    _, plane, organization, tenant, _, scope, first, _ = _fixture()
    raw_payload = _upstream(tenant, 3).model_dump()
    raw_payload["endpoint_ref"] = 'body:{"raw":true}'
    with pytest.raises(ValidationError):
        GatewayUpstreamVersion.model_validate(raw_payload)

    other_tenant = _tenant(organization, "staging")
    plane.register_tenant(other_tenant)
    cross_scope_route = _finalize(
        GatewayRouteVersion,
        "record_digest",
        record_version="gateway-route.v1",
        route_id=route_id_for(tenant.tenant_id, "gateway"),
        organization_id=tenant.organization_id,
        tenant_id=tenant.tenant_id,
        route_name="gateway",
        version=1,
        path_template="/api",
        upstream_id=_upstream(other_tenant, 1).upstream_id,
        upstream_version=1,
        auth_policy_ref="policy:default",
        lifecycle=_lifecycle(),
    )
    with pytest.raises(FoundationLifecycleError, match="route_upstream_scope_or_version_missing"):
        plane.register_route(scope, cross_scope_route)

    assert "body:{" not in first.model_dump_json()


def test_immutable_versions_and_release_cas_rollback() -> None:
    repository, plane, _, tenant, principal, scope, first, second = _fixture()
    with pytest.raises(FoundationRepositoryError, match="immutable"):
        repository.put_upstream(
            first.model_copy(
                update={
                    "endpoint_ref": "endpoint:mutated",
                    "record_digest": _digest({"mutated": True}),
                }
            )
        )

    pointer, activation = plane.activate(
        scope,
        target_kind="upstream",
        target_id=first.upstream_id,
        target_version=1,
        actor_principal_id=principal.principal_id,
        expected_revision=0,
        expected_pointer_digest=None,
        change_ref="change:release:1",
        activated_at="2026-08-19T00:00:00Z",
    )
    assert pointer.revision == 1
    assert activation.operation == "activate"

    pointer, _ = plane.activate(
        scope,
        target_kind="upstream",
        target_id=second.upstream_id,
        target_version=2,
        actor_principal_id=principal.principal_id,
        expected_revision=pointer.revision,
        expected_pointer_digest=pointer.pointer_digest,
        change_ref="change:release:2",
        activated_at="2026-08-19T00:01:00Z",
    )
    with pytest.raises(FoundationLifecycleError, match="stale_cas"):
        plane.activate(
            scope,
            target_kind="upstream",
            target_id=first.upstream_id,
            target_version=1,
            actor_principal_id=principal.principal_id,
            expected_revision=1,
            expected_pointer_digest=None,
            change_ref="change:stale",
            activated_at="2026-08-19T00:02:00Z",
        )
    pointer, rollback = plane.activate(
        scope,
        target_kind="upstream",
        target_id=first.upstream_id,
        target_version=1,
        actor_principal_id=principal.principal_id,
        expected_revision=pointer.revision,
        expected_pointer_digest=pointer.pointer_digest,
        change_ref="change:rollback:1",
        activated_at="2026-08-19T00:03:00Z",
        operation="rollback",
    )
    assert pointer.target_version == 1
    assert rollback.operation == "rollback"
    assert rollback.previous_target_version == 2


def test_route_config_and_feature_bind_exact_versions() -> None:
    _, plane, _, tenant, _, scope, first, _ = _fixture()
    route = _finalize(
        GatewayRouteVersion,
        "record_digest",
        record_version="gateway-route.v1",
        route_id=route_id_for(tenant.tenant_id, "gateway"),
        organization_id=tenant.organization_id,
        tenant_id=tenant.tenant_id,
        route_name="gateway",
        version=1,
        path_template="/api",
        upstream_id=first.upstream_id,
        upstream_version=first.version,
        auth_policy_ref="policy:default",
        lifecycle=_lifecycle(),
    )
    plane.register_route(scope, route)
    route_ref = _version_ref(
        tenant, "route", route.route_id, route.version, route.record_digest
    )
    upstream_ref = _version_ref(
        tenant, "upstream", first.upstream_id, first.version, first.record_digest
    )
    component_refs = tuple(sorted((route_ref, upstream_ref), key=lambda ref: (ref.target_kind, ref.target_id, ref.target_version)))
    config = _finalize(
        GatewayConfigVersion,
        "record_digest",
        record_version="gateway-config.v1",
        config_id=config_id_for(tenant.tenant_id, "default"),
        organization_id=tenant.organization_id,
        tenant_id=tenant.tenant_id,
        config_name="default",
        version=1,
        component_refs=component_refs,
        lifecycle=_lifecycle(),
    )
    plane.register_config(scope, config)
    feature = _finalize(
        GatewayFeatureVersion,
        "record_digest",
        record_version="gateway-feature.v1",
        feature_id=feature_id_for(tenant.tenant_id, "streaming"),
        organization_id=tenant.organization_id,
        tenant_id=tenant.tenant_id,
        feature_name="streaming",
        version=1,
        enabled=True,
        config_ref=_version_ref(
            tenant, "config", config.config_id, config.version, config.record_digest
        ),
        lifecycle=_lifecycle(),
    )
    plane.register_feature(scope, feature)


def test_tombstone_blocks_active_release_and_later_reactivation() -> None:
    repository, plane, _, tenant, principal, scope, first, second = _fixture()
    pointer, _ = plane.activate(
        scope,
        target_kind="upstream",
        target_id=first.upstream_id,
        target_version=1,
        actor_principal_id=principal.principal_id,
        expected_revision=0,
        expected_pointer_digest=None,
        change_ref="change:release:1",
        activated_at="2026-08-19T00:00:00Z",
    )
    tombstone = _finalize(
        TombstoneRecord,
        "tombstone_digest",
        tombstone_version="tombstone.v1",
        tombstone_id=tombstone_id(tenant.tenant_id, "upstream", first.upstream_id, 1),
        organization_id=tenant.organization_id,
        tenant_id=tenant.tenant_id,
        resource_kind="upstream",
        resource_id=first.upstream_id,
        resource_version=1,
        retention_policy_ref="retention:default",
        reason_ref="reason:operator",
        tombstoned_at="2026-08-19T00:04:00Z",
        purge_after="2026-09-19T00:04:00Z",
    )
    with pytest.raises(FoundationLifecycleError, match="active_release"):
        plane.register_tombstone(scope, tombstone)

    pointer, _ = plane.activate(
        scope,
        target_kind="upstream",
        target_id=second.upstream_id,
        target_version=2,
        actor_principal_id=principal.principal_id,
        expected_revision=pointer.revision,
        expected_pointer_digest=pointer.pointer_digest,
        change_ref="change:release:2",
        activated_at="2026-08-19T00:05:00Z",
    )
    plane.register_tombstone(scope, tombstone)
    assert repository.get_tombstone(scope, "upstream", first.upstream_id, 1) == tombstone
    with pytest.raises(FoundationLifecycleError, match="tombstoned"):
        plane.activate(
            scope,
            target_kind="upstream",
            target_id=first.upstream_id,
            target_version=1,
            actor_principal_id=principal.principal_id,
            expected_revision=pointer.revision,
            expected_pointer_digest=pointer.pointer_digest,
            change_ref="change:reactivate-tombstone",
            activated_at="2026-08-19T00:06:00Z",
        )
