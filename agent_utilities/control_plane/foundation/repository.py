"""Typed repository and CAS seams for the control-plane foundation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    ActivationRecord,
    ApiClientIdentity,
    CircuitHistory,
    Entitlement,
    GatewayConfigVersion,
    GatewayFeatureVersion,
    GatewayRouteVersion,
    GatewayUpstreamVersion,
    Membership,
    OrganizationIdentity,
    Permission,
    PrincipalIdentity,
    QuotaContract,
    ReleaseMutation,
    ReleasePointer,
    RetentionPolicy,
    Role,
    TenantIdentity,
    TenantScope,
    TombstoneRecord,
)


class FoundationRepositoryError(RuntimeError):
    """A repository returned data outside the immutable foundation contract."""


class ReleaseConflict(FoundationRepositoryError):
    """A release pointer CAS was fenced by a newer revision."""


@runtime_checkable
class FoundationRepository(Protocol):
    """Persistence-independent authority for identities and release state.

    Implementations must enforce organization/tenant scope before returning a
    record, retain immutable records idempotently, and apply release pointer plus
    activation evidence in one revision-fenced CAS operation.  This interface
    intentionally has no methods for rate-limit buckets, health samples, secret
    values, or raw gateway payloads.
    """

    def put_organization(self, record: OrganizationIdentity) -> None: ...

    def get_organization(self, organization_id: str) -> OrganizationIdentity | None: ...

    def put_tenant(self, record: TenantIdentity) -> None: ...

    def get_tenant(
        self, scope: TenantScope, tenant_id: str
    ) -> TenantIdentity | None: ...

    def put_principal(self, record: PrincipalIdentity) -> None: ...

    def get_principal(
        self, scope: TenantScope, principal_id: str
    ) -> PrincipalIdentity | None: ...

    def put_client(self, record: ApiClientIdentity) -> None: ...

    def get_client(
        self, scope: TenantScope, client_id: str
    ) -> ApiClientIdentity | None: ...

    def put_retention_policy(self, record: RetentionPolicy) -> None: ...

    def get_retention_policy(
        self, scope: TenantScope, policy_id: str
    ) -> RetentionPolicy | None: ...

    def put_permission(self, record: Permission) -> None: ...

    def get_permission(
        self, scope: TenantScope, permission_id: str
    ) -> Permission | None: ...

    def put_role(self, record: Role) -> None: ...

    def get_role(self, scope: TenantScope, role_id: str) -> Role | None: ...

    def put_membership(self, record: Membership) -> None: ...

    def get_membership(
        self, scope: TenantScope, membership_id: str
    ) -> Membership | None: ...

    def put_entitlement(self, record: Entitlement) -> None: ...

    def get_entitlement(
        self, scope: TenantScope, entitlement_id: str
    ) -> Entitlement | None: ...

    def put_quota(self, record: QuotaContract) -> None: ...

    def get_quota(self, scope: TenantScope, quota_id: str) -> QuotaContract | None: ...

    def put_upstream(self, record: GatewayUpstreamVersion) -> None: ...

    def get_upstream(
        self, scope: TenantScope, upstream_id: str, version: int
    ) -> GatewayUpstreamVersion | None: ...

    def put_route(self, record: GatewayRouteVersion) -> None: ...

    def get_route(
        self, scope: TenantScope, route_id: str, version: int
    ) -> GatewayRouteVersion | None: ...

    def put_config(self, record: GatewayConfigVersion) -> None: ...

    def get_config(
        self, scope: TenantScope, config_id: str, version: int
    ) -> GatewayConfigVersion | None: ...

    def put_feature(self, record: GatewayFeatureVersion) -> None: ...

    def get_feature(
        self, scope: TenantScope, feature_id: str, version: int
    ) -> GatewayFeatureVersion | None: ...

    def put_circuit_history(self, record: CircuitHistory) -> None: ...

    def put_tombstone(self, record: TombstoneRecord) -> None: ...

    def get_tombstone(
        self, scope: TenantScope, resource_kind: str, resource_id: str, version: int
    ) -> TombstoneRecord | None: ...

    def get_release_pointer(
        self, scope: TenantScope, pointer_id: str
    ) -> ReleasePointer | None: ...

    def get_activation_history(
        self, scope: TenantScope, pointer_id: str
    ) -> tuple[ActivationRecord, ...]: ...

    def has_activation_target(
        self, scope: TenantScope, pointer_id: str, target_version: int
    ) -> bool: ...

    def compare_and_swap_release(
        self,
        scope: TenantScope,
        mutation: ReleaseMutation,
        pointer: ReleasePointer,
        activation: ActivationRecord,
    ) -> ReleasePointer: ...


__all__ = [
    "FoundationRepository",
    "FoundationRepositoryError",
    "ReleaseConflict",
]
