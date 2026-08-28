"""In-memory contract implementation and lifecycle-gated foundation service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from .models import (
    ActivationRecord,
    ApiClientIdentity,
    CircuitHistory,
    Entitlement,
    GatewayConfigVersion,
    GatewayFeatureVersion,
    GatewayRouteVersion,
    GatewayUpstreamVersion,
    GatewayVersionRef,
    Membership,
    OrganizationIdentity,
    Permission,
    PrincipalIdentity,
    QuotaContract,
    ReleaseMutation,
    ReleaseOperation,
    ReleasePointer,
    RetentionPolicy,
    Role,
    TargetKind,
    TenantIdentity,
    TenantScope,
    TombstoneRecord,
    _digest_payload,
    activation_id_for,
    release_pointer_id,
)
from .repository import (
    FoundationRepository,
    FoundationRepositoryError,
    ReleaseConflict,
)


class FoundationLifecycleError(ValueError):
    """A requested identity, reference, or release transition is not admissible."""


class _ScopedRecord(Protocol):
    organization_id: str
    tenant_id: str


ScopedT = TypeVar("ScopedT", bound=_ScopedRecord)
KeyT = TypeVar("KeyT")
RetainT = TypeVar("RetainT")


def _scope_record(scope: TenantScope, record: ScopedT | None) -> ScopedT | None:
    if record is None:
        return None
    if (
        record.organization_id != scope.organization_id
        or record.tenant_id != scope.tenant_id
    ):
        return None
    return record


def _retain(mapping: dict[KeyT, RetainT], key: KeyT, value: RetainT) -> None:
    prior = mapping.get(key)
    if prior is not None and prior != value:
        raise FoundationRepositoryError("immutable foundation record mutation")
    mapping[key] = value


@dataclass(slots=True, frozen=True)
class _ReleaseTarget:
    """A release target's identity + digest, bundled for CAS record construction.

    Extracted so :meth:`FoundationControlPlane.activate`'s helper methods carry
    one object instead of four separate positional fields each.
    """

    target_kind: TargetKind
    target_id: str
    target_version: int
    target_digest: str


@dataclass(slots=True, frozen=True)
class _ActivationRequest:
    """:meth:`FoundationControlPlane.activate`'s own parameters, bundled.

    ``activate``'s public signature is unchanged (pre-existing callers depend on
    it); this wraps the call's arguments once at the top of the method so the
    decomposed helpers stay under the lane's <=7-parameter cap.
    """

    target_kind: TargetKind
    target_id: str
    target_version: int
    actor_principal_id: str
    expected_revision: int
    expected_pointer_digest: str | None
    change_ref: str
    activated_at: str
    operation: str


@dataclass(slots=True)
class InMemoryFoundationRepository:
    """Small deterministic repository used by contract fixtures and local wiring.

    It models the production guarantees, not a production storage engine: scoped
    reads, immutable version keys, append-only activation history, and atomic
    release pointer CAS.  No payload, secret, counter, or health sample storage
    exists here by design.
    """

    organizations: dict[str, OrganizationIdentity] = field(default_factory=dict)
    tenants: dict[str, TenantIdentity] = field(default_factory=dict)
    principals: dict[str, PrincipalIdentity] = field(default_factory=dict)
    clients: dict[str, ApiClientIdentity] = field(default_factory=dict)
    retention_policies: dict[str, RetentionPolicy] = field(default_factory=dict)
    permissions: dict[str, Permission] = field(default_factory=dict)
    roles: dict[str, Role] = field(default_factory=dict)
    memberships: dict[str, Membership] = field(default_factory=dict)
    entitlements: dict[str, Entitlement] = field(default_factory=dict)
    quotas: dict[str, QuotaContract] = field(default_factory=dict)
    upstreams: dict[tuple[str, int], GatewayUpstreamVersion] = field(
        default_factory=dict
    )
    routes: dict[tuple[str, int], GatewayRouteVersion] = field(default_factory=dict)
    configs: dict[tuple[str, int], GatewayConfigVersion] = field(default_factory=dict)
    features: dict[tuple[str, int], GatewayFeatureVersion] = field(default_factory=dict)
    circuit_histories: dict[str, CircuitHistory] = field(default_factory=dict)
    tombstones: dict[tuple[str, str, int], TombstoneRecord] = field(
        default_factory=dict
    )
    release_pointers: dict[str, ReleasePointer] = field(default_factory=dict)
    activation_history: dict[str, list[ActivationRecord]] = field(default_factory=dict)

    def put_organization(self, record: OrganizationIdentity) -> None:
        _retain(self.organizations, record.organization_id, record)

    def get_organization(self, organization_id: str) -> OrganizationIdentity | None:
        return self.organizations.get(organization_id)

    def put_tenant(self, record: TenantIdentity) -> None:
        _retain(self.tenants, record.tenant_id, record)

    def get_tenant(self, scope: TenantScope, tenant_id: str) -> TenantIdentity | None:
        return _scope_record(scope, self.tenants.get(tenant_id))

    def put_principal(self, record: PrincipalIdentity) -> None:
        _retain(self.principals, record.principal_id, record)

    def get_principal(
        self, scope: TenantScope, principal_id: str
    ) -> PrincipalIdentity | None:
        return _scope_record(scope, self.principals.get(principal_id))

    def put_client(self, record: ApiClientIdentity) -> None:
        _retain(self.clients, record.client_id, record)

    def get_client(
        self, scope: TenantScope, client_id: str
    ) -> ApiClientIdentity | None:
        return _scope_record(scope, self.clients.get(client_id))

    def put_retention_policy(self, record: RetentionPolicy) -> None:
        _retain(self.retention_policies, record.retention_policy_id, record)

    def get_retention_policy(
        self, scope: TenantScope, policy_id: str
    ) -> RetentionPolicy | None:
        return _scope_record(scope, self.retention_policies.get(policy_id))

    def put_permission(self, record: Permission) -> None:
        _retain(self.permissions, record.permission_id, record)

    def get_permission(
        self, scope: TenantScope, permission_id: str
    ) -> Permission | None:
        return _scope_record(scope, self.permissions.get(permission_id))

    def put_role(self, record: Role) -> None:
        _retain(self.roles, record.role_id, record)

    def get_role(self, scope: TenantScope, role_id: str) -> Role | None:
        return _scope_record(scope, self.roles.get(role_id))

    def put_membership(self, record: Membership) -> None:
        _retain(self.memberships, record.membership_id, record)

    def get_membership(
        self, scope: TenantScope, membership_id: str
    ) -> Membership | None:
        return _scope_record(scope, self.memberships.get(membership_id))

    def put_entitlement(self, record: Entitlement) -> None:
        _retain(self.entitlements, record.entitlement_id, record)

    def get_entitlement(
        self, scope: TenantScope, entitlement_id: str
    ) -> Entitlement | None:
        return _scope_record(scope, self.entitlements.get(entitlement_id))

    def put_quota(self, record: QuotaContract) -> None:
        _retain(self.quotas, record.quota_id, record)

    def get_quota(self, scope: TenantScope, quota_id: str) -> QuotaContract | None:
        return _scope_record(scope, self.quotas.get(quota_id))

    def put_upstream(self, record: GatewayUpstreamVersion) -> None:
        _retain(self.upstreams, (record.upstream_id, record.version), record)

    def get_upstream(
        self, scope: TenantScope, upstream_id: str, version: int
    ) -> GatewayUpstreamVersion | None:
        return _scope_record(scope, self.upstreams.get((upstream_id, version)))

    def put_route(self, record: GatewayRouteVersion) -> None:
        _retain(self.routes, (record.route_id, record.version), record)

    def get_route(
        self, scope: TenantScope, route_id: str, version: int
    ) -> GatewayRouteVersion | None:
        return _scope_record(scope, self.routes.get((route_id, version)))

    def put_config(self, record: GatewayConfigVersion) -> None:
        _retain(self.configs, (record.config_id, record.version), record)

    def get_config(
        self, scope: TenantScope, config_id: str, version: int
    ) -> GatewayConfigVersion | None:
        return _scope_record(scope, self.configs.get((config_id, version)))

    def put_feature(self, record: GatewayFeatureVersion) -> None:
        _retain(self.features, (record.feature_id, record.version), record)

    def get_feature(
        self, scope: TenantScope, feature_id: str, version: int
    ) -> GatewayFeatureVersion | None:
        return _scope_record(scope, self.features.get((feature_id, version)))

    def put_circuit_history(self, record: CircuitHistory) -> None:
        _retain(self.circuit_histories, record.circuit_id, record)

    def put_tombstone(self, record: TombstoneRecord) -> None:
        _retain(
            self.tombstones,
            (record.resource_kind, record.resource_id, record.resource_version),
            record,
        )

    def get_tombstone(
        self, scope: TenantScope, resource_kind: str, resource_id: str, version: int
    ) -> TombstoneRecord | None:
        return _scope_record(
            scope, self.tombstones.get((resource_kind, resource_id, version))
        )

    def get_release_pointer(
        self, scope: TenantScope, pointer_id: str
    ) -> ReleasePointer | None:
        return _scope_record(scope, self.release_pointers.get(pointer_id))

    def get_activation_history(
        self, scope: TenantScope, pointer_id: str
    ) -> tuple[ActivationRecord, ...]:
        pointer = self.release_pointers.get(pointer_id)
        if pointer is None or _scope_record(scope, pointer) is None:
            return ()
        return tuple(self.activation_history.get(pointer_id, ()))

    def has_activation_target(
        self, scope: TenantScope, pointer_id: str, target_version: int
    ) -> bool:
        return any(
            event.target_version == target_version
            for event in self.get_activation_history(scope, pointer_id)
        )

    @staticmethod
    def _assert_cas_identity(
        pointer: ReleasePointer,
        mutation: ReleaseMutation,
        activation: ActivationRecord,
    ) -> None:
        if (
            pointer.pointer_id != mutation.pointer_id
            or activation.pointer_id != pointer.pointer_id
        ):
            raise FoundationRepositoryError("release CAS record identity mismatch")

    @staticmethod
    def _assert_cas_scope(
        scope: TenantScope, pointer: ReleasePointer, mutation: ReleaseMutation
    ) -> None:
        if (
            pointer.organization_id != scope.organization_id
            or pointer.tenant_id != scope.tenant_id
            or mutation.organization_id != scope.organization_id
            or mutation.tenant_id != scope.tenant_id
        ):
            raise FoundationRepositoryError("release CAS scope mismatch")

    @staticmethod
    def _assert_cas_target(
        mutation: ReleaseMutation,
        pointer: ReleasePointer,
        activation: ActivationRecord,
    ) -> None:
        if (
            mutation.operation != activation.operation
            or mutation.target_kind != pointer.target_kind
            or mutation.target_id != pointer.target_id
            or mutation.target_version != pointer.target_version
            or mutation.target_digest != pointer.target_digest
            or activation.target_digest != pointer.target_digest
            or activation.release_revision != pointer.revision
        ):
            raise FoundationRepositoryError("release CAS target mismatch")

    def _assert_cas_revision(
        self, mutation: ReleaseMutation, pointer: ReleasePointer
    ) -> None:
        current = self.release_pointers.get(pointer.pointer_id)
        current_revision = current.revision if current is not None else 0
        current_digest = current.pointer_digest if current is not None else None
        if (
            mutation.expected_revision != current_revision
            or mutation.expected_pointer_digest != current_digest
        ):
            raise ReleaseConflict("release pointer CAS conflict")
        if pointer.revision != current_revision + 1:
            raise FoundationRepositoryError("release pointer revision is not monotonic")

    def _cas_prior_event(
        self, pointer: ReleasePointer, activation: ActivationRecord
    ) -> ActivationRecord | None:
        prior_event = next(
            (
                event
                for event in self.activation_history.get(pointer.pointer_id, ())
                if event.activation_id == activation.activation_id
            ),
            None,
        )
        if prior_event is not None and prior_event != activation:
            raise FoundationRepositoryError("activation evidence mutation")
        return prior_event

    def compare_and_swap_release(
        self,
        scope: TenantScope,
        mutation: ReleaseMutation,
        pointer: ReleasePointer,
        activation: ActivationRecord,
    ) -> ReleasePointer:
        self._assert_cas_identity(pointer, mutation, activation)
        self._assert_cas_scope(scope, pointer, mutation)
        self._assert_cas_target(mutation, pointer, activation)
        self._assert_cas_revision(mutation, pointer)
        prior_event = self._cas_prior_event(pointer, activation)
        self.release_pointers[pointer.pointer_id] = pointer
        if prior_event is None:
            self.activation_history.setdefault(pointer.pointer_id, []).append(
                activation
            )
        return pointer


class FoundationControlPlane:
    """Scope- and lifecycle-gated service over :class:`FoundationRepository`."""

    def __init__(self, repository: FoundationRepository) -> None:
        self._repository = repository

    def register_organization(self, record: OrganizationIdentity) -> None:
        self._repository.put_organization(record)

    def register_tenant(self, record: TenantIdentity) -> None:
        organization = self._repository.get_organization(record.organization_id)
        if organization is None:
            raise FoundationLifecycleError("organization_unavailable")
        if organization.lifecycle.status != "active":
            raise FoundationLifecycleError("organization_not_active")
        self._repository.put_tenant(record)

    def register_principal(self, scope: TenantScope, record: PrincipalIdentity) -> None:
        self._require_tenant_scope(
            scope, record.organization_id, record.tenant_id, require_principal=False
        )
        self._repository.put_principal(record)

    def register_client(self, scope: TenantScope, record: ApiClientIdentity) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        if record.principal_id != scope.principal_id:
            raise FoundationLifecycleError("client_principal_scope_mismatch")
        principal = self._repository.get_principal(scope, record.principal_id)
        if principal is None:
            raise FoundationLifecycleError("client_principal_unavailable")
        if principal.lifecycle.status != "active":
            raise FoundationLifecycleError("client_principal_not_active")
        self._repository.put_client(record)

    def register_retention_policy(
        self, scope: TenantScope, record: RetentionPolicy
    ) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        self._repository.put_retention_policy(record)

    def register_permission(self, scope: TenantScope, record: Permission) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        self._repository.put_permission(record)

    def register_role(self, scope: TenantScope, record: Role) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        for permission_id in record.permission_ids:
            if self._repository.get_permission(scope, permission_id) is None:
                raise FoundationLifecycleError("role_permission_scope_or_missing")
        self._repository.put_role(record)

    def register_membership(self, scope: TenantScope, record: Membership) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        principal = self._repository.get_principal(scope, record.principal_id)
        role = self._repository.get_role(scope, record.role_id)
        if principal is None or role is None:
            raise FoundationLifecycleError("membership_scope_or_reference_missing")
        if principal.lifecycle.status != "active" or role.lifecycle.status != "active":
            raise FoundationLifecycleError("membership_reference_not_active")
        self._repository.put_membership(record)

    def register_entitlement(self, scope: TenantScope, record: Entitlement) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        self._require_subject(scope, record.subject_kind, record.subject_id)
        self._repository.put_entitlement(record)

    def register_quota(self, scope: TenantScope, record: QuotaContract) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        self._require_subject(scope, record.subject_kind, record.subject_id)
        self._repository.put_quota(record)

    def register_upstream(
        self, scope: TenantScope, record: GatewayUpstreamVersion
    ) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        self._repository.put_upstream(record)

    def register_route(self, scope: TenantScope, record: GatewayRouteVersion) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        upstream = self._repository.get_upstream(
            scope, record.upstream_id, record.upstream_version
        )
        if upstream is None or upstream.lifecycle.status != "active":
            raise FoundationLifecycleError("route_upstream_scope_or_version_missing")
        self._repository.put_route(record)

    def register_config(self, scope: TenantScope, record: GatewayConfigVersion) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        for component in record.component_refs:
            if component.target_kind not in {"route", "upstream"}:
                raise FoundationLifecycleError("config_component_kind_not_allowed")
            component_record = self._get_version(scope, component)
            if (
                component_record is None
                or component_record.lifecycle.status != "active"
            ):
                raise FoundationLifecycleError(
                    "config_component_scope_or_version_missing"
                )
        self._repository.put_config(record)

    def register_feature(
        self, scope: TenantScope, record: GatewayFeatureVersion
    ) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        if record.config_ref is not None:
            config = self._get_version(scope, record.config_ref)
            if config is None or config.lifecycle.status != "active":
                raise FoundationLifecycleError(
                    "feature_config_scope_or_version_missing"
                )
        self._repository.put_feature(record)

    def register_circuit_history(
        self, scope: TenantScope, record: CircuitHistory
    ) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        self._repository.put_circuit_history(record)

    def register_tombstone(self, scope: TenantScope, record: TombstoneRecord) -> None:
        self._require_tenant_scope(scope, record.organization_id, record.tenant_id)
        if (
            self._repository.get_retention_policy(scope, record.retention_policy_ref)
            is None
        ):
            raise FoundationLifecycleError("tombstone_retention_policy_missing")
        target = self._lookup_version(
            scope, record.resource_kind, record.resource_id, record.resource_version
        )
        if target is None:
            raise FoundationLifecycleError("tombstone_target_scope_or_version_missing")
        pointer_id = release_pointer_id(
            scope.tenant_id, record.resource_kind, record.resource_id
        )
        pointer = self._repository.get_release_pointer(scope, pointer_id)
        if pointer is not None and pointer.target_version == record.resource_version:
            raise FoundationLifecycleError("active_release_cannot_be_tombstoned")
        self._repository.put_tombstone(record)

    @staticmethod
    def _resolve_release_operation(operation: str) -> ReleaseOperation:
        if operation == "activate":
            return "activate"
        if operation == "rollback":
            return "rollback"
        raise FoundationLifecycleError("unknown_release_operation")

    def _assert_release_actor(
        self, scope: TenantScope, actor_principal_id: str
    ) -> None:
        if actor_principal_id != scope.principal_id:
            raise FoundationLifecycleError("release_actor_scope_mismatch")
        principal = self._repository.get_principal(scope, actor_principal_id)
        if principal is None or principal.lifecycle.status != "active":
            raise FoundationLifecycleError("release_actor_not_active_in_scope")

    def _resolve_release_target(
        self,
        scope: TenantScope,
        target_kind: TargetKind,
        target_id: str,
        target_version: int,
    ) -> (
        GatewayUpstreamVersion
        | GatewayRouteVersion
        | GatewayConfigVersion
        | GatewayFeatureVersion
    ):
        target = self._lookup_version(scope, target_kind, target_id, target_version)
        if target is None:
            raise FoundationLifecycleError("release_target_scope_or_version_missing")
        if target.lifecycle.status != "active":
            raise FoundationLifecycleError("release_target_not_active")
        if (
            self._repository.get_tombstone(
                scope, target_kind, target_id, target_version
            )
            is not None
        ):
            raise FoundationLifecycleError("release_target_tombstoned")
        return target

    def _pointer_cas_state(
        self,
        scope: TenantScope,
        pointer_id: str,
        *,
        expected_revision: int,
        expected_pointer_digest: str | None,
        target_version: int,
    ) -> tuple[ReleasePointer | None, int]:
        current = self._repository.get_release_pointer(scope, pointer_id)
        current_revision = current.revision if current is not None else 0
        current_digest = current.pointer_digest if current is not None else None
        if (
            expected_revision != current_revision
            or expected_pointer_digest != current_digest
        ):
            raise FoundationLifecycleError("release_pointer_stale_cas")
        if current is not None and current.target_version == target_version:
            raise FoundationLifecycleError("release_target_already_active")
        return current, current_revision + 1

    def _assert_rollback_valid(
        self,
        scope: TenantScope,
        pointer_id: str,
        *,
        operation: str,
        current: ReleasePointer | None,
        target_version: int,
    ) -> None:
        if operation != "rollback":
            return
        if current is None:
            raise FoundationLifecycleError("rollback_requires_existing_release")
        if not self._repository.has_activation_target(
            scope, pointer_id, target_version
        ):
            raise FoundationLifecycleError("rollback_target_not_in_history")

    @staticmethod
    def _build_release_pointer(
        scope: TenantScope,
        pointer_id: str,
        target: _ReleaseTarget,
        *,
        next_revision: int,
        actor_principal_id: str,
    ) -> ReleasePointer:
        payload = {
            "pointer_version": "release-pointer.v1",
            "pointer_id": pointer_id,
            "organization_id": scope.organization_id,
            "tenant_id": scope.tenant_id,
            "target_kind": target.target_kind,
            "target_id": target.target_id,
            "target_version": target.target_version,
            "target_digest": target.target_digest,
            "revision": next_revision,
            "updated_by_principal_id": actor_principal_id,
        }
        return ReleasePointer(
            pointer_version="release-pointer.v1",
            pointer_id=pointer_id,
            organization_id=scope.organization_id,
            tenant_id=scope.tenant_id,
            target_kind=target.target_kind,
            target_id=target.target_id,
            target_version=target.target_version,
            target_digest=target.target_digest,
            revision=next_revision,
            updated_by_principal_id=actor_principal_id,
            pointer_digest=_digest_payload(payload),
        )

    @staticmethod
    def _build_activation_record(
        scope: TenantScope,
        pointer_id: str,
        target: _ReleaseTarget,
        req: _ActivationRequest,
        *,
        release_operation: ReleaseOperation,
        previous_target_version: int | None,
        next_revision: int,
    ) -> ActivationRecord:
        payload = {
            "activation_version": "activation-record.v1",
            "pointer_id": pointer_id,
            "organization_id": scope.organization_id,
            "tenant_id": scope.tenant_id,
            "operation": req.operation,
            "target_kind": target.target_kind,
            "target_id": target.target_id,
            "target_version": target.target_version,
            "target_digest": target.target_digest,
            "previous_target_version": previous_target_version,
            "release_revision": next_revision,
            "actor_principal_id": req.actor_principal_id,
            "change_ref": req.change_ref,
            "activated_at": req.activated_at,
        }
        activation_digest = _digest_payload(payload)
        return ActivationRecord(
            activation_version="activation-record.v1",
            pointer_id=pointer_id,
            organization_id=scope.organization_id,
            tenant_id=scope.tenant_id,
            operation=release_operation,
            target_kind=target.target_kind,
            target_id=target.target_id,
            target_version=target.target_version,
            target_digest=target.target_digest,
            previous_target_version=previous_target_version,
            release_revision=next_revision,
            actor_principal_id=req.actor_principal_id,
            change_ref=req.change_ref,
            activated_at=req.activated_at,
            activation_id=activation_id_for(
                pointer_id, next_revision, activation_digest
            ),
            activation_digest=activation_digest,
        )

    @staticmethod
    def _build_release_mutation(
        scope: TenantScope,
        pointer_id: str,
        target: _ReleaseTarget,
        req: _ActivationRequest,
        *,
        release_operation: ReleaseOperation,
    ) -> ReleaseMutation:
        return ReleaseMutation(
            mutation_version="release-mutation.v1",
            pointer_id=pointer_id,
            organization_id=scope.organization_id,
            tenant_id=scope.tenant_id,
            expected_revision=req.expected_revision,
            expected_pointer_digest=req.expected_pointer_digest,
            operation=release_operation,
            target_kind=target.target_kind,
            target_id=target.target_id,
            target_version=target.target_version,
            target_digest=target.target_digest,
            actor_principal_id=req.actor_principal_id,
            change_ref=req.change_ref,
        )

    def _apply_cas(
        self,
        scope: TenantScope,
        mutation: ReleaseMutation,
        pointer: ReleasePointer,
        activation: ActivationRecord,
    ) -> tuple[ReleasePointer, ActivationRecord]:
        try:
            applied = self._repository.compare_and_swap_release(
                scope, mutation, pointer, activation
            )
        except ReleaseConflict as exc:
            raise FoundationLifecycleError("release_pointer_stale_cas") from exc
        if applied != pointer:
            raise FoundationRepositoryError(
                "repository returned a different release pointer"
            )
        return pointer, activation

    def activate(
        self,
        scope: TenantScope,
        *,
        target_kind: TargetKind,
        target_id: str,
        target_version: int,
        actor_principal_id: str,
        expected_revision: int,
        expected_pointer_digest: str | None,
        change_ref: str,
        activated_at: str,
        operation: str = "activate",
    ) -> tuple[ReleasePointer, ActivationRecord]:
        req = _ActivationRequest(
            target_kind=target_kind,
            target_id=target_id,
            target_version=target_version,
            actor_principal_id=actor_principal_id,
            expected_revision=expected_revision,
            expected_pointer_digest=expected_pointer_digest,
            change_ref=change_ref,
            activated_at=activated_at,
            operation=operation,
        )
        release_operation = self._resolve_release_operation(req.operation)
        self._assert_release_actor(scope, req.actor_principal_id)
        target = self._resolve_release_target(
            scope, req.target_kind, req.target_id, req.target_version
        )
        target_ref = _ReleaseTarget(
            target_kind=req.target_kind,
            target_id=req.target_id,
            target_version=req.target_version,
            target_digest=target.record_digest,
        )
        pointer_id = release_pointer_id(scope.tenant_id, req.target_kind, req.target_id)
        current, next_revision = self._pointer_cas_state(
            scope,
            pointer_id,
            expected_revision=req.expected_revision,
            expected_pointer_digest=req.expected_pointer_digest,
            target_version=req.target_version,
        )
        self._assert_rollback_valid(
            scope,
            pointer_id,
            operation=req.operation,
            current=current,
            target_version=req.target_version,
        )
        pointer = self._build_release_pointer(
            scope,
            pointer_id,
            target_ref,
            next_revision=next_revision,
            actor_principal_id=req.actor_principal_id,
        )
        activation = self._build_activation_record(
            scope,
            pointer_id,
            target_ref,
            req,
            release_operation=release_operation,
            previous_target_version=current.target_version if current else None,
            next_revision=next_revision,
        )
        mutation = self._build_release_mutation(
            scope, pointer_id, target_ref, req, release_operation=release_operation
        )
        return self._apply_cas(scope, mutation, pointer, activation)

    def _assert_scope_principal(self, scope: TenantScope) -> None:
        principal = self._repository.get_principal(scope, scope.principal_id)
        if principal is None:
            raise FoundationLifecycleError("scope_principal_unavailable")
        if principal.lifecycle.status != "active":
            raise FoundationLifecycleError("scope_principal_not_active")
        if scope.client_id is not None:
            client = self._repository.get_client(scope, scope.client_id)
            if client is None or client.principal_id != scope.principal_id:
                raise FoundationLifecycleError("scope_client_mismatch")

    def _require_tenant_scope(
        self,
        scope: TenantScope,
        organization_id: str,
        tenant_id: str,
        *,
        require_principal: bool = True,
    ) -> TenantIdentity:
        if scope.organization_id != organization_id or scope.tenant_id != tenant_id:
            raise FoundationLifecycleError("cross_tenant_scope")
        tenant = self._repository.get_tenant(scope, tenant_id)
        if tenant is None:
            raise FoundationLifecycleError("tenant_unavailable")
        if tenant.lifecycle.status != "active":
            raise FoundationLifecycleError("tenant_not_active")
        if require_principal:
            self._assert_scope_principal(scope)
        return tenant

    def _require_subject(
        self, scope: TenantScope, subject_kind: str, subject_id: str
    ) -> None:
        if subject_kind == "tenant":
            if subject_id != scope.tenant_id:
                raise FoundationLifecycleError("cross_tenant_subject")
        elif subject_kind == "principal":
            principal = self._repository.get_principal(scope, subject_id)
            if principal is None:
                raise FoundationLifecycleError("subject_scope_or_missing")
        elif subject_kind == "client":
            client = self._repository.get_client(scope, subject_id)
            if client is None:
                raise FoundationLifecycleError("subject_scope_or_missing")
        else:
            raise FoundationLifecycleError("unknown_subject_kind")

    def _version_ref(
        self,
        scope: TenantScope,
        target_kind: TargetKind,
        target_id: str,
        target_version: int,
        target_digest: str,
    ) -> GatewayVersionRef:
        payload = {
            "ref_version": "gateway-version-ref.v1",
            "organization_id": scope.organization_id,
            "tenant_id": scope.tenant_id,
            "target_kind": target_kind,
            "target_id": target_id,
            "target_version": target_version,
            "target_digest": target_digest,
        }
        return GatewayVersionRef(
            ref_version="gateway-version-ref.v1",
            organization_id=scope.organization_id,
            tenant_id=scope.tenant_id,
            target_kind=target_kind,
            target_id=target_id,
            target_version=target_version,
            target_digest=target_digest,
            ref_digest=_digest_payload(payload),
        )

    def _lookup_version(
        self,
        scope: TenantScope,
        target_kind: TargetKind,
        target_id: str,
        target_version: int,
    ) -> (
        GatewayUpstreamVersion
        | GatewayRouteVersion
        | GatewayConfigVersion
        | GatewayFeatureVersion
        | None
    ):
        if target_kind == "upstream":
            return self._repository.get_upstream(scope, target_id, target_version)
        if target_kind == "route":
            return self._repository.get_route(scope, target_id, target_version)
        if target_kind == "config":
            return self._repository.get_config(scope, target_id, target_version)
        return self._repository.get_feature(scope, target_id, target_version)

    def _get_version(
        self, scope: TenantScope, reference: GatewayVersionRef
    ) -> (
        GatewayUpstreamVersion
        | GatewayRouteVersion
        | GatewayConfigVersion
        | GatewayFeatureVersion
        | None
    ):
        if (
            reference.organization_id != scope.organization_id
            or reference.tenant_id != scope.tenant_id
        ):
            return None
        target = self._lookup_version(
            scope, reference.target_kind, reference.target_id, reference.target_version
        )
        if target is None or target.record_digest != reference.target_digest:
            return None
        return target


__all__ = [
    "FoundationControlPlane",
    "FoundationLifecycleError",
    "InMemoryFoundationRepository",
]
