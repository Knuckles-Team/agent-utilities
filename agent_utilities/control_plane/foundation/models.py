"""Typed identity, gateway-version, and release-control contracts.

This module is the persistence-independent foundation for the control plane.  It
records identities, policy contracts, immutable gateway versions, and bounded
release evidence; it deliberately does not contain request counters, health
samples, raw configuration bodies, or secret values.  Secret/key fields are
opaque references resolved by a later deployment boundary.

(CONCEPT:AU-OS.identity.tenant-rbac-admission,
AU-OS.governance.verified-write-state-advance)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Annotated, Literal, TypeAlias

from pydantic import Field, model_validator

from agent_utilities.protocols.epistemic_operations import ProtocolModel

Identifier: TypeAlias = Annotated[
    str,
    Field(
        min_length=1,
        max_length=192,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$",
    ),
]
NameText: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=96, pattern=r"^[a-z0-9][a-z0-9._-]*$"),
]
Digest: TypeAlias = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
Timestamp: TypeAlias = Annotated[
    str,
    Field(
        min_length=20,
        max_length=64,
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|z|[+-][0-9]{2}:[0-9]{2})$",
    ),
]
OpaqueRef: TypeAlias = Annotated[
    str,
    Field(
        min_length=3,
        max_length=192,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$",
    ),
]
PathText: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=512, pattern=r"^[^\x00\r\n]+$"),
]

IdentityKind = Literal["user", "service", "agent"]
RecordStatus = Literal["active", "retired"]
TargetKind = Literal["upstream", "route", "config", "feature"]
ReleaseOperation = Literal["activate", "rollback"]
SubjectKind = Literal["tenant", "principal", "client"]
CircuitState = Literal["closed", "open", "half_open"]

_MAX_LIST = 128
_MAX_CIRCUIT_REFS = 32
_MAX_RETENTION_DAYS = 36_500


def _digest_payload(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _opaque_ref(value: str, field_name: str) -> str:
    """Reject payloads while retaining opaque provider-owned references."""

    lowered = value.casefold()
    forbidden_prefixes = (
        "http:",
        "https:",
        "file:",
        "data:",
        "body:",
        "result:",
        "base64:",
    )
    inline_markers = (
        "password=",
        "authorization:",
        "bearer ",
        "-----begin",
        '{"',
        "[{",
    )
    if (
        lowered.startswith(forbidden_prefixes)
        or any(marker in lowered for marker in inline_markers)
        or ":" not in value
    ):
        raise ValueError(
            f"{field_name} must be an opaque reference, not payload material"
        )
    return value


def _key_ref(value: str, field_name: str) -> str:
    _opaque_ref(value, field_name)
    if not value.casefold().startswith(("keyref:", "secretref:")):
        raise ValueError(f"{field_name} must be a provider-owned key/secret reference")
    return value


def _sorted_unique(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must be unique")
    if tuple(sorted(result)) != result:
        raise ValueError(f"{field_name} must be sorted")
    return result


def _record_digest(model: ProtocolModel) -> str:
    payload = model.model_dump(mode="json")
    for digest_name in (
        "record_digest",
        "identity_digest",
        "pointer_digest",
        "activation_id",
        "activation_digest",
        "policy_digest",
        "ref_digest",
        "tombstone_digest",
    ):
        payload.pop(digest_name, None)
    return _digest_payload(payload)


def record_digest_for(model: ProtocolModel) -> str:
    """Return the canonical digest for a model with its digest field omitted."""

    return _record_digest(model)


def organization_id_for(organization_slug: str) -> str:
    return "org:" + hashlib.sha256(organization_slug.encode("utf-8")).hexdigest()


def tenant_id_for(organization_id: str, tenant_slug: str) -> str:
    return (
        "tenant:"
        + hashlib.sha256(f"{organization_id}\x1f{tenant_slug}".encode()).hexdigest()
    )


def principal_id_for(tenant_id: str, subject_ref: str) -> str:
    return (
        "principal:"
        + hashlib.sha256(f"{tenant_id}\x1f{subject_ref}".encode()).hexdigest()
    )


def client_id_for(tenant_id: str, client_name: str) -> str:
    return (
        "client:" + hashlib.sha256(f"{tenant_id}\x1f{client_name}".encode()).hexdigest()
    )


def permission_id_for(tenant_id: str, action: str, resource_ref: str) -> str:
    return (
        "permission:"
        + hashlib.sha256(
            f"{tenant_id}\x1f{action}\x1f{resource_ref}".encode()
        ).hexdigest()
    )


def role_id_for(tenant_id: str, role_name: str) -> str:
    return "role:" + hashlib.sha256(f"{tenant_id}\x1f{role_name}".encode()).hexdigest()


def membership_id_for(tenant_id: str, principal_id: str, role_id: str) -> str:
    return (
        "membership:"
        + hashlib.sha256(
            f"{tenant_id}\x1f{principal_id}\x1f{role_id}".encode()
        ).hexdigest()
    )


def entitlement_id_for(
    tenant_id: str, subject_kind: SubjectKind, subject_id: str, name: str
) -> str:
    return (
        "entitlement:"
        + hashlib.sha256(
            f"{tenant_id}\x1f{subject_kind}\x1f{subject_id}\x1f{name}".encode()
        ).hexdigest()
    )


def quota_id_for(tenant_id: str, subject_kind: SubjectKind, subject_id: str) -> str:
    return (
        "quota:"
        + hashlib.sha256(
            f"{tenant_id}\x1f{subject_kind}\x1f{subject_id}".encode()
        ).hexdigest()
    )


def upstream_id_for(tenant_id: str, upstream_name: str) -> str:
    return (
        "upstream:"
        + hashlib.sha256(f"{tenant_id}\x1f{upstream_name}".encode()).hexdigest()
    )


def route_id_for(tenant_id: str, route_name: str) -> str:
    return (
        "route:" + hashlib.sha256(f"{tenant_id}\x1f{route_name}".encode()).hexdigest()
    )


def config_id_for(tenant_id: str, config_name: str) -> str:
    return (
        "config:" + hashlib.sha256(f"{tenant_id}\x1f{config_name}".encode()).hexdigest()
    )


def feature_id_for(tenant_id: str, feature_name: str) -> str:
    return (
        "feature:"
        + hashlib.sha256(f"{tenant_id}\x1f{feature_name}".encode()).hexdigest()
    )


def release_pointer_id(tenant_id: str, target_kind: TargetKind, target_id: str) -> str:
    return (
        "release:"
        + hashlib.sha256(
            f"{tenant_id}\x1f{target_kind}\x1f{target_id}".encode()
        ).hexdigest()
    )


def tombstone_id(
    tenant_id: str, resource_kind: TargetKind, resource_id: str, version: int
) -> str:
    return (
        "tombstone:"
        + hashlib.sha256(
            f"{tenant_id}\x1f{resource_kind}\x1f{resource_id}\x1f{version}".encode()
        ).hexdigest()
    )


def activation_id_for(
    pointer_id: str, release_revision: int, activation_digest: str
) -> str:
    return (
        "activation:"
        + hashlib.sha256(
            f"{pointer_id}\x1f{release_revision}\x1f{activation_digest}".encode()
        ).hexdigest()
    )


class LifecycleState(ProtocolModel):
    """Retention metadata attached to every immutable record."""

    lifecycle_version: Literal["lifecycle.v1"]
    status: RecordStatus
    retention_policy_ref: OpaqueRef

    @model_validator(mode="after")
    def references_are_opaque(self) -> LifecycleState:
        _opaque_ref(self.retention_policy_ref, "retention_policy_ref")
        return self


class TenantScope(ProtocolModel):
    """Composite organization/tenant/principal/client authorization scope."""

    scope_version: Literal["tenant-scope.v1"]
    organization_id: Identifier
    tenant_id: Identifier
    principal_id: Identifier
    client_id: Identifier | None = None
    grant_digests: tuple[Digest, ...] = Field(default=(), max_length=_MAX_LIST)
    scope_digest: Digest

    @model_validator(mode="after")
    def scope_is_bound_and_normalized(self) -> TenantScope:
        _sorted_unique(self.grant_digests, "grant_digests")
        expected = _digest_payload(
            {
                "organization_id": self.organization_id,
                "tenant_id": self.tenant_id,
                "principal_id": self.principal_id,
                "client_id": self.client_id,
                "grant_digests": self.grant_digests,
            }
        )
        if self.scope_digest != expected:
            raise ValueError("tenant scope digest does not match its subject")
        return self


class OrganizationIdentity(ProtocolModel):
    identity_version: Literal["organization-identity.v1"]
    organization_id: Identifier
    organization_slug: NameText
    revision: int = Field(ge=1)
    lifecycle: LifecycleState
    identity_digest: Digest

    @model_validator(mode="after")
    def identity_is_stable(self) -> OrganizationIdentity:
        if self.organization_id != organization_id_for(self.organization_slug):
            raise ValueError("organization identity is not content-derived")
        if self.identity_digest != _record_digest(self):
            raise ValueError("organization identity digest drift")
        return self


class TenantIdentity(ProtocolModel):
    identity_version: Literal["tenant-identity.v1"]
    tenant_id: Identifier
    organization_id: Identifier
    tenant_slug: NameText
    revision: int = Field(ge=1)
    lifecycle: LifecycleState
    identity_digest: Digest

    @model_validator(mode="after")
    def identity_is_stable(self) -> TenantIdentity:
        if self.tenant_id != tenant_id_for(self.organization_id, self.tenant_slug):
            raise ValueError("tenant identity is not content-derived")
        if self.identity_digest != _record_digest(self):
            raise ValueError("tenant identity digest drift")
        return self


class PrincipalIdentity(ProtocolModel):
    identity_version: Literal["principal-identity.v1"]
    principal_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    principal_kind: IdentityKind
    subject_ref: OpaqueRef
    revision: int = Field(ge=1)
    lifecycle: LifecycleState
    identity_digest: Digest

    @model_validator(mode="after")
    def identity_is_stable_and_opaque(self) -> PrincipalIdentity:
        _opaque_ref(self.subject_ref, "subject_ref")
        if self.principal_id != principal_id_for(self.tenant_id, self.subject_ref):
            raise ValueError("principal identity is not content-derived")
        if self.identity_digest != _record_digest(self):
            raise ValueError("principal identity digest drift")
        return self


class ApiClientIdentity(ProtocolModel):
    identity_version: Literal["api-client-identity.v1"]
    client_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    principal_id: Identifier
    client_name: NameText
    key_ref: OpaqueRef
    revision: int = Field(ge=1)
    lifecycle: LifecycleState
    identity_digest: Digest

    @model_validator(mode="after")
    def identity_is_stable_and_secret_free(self) -> ApiClientIdentity:
        _key_ref(self.key_ref, "key_ref")
        if self.client_id != client_id_for(self.tenant_id, self.client_name):
            raise ValueError("API client identity is not content-derived")
        if self.identity_digest != _record_digest(self):
            raise ValueError("API client identity digest drift")
        return self


class RetentionPolicy(ProtocolModel):
    retention_version: Literal["retention-policy.v1"]
    retention_policy_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    retention_days: int = Field(ge=1, le=_MAX_RETENTION_DAYS)
    legal_hold: bool = False
    revision: int = Field(ge=1)
    policy_digest: Digest

    @model_validator(mode="after")
    def policy_is_content_addressed(self) -> RetentionPolicy:
        if self.policy_digest != _record_digest(self):
            raise ValueError("retention policy digest drift")
        return self


class Permission(ProtocolModel):
    record_version: Literal["permission.v1"]
    permission_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    action: NameText
    resource_ref: OpaqueRef
    lifecycle: LifecycleState
    record_digest: Digest

    @model_validator(mode="after")
    def permission_is_stable(self) -> Permission:
        _opaque_ref(self.resource_ref, "resource_ref")
        if self.permission_id != permission_id_for(
            self.tenant_id, self.action, self.resource_ref
        ):
            raise ValueError("permission identity is not content-derived")
        if self.record_digest != _record_digest(self):
            raise ValueError("permission digest drift")
        return self


class Role(ProtocolModel):
    record_version: Literal["role.v1"]
    role_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    role_name: NameText
    permission_ids: tuple[Identifier, ...] = Field(max_length=_MAX_LIST)
    lifecycle: LifecycleState
    record_digest: Digest

    @model_validator(mode="after")
    def role_is_stable(self) -> Role:
        _sorted_unique(self.permission_ids, "permission_ids")
        if self.role_id != role_id_for(self.tenant_id, self.role_name):
            raise ValueError("role identity is not content-derived")
        if self.record_digest != _record_digest(self):
            raise ValueError("role digest drift")
        return self


class Membership(ProtocolModel):
    record_version: Literal["membership.v1"]
    membership_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    principal_id: Identifier
    role_id: Identifier
    lifecycle: LifecycleState
    record_digest: Digest

    @model_validator(mode="after")
    def membership_is_stable(self) -> Membership:
        if self.membership_id != membership_id_for(
            self.tenant_id, self.principal_id, self.role_id
        ):
            raise ValueError("membership identity is not content-derived")
        if self.record_digest != _record_digest(self):
            raise ValueError("membership digest drift")
        return self


class Entitlement(ProtocolModel):
    record_version: Literal["entitlement.v1"]
    entitlement_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    subject_kind: SubjectKind
    subject_id: Identifier
    entitlement_name: NameText
    value_ref: OpaqueRef
    lifecycle: LifecycleState
    record_digest: Digest

    @model_validator(mode="after")
    def entitlement_is_stable(self) -> Entitlement:
        _opaque_ref(self.value_ref, "value_ref")
        if self.entitlement_id != entitlement_id_for(
            self.tenant_id, self.subject_kind, self.subject_id, self.entitlement_name
        ):
            raise ValueError("entitlement identity is not content-derived")
        if self.record_digest != _record_digest(self):
            raise ValueError("entitlement digest drift")
        return self


class QuotaDimension(ProtocolModel):
    dimension_name: NameText
    limit_value: int = Field(ge=1, le=10**12)
    window_seconds: int = Field(ge=1, le=31_536_000)


class QuotaContract(ProtocolModel):
    record_version: Literal["quota-contract.v1"]
    quota_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    subject_kind: SubjectKind
    subject_id: Identifier
    dimensions: tuple[QuotaDimension, ...] = Field(min_length=1, max_length=_MAX_LIST)
    lifecycle: LifecycleState
    record_digest: Digest

    @model_validator(mode="after")
    def quota_is_stable_without_hot_state(self) -> QuotaContract:
        names = tuple(item.dimension_name for item in self.dimensions)
        _sorted_unique(names, "quota dimension names")
        if self.quota_id != quota_id_for(
            self.tenant_id, self.subject_kind, self.subject_id
        ):
            raise ValueError("quota identity is not content-derived")
        if self.record_digest != _record_digest(self):
            raise ValueError("quota digest drift")
        return self


class GatewayUpstreamVersion(ProtocolModel):
    record_version: Literal["gateway-upstream.v1"]
    upstream_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    upstream_name: NameText
    version: int = Field(ge=1)
    endpoint_ref: OpaqueRef
    tls_key_ref: OpaqueRef | None = None
    ca_ref: OpaqueRef | None = None
    lifecycle: LifecycleState
    record_digest: Digest

    @model_validator(mode="after")
    def upstream_is_stable_and_secret_free(self) -> GatewayUpstreamVersion:
        _opaque_ref(self.endpoint_ref, "endpoint_ref")
        if self.tls_key_ref is not None:
            _key_ref(self.tls_key_ref, "tls_key_ref")
        if self.ca_ref is not None:
            _opaque_ref(self.ca_ref, "ca_ref")
        if self.upstream_id != upstream_id_for(self.tenant_id, self.upstream_name):
            raise ValueError("upstream identity is not content-derived")
        if self.record_digest != _record_digest(self):
            raise ValueError("upstream digest drift")
        return self


class GatewayRouteVersion(ProtocolModel):
    record_version: Literal["gateway-route.v1"]
    route_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    route_name: NameText
    version: int = Field(ge=1)
    path_template: PathText
    upstream_id: Identifier
    upstream_version: int = Field(ge=1)
    auth_policy_ref: OpaqueRef
    lifecycle: LifecycleState
    record_digest: Digest

    @model_validator(mode="after")
    def route_is_stable_and_safe(self) -> GatewayRouteVersion:
        if not self.path_template.startswith("/") or ".." in self.path_template.split(
            "/"
        ):
            raise ValueError("gateway route path must be absolute and escape-free")
        _opaque_ref(self.auth_policy_ref, "auth_policy_ref")
        if self.route_id != route_id_for(self.tenant_id, self.route_name):
            raise ValueError("route identity is not content-derived")
        if self.record_digest != _record_digest(self):
            raise ValueError("route digest drift")
        return self


class GatewayVersionRef(ProtocolModel):
    ref_version: Literal["gateway-version-ref.v1"]
    organization_id: Identifier
    tenant_id: Identifier
    target_kind: TargetKind
    target_id: Identifier
    target_version: int = Field(ge=1)
    target_digest: Digest
    ref_digest: Digest

    @model_validator(mode="after")
    def reference_is_content_addressed(self) -> GatewayVersionRef:
        if self.ref_digest != _record_digest(self):
            raise ValueError("gateway version reference digest drift")
        return self


class GatewayConfigVersion(ProtocolModel):
    record_version: Literal["gateway-config.v1"]
    config_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    config_name: NameText
    version: int = Field(ge=1)
    component_refs: tuple[GatewayVersionRef, ...] = Field(
        min_length=1, max_length=_MAX_LIST
    )
    lifecycle: LifecycleState
    record_digest: Digest

    @model_validator(mode="after")
    def config_is_stable_and_normalized(self) -> GatewayConfigVersion:
        refs = tuple(
            (ref.target_kind, ref.target_id, ref.target_version)
            for ref in self.component_refs
        )
        if tuple(sorted(refs)) != refs or len(set(refs)) != len(refs):
            raise ValueError(
                "gateway config component references must be unique and sorted"
            )
        if self.config_id != config_id_for(self.tenant_id, self.config_name):
            raise ValueError("config identity is not content-derived")
        if self.record_digest != _record_digest(self):
            raise ValueError("config digest drift")
        return self


class GatewayFeatureVersion(ProtocolModel):
    record_version: Literal["gateway-feature.v1"]
    feature_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    feature_name: NameText
    version: int = Field(ge=1)
    enabled: bool
    config_ref: GatewayVersionRef | None = None
    lifecycle: LifecycleState
    record_digest: Digest

    @model_validator(mode="after")
    def feature_is_stable(self) -> GatewayFeatureVersion:
        if self.config_ref is not None and self.config_ref.target_kind != "config":
            raise ValueError("feature config_ref must target a config version")
        if self.feature_id != feature_id_for(self.tenant_id, self.feature_name):
            raise ValueError("feature identity is not content-derived")
        if self.record_digest != _record_digest(self):
            raise ValueError("feature digest drift")
        return self


class CircuitHistory(ProtocolModel):
    record_version: Literal["circuit-history.v1"]
    circuit_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    state: CircuitState
    history_refs: tuple[OpaqueRef, ...] = Field(
        default=(), max_length=_MAX_CIRCUIT_REFS
    )
    transition_count: int = Field(ge=0, le=10**9)
    observed_at: Timestamp
    record_digest: Digest

    @model_validator(mode="after")
    def history_is_bounded_and_opaque(self) -> CircuitHistory:
        _sorted_unique(self.history_refs, "history_refs")
        for ref in self.history_refs:
            _opaque_ref(ref, "history_ref")
        if self.record_digest != _record_digest(self):
            raise ValueError("circuit history digest drift")
        return self


class TombstoneRecord(ProtocolModel):
    tombstone_version: Literal["tombstone.v1"]
    tombstone_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    resource_kind: TargetKind
    resource_id: Identifier
    resource_version: int = Field(ge=1)
    retention_policy_ref: OpaqueRef
    reason_ref: OpaqueRef
    tombstoned_at: Timestamp
    purge_after: Timestamp
    tombstone_digest: Digest

    @model_validator(mode="after")
    def tombstone_is_stable_and_opaque(self) -> TombstoneRecord:
        _opaque_ref(self.retention_policy_ref, "retention_policy_ref")
        _opaque_ref(self.reason_ref, "reason_ref")
        if self.tombstone_id != tombstone_id(
            self.tenant_id, self.resource_kind, self.resource_id, self.resource_version
        ):
            raise ValueError("tombstone identity is not content-derived")
        if self.tombstone_digest != _record_digest(self):
            raise ValueError("tombstone digest drift")
        return self


class ReleasePointer(ProtocolModel):
    pointer_version: Literal["release-pointer.v1"]
    pointer_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    target_kind: TargetKind
    target_id: Identifier
    target_version: int = Field(ge=1)
    target_digest: Digest
    revision: int = Field(ge=1)
    updated_by_principal_id: Identifier
    pointer_digest: Digest

    @model_validator(mode="after")
    def pointer_is_content_addressed(self) -> ReleasePointer:
        if self.pointer_id != release_pointer_id(
            self.tenant_id, self.target_kind, self.target_id
        ):
            raise ValueError("release pointer identity is not content-derived")
        if self.pointer_digest != _record_digest(self):
            raise ValueError("release pointer digest drift")
        return self


class ActivationRecord(ProtocolModel):
    activation_version: Literal["activation-record.v1"]
    activation_id: Identifier
    pointer_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    operation: ReleaseOperation
    target_kind: TargetKind
    target_id: Identifier
    target_version: int = Field(ge=1)
    target_digest: Digest
    previous_target_version: int | None = Field(default=None, ge=1)
    release_revision: int = Field(ge=1)
    actor_principal_id: Identifier
    change_ref: OpaqueRef
    activated_at: Timestamp
    activation_digest: Digest

    @model_validator(mode="after")
    def activation_is_immutable_and_opaque(self) -> ActivationRecord:
        _opaque_ref(self.change_ref, "change_ref")
        if self.operation == "rollback" and self.previous_target_version is None:
            raise ValueError("rollback activation requires a previous target")
        if self.activation_id != activation_id_for(
            self.pointer_id, self.release_revision, self.activation_digest
        ):
            raise ValueError("activation identity is not content-derived")
        if self.activation_digest != _record_digest(self):
            raise ValueError("activation digest drift")
        return self


class ReleaseMutation(ProtocolModel):
    mutation_version: Literal["release-mutation.v1"]
    pointer_id: Identifier
    organization_id: Identifier
    tenant_id: Identifier
    expected_revision: int = Field(ge=0)
    expected_pointer_digest: Digest | None = None
    operation: ReleaseOperation
    target_kind: TargetKind
    target_id: Identifier
    target_version: int = Field(ge=1)
    target_digest: Digest
    actor_principal_id: Identifier
    change_ref: OpaqueRef

    @model_validator(mode="after")
    def mutation_is_opaque(self) -> ReleaseMutation:
        _opaque_ref(self.change_ref, "change_ref")
        if self.expected_revision == 0 and self.expected_pointer_digest is not None:
            raise ValueError("an empty release pointer cannot have a prior digest")
        return self


__all__ = [
    "ActivationRecord",
    "ApiClientIdentity",
    "CircuitHistory",
    "Digest",
    "Entitlement",
    "GatewayConfigVersion",
    "GatewayFeatureVersion",
    "GatewayRouteVersion",
    "GatewayUpstreamVersion",
    "GatewayVersionRef",
    "Identifier",
    "IdentityKind",
    "LifecycleState",
    "Membership",
    "NameText",
    "OpaqueRef",
    "OrganizationIdentity",
    "PathText",
    "Permission",
    "PrincipalIdentity",
    "QuotaContract",
    "QuotaDimension",
    "RecordStatus",
    "ReleaseMutation",
    "ReleaseOperation",
    "ReleasePointer",
    "RetentionPolicy",
    "Role",
    "SubjectKind",
    "TargetKind",
    "TenantIdentity",
    "TenantScope",
    "Timestamp",
    "TombstoneRecord",
    "activation_id_for",
    "client_id_for",
    "config_id_for",
    "entitlement_id_for",
    "feature_id_for",
    "membership_id_for",
    "organization_id_for",
    "permission_id_for",
    "principal_id_for",
    "quota_id_for",
    "record_digest_for",
    "release_pointer_id",
    "role_id_for",
    "route_id_for",
    "tenant_id_for",
    "tombstone_id",
    "upstream_id_for",
]
