"""Strict connector control-plane protocol models.

This is a domain contract, not a database schema.  It keeps connector
identity, immutable releases, desired registration, observations, capability
bindings, authorization decisions and graph projections separate so a
transport or repository cannot accidentally turn a failed probe into a state
mutation.  No model contains provider credentials, command arguments, raw
probe errors or private endpoints.

(CONCEPT:AU-ECO.connector.factory-ingestion-adaptor,
AU-OS.config.desired-state-fleet-reconciler,
AU-KG.ingest.fleet-catalog-relational-tables)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator

from agent_utilities.protocols.epistemic_operations import ProtocolModel

Identifier: TypeAlias = Annotated[
    str,
    Field(
        min_length=1,
        max_length=192,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$",
    ),
]
Digest: TypeAlias = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
VersionText: TypeAlias = Annotated[
    str,
    Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]*$"),
]
Timestamp: TypeAlias = Annotated[
    str,
    Field(
        min_length=20,
        max_length=64,
        pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|z|[+-][0-9]{2}:[0-9]{2})$",
    ),
]

ProtocolName = Literal["mcp", "a2a", "source"]
CapabilityKind = Literal["tool", "resource", "prompt"]
DesiredStatus = Literal["disabled", "enabled", "quarantined"]
ObservationStatus = Literal["ready", "empty", "unreachable", "failed", "quarantined"]
DriftStatus = Literal["clear", "degraded", "drifted", "quarantined"]
AuthorizationKind = Literal["approval", "install", "credential_access", "enable"]
AuthorizationOutcome = Literal["pending", "approved", "denied", "revoked"]

_MAX_CAPABILITIES = 512
_MAX_GRANTS = 64
_MAX_COMPATIBILITY_VALUES = 32
_MAX_PAGE_SIZE = 100


def _canonical(value: str) -> str:
    return str(value).strip().casefold()


def _digest_payload(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def connector_id_for(
    publisher: str,
    package_name: str,
    protocol: ProtocolName = "mcp",
) -> str:
    """Return a stable connector identity without embedding a host or secret."""

    return (
        "connector:"
        + hashlib.sha256(
            "\x1f".join(
                (_canonical(publisher), _canonical(package_name), protocol)
            ).encode("utf-8")
        ).hexdigest()
    )


def server_id_for(
    tenant_id: str, connector_id: str, instance_ref: str = "default"
) -> str:
    """Return a tenant-scoped stable server identity."""

    return (
        "server:"
        + hashlib.sha256(
            "\x1f".join(
                (
                    _canonical(tenant_id),
                    _canonical(connector_id),
                    _canonical(instance_ref),
                )
            ).encode("utf-8")
        ).hexdigest()
    )


def activation_digest(value: Mapping[str, object] | object) -> str:
    """Hash an allow-listed binding payload for immutable version evidence."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        raise TypeError("activation digest requires a mapping or protocol model")
    return _digest_payload(dict(value))


def capability_id_for(kind: CapabilityKind, name: str, version: str) -> str:
    return (
        "capability:"
        + hashlib.sha256(
            "\x1f".join(
                (_canonical(kind), _canonical(name), _canonical(version))
            ).encode("utf-8")
        ).hexdigest()
    )


def capability_set_digest(capabilities: Iterable[CapabilityBinding]) -> str:
    ordered = [
        capability.model_dump(mode="json")
        for capability in sorted(capabilities, key=lambda item: item.binding_id)
    ]
    return _digest_payload(ordered)


def version_id_for(
    connector_id: str,
    version: str,
    manifest_digest: str,
    mapping_digest: str,
    capability_digest: str,
) -> str:
    return (
        "version:"
        + hashlib.sha256(
            "\x1f".join(
                (
                    connector_id,
                    version,
                    manifest_digest,
                    mapping_digest,
                    capability_digest,
                )
            ).encode("utf-8")
        ).hexdigest()
    )


class ConnectorIdentity(ProtocolModel):
    """Stable package identity independent of a configured server instance."""

    identity_version: Literal["connector-identity.v1"]
    connector_id: Identifier
    publisher: Identifier
    package_name: Identifier
    protocol: ProtocolName

    @model_validator(mode="after")
    def id_is_derived(self) -> ConnectorIdentity:
        if self.connector_id != connector_id_for(
            self.publisher, self.package_name, self.protocol
        ):
            raise ValueError("connector id is not derived from its identity fields")
        return self


class ServerIdentity(ProtocolModel):
    """Stable tenant/server identity; instance labels never contain endpoints."""

    identity_version: Literal["connector-server-identity.v1"]
    server_id: Identifier
    connector: ConnectorIdentity
    tenant_id: Identifier
    instance_ref: Identifier
    display_name: Identifier

    @model_validator(mode="after")
    def id_is_derived(self) -> ServerIdentity:
        if self.server_id != server_id_for(
            self.tenant_id, self.connector.connector_id, self.instance_ref
        ):
            raise ValueError("server id is not derived from its tenant connector scope")
        return self


class CompatibilityProfile(ProtocolModel):
    """Normalized, bounded compatibility declarations."""

    compatibility_version: Literal["connector-compatibility.v1"]
    protocol: ProtocolName
    protocol_versions: tuple[VersionText, ...] = Field(
        min_length=1, max_length=_MAX_COMPATIBILITY_VALUES
    )
    engine_versions: tuple[VersionText, ...] = Field(
        default=(), max_length=_MAX_COMPATIBILITY_VALUES
    )
    platforms: tuple[Identifier, ...] = Field(
        default=(), max_length=_MAX_COMPATIBILITY_VALUES
    )
    features: tuple[Identifier, ...] = Field(
        default=(), max_length=_MAX_COMPATIBILITY_VALUES
    )

    @model_validator(mode="after")
    def values_are_unique_and_normalized(self) -> CompatibilityProfile:
        for values in (
            self.protocol_versions,
            self.engine_versions,
            self.platforms,
            self.features,
        ):
            if len(set(values)) != len(values) or tuple(sorted(values)) != values:
                raise ValueError("compatibility values must be sorted and unique")
        return self


class CapabilityBinding(ProtocolModel):
    """Exact tool/resource/prompt and schema version binding."""

    binding_id: Identifier
    kind: CapabilityKind
    name: Identifier
    version: VersionText
    schema_digest: Digest
    protocol_version: VersionText
    capability_digest: Digest

    @model_validator(mode="after")
    def binding_is_immutable_and_exact(self) -> CapabilityBinding:
        if self.binding_id != capability_id_for(self.kind, self.name, self.version):
            raise ValueError("capability binding id is not stable")
        expected = activation_digest(
            {
                "kind": self.kind,
                "name": self.name,
                "version": self.version,
                "schema_digest": self.schema_digest,
                "protocol_version": self.protocol_version,
            }
        )
        if self.capability_digest != expected:
            raise ValueError("capability digest does not match its binding")
        return self


def normalize_capabilities(
    capabilities: Iterable[CapabilityBinding],
) -> tuple[CapabilityBinding, ...]:
    """Return one deterministic capability set and reject duplicate identities."""

    values = tuple(capabilities)
    if len(values) > _MAX_CAPABILITIES:
        raise ValueError("connector capability set exceeds its bounded limit")
    ids = [item.binding_id for item in values]
    if len(set(ids)) != len(ids):
        raise ValueError("connector capability identities must be unique")
    return tuple(sorted(values, key=lambda item: item.binding_id))


class InventoryReference(ProtocolModel):
    """Controlled package inventory coordinate; never a filesystem or secret."""

    inventory_version: Literal["connector-inventory-ref.v1"]
    package_ref: Identifier
    package_version: VersionText
    manifest_ref: Identifier
    manifest_digest: Digest
    source_ref: Identifier
    signer_ref: Identifier | None = None

    @model_validator(mode="after")
    def references_are_controlled(self) -> InventoryReference:
        values = (self.package_ref, self.manifest_ref, self.source_ref, self.signer_ref)
        if any(
            value is not None
            and value.casefold().startswith(
                ("http:", "https:", "file:", "env:", "secret:", "vault:")
            )
            for value in values
        ):
            raise ValueError(
                "inventory references must use controlled opaque coordinates"
            )
        return self


class ConnectorVersion(ProtocolModel):
    """Immutable, digest-pinned connector release record."""

    version_record_version: Literal["connector-version.v1"]
    version_id: Identifier
    connector_id: Identifier
    version: VersionText
    manifest_ref: Identifier
    manifest_digest: Digest
    mapping_digest: Digest
    compatibility: CompatibilityProfile
    capabilities: tuple[CapabilityBinding, ...] = Field(
        min_length=0, max_length=_MAX_CAPABILITIES
    )
    capability_set_digest: Digest
    inventory: InventoryReference

    @model_validator(mode="after")
    def release_is_self_consistent(self) -> ConnectorVersion:
        if self.inventory.manifest_digest != self.manifest_digest:
            raise ValueError("inventory and connector manifest digests differ")
        if self.inventory.manifest_ref != self.manifest_ref:
            raise ValueError("inventory and connector manifest references differ")
        if self.inventory.package_version != self.version:
            raise ValueError("inventory and connector versions differ")
        normalized = normalize_capabilities(self.capabilities)
        if normalized != self.capabilities:
            raise ValueError("connector capabilities must be normalized")
        if self.capability_set_digest != capability_set_digest(self.capabilities):
            raise ValueError("connector capability-set digest does not match")
        expected_id = version_id_for(
            self.connector_id,
            self.version,
            self.manifest_digest,
            self.mapping_digest,
            self.capability_set_digest,
        )
        if self.version_id != expected_id:
            raise ValueError("connector version id is not immutable")
        return self


class DesiredConnectorState(ProtocolModel):
    """Operator-owned target state; observations never overwrite this record."""

    state_version: Literal["connector-desired-state.v1"]
    server: ServerIdentity
    version_id: Identifier
    inventory_ref: Identifier
    desired_status: DesiredStatus
    revision: int = Field(ge=1)
    change_ref: Identifier


ProbeFailureCode = Literal[
    "timeout",
    "transport_unavailable",
    "authentication_unavailable",
    "schema_unavailable",
    "schema_mismatch",
    "policy_denied",
    "malformed_response",
]


class Observation(ProtocolModel):
    """Immutable probe outcome with explicit empty-snapshot proof."""

    observation_version: Literal["connector-observation.v1"]
    observation_ref: Identifier
    server_id: Identifier
    version_id: Identifier
    manifest_digest: Digest | None = None
    status: ObservationStatus
    capabilities: tuple[CapabilityBinding, ...] = Field(
        default=(), max_length=_MAX_CAPABILITIES
    )
    compatibility: CompatibilityProfile | None = None
    observed_at: Timestamp
    probe_complete: bool
    authoritative_snapshot: bool = False
    verified_empty: bool = False
    failure_code: ProbeFailureCode | None = None

    @model_validator(mode="after")
    def probe_result_is_honest(self) -> Observation:
        normalized = normalize_capabilities(self.capabilities)
        if normalized != self.capabilities:
            raise ValueError("observed capabilities must be normalized")
        if self.verified_empty and not (
            self.status == "empty"
            and self.probe_complete
            and self.authoritative_snapshot
        ):
            raise ValueError(
                "only a complete authoritative empty snapshot may be verified"
            )
        if self.status in {"unreachable", "failed"} and self.verified_empty:
            raise ValueError("failed or transient probes cannot be verified empty")
        if (
            self.status == "empty"
            and self.authoritative_snapshot
            and not self.probe_complete
        ):
            raise ValueError("authoritative empty snapshots must be complete")
        if self.status in {"unreachable", "failed"} and self.failure_code is None:
            raise ValueError("failed observations require a bounded failure code")
        if self.status == "ready" and self.failure_code is not None:
            raise ValueError("ready observations cannot carry a failure code")
        if self.status == "ready" and (
            self.manifest_digest is None or self.compatibility is None
        ):
            raise ValueError(
                "ready observations require manifest and compatibility evidence"
            )
        return self


DriftCode = Literal[
    "no_drift",
    "no_observation",
    "probe_failed",
    "probe_unreachable",
    "empty_snapshot_unverified",
    "verified_empty",
    "version_mismatch",
    "manifest_mismatch",
    "capability_mismatch",
    "compatibility_mismatch",
    "observation_quarantined",
]

_DRIFT_SUMMARIES: dict[str, str] = {
    "no_drift": "connector matches the approved desired state",
    "no_observation": "no connector observation is available",
    "probe_failed": "connector probe failed; desired state was retained",
    "probe_unreachable": "connector probe was unreachable; desired state was retained",
    "empty_snapshot_unverified": "empty probe was not an authoritative snapshot",
    "verified_empty": "authoritative connector snapshot is empty",
    "version_mismatch": "observed connector version differs from desired state",
    "manifest_mismatch": "observed manifest digest differs from the approved release",
    "capability_mismatch": "observed capabilities differ from the approved release",
    "compatibility_mismatch": "observed compatibility differs from the approved release",
    "observation_quarantined": "connector observation is quarantined",
}


class DriftRecord(ProtocolModel):
    """Privacy-safe reconciliation result; raw probe text is never retained."""

    drift_version: Literal["connector-drift.v1"]
    server_id: Identifier
    status: DriftStatus
    codes: tuple[DriftCode, ...] = Field(min_length=1, max_length=16)
    quarantine_ref: Identifier | None = None
    observation_ref: Identifier | None = None

    @model_validator(mode="after")
    def codes_are_unique_and_allowlisted(self) -> DriftRecord:
        if len(set(self.codes)) != len(self.codes):
            raise ValueError("drift codes must be unique")
        for code in self.codes:
            if code not in _DRIFT_SUMMARIES:
                raise ValueError("drift code is not allow-listed")
        if self.status == "quarantined" and self.quarantine_ref is None:
            raise ValueError("quarantined drift requires an opaque quarantine ref")
        return self


class ConnectorLifecycle(ProtocolModel):
    """Combined read model; desired and observed fields remain distinguishable."""

    lifecycle_version: Literal["connector-lifecycle.v1"]
    server_id: Identifier
    desired_status: DesiredStatus
    observed_status: ObservationStatus | Literal["unknown"]
    drift_status: DriftStatus
    desired_revision: int = Field(ge=1)
    observation_ref: Identifier | None = None
    quarantine_ref: Identifier | None = None


class AuthorizationDecision(ProtocolModel):
    """One independently auditable approval/install/credential/enable decision."""

    decision_version: Literal["connector-authorization.v1"]
    decision_id: Identifier
    kind: AuthorizationKind
    outcome: AuthorizationOutcome
    tenant_id: Identifier
    principal_id: Identifier
    server_id: Identifier
    version_id: Identifier
    grant_digest: Digest | None = None
    evidence_ref: Identifier
    decided_at: Timestamp
    expires_at: Timestamp | None = None

    @model_validator(mode="after")
    def grant_is_scoped(self) -> AuthorizationDecision:
        if self.kind == "credential_access" and self.grant_digest is None:
            raise ValueError("credential authorization requires a grant digest")
        if self.kind != "credential_access" and self.grant_digest is not None:
            raise ValueError("only credential authorization may carry a grant digest")
        return self


class AuthorizationSet(ProtocolModel):
    """One or more separate decisions for one server/version scope."""

    decisions: tuple[AuthorizationDecision, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def has_one_decision_per_kind(self) -> AuthorizationSet:
        kinds = {item.kind for item in self.decisions}
        if len(kinds) != len(self.decisions):
            raise ValueError("authorization set must contain one decision per kind")
        first = self.decisions[0]
        if any(
            (
                item.tenant_id != first.tenant_id
                or item.principal_id != first.principal_id
                or item.server_id != first.server_id
                or item.version_id != first.version_id
            )
            for item in self.decisions[1:]
        ):
            raise ValueError("authorization decisions must share one exact scope")
        return self


class AuthorizationEvaluation(ProtocolModel):
    """Bounded decision result with no policy prose or credential material."""

    evaluation_version: Literal["connector-authorization-evaluation.v1"]
    server_id: Identifier
    version_id: Identifier
    allowed: bool
    missing_or_denied: tuple[AuthorizationKind, ...] = Field(default=(), max_length=4)


def scope_digest_for(
    tenant_id: str,
    principal_id: str,
    grant_digests: Iterable[str],
) -> str:
    return _digest_payload(
        {
            "tenant_id": tenant_id,
            "principal_id": principal_id,
            "grant_digests": sorted(set(grant_digests)),
        }
    )


class AccessScope(ProtocolModel):
    """Tenant/principal/grant visibility boundary for every list/read."""

    scope_version: Literal["connector-access-scope.v1"]
    tenant_id: Identifier
    principal_id: Identifier
    grant_digests: tuple[Digest, ...] = Field(default=(), max_length=_MAX_GRANTS)
    scope_digest: Digest

    @model_validator(mode="after")
    def scope_is_self_consistent(self) -> AccessScope:
        if len(set(self.grant_digests)) != len(self.grant_digests):
            raise ValueError("grant digests must be unique")
        if tuple(sorted(self.grant_digests)) != self.grant_digests:
            raise ValueError("grant digests must be sorted")
        if self.scope_digest != scope_digest_for(
            self.tenant_id, self.principal_id, self.grant_digests
        ):
            raise ValueError(
                "access scope digest does not match its subject and grants"
            )
        return self


class KeysetCursor(ProtocolModel):
    """Bounded keyset position; repositories may wrap it in a signed token."""

    cursor_version: Literal["connector-keyset-cursor.v1"]
    scope_digest: Digest
    after_server_id: Identifier
    after_version_id: Identifier


class ConnectorListRequest(ProtocolModel):
    """Bounded, scope-bound connector list request."""

    scope: AccessScope
    limit: int = Field(default=50, ge=1, le=_MAX_PAGE_SIZE)
    cursor: KeysetCursor | None = None

    @model_validator(mode="after")
    def cursor_is_scope_bound(self) -> ConnectorListRequest:
        if (
            self.cursor is not None
            and self.cursor.scope_digest != self.scope.scope_digest
        ):
            raise ValueError(
                "connector cursor is bound to a different visibility scope"
            )
        return self


class ConnectorPageItem(ProtocolModel):
    """Privacy-safe connector summary suitable for a bounded list response."""

    tenant_id: Identifier
    server_id: Identifier
    connector_id: Identifier
    version_id: Identifier
    desired_status: DesiredStatus
    observed_status: ObservationStatus | Literal["unknown"]
    drift_status: DriftStatus
    capability_count: int = Field(ge=0, le=_MAX_CAPABILITIES)
    capability_set_digest: Digest
    manifest_digest: Digest
    package_ref: Identifier
    observed_at: Timestamp | None = None


class ConnectorPage(ProtocolModel):
    """Bounded page returned by a repository implementation."""

    page_version: Literal["connector-page.v1"]
    items: tuple[ConnectorPageItem, ...] = Field(max_length=_MAX_PAGE_SIZE)
    next_cursor: KeysetCursor | None = None

    @model_validator(mode="after")
    def page_is_bounded(self) -> ConnectorPage:
        if self.next_cursor is not None and self.items == ():
            raise ValueError("an empty connector page cannot advance a cursor")
        return self


class ConnectorGraphProjection(ProtocolModel):
    """Allow-listed projection for graph reasoning; no endpoint or probe text."""

    projection_version: Literal["connector-graph-projection.v1"]
    tenant_id: Identifier
    server_id: Identifier
    connector_id: Identifier
    version_id: Identifier
    desired_status: DesiredStatus
    observed_status: ObservationStatus | Literal["unknown"]
    drift_status: DriftStatus
    capability_count: int = Field(ge=0, le=_MAX_CAPABILITIES)
    capability_set_digest: Digest
    manifest_digest: Digest
    package_ref: Identifier
    drift_codes: tuple[DriftCode, ...] = Field(default=(), max_length=16)
    observation_ref: Identifier | None = None
    quarantine_ref: Identifier | None = None
