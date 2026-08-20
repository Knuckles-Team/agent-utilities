"""Focused NE-084 connector control-plane contract fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from agent_utilities.control_plane.connectors import (
    AccessScope,
    AuthorizationDecision,
    AuthorizationSet,
    CapabilityBinding,
    CompatibilityProfile,
    ConnectorControlPlane,
    ConnectorIdentity,
    ConnectorListRequest,
    ConnectorPage,
    ConnectorPageItem,
    ConnectorRegistration,
    ConnectorVersion,
    DesiredConnectorState,
    InventoryReference,
    KeysetCursor,
    Observation,
    ServerIdentity,
    activation_digest,
    capability_id_for,
    capability_set_digest,
    connector_id_for,
    normalize_capabilities,
    scope_digest_for,
    server_id_for,
    version_id_for,
)


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _capability(kind: str, name: str, version: str = "1") -> CapabilityBinding:
    payload = {
        "kind": kind,
        "name": name,
        "version": version,
        "schema_digest": _digest("a"),
        "protocol_version": "2026-06",
    }
    return CapabilityBinding(
        binding_id=capability_id_for(kind, name, version),
        **payload,
        capability_digest=activation_digest(payload),
    )


@dataclass
class _MemoryRepository:
    identities: dict[str, ConnectorIdentity] = field(default_factory=dict)
    servers: dict[str, ServerIdentity] = field(default_factory=dict)
    inventories: dict[str, InventoryReference] = field(default_factory=dict)
    versions: dict[str, ConnectorVersion] = field(default_factory=dict)
    desired: dict[str, DesiredConnectorState] = field(default_factory=dict)
    observations: dict[str, Observation] = field(default_factory=dict)
    authorizations: dict[tuple[str, str], AuthorizationSet] = field(
        default_factory=dict
    )
    authorization_decisions: dict[tuple[str, str], dict[str, AuthorizationDecision]] = (
        field(default_factory=dict)
    )
    page: ConnectorPage | None = None
    append_count: int = 0

    def put_connector(self, identity: ConnectorIdentity) -> None:
        self.identities[identity.connector_id] = identity

    def put_server(self, server: ServerIdentity) -> None:
        self.servers[server.server_id] = server

    def put_inventory(self, inventory: InventoryReference) -> None:
        self.inventories[inventory.manifest_ref] = inventory

    def put_version(self, version: ConnectorVersion) -> None:
        self.versions[version.version_id] = version

    def put_desired(self, desired: DesiredConnectorState) -> None:
        self.desired[desired.server.server_id] = desired

    def append_observation(self, observation: Observation) -> None:
        self.observations[observation.server_id] = observation
        self.append_count += 1

    def put_authorization(self, decision: AuthorizationDecision) -> None:
        key = (decision.server_id, decision.version_id)
        current = self.authorization_decisions.setdefault(key, {})
        current[decision.kind] = decision
        ordered = tuple(
            current[kind]
            for kind in ("approval", "install", "credential_access", "enable")
            if kind in current
        )
        self.authorizations[key] = AuthorizationSet(decisions=ordered)

    def get_server(self, scope: AccessScope, server_id: str) -> ServerIdentity | None:
        value = self.servers.get(server_id)
        return (
            value if value is not None and value.tenant_id == scope.tenant_id else None
        )

    def get_version(
        self, scope: AccessScope, version_id: str
    ) -> ConnectorVersion | None:
        del scope
        return self.versions.get(version_id)

    def get_desired(
        self, scope: AccessScope, server_id: str
    ) -> DesiredConnectorState | None:
        value = self.desired.get(server_id)
        return (
            value
            if value is not None and value.server.tenant_id == scope.tenant_id
            else None
        )

    def get_latest_observation(
        self, scope: AccessScope, server_id: str
    ) -> Observation | None:
        del scope
        return self.observations.get(server_id)

    def get_authorizations(
        self, scope: AccessScope, server_id: str, version_id: str
    ) -> AuthorizationSet | None:
        value = self.authorizations.get((server_id, version_id))
        if value is None or any(
            item.tenant_id != scope.tenant_id for item in value.decisions
        ):
            return None
        return value

    def list_connectors(self, request: ConnectorListRequest) -> ConnectorPage:
        del request
        assert self.page is not None
        return self.page

    def cursor_for(
        self, scope: AccessScope, after_server_id: str, after_version_id: str
    ) -> KeysetCursor:
        return KeysetCursor(
            cursor_version="connector-keyset-cursor.v1",
            scope_digest=scope.scope_digest,
            after_server_id=after_server_id,
            after_version_id=after_version_id,
        )


def _fixture() -> tuple[
    _MemoryRepository, ConnectorControlPlane, AccessScope, ConnectorRegistration
]:
    tenant = "tenant:alpha"
    connector_id = connector_id_for("knuckles", "gitlab-api")
    identity = ConnectorIdentity(
        identity_version="connector-identity.v1",
        connector_id=connector_id,
        publisher="knuckles",
        package_name="gitlab-api",
        protocol="mcp",
    )
    server = ServerIdentity(
        identity_version="connector-server-identity.v1",
        server_id=server_id_for(tenant, connector_id, "primary"),
        connector=identity,
        tenant_id=tenant,
        instance_ref="primary",
        display_name="gitlab-primary",
    )
    inventory = InventoryReference(
        inventory_version="connector-inventory-ref.v1",
        package_ref="pkg:gitlab-api",
        package_version="1.4.0",
        manifest_ref="manifest:gitlab:v1",
        manifest_digest=_digest("b"),
        source_ref="repo:gitlab-api",
        signer_ref="signer:release",
    )
    compatibility = CompatibilityProfile(
        compatibility_version="connector-compatibility.v1",
        protocol="mcp",
        protocol_versions=("2026-06",),
        engine_versions=("1",),
        platforms=("linux",),
        features=("schema-pinning",),
    )
    capabilities = normalize_capabilities(
        (
            _capability("tool", "issues.list"),
            _capability("resource", "issues://project"),
        )
    )
    version = ConnectorVersion(
        version_record_version="connector-version.v1",
        version_id=version_id_for(
            connector_id,
            inventory.package_version,
            inventory.manifest_digest,
            _digest("c"),
            capability_set_digest(capabilities),
        ),
        connector_id=connector_id,
        version=inventory.package_version,
        manifest_ref=inventory.manifest_ref,
        manifest_digest=inventory.manifest_digest,
        mapping_digest=_digest("c"),
        compatibility=compatibility,
        capabilities=capabilities,
        capability_set_digest=capability_set_digest(capabilities),
        inventory=inventory,
    )
    desired = DesiredConnectorState(
        state_version="connector-desired-state.v1",
        server=server,
        version_id=version.version_id,
        inventory_ref=inventory.manifest_ref,
        desired_status="enabled",
        revision=1,
        change_ref="change:initial",
    )
    scope = AccessScope(
        scope_version="connector-access-scope.v1",
        tenant_id=tenant,
        principal_id="principal:operator",
        grant_digests=(_digest("d"),),
        scope_digest=scope_digest_for(tenant, "principal:operator", (_digest("d"),)),
    )
    registration = ConnectorRegistration(
        registration_version="connector-registration.v1",
        identity=identity,
        server=server,
        inventory=inventory,
        version=version,
        desired=desired,
    )
    repository = _MemoryRepository()
    plane = ConnectorControlPlane(repository)
    plane.register(registration)
    return repository, plane, scope, registration


def _observation(
    registration: ConnectorRegistration, **overrides: object
) -> Observation:
    values: dict[str, object] = {
        "observation_version": "connector-observation.v1",
        "observation_ref": "observation:1",
        "server_id": registration.server.server_id,
        "version_id": registration.version.version_id,
        "manifest_digest": registration.version.manifest_digest,
        "status": "ready",
        "capabilities": registration.version.capabilities,
        "compatibility": registration.version.compatibility,
        "observed_at": "2026-08-19T00:00:00Z",
        "probe_complete": True,
    }
    values.update(overrides)
    return Observation(**values)


def _decision(
    registration: ConnectorRegistration,
    scope: AccessScope,
    kind: str,
    *,
    outcome: str = "approved",
) -> AuthorizationDecision:
    return AuthorizationDecision(
        decision_version="connector-authorization.v1",
        decision_id=f"decision:{kind}",
        kind=kind,
        outcome=outcome,
        tenant_id=scope.tenant_id,
        principal_id=scope.principal_id,
        server_id=registration.server.server_id,
        version_id=registration.version.version_id,
        grant_digest=scope.grant_digests[0] if kind == "credential_access" else None,
        evidence_ref=f"evidence:{kind}",
        decided_at="2026-08-19T00:00:00Z",
    )


def test_identity_and_version_records_are_stable_and_immutable() -> None:
    _, _, _, registration = _fixture()
    assert registration.server.server_id == server_id_for(
        registration.server.tenant_id,
        registration.identity.connector_id,
        registration.server.instance_ref,
    )
    assert registration.version.version_id.startswith("version:")
    with pytest.raises(ValidationError):
        registration.version.model_validate(
            {
                **registration.version.model_dump(mode="json"),
                "manifest_digest": _digest("e"),
            }
        )


def test_empty_or_failed_probe_never_removes_or_rewrites_desired_state() -> None:
    repository, plane, scope, registration = _fixture()
    initial = repository.desired[registration.server.server_id]
    empty = _observation(
        registration,
        status="empty",
        probe_complete=True,
        authoritative_snapshot=False,
        verified_empty=False,
        capabilities=(),
        compatibility=None,
        manifest_digest=None,
    )
    result = plane.record_observation(scope, empty)
    assert result.drift.codes == ("empty_snapshot_unverified",)
    assert repository.desired[registration.server.server_id] == initial

    failed = _observation(
        registration,
        observation_ref="observation:2",
        status="failed",
        probe_complete=False,
        capabilities=(),
        compatibility=None,
        manifest_digest=None,
        failure_code="timeout",
    )
    result = plane.record_observation(scope, failed)
    assert result.drift.codes == ("probe_failed",)
    assert repository.desired[registration.server.server_id] == initial
    with pytest.raises(ValidationError):
        _observation(
            registration,
            status="empty",
            verified_empty=True,
            authoritative_snapshot=False,
        )


def test_ready_probe_matching_release_is_clear_and_projection_is_privacy_safe() -> None:
    _, plane, scope, registration = _fixture()
    result = plane.record_observation(scope, _observation(registration))
    assert result.drift.codes == ("no_drift",)
    assert result.lifecycle.desired_status == "enabled"
    assert result.lifecycle.observed_status == "ready"
    assert result.projection.drift_status == "clear"
    projection_text = result.projection.model_dump_json()
    assert "https://" not in projection_text
    assert "timeout" not in projection_text
    assert "principal:operator" not in projection_text


def test_separate_authorization_decisions_require_all_four_and_scope_grant() -> None:
    _, plane, scope, registration = _fixture()
    plane.authorize(_decision(registration, scope, "approval"))
    partial = plane.evaluate_authorization(
        scope,
        registration.server.server_id,
        registration.version.version_id,
        at="2026-08-19T01:00:00Z",
    )
    assert not partial.allowed
    assert partial.missing_or_denied == ("install", "credential_access", "enable")
    for kind in ("approval", "install", "credential_access", "enable"):
        plane.authorize(_decision(registration, scope, kind))
    allowed = plane.evaluate_authorization(
        scope,
        registration.server.server_id,
        registration.version.version_id,
        at="2026-08-19T01:00:00Z",
    )
    assert allowed.allowed
    assert allowed.missing_or_denied == ()

    plane.authorize(_decision(registration, scope, "enable", outcome="revoked"))
    denied = plane.evaluate_authorization(
        scope,
        registration.server.server_id,
        registration.version.version_id,
        at="2026-08-19T01:00:00Z",
    )
    assert not denied.allowed
    assert denied.missing_or_denied == ("enable",)


def test_cursor_is_scope_bound_and_page_is_bounded_before_return() -> None:
    repository, plane, scope, registration = _fixture()
    wrong_scope = AccessScope(
        scope_version="connector-access-scope.v1",
        tenant_id=scope.tenant_id,
        principal_id="principal:other",
        grant_digests=scope.grant_digests,
        scope_digest=scope_digest_for(
            scope.tenant_id, "principal:other", scope.grant_digests
        ),
    )
    cursor = KeysetCursor(
        cursor_version="connector-keyset-cursor.v1",
        scope_digest=wrong_scope.scope_digest,
        after_server_id=registration.server.server_id,
        after_version_id=registration.version.version_id,
    )
    with pytest.raises(ValidationError):
        ConnectorListRequest(scope=scope, cursor=cursor)

    repository.page = ConnectorPage(
        page_version="connector-page.v1",
        items=(
            ConnectorPageItem(
                tenant_id=scope.tenant_id,
                server_id=registration.server.server_id,
                connector_id=registration.identity.connector_id,
                version_id=registration.version.version_id,
                desired_status="enabled",
                observed_status="unknown",
                drift_status="degraded",
                capability_count=0,
                capability_set_digest=registration.version.capability_set_digest,
                manifest_digest=registration.version.manifest_digest,
                package_ref=registration.version.inventory.package_ref,
            ),
        ),
    )
    page = plane.list(ConnectorListRequest(scope=scope, limit=1))
    assert len(page.items) == 1
