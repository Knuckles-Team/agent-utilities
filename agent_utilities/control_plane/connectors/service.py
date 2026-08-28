"""Pure connector control-plane orchestration over the repository protocol."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import model_validator

from agent_utilities.protocols.epistemic_operations import ProtocolModel

from .models import (
    AccessScope,
    AuthorizationDecision,
    AuthorizationEvaluation,
    ConnectorGraphProjection,
    ConnectorIdentity,
    ConnectorLifecycle,
    ConnectorListRequest,
    ConnectorPage,
    ConnectorPageItem,
    ConnectorVersion,
    DesiredConnectorState,
    DriftCode,
    DriftRecord,
    DriftStatus,
    InventoryReference,
    Observation,
    ObservationStatus,
    ServerIdentity,
    capability_set_digest,
)
from .repository import ConnectorRepository, RepositoryContractError


class ConnectorRegistration(ProtocolModel):
    """Atomic registration bundle passed to a repository adapter."""

    registration_version: Literal["connector-registration.v1"]
    identity: ConnectorIdentity
    server: ServerIdentity
    inventory: InventoryReference
    version: ConnectorVersion
    desired: DesiredConnectorState

    @model_validator(mode="after")
    def bundle_is_consistent(self) -> ConnectorRegistration:
        if self.server.connector != self.identity:
            raise ValueError("registration server does not belong to its connector")
        if self.version.connector_id != self.identity.connector_id:
            raise ValueError("registration version belongs to another connector")
        if self.inventory != self.version.inventory:
            raise ValueError("registration inventory differs from immutable version")
        if self.server.server_id != self.desired.server.server_id:
            raise ValueError("registration desired state targets another server")
        if self.desired.version_id != self.version.version_id:
            raise ValueError("registration desired state targets another version")
        if self.desired.inventory_ref != self.inventory.manifest_ref:
            raise ValueError("registration desired state is not bound to its inventory")
        return self


class ConnectorReconciliation(ProtocolModel):
    """Explicit desired/observed/drift read model returned by reconciliation."""

    reconciliation_version: Literal["connector-reconciliation.v1"]
    lifecycle: ConnectorLifecycle
    desired: DesiredConnectorState
    drift: DriftRecord
    projection: ConnectorGraphProjection
    observation: Observation | None = None


def _quarantine_ref(
    server_id: str, observation_ref: str | None, codes: tuple[DriftCode, ...]
) -> str:
    payload = json.dumps(
        {"server_id": server_id, "observation_ref": observation_ref, "codes": codes},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "quarantine:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scope_matches(scope: AccessScope, server: ServerIdentity) -> None:
    if server.tenant_id != scope.tenant_id:
        raise RepositoryContractError("repository returned a cross-tenant connector")


def _status_only_drift_codes(observation: Observation) -> tuple[DriftCode, ...] | None:
    """Drift codes decidable from ``observation.status`` alone, or ``None``."""
    if observation.status == "failed":
        return ("probe_failed",)
    if observation.status == "unreachable":
        return ("probe_unreachable",)
    if observation.status == "quarantined":
        return ("observation_quarantined",)
    if observation.status == "empty" and not observation.verified_empty:
        return ("empty_snapshot_unverified",)
    if observation.status == "empty" and observation.verified_empty:
        return ("verified_empty",)
    return None


def _version_drift_codes(
    desired: DesiredConnectorState,
    version: ConnectorVersion | None,
    observation: Observation,
) -> list[DriftCode]:
    codes: list[DriftCode] = []
    if observation.version_id != desired.version_id:
        codes.append("version_mismatch")
    if version is None:
        codes.append("version_mismatch")
    else:
        if observation.manifest_digest != version.manifest_digest:
            codes.append("manifest_mismatch")
        if (
            capability_set_digest(observation.capabilities)
            != version.capability_set_digest
        ):
            codes.append("capability_mismatch")
        if observation.compatibility != version.compatibility:
            codes.append("compatibility_mismatch")
    return codes


def _codes_for(
    desired: DesiredConnectorState,
    version: ConnectorVersion | None,
    observation: Observation | None,
) -> tuple[DriftCode, ...]:
    if observation is None:
        return ("no_observation",)
    status_only = _status_only_drift_codes(observation)
    if status_only is not None:
        return status_only
    codes = _version_drift_codes(desired, version, observation)
    return tuple(dict.fromkeys(codes)) or ("no_drift",)


def _drift_status(
    observation: Observation | None, codes: tuple[DriftCode, ...]
) -> DriftStatus:
    if observation is not None and observation.status == "quarantined":
        return "quarantined"
    if codes == ("no_drift",):
        return "clear"
    if codes in {
        ("probe_failed",),
        ("probe_unreachable",),
        ("empty_snapshot_unverified",),
        ("no_observation",),
    }:
        return "degraded"
    return "drifted"


def _observation_status(
    observation: Observation | None,
) -> ObservationStatus | Literal["unknown"]:
    return observation.status if observation is not None else "unknown"


def _now_or_expired(expires_at: str | None, at: str) -> bool:
    if expires_at is None:
        return True

    def parse(value: str) -> datetime:
        normalized = value[:-1] + "+00:00" if value[-1:] in {"Z", "z"} else value
        parsed = datetime.fromisoformat(normalized)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    try:
        return parse(expires_at) > parse(at)
    except ValueError:
        return False


class ConnectorControlPlane:
    """State machine over typed repository seams, with no transport side effects."""

    def __init__(self, repository: ConnectorRepository) -> None:
        self._repository = repository

    def register(self, registration: ConnectorRegistration) -> None:
        """Publish immutable identity/version records and one desired target.

        A concrete repository should implement these calls transactionally.  No
        observation or probe result is consulted, and this method never starts
        a provider client.
        """

        self._repository.put_connector(registration.identity)
        self._repository.put_server(registration.server)
        self._repository.put_inventory(registration.inventory)
        self._repository.put_version(registration.version)
        self._repository.put_desired(registration.desired)

    def set_desired(
        self,
        scope: AccessScope,
        desired: DesiredConnectorState,
    ) -> None:
        """Apply explicit operator desired state; observations cannot call this."""

        _scope_matches(scope, desired.server)
        current = self._repository.get_desired(scope, desired.server.server_id)
        if current is not None and desired.revision <= current.revision:
            raise RepositoryContractError("desired connector revision is stale")
        version = self._repository.get_version(scope, desired.version_id)
        if version is None:
            raise RepositoryContractError("desired connector version is unavailable")
        if version.connector_id != desired.server.connector.connector_id:
            raise RepositoryContractError(
                "desired version belongs to another connector"
            )
        if version.manifest_ref != desired.inventory_ref:
            raise RepositoryContractError("desired state is not bound to its inventory")
        self._repository.put_desired(desired)

    def record_observation(
        self,
        scope: AccessScope,
        observation: Observation,
    ) -> ConnectorReconciliation:
        """Append a probe result, then reconcile without mutating desired state."""

        server = self._repository.get_server(scope, observation.server_id)
        if server is None:
            raise RepositoryContractError("observation targets an unknown server")
        _scope_matches(scope, server)
        self._repository.append_observation(observation)
        return self.reconcile(scope, observation.server_id, observation=observation)

    def _load_reconciliation_inputs(
        self,
        scope: AccessScope,
        server_id: str,
        observation: Observation | None,
    ) -> tuple[
        ServerIdentity, DesiredConnectorState, Observation | None, ConnectorVersion
    ]:
        server = self._repository.get_server(scope, server_id)
        desired = self._repository.get_desired(scope, server_id)
        if server is None or desired is None:
            raise RepositoryContractError("connector control state is incomplete")
        _scope_matches(scope, server)
        if desired.server.server_id != server_id:
            raise RepositoryContractError("desired state targets another server")
        current_observation = (
            observation
            if observation is not None
            else self._repository.get_latest_observation(scope, server_id)
        )
        if (
            current_observation is not None
            and current_observation.server_id != server_id
        ):
            raise RepositoryContractError(
                "repository returned a cross-server observation"
            )
        version = self._repository.get_version(scope, desired.version_id)
        if version is None:
            raise RepositoryContractError("desired connector version is unavailable")
        if version.connector_id != server.connector.connector_id:
            raise RepositoryContractError(
                "desired version belongs to another connector"
            )
        return server, desired, current_observation, version

    def reconcile(
        self,
        scope: AccessScope,
        server_id: str,
        *,
        observation: Observation | None = None,
    ) -> ConnectorReconciliation:
        """Compute drift while retaining desired state on empty or failed probes."""

        server, desired, current_observation, version = (
            self._load_reconciliation_inputs(scope, server_id, observation)
        )
        codes = _codes_for(desired, version, current_observation)
        status = _drift_status(current_observation, codes)
        quarantine_ref = (
            _quarantine_ref(
                server_id, getattr(current_observation, "observation_ref", None), codes
            )
            if status == "quarantined"
            else None
        )
        drift = DriftRecord(
            drift_version="connector-drift.v1",
            server_id=server_id,
            status=status,
            codes=codes,
            quarantine_ref=quarantine_ref,
            observation_ref=(
                current_observation.observation_ref
                if current_observation is not None
                else None
            ),
        )
        lifecycle = ConnectorLifecycle(
            lifecycle_version="connector-lifecycle.v1",
            server_id=server_id,
            desired_status=desired.desired_status,
            observed_status=_observation_status(current_observation),
            drift_status=status,
            desired_revision=desired.revision,
            observation_ref=drift.observation_ref,
            quarantine_ref=quarantine_ref,
        )
        projection = ConnectorGraphProjection(
            projection_version="connector-graph-projection.v1",
            tenant_id=server.tenant_id,
            server_id=server.server_id,
            connector_id=server.connector.connector_id,
            version_id=version.version_id,
            desired_status=desired.desired_status,
            observed_status=_observation_status(current_observation),
            drift_status=status,
            capability_count=len(version.capabilities),
            capability_set_digest=version.capability_set_digest,
            manifest_digest=version.manifest_digest,
            package_ref=version.inventory.package_ref,
            drift_codes=drift.codes,
            observation_ref=drift.observation_ref,
            quarantine_ref=quarantine_ref,
        )
        return ConnectorReconciliation(
            reconciliation_version="connector-reconciliation.v1",
            lifecycle=lifecycle,
            desired=desired,
            observation=current_observation,
            drift=drift,
            projection=projection,
        )

    def authorize(self, decision: AuthorizationDecision) -> None:
        """Persist one independent decision without accepting credential values."""

        self._repository.put_authorization(decision)

    @staticmethod
    def _apply_authorization_decision(
        decision: Any,
        scope: AccessScope,
        server_id: str,
        version_id: str,
        at: str,
        present: set[Literal["approval", "install", "credential_access", "enable"]],
        missing: list[Literal["approval", "install", "credential_access", "enable"]],
    ) -> None:
        if (
            decision.tenant_id != scope.tenant_id
            or decision.principal_id != scope.principal_id
        ):
            raise RepositoryContractError(
                "repository returned cross-scope authorization"
            )
        if decision.server_id != server_id or decision.version_id != version_id:
            raise RepositoryContractError(
                "repository returned authorization for another connector"
            )
        present.add(decision.kind)
        if (
            decision.kind == "credential_access"
            and decision.grant_digest not in scope.grant_digests
        ):
            missing.append(decision.kind)
        elif decision.outcome != "approved" or not _now_or_expired(
            decision.expires_at, at
        ):
            missing.append(decision.kind)

    def evaluate_authorization(
        self,
        scope: AccessScope,
        server_id: str,
        version_id: str,
        *,
        at: str,
    ) -> AuthorizationEvaluation:
        """Require approval, package install, credential access and enablement."""

        decisions = self._repository.get_authorizations(scope, server_id, version_id)
        required: tuple[
            Literal["approval", "install", "credential_access", "enable"], ...
        ] = ("approval", "install", "credential_access", "enable")
        missing: list[
            Literal["approval", "install", "credential_access", "enable"]
        ] = []
        if decisions is None:
            missing.extend(required)
        else:
            present: set[
                Literal["approval", "install", "credential_access", "enable"]
            ] = set()
            for decision in decisions.decisions:
                self._apply_authorization_decision(
                    decision, scope, server_id, version_id, at, present, missing
                )
            missing.extend(kind for kind in required if kind not in present)
        return AuthorizationEvaluation(
            evaluation_version="connector-authorization-evaluation.v1",
            server_id=server_id,
            version_id=version_id,
            allowed=not missing,
            missing_or_denied=tuple(dict.fromkeys(missing)),
        )

    def list(self, request: ConnectorListRequest) -> ConnectorPage:
        """Return a bounded page and reject scope/cursor drift from adapters."""

        page = self._repository.list_connectors(request)
        if len(page.items) > request.limit:
            raise RepositoryContractError(
                "repository returned an oversized connector page"
            )
        if any(item.tenant_id != request.scope.tenant_id for item in page.items):
            raise RepositoryContractError(
                "repository returned a cross-tenant connector page"
            )
        if (
            page.next_cursor is not None
            and page.next_cursor.scope_digest != request.scope.scope_digest
        ):
            raise RepositoryContractError(
                "repository returned an unbound connector cursor"
            )
        return page

    def project(
        self,
        scope: AccessScope,
        server_id: str,
    ) -> ConnectorGraphProjection:
        """Build the privacy-safe graph projection from the same reconciliation."""

        return self.reconcile(scope, server_id).projection

    def summarize_page_item(
        self,
        scope: AccessScope,
        server_id: str,
    ) -> ConnectorPageItem:
        """Produce one bounded list projection without exposing observation prose."""

        reconciliation = self.reconcile(scope, server_id)
        projection = reconciliation.projection
        version = self._repository.get_version(scope, projection.version_id)
        if version is None:
            raise RepositoryContractError(
                "connector version disappeared during projection"
            )
        return ConnectorPageItem(
            tenant_id=projection.tenant_id,
            server_id=projection.server_id,
            connector_id=projection.connector_id,
            version_id=projection.version_id,
            desired_status=projection.desired_status,
            observed_status=projection.observed_status,
            drift_status=projection.drift_status,
            capability_count=projection.capability_count,
            capability_set_digest=projection.capability_set_digest,
            manifest_digest=projection.manifest_digest,
            package_ref=projection.package_ref,
            observed_at=(
                reconciliation.observation.observed_at
                if reconciliation.observation is not None
                else None
            ),
        )
