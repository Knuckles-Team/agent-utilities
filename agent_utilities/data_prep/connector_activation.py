"""Fail-closed connector activation binding for native ChangeEnvelope admission.

The preparation contract owns how a page was cleaned and mapped; this module
only binds the already-approved mapping, SHACL and ICV artifacts to a
connector/version, tenant and target graph.  It deliberately does not store
activation state or implement a writer.  ``admit`` delegates the final write
to :func:`knowledge_graph.ingestion.envelope_ingest.ingest_envelope`, whose
native ``ApplyChangeEnvelope`` path is the authority for SHACL/ICV and
durability.

(CONCEPT:AU-KG.ontology.activation-fails-closed,
AU-KG.ontology.activation-icv-fallback, AU-KG.ingest.change-envelope)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal

from pydantic import Field, model_validator

from agent_utilities.protocols.epistemic_operations import ProtocolModel

from .models import Digest, OpaqueReference

__all__ = [
    "ActivationAdmissionAdapter",
    "ActivationAdmissionReport",
    "ActivationArtifact",
    "ActivationBinding",
    "ActivationFinding",
    "ActivationState",
    "ActivationStateError",
    "begin_activation_rotation",
    "bind_activation",
    "commit_activation_rotation",
    "initial_activation",
    "activation_binding_digest",
    "rollback_activation",
]


ActivationStatus = Literal["active", "rotating", "rolled_back"]
AdmissionOutcome = Literal["accepted", "rejected"]
NativeStatus = Literal[
    "bound",
    "success",
    "skipped",
    "rejected",
    "failed",
    "unavailable",
]


class ActivationArtifact(ProtocolModel):
    """One approved artifact identity; content is never carried here."""

    kind: Literal["mapping", "shacl", "icv"]
    ref: OpaqueReference
    digest: Digest


def _dump(value: Any) -> Any:
    """Return a JSON-like value without invoking arbitrary serialization."""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _artifact_payload(value: Any) -> dict[str, Any] | None:
    value = _dump(value)
    if not isinstance(value, Mapping):
        return None
    result = {
        "kind": value.get("kind"),
        "ref": value.get("ref"),
        "digest": value.get("digest"),
    }
    if any(item is None for item in result.values()):
        return None
    return result


def _binding_payload(value: Any) -> dict[str, Any]:
    """Select the exact fields covered by the activation digest.

    Selecting fields instead of hashing an arbitrary model dump prevents a
    future optional field from silently changing the wire fingerprint and
    prevents an unapproved caller field from becoming part of the binding.
    """

    data = _dump(value)
    if not isinstance(data, Mapping):
        raise TypeError("activation binding must be a mapping or protocol model")
    payload: dict[str, Any] = {
        "activation_version": data.get("activation_version"),
        "connector": data.get("connector"),
        "connector_version": data.get("connector_version"),
        "tenant": data.get("tenant"),
        "target_graph": data.get("target_graph"),
        "binding_ref": data.get("binding_ref"),
    }
    for kind in ("mapping", "shacl", "icv"):
        payload[kind] = _artifact_payload(data.get(kind))
    return payload


def activation_binding_digest(value: Any) -> str:
    """Compute the deterministic digest for an activation binding."""

    encoded = json.dumps(
        _binding_payload(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class ActivationBinding(ProtocolModel):
    """Exact connector/version/tenant/graph to governance-artifact binding."""

    activation_version: Literal["connector-activation.v1"]
    connector: OpaqueReference
    connector_version: OpaqueReference
    tenant: OpaqueReference
    target_graph: OpaqueReference
    binding_ref: OpaqueReference
    mapping: ActivationArtifact
    shacl: ActivationArtifact
    icv: ActivationArtifact
    binding_digest: Digest

    @model_validator(mode="after")
    def binding_is_self_consistent(self) -> ActivationBinding:
        expected = {
            "mapping": "mapping",
            "shacl": "shacl",
            "icv": "icv",
        }
        for field_name, kind in expected.items():
            if getattr(self, field_name).kind != kind:
                raise ValueError("activation artifact kind does not match its slot")
        if self.binding_digest != activation_binding_digest(self):
            raise ValueError("activation binding digest does not match its contents")
        return self


class ActivationStateError(ValueError):
    """Stable error for an invalid pure activation-state transition."""


class ActivationState(ProtocolModel):
    """Versioned in-memory control-plane state; no persistence authority."""

    state_version: Literal["connector-activation-state.v1"]
    status: ActivationStatus
    generation: int = Field(ge=1)
    active: ActivationBinding
    pending: ActivationBinding | None = None
    previous: ActivationBinding | None = None
    rotation_ref: OpaqueReference | None = None

    @model_validator(mode="after")
    def state_shape_is_consistent(self) -> ActivationState:
        if self.status == "rotating":
            if self.pending is None or self.rotation_ref is None:
                raise ValueError("rotating activation requires pending state and a ref")
        elif self.pending is not None:
            raise ValueError("non-rotating activation cannot retain pending state")
        for prior in (self.pending, self.previous):
            if prior is not None and (
                prior.connector != self.active.connector
                or prior.tenant != self.active.tenant
            ):
                raise ValueError("activation generations must retain connector scope")
        return self


def initial_activation(binding: ActivationBinding) -> ActivationState:
    """Create the first active generation without persisting it."""

    return ActivationState(
        state_version="connector-activation-state.v1",
        status="active",
        generation=1,
        active=binding,
    )


def _state_copy(state: ActivationState, **updates: Any) -> ActivationState:
    values = state.model_dump(mode="json")
    values.update(updates)
    return ActivationState.model_validate(values)


def begin_activation_rotation(
    state: ActivationState,
    candidate: ActivationBinding,
    rotation_ref: OpaqueReference,
) -> ActivationState:
    """Stage a same-connector, same-tenant candidate for controlled rotation."""

    if state.status == "rotating" or state.pending is not None:
        raise ActivationStateError("activation rotation is already in progress")
    if candidate == state.active:
        raise ActivationStateError("activation rotation candidate is unchanged")
    if (
        candidate.connector != state.active.connector
        or candidate.tenant != state.active.tenant
    ):
        raise ActivationStateError("activation rotation identity is not compatible")
    return _state_copy(
        state,
        status="rotating",
        pending=candidate.model_dump(mode="json"),
        rotation_ref=rotation_ref,
    )


def commit_activation_rotation(state: ActivationState) -> ActivationState:
    """Promote a staged candidate and retain the prior binding for rollback."""

    if state.status != "rotating" or state.pending is None:
        raise ActivationStateError("activation rotation has no pending candidate")
    return _state_copy(
        state,
        status="active",
        generation=state.generation + 1,
        active=state.pending.model_dump(mode="json"),
        previous=state.active.model_dump(mode="json"),
        pending=None,
        rotation_ref=None,
    )


def rollback_activation(state: ActivationState) -> ActivationState:
    """Reject a pending candidate or restore the retained prior generation."""

    if state.status == "rotating":
        if state.pending is None:
            raise ActivationStateError("activation rotation has no pending candidate")
        return _state_copy(
            state,
            status="rolled_back",
            generation=state.generation + 1,
            pending=None,
            rotation_ref=None,
        )
    if state.previous is None:
        raise ActivationStateError("activation state has no prior generation")
    return _state_copy(
        state,
        status="rolled_back",
        generation=state.generation + 1,
        active=state.previous.model_dump(mode="json"),
        previous=state.active.model_dump(mode="json"),
        pending=None,
        rotation_ref=None,
    )


FindingCode = Literal[
    "activation_missing",
    "activation_rotating",
    "activation_stale",
    "activation_claim_missing",
    "connector_mismatch",
    "connector_version_mismatch",
    "tenant_mismatch",
    "graph_mismatch",
    "mapping_ref_mismatch",
    "mapping_digest_mismatch",
    "shacl_ref_mismatch",
    "shacl_digest_mismatch",
    "icv_ref_mismatch",
    "icv_digest_mismatch",
    "prep_artifact_missing",
    "prep_artifact_mismatch",
    "session_missing",
    "session_tenant_mismatch",
    "session_graph_mismatch",
    "engine_rejected",
    "engine_failed",
    "native_unavailable",
]


_FINDING_SUMMARIES: dict[str, str] = {
    "activation_missing": "no active connector activation is available",
    "activation_rotating": "connector activation rotation is not committed",
    "activation_stale": "connector activation claim is stale",
    "activation_claim_missing": "connector activation claim is missing",
    "connector_mismatch": "connector is outside the approved activation",
    "connector_version_mismatch": "connector version is outside the approved activation",
    "tenant_mismatch": "tenant is outside the approved activation",
    "graph_mismatch": "target graph is outside the approved activation",
    "mapping_ref_mismatch": "mapping artifact reference is not approved",
    "mapping_digest_mismatch": "mapping artifact digest is not approved",
    "shacl_ref_mismatch": "SHACL artifact reference is not approved",
    "shacl_digest_mismatch": "SHACL artifact digest is not approved",
    "icv_ref_mismatch": "ICV artifact reference is not approved",
    "icv_digest_mismatch": "ICV artifact digest is not approved",
    "prep_artifact_missing": "prepared governance artifact is missing",
    "prep_artifact_mismatch": "prepared governance artifact is not approved",
    "session_missing": "verified write session is unavailable",
    "session_tenant_mismatch": "verified session tenant is not approved",
    "session_graph_mismatch": "verified session graph is not approved",
    "engine_rejected": "native admission rejected the envelope",
    "engine_failed": "native admission failed without a commit",
    "native_unavailable": "native admission authority is unavailable",
}


_POINTER_PATTERN = r"^/[A-Za-z0-9_.~/-]*$"


class ActivationFinding(ProtocolModel):
    """One fixed, privacy-safe admission code."""

    code: FindingCode
    field_pointer: str | None = Field(default=None, pattern=_POINTER_PATTERN)
    summary: str = Field(min_length=1, max_length=96)

    @model_validator(mode="after")
    def summary_is_allowlisted(self) -> ActivationFinding:
        if self.summary != _FINDING_SUMMARIES[self.code]:
            raise ValueError("activation finding summary is not allow-listed")
        return self


class ActivationAdmissionReport(ProtocolModel):
    """Bounded outcome returned by binding and native admission."""

    report_version: Literal["connector-activation-report.v1"]
    outcome: AdmissionOutcome
    report_ref: OpaqueReference
    binding_digest: Digest | None = None
    findings: tuple[ActivationFinding, ...] = Field(default_factory=tuple, max_length=16)
    native_status: NativeStatus | None = None

    @model_validator(mode="after")
    def outcome_matches_findings(self) -> ActivationAdmissionReport:
        if (self.outcome == "accepted") != (not self.findings):
            raise ValueError("activation report outcome does not match findings")
        return self


def _finding(code: FindingCode, pointer: str | None = None) -> ActivationFinding:
    return ActivationFinding(
        code=code,
        field_pointer=pointer,
        summary=_FINDING_SUMMARIES[code],
    )


def _digest_text(value: Any) -> str:
    rendered = "" if value is None else str(value)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _report(
    *,
    binding_digest: str | None,
    idempotency_key: str | None,
    findings: list[tuple[FindingCode, str | None]] | None = None,
    native_status: NativeStatus | None = None,
) -> ActivationAdmissionReport:
    """Build a deterministic report without including raw identity values."""

    unique: list[tuple[FindingCode, str | None]] = []
    for item in findings or []:
        if item not in unique:
            unique.append(item)
        if len(unique) >= 16:
            break
    rendered_findings = tuple(_finding(code, pointer) for code, pointer in unique)
    outcome: AdmissionOutcome = "accepted" if not rendered_findings else "rejected"
    report_material = {
        "report_version": "connector-activation-report.v1",
        "outcome": outcome,
        "binding_digest": binding_digest,
        "idempotency_digest": _digest_text(idempotency_key),
        "findings": [
            {"code": item.code, "field_pointer": item.field_pointer}
            for item in rendered_findings
        ],
        "native_status": native_status,
    }
    report_digest = hashlib.sha256(
        json.dumps(
            report_material,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ActivationAdmissionReport(
        report_version="connector-activation-report.v1",
        outcome=outcome,
        report_ref=f"report:{report_digest}",
        binding_digest=binding_digest,
        findings=rendered_findings,
        native_status=native_status,
    )


def _identity_findings(
    envelope: Any,
    binding: ActivationBinding,
) -> list[tuple[FindingCode, str | None]]:
    findings: list[tuple[FindingCode, str | None]] = []
    if getattr(envelope, "connector", None) != binding.connector:
        findings.append(("connector_mismatch", "/connector"))
    if getattr(envelope, "schema_version", None) != binding.connector_version:
        findings.append(("connector_version_mismatch", "/schema_version"))
    if getattr(envelope, "tenant", None) != binding.tenant:
        findings.append(("tenant_mismatch", "/tenant"))
    return findings


def _artifact_findings(
    artifacts: Mapping[str, Any] | None,
    binding: ActivationBinding,
    *,
    prefix: str,
) -> list[tuple[FindingCode, str | None]]:
    findings: list[tuple[FindingCode, str | None]] = []
    expected = {
        "mapping": binding.mapping,
        "shacl": binding.shacl,
        "icv": binding.icv,
    }
    if artifacts is None:
        return [("prep_artifact_missing", f"{prefix}/artifacts")]
    for kind, approved in expected.items():
        pointer = f"{prefix}/artifacts/{kind}"
        actual = _artifact_payload(artifacts.get(kind))
        if actual is None:
            findings.append(("prep_artifact_missing", pointer))
            continue
        if actual.get("kind") != kind:
            findings.append(("prep_artifact_mismatch", f"{pointer}/kind"))
        if actual.get("ref") != approved.ref:
            findings.append((f"{kind}_ref_mismatch", f"{pointer}/ref"))
        if actual.get("digest") != approved.digest:
            findings.append((f"{kind}_digest_mismatch", f"{pointer}/digest"))
    return findings


def _contract_data(contract: Any) -> Mapping[str, Any] | None:
    data = _dump(contract)
    if isinstance(data, Mapping):
        return data
    return None


def _contract_artifacts(contract: Any) -> Mapping[str, Any] | None:
    data = _contract_data(contract)
    if data is None:
        return None
    artifacts = data.get("artifacts")
    artifacts = _dump(artifacts)
    if isinstance(artifacts, Mapping):
        return artifacts
    return None


def _preparation_artifacts(
    envelope: Any,
    prepared_contract: Any | None,
) -> tuple[Mapping[str, Any] | None, str]:
    """Read prepared artifacts from the NE-110 contract or its envelope claim."""

    if prepared_contract is not None:
        return _contract_artifacts(prepared_contract), "/prepared_contract"
    provenance = getattr(envelope, "provenance", None)
    if not isinstance(provenance, Mapping):
        return None, "/provenance/connector_preparation"
    preparation = provenance.get("connector_preparation")
    if not isinstance(preparation, Mapping):
        return None, "/provenance/connector_preparation"
    artifacts = preparation.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None, "/provenance/connector_preparation"
    return artifacts, "/provenance/connector_preparation"


def _contract_identity_findings(
    envelope: Any,
    binding: ActivationBinding,
    prepared_contract: Any,
) -> list[tuple[FindingCode, str | None]]:
    data = _contract_data(prepared_contract)
    if data is None:
        return [("prep_artifact_missing", "/prepared_contract")]
    findings: list[tuple[FindingCode, str | None]] = []
    if data.get("connector") != binding.connector:
        findings.append(("connector_mismatch", "/prepared_contract/connector"))
    if data.get("schema_version") != binding.connector_version:
        findings.append(
            ("connector_version_mismatch", "/prepared_contract/schema_version")
        )
    if data.get("tenant") != binding.tenant:
        findings.append(("tenant_mismatch", "/prepared_contract/tenant"))
    findings.extend(_identity_findings(envelope, binding))
    return findings


def _activation_claim_findings(
    envelope: Any,
    binding: ActivationBinding,
) -> list[tuple[FindingCode, str | None]]:
    provenance = getattr(envelope, "provenance", None)
    claim = provenance.get("connector_activation") if isinstance(provenance, Mapping) else None
    if not isinstance(claim, Mapping):
        return [("activation_claim_missing", "/provenance/connector_activation")]
    if claim.get("binding_digest") != binding.binding_digest:
        return [("activation_stale", "/provenance/connector_activation/binding_digest")]
    findings: list[tuple[FindingCode, str | None]] = []
    if claim.get("activation_version") != binding.activation_version:
        findings.append(
            ("activation_stale", "/provenance/connector_activation/activation_version")
        )
    if claim.get("binding_ref") != binding.binding_ref:
        findings.append(
            ("activation_stale", "/provenance/connector_activation/binding_ref")
        )
    if claim.get("connector_version") != binding.connector_version:
        findings.append(
            ("connector_version_mismatch", "/provenance/connector_activation/connector_version")
        )
    if claim.get("target_graph") != binding.target_graph:
        findings.append(
            ("graph_mismatch", "/provenance/connector_activation/target_graph")
        )
    claim_artifacts = claim.get("artifacts")
    findings.extend(
        _artifact_findings(
            claim_artifacts if isinstance(claim_artifacts, Mapping) else None,
            binding,
            prefix="/provenance/connector_activation",
        )
    )
    return findings


def _session_findings(session: Any | None, binding: ActivationBinding) -> list[tuple[FindingCode, str | None]]:
    if session is None:
        return [("session_missing", "/session")]
    findings: list[tuple[FindingCode, str | None]] = []
    if getattr(session, "tenant", None) != binding.tenant:
        findings.append(("session_tenant_mismatch", "/session/tenant"))
    if getattr(session, "graph", None) != binding.target_graph:
        findings.append(("session_graph_mismatch", "/session/graph"))
    return findings


def _activation_claim(binding: ActivationBinding) -> dict[str, Any]:
    return {
        "activation_version": binding.activation_version,
        "binding_ref": binding.binding_ref,
        "binding_digest": binding.binding_digest,
        "connector_version": binding.connector_version,
        "target_graph": binding.target_graph,
        "artifacts": {
            kind: getattr(binding, kind).model_dump(mode="json")
            for kind in ("mapping", "shacl", "icv")
        },
    }


def bind_activation(
    envelope: Any,
    binding: ActivationBinding,
    *,
    prepared_contract: Any | None = None,
) -> tuple[Any | None, ActivationAdmissionReport]:
    """Stamp a validated activation claim onto an envelope without writing.

    ``prepared_contract`` is intentionally structural so this adapter can be
    shipped alongside the NE-110 contract and composed with its canonical
    ``ConnectorPrepContract`` without defining a second preparation path.
    """

    findings = _identity_findings(envelope, binding)
    if prepared_contract is not None:
        findings.extend(_contract_identity_findings(envelope, binding, prepared_contract))
    provenance = getattr(envelope, "provenance", {})
    if isinstance(provenance, Mapping) and "connector_activation" in provenance:
        # A caller may re-bind an envelope only to the same approved binding;
        # silently overwriting a different claim would turn stale activation
        # evidence into an apparent fresh approval.
        findings.extend(_activation_claim_findings(envelope, binding))
    artifacts, prefix = _preparation_artifacts(envelope, prepared_contract)
    findings.extend(_artifact_findings(artifacts, binding, prefix=prefix))
    if findings:
        return None, _report(
            binding_digest=binding.binding_digest,
            idempotency_key=getattr(envelope, "idempotency_key", None),
            findings=findings,
            native_status=None,
        )

    if not isinstance(provenance, Mapping):
        provenance = {}
    stamped = dict(provenance)
    stamped["connector_activation"] = _activation_claim(binding)
    bound = replace(envelope, provenance=stamped)
    return bound, _report(
        binding_digest=binding.binding_digest,
        idempotency_key=getattr(envelope, "idempotency_key", None),
        native_status="bound",
    )


class ActivationAdmissionAdapter:
    """Validate activation claims and delegate one native ChangeEnvelope write."""

    def validate(
        self,
        envelope: Any,
        state: ActivationState | None,
        *,
        session: Any | None = None,
    ) -> ActivationAdmissionReport:
        """Return a bounded pre-commit report; never calls the engine."""

        if state is None:
            return _report(
                binding_digest=None,
                idempotency_key=getattr(envelope, "idempotency_key", None),
                findings=[("activation_missing", "/activation")],
            )
        if state.status == "rotating":
            return _report(
                binding_digest=state.active.binding_digest,
                idempotency_key=getattr(envelope, "idempotency_key", None),
                findings=[("activation_rotating", "/activation/status")],
            )
        binding = state.active
        findings = _identity_findings(envelope, binding)
        findings.extend(_activation_claim_findings(envelope, binding))
        preparation, prefix = _preparation_artifacts(envelope, None)
        findings.extend(_artifact_findings(preparation, binding, prefix=prefix))
        findings.extend(_session_findings(session, binding))
        return _report(
            binding_digest=binding.binding_digest,
            idempotency_key=getattr(envelope, "idempotency_key", None),
            findings=findings,
        )

    def admit(
        self,
        engine: Any,
        envelope: Any,
        state: ActivationState | None,
    ) -> ActivationAdmissionReport:
        """Resolve the verified write session, then call the sole native writer."""

        session = _verified_write_session()
        preflight = self.validate(envelope, state, session=session)
        if preflight.outcome == "rejected":
            return preflight

        try:
            result = _native_ingest(engine, envelope)
        except ImportError:
            return _report(
                binding_digest=preflight.binding_digest,
                idempotency_key=getattr(envelope, "idempotency_key", None),
                findings=[("native_unavailable", "/native")],
                native_status="unavailable",
            )
        status = result.get("status") if isinstance(result, Mapping) else None
        if status in {"success", "skipped"}:
            native_status: NativeStatus = status
            return _report(
                binding_digest=preflight.binding_digest,
                idempotency_key=getattr(envelope, "idempotency_key", None),
                native_status=native_status,
            )
        if status == "rejected":
            code: FindingCode = "engine_rejected"
            native_status = "rejected"
        elif status == "failed" and isinstance(result, Mapping) and result.get("error") == "NativeChangeEnvelopeUnavailable":
            code = "native_unavailable"
            native_status = "unavailable"
        else:
            code = "engine_failed"
            native_status = "failed"
        return _report(
            binding_digest=preflight.binding_digest,
            idempotency_key=getattr(envelope, "idempotency_key", None),
            findings=[(code, "/native")],
            native_status=native_status,
        )


def _verified_write_session() -> Any | None:
    """Resolve only the middleware-minted native write session."""

    try:
        from agent_utilities.knowledge_graph.core.session import (
            current_session,
            resolve_session,
        )

        ambient = current_session()
        if ambient is None:
            return None
        return resolve_session(ambient, required_scope="kg:write")
    except Exception:
        # The report deliberately does not expose expiry, actor, scope or
        # process details; session resolution remains the authority's concern.
        return None


def _native_ingest(engine: Any, envelope: Any) -> Mapping[str, Any]:
    """Lazy seam to the existing native ChangeEnvelope writer.

    Keeping this tiny wrapper makes the no-write-on-denial invariant directly
    fixtureable without introducing a second commit path.
    """

    from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
        ingest_envelope,
    )

    return ingest_envelope(engine, envelope)
