"""Strict, privacy-safe contracts for optional operator data-quality runs.

This package is deliberately separate from the Arrow preparation kernel.  A
certification run can describe a human-facing Data Docs artifact, but it never
accepts or returns the report body, a provider query, credentials, or source
rows.  The native engine's SHACL/ICV decision remains authoritative; this
contract is an optional operator observation only.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator, Field, model_validator

from agent_utilities.models.company_brain import DataClassification
from agent_utilities.protocols.epistemic_operations import ProtocolModel

CertificationBackend: TypeAlias = Literal["great_expectations", "pandera"]
CertificationState: TypeAlias = Literal[
    "not_requested", "unavailable", "denied", "failed", "passed"
]
FailureCode: TypeAlias = Literal[
    "arrow_dependency_unavailable",
    "provider_dependency_unavailable",
    "authorization_expired",
    "invalid_sample",
    "sample_limit_exceeded",
    "adapter_mismatch",
    "adapter_error",
    "invalid_adapter_result",
    "policy_denied",
    "policy_mismatch",
    "signer_unavailable",
    "signer_invalid",
]
Digest: TypeAlias = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
# A reference is an opaque local identifier, not a URL, path, query, token, or
# provider configuration.  In particular, ``/``, ``?``, ``#``, ``@`` and ``%``
# are intentionally absent from the grammar.  The semantic validator also
# rejects common secret/query labels so an opaque reference cannot be used as a
# convenient bearer-token or provider-config escape hatch.
def _safe_opaque_ref(value: str) -> str:
    lowered = value.casefold()
    if any(
        term in lowered
        for term in (
            "secret",
            "token",
            "bearer",
            "password",
            "api_key",
            "query",
            "http",
        )
    ):
        raise ValueError("opaque reference contains a forbidden sensitive term")
    return value


OpaqueRef: TypeAlias = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    ),
    AfterValidator(_safe_opaque_ref),
]
Signature: TypeAlias = Annotated[
    str,
    Field(
        min_length=1,
        max_length=2_048,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
]


class ArtifactPolicy(ProtocolModel):
    """The complete policy inherited by every certification report artifact."""

    schema_version: Literal["data-quality-policy.v1"]
    tenant_ref: OpaqueRef
    classification: DataClassification
    retention_policy_ref: OpaqueRef
    deletion_policy_ref: OpaqueRef
    access_policy_ref: OpaqueRef


class AuthorizedArrowSample(ProtocolModel):
    """An explicit, bounded authorization for one in-memory Arrow sample."""

    schema_version: Literal["data-quality-sample.v1"]
    sample_artifact_ref: OpaqueRef
    authorization_ref: OpaqueRef
    policy: ArtifactPolicy
    expires_at_ms: int = Field(ge=1)
    max_rows: int = Field(ge=1, le=1_000_000)
    max_columns: int = Field(ge=1, le=1_024)
    max_bytes: int = Field(ge=1, le=512 * 1024 * 1024)


class CertificationJobSpec(ProtocolModel):
    """Operator job identity and provider contract binding."""

    schema_version: Literal["data-quality-job.v1"]
    job_ref: OpaqueRef
    backend: CertificationBackend
    contract_ref: OpaqueRef
    sample: AuthorizedArrowSample


class CertificationArtifact(ProtocolModel):
    """An already-published, access-controlled human report reference."""

    schema_version: Literal["data-quality-report.v1"]
    artifact_ref: OpaqueRef
    media_type: Literal["text/html", "application/json"]
    content_digest: Digest
    byte_size: int = Field(ge=1, le=4 * 1024 * 1024)
    policy: ArtifactPolicy


class AdapterObservation(ProtocolModel):
    """Bounded provider outcome; no provider exception or result body is allowed."""

    schema_version: Literal["data-quality-observation.v1"]
    status: Literal["failed", "passed"]
    provider_version: OpaqueRef
    checks_total: int = Field(ge=0, le=10_000)
    checks_passed: int = Field(ge=0, le=10_000)
    checks_failed: int = Field(ge=0, le=10_000)
    failure_codes: list[FailureCode] = Field(default_factory=list, max_length=8)
    report: CertificationArtifact | None = None

    @model_validator(mode="after")
    def counts_and_status_are_consistent(self) -> AdapterObservation:
        if self.checks_total != self.checks_passed + self.checks_failed:
            raise ValueError("quality check counts must reconcile")
        if self.status == "passed" and (
            self.checks_failed or self.failure_codes
        ):
            raise ValueError("a passed observation cannot contain failures")
        if self.status == "failed" and not (
            self.checks_failed or self.failure_codes
        ):
            raise ValueError("a failed observation must identify a bounded failure")
        return self


class SignedResultSummary(ProtocolModel):
    """The only durable result body emitted by the optional certification job."""

    schema_version: Literal["data-quality-result.v1"]
    job_ref: OpaqueRef
    backend: CertificationBackend
    contract_ref: OpaqueRef
    sample_artifact_ref: OpaqueRef
    policy: ArtifactPolicy
    status: Literal["failed", "passed"]
    checks_total: int = Field(ge=0, le=10_000)
    checks_passed: int = Field(ge=0, le=10_000)
    checks_failed: int = Field(ge=0, le=10_000)
    failure_codes: list[FailureCode] = Field(default_factory=list, max_length=8)
    report_artifact_ref: OpaqueRef | None = None
    report_content_digest: Digest | None = None
    report_byte_size: int | None = Field(default=None, ge=1, le=4 * 1024 * 1024)
    summary_digest: Digest
    signer_ref: OpaqueRef
    signature_algorithm: Literal["ed25519"]
    signature: Signature

    @model_validator(mode="after")
    def counts_and_status_are_consistent(self) -> SignedResultSummary:
        if self.checks_total != self.checks_passed + self.checks_failed:
            raise ValueError("signed quality check counts must reconcile")
        if self.status == "passed" and (
            self.checks_failed or self.failure_codes
        ):
            raise ValueError("a signed passed result cannot contain failures")
        if self.status == "failed" and not (
            self.checks_failed or self.failure_codes
        ):
            raise ValueError("a signed failed result must identify a bounded failure")
        if self.report_artifact_ref is None and (
            self.report_content_digest is not None or self.report_byte_size is not None
        ):
            raise ValueError("report digest/size requires a report reference")
        if self.report_artifact_ref is not None and (
            self.report_content_digest is None or self.report_byte_size is None
        ):
            raise ValueError("report reference requires its digest and byte size")
        return self


class CertificationResult(ProtocolModel):
    """Honest terminal state for requested and non-requested optional work."""

    schema_version: Literal["data-quality-result-state.v1"]
    job_ref: OpaqueRef
    state: CertificationState
    failure_code: FailureCode | None = None
    report: CertificationArtifact | None = None
    signed_summary: SignedResultSummary | None = None

    @model_validator(mode="after")
    def references_match(self) -> CertificationResult:
        if self.report is not None and self.signed_summary is None:
            raise ValueError("a report reference requires a signed summary")
        if self.signed_summary is not None:
            if self.state not in {"failed", "passed"}:
                raise ValueError("only executed states may carry a signed summary")
            if self.report is not None and (
                self.report.artifact_ref
                != self.signed_summary.report_artifact_ref
            ):
                raise ValueError("report and signed summary references differ")
            if self.report is not None and (
                self.report.content_digest
                != self.signed_summary.report_content_digest
                or self.report.byte_size != self.signed_summary.report_byte_size
            ):
                raise ValueError("report and signed summary digests differ")
            if self.signed_summary.job_ref != self.job_ref:
                raise ValueError("signed summary job reference differs")
        if self.state == "passed" and self.signed_summary is None:
            raise ValueError("passed certification requires a signed summary")
        if self.state in {"not_requested", "unavailable", "denied"} and (
            self.report is not None or self.signed_summary is not None
        ):
            raise ValueError("non-executed certification cannot publish artifacts")
        return self


__all__ = [
    "AdapterObservation",
    "ArtifactPolicy",
    "AuthorizedArrowSample",
    "CertificationArtifact",
    "CertificationBackend",
    "CertificationJobSpec",
    "CertificationResult",
    "CertificationState",
    "Digest",
    "FailureCode",
    "OpaqueRef",
    "Signature",
    "SignedResultSummary",
]
