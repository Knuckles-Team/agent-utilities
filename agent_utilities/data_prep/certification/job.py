"""Bounded operator certification job orchestration.

``run_certification_job`` is the only execution helper in this package.  It
accepts an already-authorized in-memory Arrow sample, applies the authorization
limits before provider code runs, invokes one lazy optional adapter, and emits
only an opaque report reference plus a signed aggregate result.  It never calls
``ChangeEnvelope``/``MutationBatch``, writes graph facts, or changes an engine
decision retroactively.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Protocol

from agent_utilities.data_prep.kernel import (
    ArrowAdapter,
    DataPrepDependencyError,
    ProfileLimitError,
)
from agent_utilities.data_prep.models import LocalProfile

from .adapters import CertificationAdapter, CertificationDependencyUnavailable
from .models import (
    AdapterObservation,
    CertificationJobSpec,
    CertificationResult,
    CertificationState,
    FailureCode,
    SignedResultSummary,
)

__all__ = [
    "CertificationSigner",
    "mark_not_requested",
    "run_certification_job",
]


class CertificationSigner(Protocol):
    """Operator-provided Ed25519 signer; private key custody stays external."""

    signer_id: str
    algorithm: str

    def sign(self, digest_hex: str) -> str: ...


def _digest(value: dict[str, Any]) -> tuple[str, str]:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest_hex = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest_hex}", digest_hex


def _result(
    job: CertificationJobSpec,
    state: CertificationState,
    *,
    failure_code: FailureCode | None = None,
) -> CertificationResult:
    return CertificationResult(
        schema_version="data-quality-result-state.v1",
        job_ref=job.job_ref,
        state=state,
        failure_code=failure_code,
    )


def mark_not_requested(job: CertificationJobSpec) -> CertificationResult:
    """Return an honest absence state when an operator did not request a run."""

    return _result(job, "not_requested")


def _signed_result(
    job: CertificationJobSpec,
    observation: AdapterObservation,
    signer: CertificationSigner,
) -> CertificationResult:
    signer_id = str(getattr(signer, "signer_id", ""))
    algorithm = str(getattr(signer, "algorithm", ""))
    if not signer_id or not algorithm:
        return _result(job, "failed", failure_code="signer_invalid")
    if algorithm != "ed25519":
        return _result(job, "failed", failure_code="signer_invalid")

    unsigned: dict[str, Any] = {
        "schema_version": "data-quality-result.v1",
        "job_ref": job.job_ref,
        "backend": job.backend,
        "contract_ref": job.contract_ref,
        "sample_artifact_ref": job.sample.sample_artifact_ref,
        "policy": job.sample.policy.model_dump(mode="json"),
        "status": observation.status,
        "checks_total": observation.checks_total,
        "checks_passed": observation.checks_passed,
        "checks_failed": observation.checks_failed,
        "failure_codes": list(observation.failure_codes),
        "report_artifact_ref": (
            observation.report.artifact_ref if observation.report else None
        ),
        "report_content_digest": (
            observation.report.content_digest if observation.report else None
        ),
        "report_byte_size": (
            observation.report.byte_size if observation.report else None
        ),
        "signer_ref": signer_id,
        "signature_algorithm": algorithm,
    }
    summary_digest, digest_hex = _digest(unsigned)
    try:
        signature = str(signer.sign(digest_hex))
    except Exception:  # noqa: BLE001 - private signer detail never enters evidence
        return _result(job, "failed", failure_code="signer_unavailable")
    try:
        summary = SignedResultSummary(
            **{**unsigned, "policy": job.sample.policy},
            summary_digest=summary_digest,
            signature=signature,
        )
    except Exception:  # noqa: BLE001 - malformed signer output is not published
        return _result(job, "failed", failure_code="signer_invalid")
    return CertificationResult(
        schema_version="data-quality-result-state.v1",
        job_ref=job.job_ref,
        state=observation.status,
        failure_code=(
            observation.failure_codes[0] if observation.failure_codes else None
        ),
        report=observation.report,
        signed_summary=summary,
    )


def _signed_failure(
    job: CertificationJobSpec,
    failure_code: FailureCode,
    signer: CertificationSigner | None,
) -> CertificationResult:
    """Sign an executed failure without carrying provider detail or a report."""

    if signer is None:
        return _result(job, "failed", failure_code="signer_unavailable")
    observation = AdapterObservation(
        schema_version="data-quality-observation.v1",
        status="failed",
        provider_version="unknown",
        checks_total=0,
        checks_passed=0,
        checks_failed=0,
        failure_codes=[failure_code],
    )
    return _signed_result(job, observation, signer)


def run_certification_job(
    table: Any,
    job: CertificationJobSpec,
    *,
    adapter: CertificationAdapter,
    signer: CertificationSigner | None,
    now_ms: int | None = None,
) -> CertificationResult:
    """Run one optional certification under an explicit Arrow sample policy.

    The caller must provide a signer.  A missing signer or provider is reported
    as an honest non-success and cannot produce an unsigned report reference.
    The ``now_ms`` injection keeps expiry checks deterministic for job tests.
    """

    current_ms = int(time.time() * 1000) if now_ms is None else now_ms
    if current_ms >= job.sample.expires_at_ms:
        return _result(job, "denied", failure_code="authorization_expired")
    if str(getattr(adapter, "backend", "")) != job.backend:
        return _result(job, "denied", failure_code="adapter_mismatch")

    profile = LocalProfile(
        max_rows=job.sample.max_rows,
        max_columns=job.sample.max_columns,
        max_steps=1,
        max_bytes=job.sample.max_bytes,
        max_outcome_rows=min(job.sample.max_rows, 100_000),
    )
    try:
        bounded_table = ArrowAdapter.as_table(table, profile=profile)
    except DataPrepDependencyError:
        return _result(job, "unavailable", failure_code="arrow_dependency_unavailable")
    except ProfileLimitError:
        return _result(job, "denied", failure_code="sample_limit_exceeded")
    except (TypeError, ValueError):
        return _result(job, "denied", failure_code="invalid_sample")

    try:
        observation = adapter.run(
            bounded_table,
            authorization=job.sample,
        )
    except CertificationDependencyUnavailable:
        return _result(
            job,
            "unavailable",
            failure_code="provider_dependency_unavailable",
        )
    except Exception:  # noqa: BLE001 - provider detail must never cross the boundary
        return _signed_failure(job, "adapter_error", signer)
    if not isinstance(observation, AdapterObservation):
        return _signed_failure(job, "invalid_adapter_result", signer)

    if (
        observation.report is not None
        and observation.report.policy != job.sample.policy
    ):
        # A report with weaker/different tenant governance is never returned,
        # even if the provider claims success.
        return _signed_failure(job, "policy_mismatch", signer)
    if signer is None:
        return _result(job, "failed", failure_code="signer_unavailable")
    return _signed_result(job, observation, signer)
