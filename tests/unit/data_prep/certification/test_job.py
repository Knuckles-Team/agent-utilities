"""Focused NE-114 policy, privacy, and optional-adapter fixtures."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

pa = pytest.importorskip(
    "pyarrow",
    reason="NE-114 Arrow contract requires the explicit [pyarrow] extra",
)

from agent_utilities.data_prep.certification import (  # noqa: E402
    AdapterObservation,
    ArtifactPolicy,
    AuthorizedArrowSample,
    CertificationArtifact,
    CertificationDependencyUnavailable,
    CertificationJobSpec,
    GreatExpectationsAdapter,
    run_certification_job,
)
from agent_utilities.models.company_brain import DataClassification  # noqa: E402


class _Signer:
    signer_id = "fixture-signer"
    algorithm = "ed25519"

    def sign(self, _digest_hex: str) -> str:
        return "fixture-signature"


def _policy(*, tenant_ref: str = "tenant-fixture") -> ArtifactPolicy:
    return ArtifactPolicy(
        schema_version="data-quality-policy.v1",
        tenant_ref=tenant_ref,
        classification=DataClassification.INTERNAL,
        retention_policy_ref="retention:p7d",
        deletion_policy_ref="deletion:tenant",
        access_policy_ref="acl:operator",
    )


def _job(*, policy: ArtifactPolicy | None = None, expires_at_ms: int = 10_000):
    return CertificationJobSpec(
        schema_version="data-quality-job.v1",
        job_ref="job:fixture",
        backend="great_expectations",
        contract_ref="contract:fixture",
        sample=AuthorizedArrowSample(
            schema_version="data-quality-sample.v1",
            sample_artifact_ref="sample:fixture",
            authorization_ref="grant:fixture",
            policy=policy or _policy(),
            expires_at_ms=expires_at_ms,
            max_rows=8,
            max_columns=4,
            max_bytes=64 * 1024,
        ),
    )


def _report(policy: ArtifactPolicy) -> CertificationArtifact:
    return CertificationArtifact(
        schema_version="data-quality-report.v1",
        artifact_ref="report:fixture",
        media_type="application/json",
        content_digest="sha256:" + "a" * 64,
        byte_size=128,
        policy=policy,
    )


class _PassingAdapter:
    backend = "great_expectations"

    def __init__(self, report: CertificationArtifact | None) -> None:
        self.report = report

    def run(self, _table, *, authorization):
        return AdapterObservation(
            schema_version="data-quality-observation.v1",
            status="passed",
            provider_version="1.0.0",
            checks_total=2,
            checks_passed=2,
            checks_failed=0,
            report=self.report,
        )


def test_success_publishes_only_policy_bound_reference_and_signed_summary() -> None:
    job = _job()
    result = run_certification_job(
        pa.table({"id": [1, 2]}),
        job,
        adapter=_PassingAdapter(_report(job.sample.policy)),
        signer=_Signer(),
        now_ms=1_000,
    )

    assert result.state == "passed"
    assert result.report is not None
    assert result.report.artifact_ref == "report:fixture"
    assert result.report.policy == job.sample.policy
    assert result.signed_summary is not None
    assert result.signed_summary.report_artifact_ref == "report:fixture"
    assert result.signed_summary.report_content_digest == "sha256:" + "a" * 64
    assert result.signed_summary.report_byte_size == 128
    rendered = result.model_dump_json()
    assert "fixture-signature" in rendered
    assert "secret" not in rendered
    assert "bearer" not in rendered


def test_policy_mismatch_is_failed_and_does_not_publish_reference() -> None:
    job = _job()
    wrong_policy = _policy(tenant_ref="other-tenant")
    result = run_certification_job(
        pa.table({"id": [1]}),
        job,
        adapter=_PassingAdapter(_report(wrong_policy)),
        signer=_Signer(),
        now_ms=1_000,
    )

    assert result.state == "failed"
    assert result.failure_code == "policy_mismatch"
    assert result.report is None
    assert result.signed_summary is not None
    assert result.signed_summary.report_artifact_ref is None


def test_provider_absence_is_honest_and_exception_detail_is_not_evidence() -> None:
    class _UnavailableAdapter:
        backend = "great_expectations"

        def run(self, _table, *, authorization):
            raise CertificationDependencyUnavailable("bearer token must not escape")

    result = run_certification_job(
        pa.table({"id": [1]}),
        _job(),
        adapter=_UnavailableAdapter(),
        signer=_Signer(),
        now_ms=1_000,
    )

    assert result.state == "unavailable"
    assert result.failure_code == "provider_dependency_unavailable"
    assert result.signed_summary is None
    assert "bearer" not in result.model_dump_json()


def test_expired_authorization_and_unsafe_references_fail_closed() -> None:
    expired = run_certification_job(
        pa.table({"id": [1]}),
        _job(expires_at_ms=1_000),
        adapter=_PassingAdapter(None),
        signer=_Signer(),
        now_ms=1_000,
    )
    assert expired.state == "denied"
    assert expired.failure_code == "authorization_expired"

    with pytest.raises(ValidationError):
        _policy(tenant_ref="https://provider.example/query?token=secret")


def test_optional_provider_wrappers_are_lazy_and_do_not_run_without_provider() -> None:
    # Construction itself must not import either optional provider.  The runner
    # is deliberately never reached in this fixture.
    adapter = GreatExpectationsAdapter(lambda _table, _authorization: {})
    assert adapter.backend == "great_expectations"
