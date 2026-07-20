"""Current-only configuration boundaries for production certification tooling."""

from __future__ import annotations

import copy
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jsonschema import Draft202012Validator

from agent_utilities.core.config import PRODUCTION_CERTIFICATION_SCENARIOS
from scripts.certification import (
    campaign,
    collect_metrics,
    evidence,
    fault_hook,
    subprocess_boundary,
)

ROOT = Path(__file__).resolve().parents[3]


def _config(root: Path) -> SimpleNamespace:
    executable = root / "command"
    executable.write_text("synthetic executable", encoding="utf-8")
    executable.chmod(0o700)
    command = [str(executable)]
    commands = {scenario: command for scenario in PRODUCTION_CERTIFICATION_SCENARIOS}
    return SimpleNamespace(
        certification_mode="production",
        cert_release_manifest=str(root / "release.json"),
        cert_artifacts_dir=str(root / "artifacts"),
        cert_hardware_class="capacity-standard",
        cert_load_command=[*command, "--report", "{report_file}"],
        cert_metrics_command=command,
        cert_hook_commands=commands,
        cert_fault_action_commands=commands,
        cert_fault_probe_commands=commands,
        cert_evidence_signer_command=command,
        cert_evidence_verifier_command=command,
        cert_prometheus_url="https://metrics.example.test",
        cert_prometheus_bearer_token_ref=None,
        cert_prometheus_tls_profile="production-metrics",
        cert_prometheus_tls_profile_ref=None,
    )


def test_canonical_campaign_uses_exact_typed_scenario_maps(tmp_path: Path) -> None:
    value = yaml.safe_load(
        (ROOT / "deploy/release/certification-campaign.yml").read_text(encoding="utf-8")
    )
    config = _config(tmp_path)

    campaign._validate_campaign(value, config)

    assert tuple(item["id"] for item in value["scenarios"]) == (
        PRODUCTION_CERTIFICATION_SCENARIOS
    )
    assert all("hookEnv" not in item for item in value["scenarios"])
    assert "hardwareClassEnv" not in value
    assert "loadCommandEnv" not in value
    assert "metricsCommandEnv" not in value


def test_campaign_allows_only_a_bounded_duration_extension(tmp_path: Path) -> None:
    value = yaml.safe_load(
        (ROOT / "deploy/release/certification-campaign.yml").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (ROOT / "deploy/release/certification-campaign.schema.json").read_text(
            encoding="utf-8"
        )
    )
    value["durationSeconds"] = 259_200

    campaign._validate_campaign(value, _config(tmp_path))
    Draft202012Validator(schema).validate(value)


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "campaign-id",
        "metric-interval",
        "sample-coverage",
        "targets",
        "scenario-order",
        "scenario-count",
        "scenario-id",
        "scenario-invariants",
        "commit-phases",
        "unexpected-field",
        "duration-too-short",
        "duration-too-long",
        "duration-boolean",
    ],
)
def test_campaign_runtime_and_schema_reject_policy_substitution(
    tmp_path: Path,
    mutation: str,
) -> None:
    value = yaml.safe_load(
        (ROOT / "deploy/release/certification-campaign.yml").read_text(encoding="utf-8")
    )
    candidate = copy.deepcopy(value)
    if mutation == "version":
        candidate["campaignVersion"] = 2
    elif mutation == "campaign-id":
        candidate["id"] = "substituted-production-policy"
    elif mutation == "metric-interval":
        candidate["metricsIntervalSeconds"] = 60
    elif mutation == "sample-coverage":
        candidate["minimumSampleCoverage"] = 0.90
    elif mutation == "targets":
        candidate["targets"]["rtoSeconds"] = 1_000_000
    elif mutation == "scenario-order":
        candidate["scenarios"][0], candidate["scenarios"][1] = (
            candidate["scenarios"][1],
            candidate["scenarios"][0],
        )
    elif mutation == "scenario-count":
        candidate["scenarios"].pop()
    elif mutation == "scenario-id":
        candidate["scenarios"][0]["id"] = "substituted-scenario"
    elif mutation == "scenario-invariants":
        candidate["scenarios"][0]["invariants"] = ["trivial-pass"]
    elif mutation == "commit-phases":
        candidate["scenarios"][1]["phases"].pop()
    elif mutation == "unexpected-field":
        candidate["policyOverride"] = True
    elif mutation == "duration-too-short":
        candidate["durationSeconds"] = 86_399
    elif mutation == "duration-too-long":
        candidate["durationSeconds"] = 259_201
    elif mutation == "duration-boolean":
        candidate["durationSeconds"] = True
    else:  # pragma: no cover - parametrization is exact
        raise AssertionError(mutation)

    with pytest.raises(campaign.CampaignError):
        campaign._validate_campaign(candidate, _config(tmp_path))

    schema = json.loads(
        (ROOT / "deploy/release/certification-campaign.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(Draft202012Validator(schema).iter_errors(candidate))


def _metric_values(**overrides: float) -> dict[str, float]:
    values = {
        "gatewayP99Seconds": 1.0,
        "engineP99Seconds": 0.02,
        "gatewayErrorRatio": 0.005,
        "dispatchQueueDepth": 1.0,
        "ingestConsumerLag": 1.0,
        "analyticsJobsReady": 1.0,
        "reachableEngineMembers": 3.0,
        "walAppendDroppedFiveMinutes": 0.0,
        "checkpointAgeSeconds": 30.0,
        "podRestartsFiveMinutes": 0.0,
    }
    values.update(overrides)
    return values


def _campaign_targets() -> dict:
    value = yaml.safe_load(
        (ROOT / "deploy/release/certification-campaign.yml").read_text(
            encoding="utf-8"
        )
    )
    return value["targets"]


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("gatewayErrorRatio", 0.010001),
        ("podRestartsFiveMinutes", 1.0),
    ],
)
def test_metric_summary_enforces_every_canonical_health_bound(
    mutation: str,
    value: float,
) -> None:
    sample = {"values": _metric_values()}
    assert campaign._metric_summary([sample], _campaign_targets()) == (True, 30.0)

    sample["values"][mutation] = value

    assert campaign._metric_summary([sample], _campaign_targets())[0] is False


@pytest.mark.parametrize("missing", sorted(campaign._REQUIRED_METRICS))
def test_metric_summary_fails_closed_when_a_metric_is_missing(missing: str) -> None:
    values = _metric_values()
    values.pop(missing)

    assert campaign._metric_summary(
        [{"values": values}], _campaign_targets()
    ) == (False, float("inf"))


def test_load_report_requires_full_configured_wall_duration() -> None:
    report = {
        "ok": True,
        "scale": 1.0,
        "duration_seconds": 86_400.0,
        "real_duration_seconds": 86_400.0,
        "slo_pass": {"axis": {"p99": True}},
        "invariant_violation_counts": {"none": 0},
    }
    assert campaign._load_report_ok(report, configured_duration=86_400) is True

    report["real_duration_seconds"] = 82_080.0

    assert campaign._load_report_ok(report, configured_duration=86_400) is False


def test_metric_scheduler_skips_missed_slots_instead_of_backfilling_coverage() -> None:
    assert campaign._next_metric_deadline(100.0, 101.0, 15) == 115.0
    assert campaign._next_metric_deadline(100.0, 146.0, 15) == 160.0


def test_campaign_requires_a_precreated_private_empty_artifact_directory(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(campaign.CampaignError):
        campaign._validate_artifacts_directory(missing)

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    campaign._validate_artifacts_directory(private)
    (private / "occupied").write_text("synthetic", encoding="utf-8")
    with pytest.raises(campaign.CampaignError):
        campaign._validate_artifacts_directory(private)


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory modes are not authoritative")
def test_campaign_rejects_nonprivate_or_aliased_artifact_directory(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(campaign.CampaignError):
        campaign._validate_artifacts_directory(shared)

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(private, target_is_directory=True)
    with pytest.raises(campaign.CampaignError):
        campaign._validate_artifacts_directory(alias)


@pytest.mark.parametrize("stream", ("stdout", "stderr"))
def test_certification_adapter_output_is_bounded_and_failure_is_redacted(
    stream: str,
) -> None:
    private_marker = "private-adapter-content-must-not-escape"
    script = (
        "import sys; target=getattr(sys, sys.argv[1]); "
        "target.write(sys.argv[2]); target.write('x' * 4096); target.flush()"
    )
    with pytest.raises(
        subprocess_boundary.AdapterBoundaryError,
        match="adapter_output_limit",
    ) as captured:
        subprocess_boundary.run_bounded(
            [sys.executable, "-c", script, stream, private_marker],
            timeout=10,
            maximum_output_bytes=128,
        )

    assert private_marker not in str(captured.value)


def test_prometheus_response_body_is_bounded_closed_and_redacted() -> None:
    private_marker = b"private-prometheus-content-must-not-escape"

    class Response:
        headers = {}
        closed = False

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, *, chunk_size: int):
            assert chunk_size == 65_536
            yield private_marker
            yield b"x" * collect_metrics._MAX_PROMETHEUS_RESPONSE_BYTES

        def close(self) -> None:
            self.closed = True

    class Session:
        def get(self, _url, **kwargs):
            assert kwargs["stream"] is True
            return response

    response = Response()
    with pytest.raises(RuntimeError, match="size boundary") as captured:
        collect_metrics._query(
            Session(),
            "https://metrics.example.test",
            "aggregate-query",
            {},
        )

    assert response.closed is True
    assert private_marker.decode() not in str(captured.value)


def _passing_evidence() -> dict:
    policy = yaml.safe_load(
        (ROOT / "deploy/release/certification-campaign.yml").read_text(encoding="utf-8")
    )
    digest = "sha256:" + "a" * 64
    scenarios = [
        {
            "id": item["id"],
            "result": "pass",
            "faultApplied": True,
            "actionDigest": digest,
            "observationDigest": digest,
            "metricsDigest": digest,
            "recoverySeconds": 1.0,
            "invariants": {name: True for name in item["invariants"]},
        }
        for item in policy["scenarios"]
    ]
    return {
        "apiVersion": "graphos.io/v1",
        "kind": "OperationalEvidence",
        "evidenceVersion": 1,
        "release": {
            "digest": digest,
            "configurationDigest": digest,
            "componentDigests": {name: digest for name in evidence._COMPONENTS},
            "certificationDigests": {
                name: digest for name in evidence._CERTIFICATIONS
            },
        },
        "campaign": {
            "digest": evidence.digest_bytes(evidence.canonical_bytes(policy)),
            "scale": 1.0,
            "durationSeconds": 86_400,
            "observedDurationSeconds": 86_400.0,
            "startedAtUnix": 1,
            "hardwareClass": "capacity-standard",
        },
        "scenarios": scenarios,
        "metrics": {
            "rawDigest": digest,
            "loadReportDigest": digest,
            "sampleCount": 5_760,
            "sampleCoverage": 1.0,
            "sloPass": True,
        },
        "recovery": {
            "rpoTargetSeconds": 60.0,
            "rtoTargetSeconds": 300.0,
            "observedRpoSeconds": 1.0,
            "observedRtoSeconds": 1.0,
            "pass": True,
        },
        "privacy": {
            "policyDigest": digest,
            "containsDirectIdentifiers": False,
            "containsEndpoints": False,
            "containsFilesystemLocations": False,
        },
        "result": "pass",
    }


def test_passing_signed_subject_cannot_substitute_sample_coverage_for_elapsed_time() -> None:
    value = _passing_evidence()
    value["campaign"]["observedDurationSeconds"] = 82_080.0
    value["metrics"]["sampleCount"] = 5_472
    value["metrics"]["sampleCoverage"] = 0.95

    with pytest.raises(evidence.EvidenceError, match="contradicts"):
        evidence.validate_evidence(value, require_signature=False)


@pytest.mark.parametrize(
    "mutation",
    ("campaign-digest", "invariant-name", "recovery-target", "evidence-version"),
)
def test_operational_evidence_rejects_semantic_substitution(mutation: str) -> None:
    value = _passing_evidence()
    if mutation == "campaign-digest":
        value["campaign"]["digest"] = "sha256:" + "b" * 64
    elif mutation == "invariant-name":
        value["scenarios"][0]["invariants"] = {"trivial-pass": True}
    elif mutation == "recovery-target":
        value["recovery"]["rtoTargetSeconds"] = 1_000_000.0
    else:
        value["evidenceVersion"] = 2

    with pytest.raises(evidence.EvidenceError):
        evidence.validate_evidence(value, require_signature=False)


def test_campaign_rejects_paths_outside_active_agent_config(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(campaign.CampaignError, match="active AgentConfig"):
        campaign.execute(
            campaign_path=tmp_path / "unused-campaign.yml",
            release_path=tmp_path / "different-release.json",
            matrix_path=tmp_path / "unused-matrix.yml",
            artifacts_dir=Path(config.cert_artifacts_dir),
            config=config,
        )


def test_prometheus_bearer_auth_resolves_only_runtime_reference(monkeypatch) -> None:
    from agent_utilities.security import cli_secrets

    config = SimpleNamespace(
        cert_prometheus_bearer_token_ref="env://TEST_CERT_PROMETHEUS_TOKEN"
    )
    monkeypatch.setattr(
        cli_secrets,
        "resolve_runtime_secret_reference",
        lambda reference: (
            "opaque-runtime-token"
            if reference == "env://TEST_CERT_PROMETHEUS_TOKEN"
            else None
        ),
    )

    assert collect_metrics._headers(config) == {
        "Authorization": "Bearer opaque-runtime-token"
    }


@pytest.mark.parametrize("module", (campaign, collect_metrics, evidence, fault_hook))
def test_certification_consumers_have_no_direct_configuration_reads(module) -> None:
    source = inspect.getsource(module)
    assert "os.environ.get(" not in source
    assert "os.getenv(" not in source
    assert "setting(" not in source
