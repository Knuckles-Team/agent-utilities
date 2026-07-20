#!/usr/bin/python
"""Execute the 24-72h exact-release soak/chaos certification campaign.

This is intentionally production-only.  It has no mock mode, no skip path and no
implicit fault implementation: every command is resolved from the typed production
certification fields in ``AgentConfig`` during preflight, then executed while
exact-release load and raw aggregate telemetry are running.  Missing hooks,
telemetry, signatures or release pins fail before the campaign can claim evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import threading
import time
from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml

from agent_utilities.core.config import (
    PRODUCTION_CERTIFICATION_SCENARIOS,
    AgentConfig,
)
from scripts.certification import evidence
from scripts.certification.subprocess_boundary import (
    AdapterBoundaryError,
    run_bounded,
)
from scripts.release import check_compatibility as compatibility


class CampaignError(RuntimeError):
    """The certification campaign failed its executable contract."""


_REQUIRED_METRICS = {
    "gatewayP99Seconds",
    "engineP99Seconds",
    "gatewayErrorRatio",
    "dispatchQueueDepth",
    "ingestConsumerLag",
    "analyticsJobsReady",
    "reachableEngineMembers",
    "walAppendDroppedFiveMinutes",
    "checkpointAgeSeconds",
    "podRestartsFiveMinutes",
}


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignError("campaign/release input must be a mapping")
    return value


def _packaged_campaign() -> dict[str, Any]:
    """Load the immutable production policy shipped with this release."""

    value = yaml.safe_load(
        files("deploy.release")
        .joinpath("certification-campaign.yml")
        .read_text(encoding="utf-8")
    )
    if not isinstance(value, dict):
        raise CampaignError("packaged certification campaign is invalid")
    return value


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _proof_digest(value: Any, field: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", text) or text.endswith("0" * 64):
        raise CampaignError(f"fault hook returned an invalid {field}")
    return text


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _command(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, str) and v for v in value)
    ):
        raise CampaignError(f"{field} must contain a non-empty JSON argv array")
    from agent_utilities.skills.runtime_validation import (
        _validate_external_command_argv,
    )

    try:
        return _validate_external_command_argv(value)
    except RuntimeError as exc:
        raise CampaignError(f"{field} is not a safe executable argv array") from exc


def _render_command(command: list[str], values: dict[str, str]) -> list[str]:
    rendered: list[str] = []
    for part in command:
        try:
            rendered.append(part.format_map(values))
        except KeyError as exc:
            raise CampaignError(f"command has unsupported placeholder {exc}") from exc
    return rendered


def _next_metric_deadline(previous: float, completed: float, interval: int) -> float:
    """Advance to a future sampling slot without backfilling missed intervals."""

    deadline = previous + interval
    if deadline <= completed:
        deadline += (int((completed - deadline) // interval) + 1) * interval
    return deadline


def _validate_artifacts_directory(path: Path) -> None:
    """Require the pre-created private directory proven by the doctor."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CampaignError("certification artifacts directory is unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or not os.access(path, os.R_OK | os.W_OK | os.X_OK)
        or any(path.iterdir())
    ):
        raise CampaignError(
            "certification artifacts directory must be empty, private, and unaliased"
        )


def _validate_campaign(campaign: dict[str, Any], config: AgentConfig) -> None:
    canonical = _packaged_campaign()
    duration = campaign.get("durationSeconds")
    if type(duration) is not int or not 86_400 <= duration <= 259_200:
        raise CampaignError("production certification duration must be 24-72 hours")
    # The only supported policy variation is a longer bounded soak.  Normalize
    # that one field and compare canonical JSON bytes so callers cannot weaken
    # sampling, SLO/RPO/RTO targets, fault timing, phases, or invariants through
    # the optional --campaign input.
    normalized = dict(campaign)
    normalized["durationSeconds"] = canonical["durationSeconds"]
    if _canonical(normalized) != _canonical(canonical):
        raise CampaignError("production certification campaign policy is not exact")
    scenarios = campaign["scenarios"]
    ids = tuple(str(item["id"]) for item in scenarios)
    if ids != PRODUCTION_CERTIFICATION_SCENARIOS:
        raise CampaignError("production certification scenario set is not exact")
    if set(config.cert_hook_commands) != set(PRODUCTION_CERTIFICATION_SCENARIOS):
        raise CampaignError("production certification hook command set is not exact")
    if set(config.cert_fault_action_commands) != set(
        PRODUCTION_CERTIFICATION_SCENARIOS
    ) or set(config.cert_fault_probe_commands) != set(
        PRODUCTION_CERTIFICATION_SCENARIOS
    ):
        raise CampaignError("production certification fault command sets are not exact")
    _command(config.cert_load_command, "CERT_LOAD_COMMAND")
    _command(config.cert_metrics_command, "CERT_METRICS_COMMAND")
    _command(config.cert_evidence_signer_command, "CERT_EVIDENCE_SIGNER_COMMAND")
    _command(config.cert_evidence_verifier_command, "CERT_EVIDENCE_VERIFIER_COMMAND")
    for scenario in scenarios:
        _command(
            config.cert_hook_commands[str(scenario["id"])],
            f"CERT_HOOK_COMMANDS.{scenario['id']}",
        )
        # Invariants and commit phases were already compared byte-for-byte with
        # the packaged policy above.  Keep the explicit lookup here to make the
        # command-to-scenario binding obvious at the execution boundary.
        _ = scenario["invariants"]


class MetricsCollector:
    def __init__(self, command: list[str], interval_seconds: int, output: Path) -> None:
        self.command = command
        self.interval_seconds = interval_seconds
        self.output = output
        self.stop = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self._run, name="cert-metrics", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=max(30, self.interval_seconds * 2))
        if self.thread.is_alive():
            raise CampaignError("metrics collector did not stop")

    def _run(self) -> None:
        next_run = time.monotonic()
        with self.output.open("ab", buffering=0) as stream:
            while not self.stop.is_set():
                delay = max(0.0, next_run - time.monotonic())
                if self.stop.wait(delay):
                    return
                try:
                    result = run_bounded(
                        self.command,
                        timeout=max(30, self.interval_seconds),
                        maximum_output_bytes=262_144,
                    )
                except AdapterBoundaryError as exc:
                    self.error = str(exc)
                    return
                if result.returncode != 0:
                    self.error = (
                        "metrics-command-failed:"
                        + hashlib.sha256(result.stdout + result.stderr).hexdigest()
                    )
                    return
                try:
                    payload = json.loads(result.stdout)
                    sample = payload["sample"]
                    values = sample["values"]
                    if payload.get("ok") is not True or not isinstance(values, dict):
                        raise ValueError("invalid metrics response")
                    if set(values) != _REQUIRED_METRICS:
                        raise ValueError("metrics response keys are not exact")
                    numeric_values = {str(k): float(v) for k, v in values.items()}
                    if any(
                        not math.isfinite(value) or value < 0
                        for value in numeric_values.values()
                    ):
                        raise ValueError("metrics response values are invalid")
                    timestamp = sample["timestampUnix"]
                    if type(timestamp) is not int or timestamp < 1:
                        raise ValueError("metrics timestamp is invalid")
                    normalized = {
                        "timestampUnix": timestamp,
                        "values": dict(sorted(numeric_values.items())),
                    }
                except Exception:  # noqa: BLE001 - normalized aggregate contract
                    self.error = "metrics-command-returned-invalid-sample"
                    return
                line = _canonical(normalized) + b"\n"
                stream.write(line)
                self.samples.append(normalized)
                next_run = _next_metric_deadline(
                    next_run,
                    time.monotonic(),
                    self.interval_seconds,
                )


def _invoke_hook(
    scenario: dict[str, Any],
    *,
    command: list[str],
    release_digest: str,
    phase: str | None,
) -> tuple[dict[str, Any], str]:
    request = {
        "apiVersion": "graphos.io/v1",
        "kind": "CertificationFaultRequest",
        "scenario": scenario["id"],
        "phase": phase,
        "releaseDigest": release_digest,
        "timeoutSeconds": int(scenario["timeoutSeconds"]),
    }
    result = run_bounded(
        command,
        payload=_canonical(request),
        timeout=int(scenario["timeoutSeconds"]),
    )
    output_digest = _digest_bytes(result.stdout + result.stderr)
    if result.returncode != 0:
        raise CampaignError(f"fault hook failed with output {output_digest}")
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CampaignError("fault hook returned non-JSON") from exc
    if not isinstance(response, dict) or response.get("ok") is not True:
        raise CampaignError("fault hook did not report a successful recovery")
    if response.get("faultApplied") is not True:
        raise CampaignError("fault hook did not prove that a real fault was applied")
    action_digest = _proof_digest(response.get("actionDigest"), "action digest")
    observation_digest = _proof_digest(
        response.get("observationDigest"), "observation digest"
    )
    required = set(scenario["invariants"])
    observed = response.get("invariants")
    if not isinstance(observed, dict) or any(
        observed.get(name) is not True for name in required
    ):
        raise CampaignError("fault hook did not prove every required invariant")
    recovery_value = response.get("recoverySeconds", -1)
    rpo_value = response.get("observedRpoSeconds", -1)
    if isinstance(recovery_value, bool) or isinstance(rpo_value, bool):
        raise CampaignError("fault hook returned invalid recovery/RPO measurements")
    recovery = float(recovery_value)
    rpo = float(rpo_value)
    if not math.isfinite(recovery) or not math.isfinite(rpo) or recovery < 0 or rpo < 0:
        raise CampaignError("fault hook omitted recovery/RPO measurements")
    return {
        "recoverySeconds": recovery,
        "observedRpoSeconds": rpo,
        "invariants": {name: True for name in sorted(required)},
        "faultApplied": True,
        "actionDigest": action_digest,
        "observationDigest": observation_digest,
    }, output_digest


def _run_scenario(
    scenario: dict[str, Any],
    *,
    hook_command: list[str],
    release_digest: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    start_index = len(samples)
    phases = scenario.get("phases") or [None]
    action_material = bytearray()
    observation_material = bytearray()
    max_recovery = 0.0
    max_rpo = 0.0
    combined = {name: True for name in scenario["invariants"]}
    try:
        for phase in phases:
            result, output_digest = _invoke_hook(
                scenario,
                command=hook_command,
                release_digest=release_digest,
                phase=phase,
            )
            action_material.extend(output_digest.encode("ascii"))
            action_material.extend(result["actionDigest"].encode("ascii"))
            observation_material.extend(result["observationDigest"].encode("ascii"))
            max_recovery = max(max_recovery, result["recoverySeconds"])
            max_rpo = max(max_rpo, result["observedRpoSeconds"])
        status = "pass"
    except Exception as exc:  # noqa: BLE001 - retain signed failure evidence
        action_material.extend(type(exc).__name__.encode("utf-8"))
        combined = {name: False for name in scenario["invariants"]}
        status = "fail"
    scenario_samples = samples[start_index:]
    return {
        "id": scenario["id"],
        "result": status,
        "actionDigest": _digest_bytes(bytes(action_material)),
        "observationDigest": _digest_bytes(bytes(observation_material)),
        "faultApplied": status == "pass",
        "metricsDigest": _digest_bytes(_canonical(scenario_samples)),
        "recoverySeconds": max_recovery,
        "observedRpoSeconds": max_rpo,
        "invariants": combined,
    }


def _load_report_ok(report: dict[str, Any], *, configured_duration: int) -> bool:
    if report.get("ok") is not True or float(report.get("scale", 0)) != 1.0:
        return False
    if (
        float(report.get("duration_seconds", -1)) != configured_duration
        or float(report.get("real_duration_seconds", -1)) < configured_duration
    ):
        return False
    if not all(all(values.values()) for values in report.get("slo_pass", {}).values()):
        return False
    return not any((report.get("invariant_violation_counts") or {}).values())


def _numeric_map(
    value: Any,
    *,
    exact_keys: set[str],
    integer: bool,
) -> dict[str, int | float]:
    if not isinstance(value, dict) or set(value) != exact_keys:
        raise CampaignError("load report aggregate keys are not exact")
    normalized: dict[str, int | float] = {}
    for key in sorted(exact_keys):
        if isinstance(value[key], bool):
            raise CampaignError("load report aggregate is not numeric")
        numeric = float(value[key])
        if not math.isfinite(numeric) or numeric < 0:
            raise CampaignError("load report aggregate is negative")
        if integer and numeric != int(numeric):
            raise CampaignError("load report count is not an integer")
        converted: int | float = int(numeric) if integer else numeric
        normalized[key] = converted
    return normalized


def _percentile_map(value: Any, *, booleans: bool) -> dict[str, Any]:
    axes = {
        "queue_latency_ms",
        "query_latency_ms",
        "write_latency_ms",
        "end_to_end_latency_ms",
    }
    percentiles = {"p50", "p95", "p99", "p99_9"}
    if not isinstance(value, dict) or set(value) != axes:
        raise CampaignError("load report latency axes are not exact")
    normalized: dict[str, Any] = {}
    for axis in sorted(axes):
        observations = value[axis]
        if not isinstance(observations, dict) or set(observations) != percentiles:
            raise CampaignError("load report percentiles are not exact")
        if booleans:
            if any(type(observations[key]) is not bool for key in percentiles):
                raise CampaignError("load report SLO observation is not boolean")
            normalized[axis] = {key: observations[key] for key in sorted(percentiles)}
        else:
            normalized[axis] = _numeric_map(
                observations,
                exact_keys=percentiles,
                integer=False,
            )
    return normalized


def _observation_count(value: Any) -> int:
    if value in (None, False, 0, ""):
        return 0
    if value is True:
        return 1
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, int(value))
    raise CampaignError("load report invariant is not countable")


def _normalize_load_report(report: dict[str, Any]) -> dict[str, Any]:
    """Discard entity-level findings and retain a fixed aggregate-only report."""
    expected = {
        "ok",
        "scale",
        "duration_s",
        "real_duration_s",
        "turn_duration_s",
        "counts",
        "throughput",
        "latency_ms",
        "slo_target",
        "slo_pass",
        "invariants",
        "faults_applied",
    }
    if set(report) != expected or type(report.get("ok")) is not bool:
        raise CampaignError("load report structure is not exact")
    scale = float(report["scale"])
    if scale != 1.0:
        raise CampaignError("load report is not scale=1.0")
    durations = _numeric_map(
        {
            "duration_seconds": report["duration_s"],
            "real_duration_seconds": report["real_duration_s"],
            "turn_duration_seconds": report["turn_duration_s"],
        },
        exact_keys={
            "duration_seconds",
            "real_duration_seconds",
            "turn_duration_seconds",
        },
        integer=False,
    )
    invariants = report["invariants"]
    invariant_names = {
        "duplicate_side_effects",
        "falsely_completed",
        "ran_but_not_terminal",
        "stuck_leases",
        "cross_tenant_violations",
    }
    if not isinstance(invariants, dict) or not invariant_names.issubset(invariants):
        raise CampaignError("load report invariant set is incomplete")
    faults = report["faults_applied"]
    if not isinstance(faults, list):
        raise CampaignError("load report faults are not a collection")
    return {
        "ok": report["ok"],
        "scale": scale,
        **durations,
        "counts": _numeric_map(
            report["counts"],
            exact_keys={
                "turns_submitted",
                "turns_succeeded",
                "turns_dead_letter",
                "turns_failed",
                "turns_cancelled",
                "messages_sent",
                "messages_delivered",
            },
            integer=True,
        ),
        "throughput": _numeric_map(
            report["throughput"],
            exact_keys={
                "turns_per_sec_measured",
                "messages_per_sec_measured",
                "mutations_per_sec_measured",
            },
            integer=False,
        ),
        "latency_ms": _percentile_map(report["latency_ms"], booleans=False),
        "slo_target": _percentile_map(report["slo_target"], booleans=False),
        "slo_pass": _percentile_map(report["slo_pass"], booleans=True),
        "invariant_violation_counts": {
            key: _observation_count(invariants[key]) for key in sorted(invariant_names)
        },
        "faults_applied_count": len(faults),
    }


def _metric_summary(
    samples: list[dict[str, Any]], targets: dict[str, Any]
) -> tuple[bool, float]:
    if not samples:
        return False, float(targets["rpoSeconds"]) + 1.0
    values = [sample["values"] for sample in samples]
    if any(set(value) != _REQUIRED_METRICS for value in values):
        return False, float("inf")
    observed_rpo = max(value["checkpointAgeSeconds"] for value in values)
    ok = (
        max(value["gatewayP99Seconds"] for value in values)
        <= float(targets["gatewayP99Seconds"])
        and max(value["engineP99Seconds"] for value in values)
        <= float(targets["engineP99Seconds"])
        and max(value["dispatchQueueDepth"] for value in values)
        <= int(targets["maximumQueueDepth"])
        and max(value["ingestConsumerLag"] for value in values)
        <= int(targets["maximumQueueDepth"])
        and max(value["analyticsJobsReady"] for value in values)
        <= int(targets["maximumQueueDepth"])
        and min(value["reachableEngineMembers"] for value in values) >= 2
        and max(value["walAppendDroppedFiveMinutes"] for value in values) == 0
        and max(value["gatewayErrorRatio"] for value in values)
        <= float(targets["maximumGatewayErrorRatio"])
        and max(value["podRestartsFiveMinutes"] for value in values)
        <= int(targets["maximumPodRestartsFiveMinutes"])
        and observed_rpo <= float(targets["rpoSeconds"])
    )
    return ok, observed_rpo


def execute(
    *,
    campaign_path: Path,
    release_path: Path,
    matrix_path: Path,
    artifacts_dir: Path,
    config: AgentConfig | None = None,
) -> tuple[dict[str, Any], bool]:
    config = config or AgentConfig()
    if config.certification_mode != "production":
        raise CampaignError("CERTIFICATION_MODE=production is required")
    configured_release = str(config.cert_release_manifest or "").strip()
    configured_artifacts = str(config.cert_artifacts_dir or "").strip()
    if not configured_release or not configured_artifacts:
        raise CampaignError("production certification paths are not configured")
    if release_path.resolve(strict=False) != Path(configured_release).resolve(
        strict=False
    ) or artifacts_dir.resolve(strict=False) != Path(configured_artifacts).resolve(
        strict=False
    ):
        raise CampaignError("campaign paths do not match the active AgentConfig")
    campaign = _yaml(campaign_path)
    _validate_campaign(campaign, config)
    release = _yaml(release_path)
    matrix = _yaml(matrix_path)
    release_report = compatibility.verify_release_manifest(
        release,
        matrix,
        matrix_path=matrix_path,
        manifest_path=release_path,
        verify_signatures=True,
    )
    _validate_artifacts_directory(artifacts_dir)
    hardware_class = str(config.cert_hardware_class or "").strip()
    if not re.fullmatch(r"(?:capacity|tier)-[a-z0-9][a-z0-9._-]{1,63}", hardware_class):
        raise CampaignError(
            "hardware class must be a non-identifying capacity-* or tier-* label"
        )
    duration = int(campaign["durationSeconds"])
    interval = int(campaign["metricsIntervalSeconds"])
    report_path = artifacts_dir / "load-report.json"
    metrics_path = artifacts_dir / "metrics.ndjson"
    values = {
        "duration_seconds": str(duration),
        "scale": "1.0",
        "report_file": str(report_path),
        "release_digest": release_report["releaseDigest"],
    }
    load_template = _command(config.cert_load_command, "CERT_LOAD_COMMAND")
    if not any("{report_file}" in part for part in load_template):
        raise CampaignError("load command must write the exact report_file placeholder")
    load_command = _render_command(load_template, values)
    metrics_command = _render_command(
        _command(config.cert_metrics_command, "CERT_METRICS_COMMAND"), values
    )
    started_unix = int(time.time())
    started = time.monotonic()
    load_process = subprocess.Popen(
        load_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    collector = MetricsCollector(metrics_command, interval, metrics_path)
    collector.start()
    scenarios: list[dict[str, Any]] = []
    load_returncode = -1
    load_finished: float | None = None
    runtime_error: str | None = None
    try:
        for scenario in campaign["scenarios"]:
            due = started + duration * float(scenario["atFraction"])
            while time.monotonic() < due:
                wait_seconds = max(0.001, min(30.0, due - time.monotonic()))
                try:
                    load_returncode = load_process.wait(timeout=wait_seconds)
                except subprocess.TimeoutExpired:
                    continue
                load_finished = time.monotonic()
                if load_returncode is not None:
                    raise CampaignError(
                        "load generator exited before all scenarios ran"
                    )
            scenarios.append(
                _run_scenario(
                    scenario,
                    hook_command=config.cert_hook_commands[str(scenario["id"])],
                    release_digest=release_report["releaseDigest"],
                    samples=collector.samples,
                )
            )
        remaining = max(1.0, started + duration - time.monotonic())
        try:
            load_returncode = load_process.wait(timeout=remaining + 600)
            load_finished = time.monotonic()
        except subprocess.TimeoutExpired:
            load_process.terminate()
            load_process.wait(timeout=60)
            load_finished = time.monotonic()
            load_returncode = -1
    except Exception as exc:  # noqa: BLE001 - retain a signed failed campaign
        runtime_error = type(exc).__name__
    finally:
        if load_process.poll() is None:
            load_process.terminate()
            try:
                load_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                load_process.kill()
                load_process.wait(timeout=10)
        if load_finished is None:
            load_finished = time.monotonic()
        try:
            collector.close()
        except Exception as exc:  # noqa: BLE001 - retain a signed failed campaign
            runtime_error = runtime_error or type(exc).__name__
    observed_duration = max(0.0, load_finished - started)
    completed = {item["id"] for item in scenarios}
    for scenario in campaign["scenarios"]:
        if scenario["id"] in completed:
            continue
        failure_material = (runtime_error or "campaign-runtime-failure").encode("utf-8")
        scenarios.append(
            {
                "id": scenario["id"],
                "result": "fail",
                "actionDigest": _digest_bytes(failure_material),
                "observationDigest": _digest_bytes(failure_material + b"observation"),
                "faultApplied": False,
                "metricsDigest": _digest_bytes(_canonical([])),
                "recoverySeconds": 0.0,
                "observedRpoSeconds": 0.0,
                "invariants": {name: False for name in sorted(scenario["invariants"])},
            }
        )
    load_report: dict[str, Any] = {}
    try:
        if report_path.is_symlink() or not report_path.is_file():
            raise CampaignError("load report is not a regular artifact")
        raw_load_report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(raw_load_report, dict):
            raise CampaignError("load report is not a mapping")
        load_report = _normalize_load_report(raw_load_report)
    except (CampaignError, json.JSONDecodeError, OSError, TypeError, ValueError):
        load_report = {}
    finally:
        # The load generator may use entity identifiers internally to prove its
        # invariants. Never retain those raw findings in certification artifacts.
        if report_path.is_symlink():
            report_path.unlink(missing_ok=True)
        report_path.write_text(
            json.dumps(load_report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    metrics_ok, metrics_rpo = _metric_summary(collector.samples, campaign["targets"])
    expected_samples = max(1, duration // interval)
    sample_coverage = min(1.0, len(collector.samples) / expected_samples)
    scenario_rpo = max(
        (item.pop("observedRpoSeconds") for item in scenarios), default=0.0
    )
    observed_rpo = max(metrics_rpo, scenario_rpo)
    observed_rto = max((item["recoverySeconds"] for item in scenarios), default=0.0)
    duration_ok = observed_duration >= duration
    load_ok = load_returncode == 0 and _load_report_ok(
        load_report,
        configured_duration=duration,
    )
    scenarios_ok = all(item["result"] == "pass" for item in scenarios)
    coverage_ok = sample_coverage >= float(campaign["minimumSampleCoverage"])
    recovery_ok = observed_rpo <= float(
        campaign["targets"]["rpoSeconds"]
    ) and observed_rto <= float(campaign["targets"]["rtoSeconds"])
    passed = bool(
        load_ok
        and scenarios_ok
        and metrics_ok
        and coverage_ok
        and recovery_ok
        and collector.error is None
        and duration_ok
    )
    unsigned = {
        "apiVersion": "graphos.io/v1",
        "kind": "OperationalEvidence",
        "evidenceVersion": 1,
        "release": {
            "digest": release_report["releaseDigest"],
            "configurationDigest": release["configurationDigest"],
            "componentDigests": release_report["componentDigests"],
            "certificationDigests": release_report["certificationDigests"],
        },
        "campaign": {
            # Bind the parsed policy that actually executed. This avoids a
            # 24-hour path TOCTOU if an operator-supplied YAML file is replaced
            # after preflight while retaining the exact semantic policy.
            "digest": _digest_bytes(_canonical(campaign)),
            "scale": 1.0,
            "durationSeconds": duration,
            "observedDurationSeconds": round(observed_duration, 3),
            "startedAtUnix": started_unix,
            "hardwareClass": hardware_class,
        },
        "scenarios": scenarios,
        "metrics": {
            "rawDigest": _digest_bytes(metrics_path.read_bytes()),
            "loadReportDigest": _digest_bytes(_canonical(load_report)),
            "sampleCount": len(collector.samples),
            "sampleCoverage": round(sample_coverage, 6),
            "sloPass": bool(load_ok and metrics_ok),
        },
        "recovery": {
            "rpoTargetSeconds": float(campaign["targets"]["rpoSeconds"]),
            "rtoTargetSeconds": float(campaign["targets"]["rtoSeconds"]),
            "observedRpoSeconds": round(observed_rpo, 3),
            "observedRtoSeconds": round(observed_rto, 3),
            "pass": recovery_ok,
        },
        "privacy": {
            "policyDigest": _digest_bytes(
                b"aggregate-metrics;opaque-identities;no-endpoints;no-filesystem-locations"
            ),
            "containsDirectIdentifiers": False,
            "containsEndpoints": False,
            "containsFilesystemLocations": False,
        },
        "result": "pass" if passed else "fail",
    }
    signed = evidence.sign_evidence(unsigned, config=config)
    evidence.verify_evidence(signed, config=config)
    (artifacts_dir / "operational-evidence.json").write_text(
        json.dumps(signed, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return signed, passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphos-certification-campaign")
    parser.add_argument(
        "--campaign",
        type=Path,
        default=None,
        help="Campaign YAML (default: packaged production campaign)",
    )
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Compatibility matrix (default: packaged current matrix)",
    )
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with ExitStack() as resources:
            campaign_path = args.campaign or resources.enter_context(
                as_file(files("deploy.release").joinpath("certification-campaign.yml"))
            )
            matrix_path = args.matrix or resources.enter_context(
                as_file(files("deploy.release").joinpath("compatibility-matrix.yml"))
            )
            signed, passed = execute(
                campaign_path=campaign_path,
                release_path=args.release,
                matrix_path=matrix_path,
                artifacts_dir=args.artifacts_dir,
            )
    except Exception as exc:  # noqa: BLE001 - never emit environment details
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "ok": passed,
                "result": signed["result"],
                "subjectDigest": signed["signature"]["subjectDigest"],
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
