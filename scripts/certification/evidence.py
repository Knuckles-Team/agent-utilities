#!/usr/bin/python
"""Privacy gate plus external signing/verification for operational evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from scripts.certification.subprocess_boundary import (
    AdapterBoundaryError,
    run_bounded,
)

_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_HARDWARE = re.compile(r"^(?:capacity|tier)-[a-z0-9][a-z0-9._-]{1,63}$")
_SIGNATURE_SCHEME = re.compile(r"^[a-z0-9][a-z0-9+._-]{1,63}$")
_SIGNATURE_VALUE = re.compile(r"^[A-Za-z0-9+/_=-]{16,16384}$")
_COMPONENTS = {
    "epistemic-operations-protocol",
    "epistemic-graph",
    "agent-utilities",
    "langfuse-agent",
    "connector-bundles",
    "prebundled-skills",
    "ontology-lock",
    "index-migrations",
}
_CERTIFICATIONS = {
    "connectorLiveCertificationLedger",
    "prebundledSkillValidationMatrix",
    "skillValidationDeployment",
    "skillValidationLifecycleEvidence",
    "exactArtifactClosureEvidence",
    "ociVulnerabilityScanEvidence",
}
_FORBIDDEN_KEYS = re.compile(
    r"(host(name)?|endpoint|url|uri|path|directory|user(name)?|email|principal|address)",
    re.IGNORECASE,
)
_FORBIDDEN_TEXT = (
    re.compile(r"(?:[A-Za-z]:\\|/home/|/Users/|/mnt/[a-z]/|file://)"),
    re.compile(r"(?:https?|tcp|unix)://", re.IGNORECASE),
    re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
)


class EvidenceError(ValueError):
    """Evidence is malformed, identifying, unsigned, or unverifiable."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest(value: Any, field: str) -> str:
    text = str(value or "")
    if not _DIGEST.fullmatch(text) or text.endswith("0" * 64):
        raise EvidenceError(f"{field} is not a non-sentinel digest")
    return text


def _campaign_policy(duration_seconds: int) -> dict[str, Any]:
    policy = yaml.safe_load(
        files("deploy.release")
        .joinpath("certification-campaign.yml")
        .read_text(encoding="utf-8")
    )
    if not isinstance(policy, dict):
        raise EvidenceError("packaged certification policy is invalid")
    policy["durationSeconds"] = duration_seconds
    return policy


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise EvidenceError(f"{field} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{field} is not numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise EvidenceError(f"{field} is not a finite non-negative number")
    return number


def _privacy_walk(value: Any, parent: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if _FORBIDDEN_KEYS.search(key_text) and key_text not in {
                "hardwareClass",
                "containsEndpoints",
                "containsFilesystemLocations",
            }:
                raise EvidenceError(f"forbidden identifying evidence key: {key_text}")
            _privacy_walk(child, key_text)
    elif isinstance(value, list):
        for child in value:
            _privacy_walk(child, parent)
    elif isinstance(value, str):
        for pattern in _FORBIDDEN_TEXT:
            if pattern.search(value):
                raise EvidenceError(f"forbidden identifying text in {parent}")


def validate_evidence(value: dict[str, Any], *, require_signature: bool) -> None:
    required = {
        "apiVersion",
        "kind",
        "evidenceVersion",
        "release",
        "campaign",
        "scenarios",
        "metrics",
        "recovery",
        "privacy",
        "result",
    }
    if not required.issubset(value):
        raise EvidenceError("evidence is missing required sections")
    allowed = required | ({"signature"} if require_signature else set())
    if set(value) != allowed:
        raise EvidenceError("evidence top-level keys are not exact")
    if value["apiVersion"] != "graphos.io/v1" or value["kind"] != "OperationalEvidence":
        raise EvidenceError("unsupported evidence apiVersion/kind")
    if type(value["evidenceVersion"]) is not int or value["evidenceVersion"] != 1:
        raise EvidenceError("unsupported evidenceVersion")
    if not isinstance(value["result"], str) or value["result"] not in {"pass", "fail"}:
        raise EvidenceError("evidence result is invalid")
    exact_sections = {
        "release": {
            "digest",
            "configurationDigest",
            "componentDigests",
            "certificationDigests",
        },
        "campaign": {
            "digest",
            "scale",
            "durationSeconds",
            "observedDurationSeconds",
            "startedAtUnix",
            "hardwareClass",
        },
        "metrics": {
            "rawDigest",
            "loadReportDigest",
            "sampleCount",
            "sampleCoverage",
            "sloPass",
        },
        "recovery": {
            "rpoTargetSeconds",
            "rtoTargetSeconds",
            "observedRpoSeconds",
            "observedRtoSeconds",
            "pass",
        },
        "privacy": {
            "policyDigest",
            "containsDirectIdentifiers",
            "containsEndpoints",
            "containsFilesystemLocations",
        },
    }
    for section_name, keys in exact_sections.items():
        section = value.get(section_name)
        if not isinstance(section, dict) or set(section) != keys:
            raise EvidenceError(f"{section_name} evidence keys are not exact")
    campaign = value["campaign"]
    if isinstance(campaign.get("scale"), bool) or campaign.get("scale") != 1.0:
        raise EvidenceError("production evidence requires scale=1.0")
    duration_value = campaign.get("durationSeconds")
    if type(duration_value) is not int:
        raise EvidenceError("production evidence duration must be an integer")
    duration = duration_value
    if not 86400 <= duration <= 259200:
        raise EvidenceError("production evidence requires a 24-72 hour campaign")
    observed_duration = _finite_nonnegative(
        campaign.get("observedDurationSeconds"), "observedDurationSeconds"
    )
    if type(campaign.get("startedAtUnix")) is not int or campaign["startedAtUnix"] < 1:
        raise EvidenceError("campaign startedAtUnix is invalid")
    if not _HARDWARE.fullmatch(str(campaign.get("hardwareClass") or "")):
        raise EvidenceError(
            "hardwareClass must be a capacity-* or tier-* non-identifying label"
        )
    for section in (value["release"], campaign, value["metrics"], value["privacy"]):
        for key, child in section.items():
            if key.casefold().endswith("digest"):
                _digest(child, key)
    policy = _campaign_policy(duration)
    if campaign["digest"] != digest_bytes(canonical_bytes(policy)):
        raise EvidenceError("campaign digest does not bind the exact packaged policy")
    component_digests = value["release"].get("componentDigests")
    if not isinstance(component_digests, dict) or set(component_digests) != _COMPONENTS:
        raise EvidenceError("component digest catalog is not exact")
    for digest in component_digests.values():
        _digest(digest, "componentDigest")
    certification_digests = value["release"].get("certificationDigests")
    if (
        not isinstance(certification_digests, dict)
        or set(certification_digests) != _CERTIFICATIONS
    ):
        raise EvidenceError("certification digest catalog is not exact")
    for digest in certification_digests.values():
        _digest(digest, "certificationDigest")
    scenarios = value["scenarios"]
    expected_scenarios = policy["scenarios"]
    if (
        not isinstance(scenarios, list)
        or len(scenarios) != len(expected_scenarios)
        or [item.get("id") for item in scenarios if isinstance(item, dict)]
        != [item["id"] for item in expected_scenarios]
    ):
        raise EvidenceError("evidence does not cover the complete fault campaign")
    scenario_keys = {
        "id",
        "result",
        "faultApplied",
        "actionDigest",
        "observationDigest",
        "metricsDigest",
        "recoverySeconds",
        "invariants",
    }
    for scenario, expected_scenario in zip(scenarios, expected_scenarios, strict=True):
        if not isinstance(scenario, dict) or set(scenario) != scenario_keys:
            raise EvidenceError("scenario evidence keys are not exact")
        for key in ("actionDigest", "observationDigest", "metricsDigest"):
            _digest(scenario.get(key), f"scenario.{key}")
        if scenario.get("result") not in {"pass", "fail"}:
            raise EvidenceError("scenario result is invalid")
        if type(scenario.get("faultApplied")) is not bool:
            raise EvidenceError("scenario fault-applied observation is not boolean")
        if (
            scenario.get("result") == "pass"
            and scenario.get("faultApplied") is not True
        ):
            raise EvidenceError("passing scenario did not apply a real fault")
        recovery_seconds = _finite_nonnegative(
            scenario.get("recoverySeconds"), "scenario.recoverySeconds"
        )
        if recovery_seconds > float(policy["targets"]["rtoSeconds"]) and scenario.get(
            "result"
        ) == "pass":
            raise EvidenceError("passing scenario exceeded the canonical RTO")
        invariants = scenario.get("invariants")
        if (
            not isinstance(invariants, dict)
            or set(invariants) != set(expected_scenario["invariants"])
            or any(type(observed) is not bool for observed in invariants.values())
        ):
            raise EvidenceError("scenario invariant observations are not exact")
        if scenario.get("result") == "pass" and not all(invariants.values()):
            raise EvidenceError("passing scenario contradicts its invariants")
    metrics = value["metrics"]
    if type(metrics.get("sampleCount")) is not int or metrics["sampleCount"] < 0:
        raise EvidenceError("metrics sampleCount is invalid")
    sample_coverage = _finite_nonnegative(
        metrics.get("sampleCoverage"), "metrics.sampleCoverage"
    )
    if sample_coverage > 1:
        raise EvidenceError("metrics sampleCoverage exceeds one")
    expected_samples = max(1, duration // int(policy["metricsIntervalSeconds"]))
    expected_coverage = round(min(1.0, metrics["sampleCount"] / expected_samples), 6)
    if sample_coverage != expected_coverage:
        raise EvidenceError("metrics sampleCoverage is inconsistent with sampleCount")
    if type(metrics.get("sloPass")) is not bool:
        raise EvidenceError("metrics sloPass is not boolean")
    recovery = value["recovery"]
    rpo_target = _finite_nonnegative(
        recovery.get("rpoTargetSeconds"), "recovery.rpoTargetSeconds"
    )
    rto_target = _finite_nonnegative(
        recovery.get("rtoTargetSeconds"), "recovery.rtoTargetSeconds"
    )
    if rpo_target != float(policy["targets"]["rpoSeconds"]) or rto_target != float(
        policy["targets"]["rtoSeconds"]
    ):
        raise EvidenceError("recovery targets are not canonical")
    observed_rpo = _finite_nonnegative(
        recovery.get("observedRpoSeconds"), "recovery.observedRpoSeconds"
    )
    observed_rto = _finite_nonnegative(
        recovery.get("observedRtoSeconds"), "recovery.observedRtoSeconds"
    )
    if type(recovery.get("pass")) is not bool:
        raise EvidenceError("recovery pass is not boolean")
    if recovery["pass"] != (observed_rpo <= rpo_target and observed_rto <= rto_target):
        raise EvidenceError("recovery pass contradicts its observations")
    privacy = value["privacy"]
    if any(
        privacy.get(key) is not False
        for key in (
            "containsDirectIdentifiers",
            "containsEndpoints",
            "containsFilesystemLocations",
        )
    ):
        raise EvidenceError("privacy assertions must all be false")
    unsigned = {key: child for key, child in value.items() if key != "signature"}
    _privacy_walk(unsigned)
    if value.get("result") == "pass" and (
        not all(scenario["result"] == "pass" for scenario in scenarios)
        or observed_duration < duration
        or sample_coverage < float(policy["minimumSampleCoverage"])
        or value["metrics"].get("sloPass") is not True
        or value["recovery"].get("pass") is not True
    ):
        raise EvidenceError("passing evidence contradicts its observations")
    if require_signature:
        signature = value.get("signature")
        signature_keys = {
            "scheme",
            "subjectDigest",
            "bundleDigest",
            "signerIdentityDigest",
            "value",
        }
        if not isinstance(signature, dict) or set(signature) != signature_keys:
            raise EvidenceError("signed evidence has no signature")
        for key in ("subjectDigest", "bundleDigest", "signerIdentityDigest"):
            _digest(signature.get(key), f"signature.{key}")
        if not _SIGNATURE_SCHEME.fullmatch(str(signature.get("scheme") or "")):
            raise EvidenceError("signature scheme is not an opaque algorithm label")
        if not _SIGNATURE_VALUE.fullmatch(str(signature.get("value") or "")):
            raise EvidenceError("signature value is not an opaque encoded signature")
        if signature["subjectDigest"] != digest_bytes(canonical_bytes(unsigned)):
            raise EvidenceError("signature subject digest does not bind the evidence")
        _privacy_walk(signature)


def _command(value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, str) and v for v in value)
    ):
        raise EvidenceError(f"{field} must contain a non-empty JSON argv array")
    from agent_utilities.skills.runtime_validation import (
        _validate_external_command_argv,
    )

    try:
        return _validate_external_command_argv(value)
    except RuntimeError as exc:
        raise EvidenceError(f"{field} is not a safe executable argv array") from exc


def _invoke(command: list[str], payload: bytes) -> dict[str, Any]:
    try:
        result = run_bounded(command, payload=payload, timeout=120)
    except AdapterBoundaryError as exc:
        raise EvidenceError("external evidence operation violated its boundary") from exc
    if result.returncode != 0:
        raise EvidenceError(
            f"external evidence operation failed; output_digest="
            f"{hashlib.sha256(result.stdout + result.stderr).hexdigest()}"
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise EvidenceError("external evidence operation returned non-JSON") from exc
    if not isinstance(value, dict):
        raise EvidenceError("external evidence operation returned a non-object")
    return value


def sign_evidence(
    unsigned: dict[str, Any], *, config: Any | None = None
) -> dict[str, Any]:
    if config is None:
        from agent_utilities.core.config import AgentConfig

        config = AgentConfig()
    unsigned.pop("signature", None)
    validate_evidence(unsigned, require_signature=False)
    payload = canonical_bytes(unsigned)
    subject_digest = digest_bytes(payload)
    response = _invoke(
        _command(
            config.cert_evidence_signer_command,
            "CERT_EVIDENCE_SIGNER_COMMAND",
        ),
        payload,
    )
    if response.get("subjectDigest") != subject_digest:
        raise EvidenceError("external signer did not bind the canonical subject")
    signature = {
        "scheme": str(response.get("scheme") or ""),
        "subjectDigest": subject_digest,
        "bundleDigest": _digest(response.get("bundleDigest"), "bundleDigest"),
        "signerIdentityDigest": _digest(
            response.get("signerIdentityDigest"), "signerIdentityDigest"
        ),
        "value": str(response.get("signature") or ""),
    }
    if not _SIGNATURE_SCHEME.fullmatch(
        signature["scheme"]
    ) or not _SIGNATURE_VALUE.fullmatch(signature["value"]):
        raise EvidenceError("external signer returned an incomplete signature")
    signed = {**unsigned, "signature": signature}
    validate_evidence(signed, require_signature=True)
    return signed


def verify_evidence(signed: dict[str, Any], *, config: Any | None = None) -> None:
    if config is None:
        from agent_utilities.core.config import AgentConfig

        config = AgentConfig()
    validate_evidence(signed, require_signature=True)
    response = _invoke(
        _command(
            config.cert_evidence_verifier_command,
            "CERT_EVIDENCE_VERIFIER_COMMAND",
        ),
        canonical_bytes(signed),
    )
    if response.get("verified") is not True:
        raise EvidenceError("external verifier rejected operational evidence")
    if response.get("subjectDigest") != signed["signature"]["subjectDigest"]:
        raise EvidenceError("external verifier returned a different subject digest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphos-operational-evidence")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("sign", "verify"):
        command = subparsers.add_parser(operation)
        command.add_argument("--input", type=Path, required=True)
        if operation == "sign":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = json.loads(args.input.read_text(encoding="utf-8"))
        if args.operation == "sign":
            signed = sign_evidence(value)
            args.output.write_text(
                json.dumps(signed, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
        else:
            verify_evidence(value)
    except Exception as exc:  # noqa: BLE001 - privacy-safe command boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, "operation": args.operation}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
