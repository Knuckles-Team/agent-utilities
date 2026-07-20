"""Static contracts for the signed production-cell deployment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_production_template_satisfies_fail_closed_source_contract():
    gate = _load(
        "graphos_production_assets_test",
        "scripts/deployment/check_production_assets.py",
    )
    report = gate.check(
        ROOT / "deploy" / "k8s" / "production-cell",
        rendered=False,
        repository_root=ROOT,
    )
    assert report["ok"] is True
    assert report["rendered"] is False


def test_workload_identity_audience_is_a_required_deployment_substitution():
    expected = "${GRAPHOS_WORKLOAD_IDENTITY_AUDIENCE:?required}"

    def audiences(value: Any):
        if isinstance(value, dict):
            token = value.get("serviceAccountToken")
            if isinstance(token, dict):
                yield token.get("audience")
            for child in value.values():
                yield from audiences(child)
        elif isinstance(value, list):
            for child in value:
                yield from audiences(child)

    found: list[str] = []
    source = ROOT / "deploy" / "k8s" / "production-cell"
    for path in source.glob("*.yaml"):
        for document in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            found.extend(value for value in audiences(document) if value is not None)

    assert found
    assert set(found) == {expected}


def test_rendered_gate_rejects_sentinel_image_pins():
    gate = _load(
        "graphos_production_assets_rendered_test",
        "scripts/deployment/check_production_assets.py",
    )
    with pytest.raises(gate.ProductionAssetError, match="digest-pinned"):
        gate.check(
            ROOT / "deploy" / "k8s" / "production-cell",
            rendered=True,
            repository_root=ROOT,
        )


def test_operational_evidence_rejects_raw_identity_and_location_fields():
    evidence = _load(
        "graphos_operational_evidence_test",
        "scripts/certification/evidence.py",
    )
    digest = "sha256:" + "1" * 64
    policy = yaml.safe_load(
        (ROOT / "deploy/release/certification-campaign.yml").read_text(
            encoding="utf-8"
        )
    )
    value = {
        "apiVersion": "graphos.io/v1",
        "kind": "OperationalEvidence",
        "evidenceVersion": 1,
        "release": {
            "digest": digest,
            "configurationDigest": digest,
            "componentDigests": {
                name: digest
                for name in {
                    "epistemic-operations-protocol",
                    "epistemic-graph",
                    "agent-utilities",
                    "langfuse-agent",
                    "connector-bundles",
                    "prebundled-skills",
                    "ontology-lock",
                    "index-migrations",
                }
            },
            "certificationDigests": {
                "connectorLiveCertificationLedger": digest,
                "prebundledSkillValidationMatrix": digest,
                "skillValidationDeployment": digest,
                "skillValidationLifecycleEvidence": digest,
                "exactArtifactClosureEvidence": digest,
                "ociVulnerabilityScanEvidence": digest,
            },
        },
        "campaign": {
            "digest": evidence.digest_bytes(evidence.canonical_bytes(policy)),
            "scale": 1.0,
            "durationSeconds": 86400,
            "observedDurationSeconds": 86400.0,
            "startedAtUnix": 1,
            "hardwareClass": "capacity-standard",
        },
        "scenarios": [
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
        ],
        "metrics": {
            "rawDigest": digest,
            "loadReportDigest": digest,
            "sampleCount": 5760,
            "sampleCoverage": 1.0,
            "sloPass": True,
        },
        "recovery": {
            "rpoTargetSeconds": 60,
            "rtoTargetSeconds": 300,
            "observedRpoSeconds": 1,
            "observedRtoSeconds": 1,
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
    evidence.validate_evidence(value, require_signature=False)
    value["metrics"]["endpoint"] = "https://example.invalid"
    with pytest.raises(evidence.EvidenceError):
        evidence.validate_evidence(value, require_signature=False)
