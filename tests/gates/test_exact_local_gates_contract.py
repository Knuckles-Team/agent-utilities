"""Behavioral source gates for the exact-installed certification campaign."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str) -> Any:
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


campaign = _load(
    "exact_local_gates_contract_campaign",
    "scripts/certification/exact_local_gates.py",
)
checker = _load(
    "exact_local_gates_contract_checker",
    "scripts/check_exact_local_gates_harness.py",
)


def _checker_sources() -> tuple[str, str, str, str, str]:
    return (
        checker.HARNESS.read_text(encoding="utf-8"),
        checker.DOCUMENTATION.read_text(encoding="utf-8"),
        checker.NAVIGATION.read_text(encoding="utf-8"),
        checker.WORKFLOW.read_text(encoding="utf-8"),
        checker.A2A_TEST.read_text(encoding="utf-8"),
    )


def test_current_harness_passes_the_ast_contract() -> None:
    assert checker.check_contract(*_checker_sources()) == ()


def test_runtime_gate_case_catalogs_are_fixed_and_path_free() -> None:
    gates = {
        "g08": {
            "case_count": len(campaign.G08_RUNTIME_CASES),
            "cases": list(campaign.G08_RUNTIME_CASES),
            "status": "pass",
        },
        "g09": {
            "case_count": len(campaign.G09_RUNTIME_CASES),
            "cases": list(campaign.G09_RUNTIME_CASES),
            "status": "pass",
        },
        "g35": {
            "case_count": len(campaign.G35_RUNTIME_CASES),
            "cases": list(campaign.G35_RUNTIME_CASES),
            "status": "pass",
        },
    }

    assert gates["g08"]["case_count"] == 8
    assert gates["g09"]["case_count"] == 2
    assert gates["g35"]["case_count"] == 8
    campaign._assert_evidence_safe({"gates": gates})


def test_permission_governance_worker_emits_only_fixed_safe_cases(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("AU_EXACT_PERMISSION_AUTHORITY", "k" * 48)
    result = tmp_path / "permission-result.json"

    campaign._permission_governance_worker(tmp_path / "runtime", result)

    evidence = json.loads(result.read_text(encoding="utf-8"))
    assert evidence == {
        "case_count": 8,
        "cases": list(campaign.G35_RUNTIME_CASES),
        "status": "pass",
    }
    campaign._assert_evidence_safe({"g35": evidence})


def test_manifest_is_closed_and_digest_pinned(tmp_path: Path) -> None:
    manifest = {
        "agent_utilities_sha256": "a" * 64,
        "distribution_closure_sha256": "b" * 64,
        "engine_sha256": "c" * 64,
        "evidence_schema_version": 1,
        "graphos_sha256": "d" * 64,
        "harness_sha256": "e" * 64,
        "promotion_evidence_sha256": "2" * 64,
        "release_id": "release-certification-v1",
        "release_python_sha256": "1" * 64,
        "release_spec_sha256": "3" * 64,
        "schema_version": 1,
        "test_catalog_sha256": "f" * 64,
    }
    path = tmp_path / "manifest.json"
    raw = json.dumps(manifest, sort_keys=True).encode()
    path.write_bytes(raw)

    parsed, digest = campaign._read_release_manifest(
        str(path), hashlib.sha256(raw).hexdigest()
    )

    assert parsed == manifest
    assert digest == hashlib.sha256(raw).hexdigest()
    with pytest.raises(campaign.CertificationError):
        campaign._read_release_manifest(str(path), "0" * 64)
    path.write_text(json.dumps({**manifest, "unexpected": True}), encoding="utf-8")
    with pytest.raises(campaign.CertificationError):
        campaign._read_release_manifest(str(path), campaign._sha256_file(path))


def test_exact_staging_rejects_digest_drift_and_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "artifact"
    source.write_bytes(b"exact artifact")
    source.chmod(0o700)
    digest = campaign._sha256_file(source)

    staged = campaign._stage_exact(
        str(source), digest, tmp_path / "staged", label="test_artifact"
    )

    assert campaign._sha256_file(staged) == digest
    with pytest.raises(campaign.CertificationError):
        campaign._stage_exact(
            str(source), "0" * 64, tmp_path / "mismatch", label="test_artifact"
        )
    link = tmp_path / "artifact-link"
    link.symlink_to(source)
    with pytest.raises(campaign.CertificationError):
        campaign._stage_exact(
            str(link), digest, tmp_path / "linked", label="test_artifact"
        )


@pytest.mark.parametrize(
    "unsafe",
    [
        {"value": "/home/operator/private"},
        {"value": "https://service.invalid/api"},
        {"value": "192.0.2.4"},
        {"agent_id": "service:operator"},
    ],
)
def test_evidence_privacy_rejects_local_network_and_identity_data(
    unsafe: dict[str, str],
) -> None:
    with pytest.raises(campaign.CertificationError):
        campaign._assert_evidence_safe(unsafe)


def test_evidence_publication_requires_private_new_destination(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    output = private / "evidence.json"

    campaign._write_evidence(output, {"schema_version": 1, "status": "pass"})

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(campaign.CertificationError):
        campaign._write_evidence(output, {"status": "pass"})
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o755)
    unsafe.chmod(0o755)
    with pytest.raises(campaign.CertificationError):
        campaign._write_evidence(unsafe / "evidence.json", {"status": "pass"})


def test_pytest_collector_requires_exact_case_names() -> None:
    collector = campaign._PytestCollector(("test_exact_case",))
    collector.collected = ["test_module.py::test_exact_case_suffix"]
    collector.passed = set(collector.collected)
    assert collector.valid(0) is False

    collector.collected = ["test_module.py::test_exact_case"]
    collector.passed = set(collector.collected)
    assert collector.valid(0) is True


class _RejectingJobs:
    def __init__(self, message: str) -> None:
        self.message = message

    def submit_program_optimization(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(self.message)


def test_budget_rejection_accepts_only_the_exact_resource_limit() -> None:
    exact = SimpleNamespace(
        jobs=_RejectingJobs("program operation exceeds resource limits")
    )
    campaign._expect_resource_limit(exact, "graph", {"budget": {}})

    arbitrary = SimpleNamespace(jobs=_RejectingJobs("transport failed"))
    with pytest.raises(campaign.CertificationError):
        campaign._expect_resource_limit(arbitrary, "graph", {"budget": {}})


def test_provider_result_requires_zero_failures_and_closed_shape() -> None:
    valid = {
        "skills": {"providers": 1, "files": 2, "failed": 0},
        "prompts": {"providers": 1, "files": 2, "failed": 0},
        "ontologies": {"providers": 1, "files": 2, "failed": 0},
        "pruned": {"skills": 0, "prompts": 0, "ontologies": 0},
        "path_free": True,
    }
    assert campaign._validate_provider_install_result(valid) is valid
    invalid = json.loads(json.dumps(valid))
    invalid["prompts"]["failed"] = 1
    with pytest.raises(campaign.CertificationError):
        campaign._validate_provider_install_result(invalid)


def test_evidence_signature_is_external_and_verifiable(monkeypatch: Any) -> None:
    private_bytes = b"k" * 32
    monkeypatch.setenv(
        "EXACT_LOCAL_TEST_SIGNING_KEY",
        base64.urlsafe_b64encode(private_bytes).decode().rstrip("="),
    )
    unsigned = {"schema_version": 1, "status": "pass"}

    signed = campaign._sign_evidence(
        unsigned,
        signer_id="signer:certification",
        signing_key_env="EXACT_LOCAL_TEST_SIGNING_KEY",
    )

    assert signed["signature"]["algorithm"] == "ed25519"
    assert "EXACT_LOCAL_TEST_SIGNING_KEY" not in repr(signed)
    assert os.environ["EXACT_LOCAL_TEST_SIGNING_KEY"] not in repr(signed)
