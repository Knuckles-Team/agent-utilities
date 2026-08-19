"""Source-only custody contracts for the connector-manifest signing Job.

These tests inspect only repository source and a synthetic input description. They
never contact Kubernetes/OpenBao, resolve a credential, build a wheel, or generate a
connector artifact.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
JOB_SOURCE = ROOT / "deploy/release/connector-manifest-signing-job.yaml"
WORKFLOW_SOURCE = ROOT / ".github/workflows/advisory.yml"
INPUT_FIXTURE = (
    ROOT / "tests/fixtures/release/connector-manifest-signing-inputs.yml"
)


def _job_document() -> dict:
    documents = list(yaml.safe_load_all(JOB_SOURCE.read_text(encoding="utf-8")))
    return next(document for document in documents if document.get("kind") == "Job")


def _signer_container(job: dict) -> dict:
    return job["spec"]["template"]["spec"]["containers"][0]


def test_fixture_contains_only_fake_digest_attested_inputs() -> None:
    fixture = yaml.safe_load(INPUT_FIXTURE.read_text(encoding="utf-8"))

    assert fixture["kind"] == "ConnectorManifestSigningInputFixture"
    assert fixture["signerImage"].startswith("registry.invalid/")
    assert fixture["fleetInput"]["image"].startswith("registry.invalid/")
    for digest in (
        fixture["lockDigest"],
        fixture["fleetInput"]["attestationSha256"],
        fixture["agentUtilitiesWheel"]["sha256"],
    ):
        assert re.fullmatch(r"[a-f]{64}", digest)
    assert fixture["output"]["inputClaim"]
    assert fixture["output"]["publicClaim"]
    assert not any(
        forbidden in INPUT_FIXTURE.read_text(encoding="utf-8").lower()
        for forbidden in ("private_key", "secretkeyref", "vault_token")
    )


def test_job_uses_attested_read_only_inputs_and_bounded_public_output() -> None:
    job = _job_document()
    pod = job["spec"]["template"]["spec"]
    container = _signer_container(job)
    volumes = {volume["name"]: volume for volume in pod["volumes"]}
    mounts = {mount["name"]: mount for mount in container["volumeMounts"]}

    assert "@sha256:" in container["image"]
    assert ":latest" not in container["image"]
    assert volumes["release-input"]["persistentVolumeClaim"]["claimName"] == (
        "connector-manifest-signing-input"
    )
    assert mounts["release-input"]["readOnly"] is True
    assert volumes["public-output"]["persistentVolumeClaim"]["claimName"] == (
        "connector-manifest-signing-public-output"
    )
    assert mounts["public-output"]["readOnly"] is False
    assert volumes["work"]["emptyDir"]["sizeLimit"] == "4Gi"
    assert "fleet" not in volumes
    assert pod["automountServiceAccountToken"] is False
    assert pod["serviceAccountName"] == "connector-manifest-signer"
    assert not any("secret" in volume for volume in pod["volumes"])
    assert not any("secretKeyRef" in item for item in container["env"])
    assert not any(item["name"] == "VAULT_TOKEN" for item in container["env"])


def test_job_installs_wheel_and_generates_bundles_before_signing() -> None:
    container = _signer_container(_job_document())
    script = "\n".join(container["args"])

    for required in (
        "--no-index",
        "--no-deps",
        "--only-binary=:all:",
        "--target \"${STAGING_ROOT}/site-packages\"",
        "direct_url.json",
        "scripts/generate_connector_capability_bundles.py",
        "--bundled-output \"${PUBLIC_OUTPUT_ROOT}/connector-bundles\"",
        "--apply",
        "scripts/release/regenerate_and_sign_connector_manifests.py",
        "--require-built-artifact",
        "--sign",
    ):
        assert required in script
    assert script.index("generate_connector_capability_bundles.py") < script.index(
        "regenerate_and_sign_connector_manifests.py"
    )
    assert "${PUBLIC_OUTPUT_ROOT}/connector-bundles" in script
    assert "${PUBLIC_OUTPUT_ROOT}/manifests" in script
    assert "${PUBLIC_OUTPUT_ROOT}/native" in script


def test_keyless_workflow_builds_non_editable_report_input() -> None:
    source = WORKFLOW_SOURCE.read_text(encoding="utf-8")
    for required in (
        "connector-manifest-diff",
        "uv build --wheel --no-build-isolation",
        "python3 -m pip install",
        "--no-index --no-deps --only-binary=:all:",
        "direct_url.json",
        "--require-built-artifact",
        "actions/upload-artifact",
        "/tmp/connector-manifest-diff-report.json",
    ):
        assert required in source
