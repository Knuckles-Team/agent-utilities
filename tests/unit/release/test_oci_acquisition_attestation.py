"""Focused tests for current-only OCI acquisition attestations."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from scripts.release import generate_oci_acquisition_attestation as acquisition

ROOT = Path(__file__).resolve().parents[3]
TLS_REFERENCE = "env://RELEASE_TLS_PROFILE"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _private_file(
    path: Path,
    payload: bytes,
    *,
    executable: bool = False,
) -> Path:
    path.write_bytes(payload)
    path.chmod(0o700 if executable else 0o600)
    return path


def _configure_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    adapter = _private_file(
        tmp_path / "attestation-adapter",
        f"""#!{sys.executable}
import hashlib
import json
import sys

payload = sys.stdin.buffer.read()
try:
    value = json.loads(payload)
except Exception:
    raise SystemExit(2)
canonical = json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode()
if payload != canonical:
    raise SystemExit(3)
digest = "sha256:" + hashlib.sha256(payload).hexdigest()
key_id = "key:" + "c" * 64
if sys.argv[1] == "sign":
    response = {{
        "algorithm": "ed25519",
        "keyId": key_id,
        "signature": "A" * 43,
        "subjectDigest": digest,
    }}
elif sys.argv[1] == "verify":
    signature = value.pop("signature")
    unsigned = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    response = {{
        "verified": signature["subjectDigest"]
        == "sha256:" + hashlib.sha256(unsigned).hexdigest(),
        "subjectDigest": signature["subjectDigest"],
        "keyId": signature["keyId"],
    }}
else:
    raise SystemExit(4)
sys.stdout.write(json.dumps(response, sort_keys=True))
""".encode(),
        executable=True,
    )
    monkeypatch.setenv(
        acquisition._ATTESTATION_SIGNER_ENV,
        json.dumps([sys.executable, str(adapter), "sign"]),
    )
    monkeypatch.setenv(
        acquisition._ATTESTATION_VERIFIER_ENV,
        json.dumps([sys.executable, str(adapter), "verify"]),
    )
    return adapter


def _scanner_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    tmp_path.chmod(0o700)
    verifier = _configure_adapters(tmp_path, monkeypatch)
    scanner_payload = b"pinned-trivy-0.72.0-linux-amd64"
    asset_payload = b"verified-trivy-release-asset"
    bundle_payload = b"verified-official-release-signature-bundle"
    scanner = _private_file(
        tmp_path / "trivy",
        scanner_payload,
        executable=True,
    )
    asset = _private_file(tmp_path / "trivy-release", asset_payload)
    bundle = _private_file(tmp_path / "trivy-release.sigstore", bundle_payload)
    return {
        "scanner_binary": scanner,
        "scanner_binary_sha256": _digest(scanner_payload),
        "release_asset": asset,
        "release_asset_sha256": _digest(asset_payload),
        "release_signature_bundle": bundle,
        "release_signature_bundle_sha256": _digest(bundle_payload),
        "signature_verifier": verifier,
        "signature_verifier_sha256": _digest(verifier.read_bytes()),
        "tls_profile_ref": TLS_REFERENCE,
        "signer_env": acquisition._ATTESTATION_SIGNER_ENV,
        "verifier_env": acquisition._ATTESTATION_VERIFIER_ENV,
    }


def _database_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    database_kind: str,
    next_update: str = "2026-08-01T00:00:00Z",
) -> dict[str, Any]:
    tmp_path.chmod(0o700)
    _configure_adapters(tmp_path, monkeypatch)
    profile = acquisition._DATABASE_PROFILES[database_kind]
    directory = tmp_path / f"{database_kind}-database"
    directory.mkdir(mode=0o700)
    metadata = {
        "Version": profile.schema_version,
        "UpdatedAt": "2026-07-18T00:00:00Z",
        "NextUpdate": next_update,
        "DownloadedAt": "2026-07-18T01:00:00Z",
    }
    metadata_payload = _canonical(metadata)
    database_payload = f"materialized-{database_kind}-database".encode()
    _private_file(directory / "metadata.json", metadata_payload)
    _private_file(directory / profile.database_file, database_payload)

    archive_payload = f"verified-{database_kind}-layer".encode()
    archive = _private_file(tmp_path / f"{database_kind}.tar.gz", archive_payload)
    config_payload = b"{}"
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "artifactType": "application/vnd.aquasec.trivy.config.v1+json",
        "config": {
            "mediaType": "application/vnd.oci.empty.v1+json",
            "digest": _digest(config_payload),
            "size": len(config_payload),
            "data": "e30=",
        },
        "layers": [
            {
                "mediaType": profile.layer_media_type,
                "digest": _digest(archive_payload),
                "size": len(archive_payload),
                "annotations": {
                    "org.opencontainers.image.title": profile.layer_title,
                },
            }
        ],
        "annotations": {
            "org.opencontainers.image.created": "2026-07-18T00:00:00Z",
        },
    }
    manifest_payload = _canonical(manifest)
    manifest_path = _private_file(
        tmp_path / f"{database_kind}-manifest.json",
        manifest_payload,
    )
    return {
        "database_kind": database_kind,
        "oci_manifest": manifest_path,
        "oci_manifest_sha256": _digest(manifest_payload),
        "layer_archive": archive,
        "layer_archive_sha256": _digest(archive_payload),
        "materialized_directory": directory,
        "metadata_sha256": _digest(metadata_payload),
        "database_sha256": _digest(database_payload),
        "tls_profile_ref": TLS_REFERENCE,
        "signer_env": acquisition._ATTESTATION_SIGNER_ENV,
        "verifier_env": acquisition._ATTESTATION_VERIFIER_ENV,
        "now": datetime(2026, 7, 19, tzinfo=UTC),
    }


def _validate_schema(document: dict[str, Any], filename: str) -> None:
    schema = json.loads((ROOT / "deploy/release" / filename).read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)


def test_scanner_attestation_is_exact_signed_and_path_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _scanner_fixture(tmp_path, monkeypatch)

    document = acquisition.generate_scanner_attestation(**values)

    asset_digest = values["release_asset_sha256"]
    bundle_digest = values["release_signature_bundle_sha256"]
    assert document["scanner"]["releaseBundleDigest"] == _digest(
        _canonical(
            {
                "releaseAssetDigest": asset_digest,
                "releaseSignatureBundleDigest": bundle_digest,
            }
        )
    )
    serialized = _canonical(document).decode()
    assert str(tmp_path) not in serialized
    assert TLS_REFERENCE not in serialized
    _validate_schema(document, "oci-scanner-attestation.schema.json")


@pytest.mark.parametrize("database_kind", ("vulnerability", "java"))
def test_database_attestation_is_exact_fresh_and_path_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    database_kind: str,
) -> None:
    values = _database_fixture(
        tmp_path,
        monkeypatch,
        database_kind=database_kind,
    )

    document = acquisition.generate_database_attestation(**values)

    serialized = _canonical(document).decode()
    assert str(tmp_path) not in serialized
    assert TLS_REFERENCE not in serialized
    assert (
        document["database"]["name"]
        == acquisition._DATABASE_PROFILES[database_kind].name
    )
    _validate_schema(
        document,
        "oci-vulnerability-database-attestation.schema.json",
    )


def test_explicit_digest_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _scanner_fixture(tmp_path, monkeypatch)
    values["release_asset_sha256"] = "sha256:" + "9" * 64

    with pytest.raises(
        acquisition.AcquisitionAttestationError,
        match="input_digest_mismatch",
    ):
        acquisition.generate_scanner_attestation(**values)


def test_aliased_scanner_inputs_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _scanner_fixture(tmp_path, monkeypatch)
    values["release_asset"] = values["scanner_binary"]
    values["release_asset_sha256"] = values["scanner_binary_sha256"]

    with pytest.raises(
        acquisition.AcquisitionAttestationError,
        match="input_alias_invalid",
    ):
        acquisition.generate_scanner_attestation(**values)


@pytest.mark.parametrize(
    ("next_update", "error"),
    (
        ("2026-07-18T23:59:59Z", "database_not_fresh"),
        ("not-a-timestamp", "database_metadata_invalid"),
    ),
)
def test_stale_or_malformed_database_metadata_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    next_update: str,
    error: str,
) -> None:
    values = _database_fixture(
        tmp_path,
        monkeypatch,
        database_kind="vulnerability",
        next_update=next_update,
    )

    with pytest.raises(acquisition.AcquisitionAttestationError, match=error):
        acquisition.generate_database_attestation(**values)


def test_database_directory_must_have_exact_private_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _database_fixture(
        tmp_path,
        monkeypatch,
        database_kind="java",
    )
    _private_file(values["materialized_directory"] / "unexpected", b"extra")

    with pytest.raises(
        acquisition.AcquisitionAttestationError,
        match="database_directory_invalid",
    ):
        acquisition.generate_database_attestation(**values)


def test_database_manifest_contract_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _database_fixture(
        tmp_path,
        monkeypatch,
        database_kind="vulnerability",
    )
    manifest = json.loads(values["oci_manifest"].read_text())
    manifest["config"]["data"] = "e30K"
    payload = _canonical(manifest)
    _private_file(values["oci_manifest"], payload)
    values["oci_manifest_sha256"] = _digest(payload)

    with pytest.raises(
        acquisition.AcquisitionAttestationError,
        match="oci_manifest_invalid",
    ):
        acquisition.generate_database_attestation(**values)


def test_cli_publishes_private_no_replace_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    values = _scanner_fixture(tmp_path, monkeypatch)
    output = tmp_path / "scanner-attestation.json"
    argv = [
        "scanner",
        "--scanner-binary",
        str(values["scanner_binary"]),
        "--scanner-binary-sha256",
        values["scanner_binary_sha256"],
        "--release-asset",
        str(values["release_asset"]),
        "--release-asset-sha256",
        values["release_asset_sha256"],
        "--release-signature-bundle",
        str(values["release_signature_bundle"]),
        "--release-signature-bundle-sha256",
        values["release_signature_bundle_sha256"],
        "--signature-verifier",
        str(values["signature_verifier"]),
        "--signature-verifier-sha256",
        values["signature_verifier_sha256"],
        "--tls-profile-ref",
        TLS_REFERENCE,
        "--output",
        str(output),
    ]

    assert acquisition.main(argv) == 0
    assert output.stat().st_mode & 0o777 == 0o600
    assert _canonical(json.loads(output.read_text())) + b"\n" == output.read_bytes()
    assert acquisition.main(argv) == 1
    assert "output_publication_failed" in capsys.readouterr().err
