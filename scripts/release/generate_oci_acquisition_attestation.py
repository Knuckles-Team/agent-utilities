#!/usr/bin/env python3
"""Sign current Trivy acquisition attestations from verified immutable inputs.

This producer has no transport or extraction capability.  Deployment first acquires
and verifies the pinned release assets through its configured trust boundary, then
passes their explicit SHA-256 identities here.  The producer reopens every input,
holds it through external signing and verification, and publishes only path-free,
privacy-safe attestations consumed by the offline OCI vulnerability campaign.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release import (  # noqa: E402
    generate_oci_vulnerability_scan_evidence as scan,
)

_ATTESTATION_SIGNER_ENV = "OCI_SCAN_ATTESTATION_SIGNER_COMMAND"
_ATTESTATION_VERIFIER_ENV = "OCI_SCAN_ATTESTATION_VERIFIER_COMMAND"
_MAX_RELEASE_ASSET_BYTES = 1024 * 1024 * 1024
_MAX_RELEASE_BUNDLE_BYTES = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_VERIFIER_BYTES = 512 * 1024 * 1024

FileIdentity = tuple[int, int]
DirectoryIdentity = tuple[int, int, int, int]


@dataclass(frozen=True)
class DatabaseProfile:
    name: str
    schema_version: int
    database_file: str
    layer_media_type: str
    layer_title: str


_DATABASE_PROFILES = {
    "vulnerability": DatabaseProfile(
        name="trivy-db",
        schema_version=2,
        database_file="trivy.db",
        layer_media_type="application/vnd.aquasec.trivy.db.layer.v1.tar+gzip",
        layer_title="db.tar.gz",
    ),
    "java": DatabaseProfile(
        name="trivy-java-db",
        schema_version=1,
        database_file="trivy-java.db",
        layer_media_type="application/vnd.aquasec.trivy.javadb.layer.v1.tar+gzip",
        layer_title="javadb.tar.gz",
    ),
}


class AcquisitionAttestationError(ValueError):
    """One bounded, privacy-safe acquisition-attestation failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> NoReturn:
    raise AcquisitionAttestationError(code)


def _expected_digest(value: str) -> str:
    try:
        return scan._digest(value, "expected digest")
    except scan.OciVulnerabilityScanError:
        _fail("expected_digest_invalid")


def _tls_binding(reference: str) -> str:
    if scan._TLS_REFERENCE.fullmatch(reference) is None or ".." in reference:
        _fail("tls_profile_reference_invalid")
    return "pref_tls_" + hashlib.sha256(reference.encode("utf-8")).hexdigest()


@contextmanager
def _private_regular(
    path: Path,
    *,
    maximum: int,
    executable: bool = False,
) -> Iterator[tuple[int, os.stat_result]]:
    yielded = False
    try:
        with scan._regular_descriptor(
            path,
            maximum=maximum,
            executable=executable,
        ) as opened:
            _descriptor, metadata = opened
            if (
                metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                _fail("input_file_not_private")
            yielded = True
            yield opened
    except AcquisitionAttestationError:
        raise
    except scan.OciVulnerabilityScanError:
        _fail("input_file_changed" if yielded else "input_file_invalid")


def _hash_descriptor(
    descriptor: int,
    metadata: os.stat_result,
    *,
    maximum: int,
) -> str:
    try:
        return scan._hash_descriptor(descriptor, metadata.st_size, maximum)
    except scan.OciVulnerabilityScanError:
        _fail("input_file_changed")


def _read_descriptor(
    descriptor: int,
    metadata: os.stat_result,
    *,
    maximum: int,
) -> bytes:
    try:
        return scan._read_descriptor(descriptor, metadata.st_size, maximum)
    except scan.OciVulnerabilityScanError:
        _fail("input_file_changed")


def _require_digest(actual: str, expected: str) -> None:
    if actual != _expected_digest(expected):
        _fail("input_digest_mismatch")


def _require_distinct(*metadata: os.stat_result) -> None:
    identities: set[FileIdentity] = {(item.st_dev, item.st_ino) for item in metadata}
    if len(identities) != len(metadata):
        _fail("input_alias_invalid")


def _json_object(payload: bytes, *, code: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                _fail(code)
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicates)
    except AcquisitionAttestationError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError):
        _fail(code)
    if not isinstance(value, dict):
        _fail(code)
    return value


def _exact(value: Any, keys: set[str], *, code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(code)
    return value


def _timestamp(value: Any, *, code: str) -> datetime:
    try:
        return scan._timestamp(value, "timestamp")
    except scan.OciVulnerabilityScanError:
        _fail(code)


def _signed(
    unsigned: dict[str, Any],
    *,
    signer_env: str,
    verifier_env: str,
    tls_binding: str,
    profile: DatabaseProfile | None,
) -> dict[str, Any]:
    if signer_env == verifier_env:
        _fail("attestation_adapters_not_distinct")
    payload = scan._canonical_bytes(unsigned)
    try:
        return_code, stdout, _stderr = scan._invoke_adapter(signer_env, payload)
    except scan.OciVulnerabilityScanError:
        _fail("attestation_signer_failed")
    if return_code != 0:
        _fail("attestation_signer_failed")
    try:
        signature = scan._signature(
            _json_object(stdout, code="attestation_signature_invalid"),
            "attestation signature",
        )
    except AcquisitionAttestationError:
        raise
    except scan.OciVulnerabilityScanError:
        _fail("attestation_signature_invalid")
    if signature["subjectDigest"] != scan._sha256_bytes(payload):
        _fail("attestation_signature_binding_invalid")
    document = {**unsigned, "signature": signature}
    try:
        if profile is None:
            scan._validate_scanner_attestation(
                document,
                tls_digest=tls_binding,
                verifier_env=verifier_env,
                invoke=True,
            )
        else:
            scan._validate_database_attestation(
                document,
                expected_name=profile.name,
                expected_schema=profile.schema_version,
                tls_digest=tls_binding,
                verifier_env=verifier_env,
                invoke=True,
            )
    except scan.OciVulnerabilityScanError:
        _fail("attestation_verification_failed")
    return document


def _release_bundle_digest(asset_digest: str, signature_bundle_digest: str) -> str:
    """Bind the release asset and its official signature bundle in one field."""

    return scan._sha256_bytes(
        scan._canonical_bytes(
            {
                "releaseAssetDigest": asset_digest,
                "releaseSignatureBundleDigest": signature_bundle_digest,
            }
        )
    )


def generate_scanner_attestation(
    *,
    scanner_binary: Path,
    scanner_binary_sha256: str,
    release_asset: Path,
    release_asset_sha256: str,
    release_signature_bundle: Path,
    release_signature_bundle_sha256: str,
    signature_verifier: Path,
    signature_verifier_sha256: str,
    tls_profile_ref: str,
    signer_env: str,
    verifier_env: str,
) -> dict[str, Any]:
    """Return a signed scanner attestation without retaining acquisition material."""

    tls_binding = _tls_binding(tls_profile_ref)
    with ExitStack() as stack:
        binary_descriptor, binary_metadata = stack.enter_context(
            _private_regular(
                scanner_binary,
                maximum=scan._MAX_SCANNER_BYTES,
                executable=True,
            )
        )
        asset_descriptor, asset_metadata = stack.enter_context(
            _private_regular(release_asset, maximum=_MAX_RELEASE_ASSET_BYTES)
        )
        bundle_descriptor, bundle_metadata = stack.enter_context(
            _private_regular(
                release_signature_bundle,
                maximum=_MAX_RELEASE_BUNDLE_BYTES,
            )
        )
        verifier_descriptor, verifier_metadata = stack.enter_context(
            _private_regular(
                signature_verifier,
                maximum=_MAX_VERIFIER_BYTES,
                executable=True,
            )
        )
        _require_distinct(
            binary_metadata,
            asset_metadata,
            bundle_metadata,
            verifier_metadata,
        )
        binary_digest = _hash_descriptor(
            binary_descriptor,
            binary_metadata,
            maximum=scan._MAX_SCANNER_BYTES,
        )
        asset_digest = _hash_descriptor(
            asset_descriptor,
            asset_metadata,
            maximum=_MAX_RELEASE_ASSET_BYTES,
        )
        signature_bundle_digest = _hash_descriptor(
            bundle_descriptor,
            bundle_metadata,
            maximum=_MAX_RELEASE_BUNDLE_BYTES,
        )
        verifier_digest = _hash_descriptor(
            verifier_descriptor,
            verifier_metadata,
            maximum=_MAX_VERIFIER_BYTES,
        )
        for actual, expected in (
            (binary_digest, scanner_binary_sha256),
            (asset_digest, release_asset_sha256),
            (signature_bundle_digest, release_signature_bundle_sha256),
            (verifier_digest, signature_verifier_sha256),
        ):
            _require_digest(actual, expected)
        unsigned = {
            "apiVersion": "graphos.io/v1",
            "kind": "OciScannerAttestation",
            "attestationVersion": 1,
            "scanner": {
                "name": scan._SCANNER_NAME,
                "version": scan._SCANNER_VERSION,
                "platform": scan._PLATFORM,
                "binaryDigest": binary_digest,
                "releaseBundleDigest": _release_bundle_digest(
                    asset_digest,
                    signature_bundle_digest,
                ),
                "verifierDigest": verifier_digest,
                "tlsProfileRef": tls_binding,
            },
        }
        return _signed(
            unsigned,
            signer_env=signer_env,
            verifier_env=verifier_env,
            tls_binding=tls_binding,
            profile=None,
        )


def _validate_manifest(
    value: dict[str, Any],
    *,
    profile: DatabaseProfile,
    archive_digest: str,
    archive_size: int,
) -> None:
    manifest = _exact(
        value,
        {
            "schemaVersion",
            "mediaType",
            "artifactType",
            "config",
            "layers",
            "annotations",
        },
        code="oci_manifest_invalid",
    )
    schema_version = manifest.get("schemaVersion")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 2
        or manifest.get("mediaType") != "application/vnd.oci.image.manifest.v1+json"
        or manifest.get("artifactType")
        != "application/vnd.aquasec.trivy.config.v1+json"
    ):
        _fail("oci_manifest_invalid")
    config = _exact(
        manifest.get("config"),
        {"mediaType", "digest", "size", "data"},
        code="oci_manifest_invalid",
    )
    if config.get("mediaType") != "application/vnd.oci.empty.v1+json":
        _fail("oci_manifest_invalid")
    config_data = config.get("data")
    if config_data != "e30=":
        _fail("oci_manifest_invalid")
    try:
        config_payload = base64.b64decode(config_data, validate=True)
    except (ValueError, binascii.Error):
        _fail("oci_manifest_invalid")
    config_size = config.get("size")
    if (
        not isinstance(config_size, int)
        or isinstance(config_size, bool)
        or config_payload != b"{}"
        or config_size != len(config_payload)
        or config.get("digest") != scan._sha256_bytes(config_payload)
    ):
        _fail("oci_manifest_invalid")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != 1:
        _fail("oci_manifest_invalid")
    layer = _exact(
        layers[0],
        {"mediaType", "digest", "size", "annotations"},
        code="oci_manifest_invalid",
    )
    annotations = _exact(
        layer.get("annotations"),
        {"org.opencontainers.image.title"},
        code="oci_manifest_invalid",
    )
    layer_size = layer.get("size")
    if (
        layer.get("mediaType") != profile.layer_media_type
        or layer.get("digest") != archive_digest
        or not isinstance(layer_size, int)
        or isinstance(layer_size, bool)
        or layer_size != archive_size
        or annotations.get("org.opencontainers.image.title") != profile.layer_title
    ):
        _fail("oci_manifest_invalid")
    top_annotations = _exact(
        manifest.get("annotations"),
        {"org.opencontainers.image.created"},
        code="oci_manifest_invalid",
    )
    _timestamp(
        top_annotations.get("org.opencontainers.image.created"),
        code="oci_manifest_invalid",
    )


def _private_database_directory(
    path: Path,
    profile: DatabaseProfile,
) -> DirectoryIdentity:
    if not path.is_absolute() or path.is_symlink():
        _fail("database_directory_invalid")
    try:
        metadata = path.stat(follow_symlinks=False)
        entries = {entry.name for entry in path.iterdir()}
    except OSError:
        _fail("database_directory_invalid")
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or entries != {"metadata.json", profile.database_file}
    ):
        _fail("database_directory_invalid")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_database_directory_unchanged(
    path: Path,
    profile: DatabaseProfile,
    expected: DirectoryIdentity,
) -> None:
    if _private_database_directory(path, profile) != expected:
        _fail("database_directory_changed")


def _metadata(
    payload: bytes,
    *,
    profile: DatabaseProfile,
    now: datetime,
) -> tuple[str, str]:
    value = _json_object(payload, code="database_metadata_invalid")
    if set(value) != {"Version", "UpdatedAt", "NextUpdate", "DownloadedAt"}:
        _fail("database_metadata_invalid")
    schema_version = value.get("Version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != profile.schema_version
    ):
        _fail("database_schema_invalid")
    updated_text = value.get("UpdatedAt")
    next_text = value.get("NextUpdate")
    downloaded_text = value.get("DownloadedAt")
    if not all(
        isinstance(item, str) for item in (updated_text, next_text, downloaded_text)
    ):
        _fail("database_metadata_invalid")
    updated = _timestamp(updated_text, code="database_metadata_invalid")
    next_update = _timestamp(next_text, code="database_metadata_invalid")
    downloaded = _timestamp(
        downloaded_text,
        code="database_metadata_invalid",
    )
    normalized_now = now.astimezone(UTC)
    if not updated <= downloaded <= normalized_now or not (
        updated < normalized_now <= next_update
    ):
        _fail("database_not_fresh")
    return str(updated_text), str(next_text)


def generate_database_attestation(
    *,
    database_kind: str,
    oci_manifest: Path,
    oci_manifest_sha256: str,
    layer_archive: Path,
    layer_archive_sha256: str,
    materialized_directory: Path,
    metadata_sha256: str,
    database_sha256: str,
    tls_profile_ref: str,
    signer_env: str,
    verifier_env: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return a signed database attestation for one exact materialized OCI layer."""

    profile = _DATABASE_PROFILES.get(database_kind)
    if profile is None:
        _fail("database_kind_invalid")
    directory_identity = _private_database_directory(materialized_directory, profile)
    tls_binding = _tls_binding(tls_profile_ref)
    with ExitStack() as stack:
        manifest_descriptor, manifest_metadata = stack.enter_context(
            _private_regular(oci_manifest, maximum=_MAX_MANIFEST_BYTES)
        )
        archive_descriptor, archive_metadata = stack.enter_context(
            _private_regular(layer_archive, maximum=scan._MAX_DATABASE_BYTES)
        )
        metadata_descriptor, metadata_metadata = stack.enter_context(
            _private_regular(
                materialized_directory / "metadata.json",
                maximum=_MAX_MANIFEST_BYTES,
            )
        )
        database_descriptor, database_metadata = stack.enter_context(
            _private_regular(
                materialized_directory / profile.database_file,
                maximum=scan._MAX_DATABASE_BYTES,
            )
        )
        _require_distinct(
            manifest_metadata,
            archive_metadata,
            metadata_metadata,
            database_metadata,
        )
        manifest_payload = _read_descriptor(
            manifest_descriptor,
            manifest_metadata,
            maximum=_MAX_MANIFEST_BYTES,
        )
        manifest_digest = scan._sha256_bytes(manifest_payload)
        archive_digest = _hash_descriptor(
            archive_descriptor,
            archive_metadata,
            maximum=scan._MAX_DATABASE_BYTES,
        )
        metadata_payload = _read_descriptor(
            metadata_descriptor,
            metadata_metadata,
            maximum=_MAX_MANIFEST_BYTES,
        )
        metadata_digest = scan._sha256_bytes(metadata_payload)
        database_digest = _hash_descriptor(
            database_descriptor,
            database_metadata,
            maximum=scan._MAX_DATABASE_BYTES,
        )
        for actual, expected in (
            (manifest_digest, oci_manifest_sha256),
            (archive_digest, layer_archive_sha256),
            (metadata_digest, metadata_sha256),
            (database_digest, database_sha256),
        ):
            _require_digest(actual, expected)
        _validate_manifest(
            _json_object(manifest_payload, code="oci_manifest_invalid"),
            profile=profile,
            archive_digest=archive_digest,
            archive_size=archive_metadata.st_size,
        )
        current = now or scan._utc_now()
        if current.tzinfo is None or current.utcoffset() is None:
            _fail("current_time_invalid")
        updated_at, next_update = _metadata(
            metadata_payload,
            profile=profile,
            now=current,
        )
        unsigned = {
            "apiVersion": "graphos.io/v1",
            "kind": "OciVulnerabilityDatabaseAttestation",
            "attestationVersion": 1,
            "purpose": "deployment-acquisition",
            "database": {
                "name": profile.name,
                "schemaVersion": profile.schema_version,
                "manifestDigest": manifest_digest,
                "archiveDigest": archive_digest,
                "metadataDigest": metadata_digest,
                "databaseDigest": database_digest,
                "updatedAt": updated_at,
                "nextUpdate": next_update,
                "tlsProfileRef": tls_binding,
            },
        }
        document = _signed(
            unsigned,
            signer_env=signer_env,
            verifier_env=verifier_env,
            tls_binding=tls_binding,
            profile=profile,
        )
        _require_database_directory_unchanged(
            materialized_directory,
            profile,
            directory_identity,
        )
        return document


def _private_output(path: Path, *, forbidden_directory: Path | None = None) -> None:
    if not path.is_absolute() or path.is_symlink():
        _fail("output_path_invalid")
    try:
        parent = path.parent.resolve(strict=True)
        metadata = parent.stat(follow_symlinks=False)
    except OSError:
        _fail("output_parent_invalid")
    if (
        parent != path.parent
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail("output_parent_invalid")
    if forbidden_directory is not None:
        try:
            parent.relative_to(forbidden_directory.resolve(strict=True))
        except ValueError:
            pass
        else:
            _fail("output_overlaps_database")


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tls-profile-ref", required=True)
    parser.add_argument("--signer-env", default=_ATTESTATION_SIGNER_ENV)
    parser.add_argument("--verifier-env", default=_ATTESTATION_VERIFIER_ENV)
    parser.add_argument("--output", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate-oci-acquisition-attestation",
        description="Sign path-free attestations for already verified Trivy inputs.",
    )
    subparsers = parser.add_subparsers(dest="subject", required=True)
    scanner_parser = subparsers.add_parser("scanner")
    scanner_parser.add_argument("--scanner-binary", required=True, type=Path)
    scanner_parser.add_argument("--scanner-binary-sha256", required=True)
    scanner_parser.add_argument("--release-asset", required=True, type=Path)
    scanner_parser.add_argument("--release-asset-sha256", required=True)
    scanner_parser.add_argument("--release-signature-bundle", required=True, type=Path)
    scanner_parser.add_argument("--release-signature-bundle-sha256", required=True)
    scanner_parser.add_argument("--signature-verifier", required=True, type=Path)
    scanner_parser.add_argument("--signature-verifier-sha256", required=True)
    _common_arguments(scanner_parser)
    database_parser = subparsers.add_parser("database")
    database_parser.add_argument(
        "--database-kind",
        required=True,
        choices=tuple(_DATABASE_PROFILES),
    )
    database_parser.add_argument("--oci-manifest", required=True, type=Path)
    database_parser.add_argument("--oci-manifest-sha256", required=True)
    database_parser.add_argument("--layer-archive", required=True, type=Path)
    database_parser.add_argument("--layer-archive-sha256", required=True)
    database_parser.add_argument("--materialized-directory", required=True, type=Path)
    database_parser.add_argument("--metadata-sha256", required=True)
    database_parser.add_argument("--database-sha256", required=True)
    _common_arguments(database_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        forbidden: Path | None = None
        if args.subject == "scanner":
            document = generate_scanner_attestation(
                scanner_binary=args.scanner_binary,
                scanner_binary_sha256=args.scanner_binary_sha256,
                release_asset=args.release_asset,
                release_asset_sha256=args.release_asset_sha256,
                release_signature_bundle=args.release_signature_bundle,
                release_signature_bundle_sha256=(args.release_signature_bundle_sha256),
                signature_verifier=args.signature_verifier,
                signature_verifier_sha256=args.signature_verifier_sha256,
                tls_profile_ref=args.tls_profile_ref,
                signer_env=args.signer_env,
                verifier_env=args.verifier_env,
            )
        else:
            forbidden = args.materialized_directory
            document = generate_database_attestation(
                database_kind=args.database_kind,
                oci_manifest=args.oci_manifest,
                oci_manifest_sha256=args.oci_manifest_sha256,
                layer_archive=args.layer_archive,
                layer_archive_sha256=args.layer_archive_sha256,
                materialized_directory=args.materialized_directory,
                metadata_sha256=args.metadata_sha256,
                database_sha256=args.database_sha256,
                tls_profile_ref=args.tls_profile_ref,
                signer_env=args.signer_env,
                verifier_env=args.verifier_env,
            )
        _private_output(args.output, forbidden_directory=forbidden)
        try:
            scan.write_no_replace(args.output, document)
        except (OSError, scan.OciVulnerabilityScanError):
            _fail("output_publication_failed")
    except AcquisitionAttestationError as exc:
        print(
            json.dumps({"errorCode": exc.code, "ok": False}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    except Exception:
        print(
            json.dumps({"errorCode": "internal_error", "ok": False}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({"kind": document["kind"], "ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
