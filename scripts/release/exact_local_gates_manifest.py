#!/usr/bin/env python3
"""Generate the closed manifest consumed by the exact-local gate campaign.

The generator accepts one already signed and independently verified promotion
record.  It derives installed-artifact identities only from that record and
derives the harness and test-catalog identities from one explicit source root.
No artifact is discovered, installed, started, or substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from pathlib import Path
from typing import Any, Final

from scripts.release.promote_local_release import (
    ReleaseError,
    verify_evidence_file,
)

SCHEMA_VERSION: Final = 1
EVIDENCE_SCHEMA_VERSION: Final = 1
RELEASE_ID = re.compile(r"^release-[a-z0-9][a-z0-9.-]{2,63}$")
HEX_64 = re.compile(r"^[a-f0-9]{64}$")
SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MAX_INPUT_BYTES: Final = 1024 * 1024

# This is the exact source catalog copied by exact_local_gates.py.  The source
# contract checker proves the two inventories stay identical.
EXACT_LOCAL_TEST_FILES: Final = (
    "unit/test_intent_surface.py",
    "unit/test_intent_surface_build_server.py",
    "unit/test_intent_selection_accuracy.py",
    "test_intent_surface_gating.py",
    "unit/core/test_provider_materialization.py",
    "test_permissions_kernel.py",
    "unit/mcp/test_tools_misc.py",
    "ontology/test_permissioning.py",
    "unit/knowledge_graph/test_ontology_actions.py",
    "retrieval/test_context_compiler_mandatory.py",
    "unit/graph/test_permission_context.py",
    "integration/protocols/test_a2a_epistemic_live.py",
    "unit/protocols/test_a2a_epistemic.py",
    "_test_engine.py",
)

MANIFEST_KEYS: Final = frozenset(
    {
        "agent_utilities_sha256",
        "distribution_closure_sha256",
        "engine_sha256",
        "evidence_schema_version",
        "graphos_sha256",
        "harness_sha256",
        "promotion_evidence_sha256",
        "release_id",
        "release_python_sha256",
        "release_spec_sha256",
        "schema_version",
        "test_catalog_sha256",
    }
)


class ManifestError(RuntimeError):
    """A stable, non-sensitive generation failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise ManifestError(code)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _read_regular(path: Path, *, limit: int = MAX_INPUT_BYTES) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= limit:
            _fail("input_file_invalid")
        raw = bytearray()
        while chunk := os.read(descriptor, min(64 * 1024, limit - len(raw) + 1)):
            raw.extend(chunk)
            if len(raw) > limit:
                _fail("input_file_invalid")
        return bytes(raw)
    except ManifestError:
        raise
    except OSError:
        _fail("input_file_invalid")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _sha256_file(path: Path, *, limit: int = MAX_INPUT_BYTES) -> str:
    return hashlib.sha256(_read_regular(path, limit=limit)).hexdigest()


def _source_root(path: Path) -> Path:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        _fail("source_root_invalid")
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        _fail("source_root_invalid")
    required = (
        resolved / "scripts" / "certification" / "exact_local_gates.py",
        resolved / "tests",
    )
    if not required[0].is_file() or not required[1].is_dir():
        _fail("source_root_invalid")
    return resolved


def source_bindings(source_root: Path) -> dict[str, str]:
    """Return the current harness and catalog identities for one source tree."""

    root = _source_root(source_root)
    harness = root / "scripts" / "certification" / "exact_local_gates.py"
    test_root = root / "tests"
    digest = hashlib.sha256(b"agent-utilities-exact-test-catalog-v1\0")
    for relative in sorted(EXACT_LOCAL_TEST_FILES):
        path = test_root / relative
        file_digest = _sha256_file(path, limit=16 * 1024 * 1024)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return {
        "harness_sha256": _sha256_file(harness, limit=4 * 1024 * 1024),
        "test_catalog_sha256": digest.hexdigest(),
    }


def manifest_from_verified_promotion(
    evidence: dict[str, Any],
    *,
    release_id: str,
    promotion_evidence_sha256: str,
    bindings: dict[str, str],
) -> dict[str, Any]:
    """Build one closed current manifest from verified promotion evidence."""

    if not RELEASE_ID.fullmatch(release_id):
        _fail("release_id_invalid")
    if (
        evidence.get("apiVersion") != "graphos.io/v2"
        or evidence.get("kind") != "ExactLocalReleaseEvidence"
        or evidence.get("releaseId") != release_id
        or evidence.get("status") != "promoted"
        or evidence.get("errorCode") is not None
        or evidence.get("privacySafe") is not True
    ):
        _fail("promotion_evidence_not_promoted")
    certification = evidence.get("certificationArtifacts")
    native = evidence.get("nativeArtifacts")
    if (
        not isinstance(certification, dict)
        or set(certification)
        != {
            "agentUtilitiesSha256",
            "agentUtilitiesFileCount",
            "distributionClosureSha256",
            "releasePythonSha256",
            "graphosSha256",
            "engineSha256",
        }
        or not isinstance(native, dict)
        or set(native) != {"epistemic-graph-server", "epistemic-graph-numeric"}
    ):
        _fail("promotion_certification_invalid")
    digest_fields = (
        "agentUtilitiesSha256",
        "distributionClosureSha256",
        "releasePythonSha256",
        "graphosSha256",
        "engineSha256",
    )
    if any(
        not isinstance(certification.get(field), str)
        or HEX_64.fullmatch(certification[field]) is None
        for field in digest_fields
    ):
        _fail("promotion_certification_invalid")
    native_engine = str(native.get("epistemic-graph-server") or "")
    if native_engine != f"sha256:{certification['engineSha256']}":
        _fail("promotion_engine_binding_invalid")
    spec_digest = str(evidence.get("specDigest") or "")
    if (
        not spec_digest.startswith("sha256:")
        or HEX_64.fullmatch(spec_digest[7:]) is None
    ):
        _fail("promotion_spec_binding_invalid")
    if HEX_64.fullmatch(promotion_evidence_sha256) is None:
        _fail("promotion_evidence_digest_invalid")
    if set(bindings) != {"harness_sha256", "test_catalog_sha256"} or any(
        not isinstance(bindings.get(field), str)
        or HEX_64.fullmatch(bindings[field]) is None
        for field in bindings
    ):
        _fail("source_binding_invalid")
    manifest = {
        "agent_utilities_sha256": certification["agentUtilitiesSha256"],
        "distribution_closure_sha256": certification["distributionClosureSha256"],
        "engine_sha256": certification["engineSha256"],
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "graphos_sha256": certification["graphosSha256"],
        "harness_sha256": bindings["harness_sha256"],
        "promotion_evidence_sha256": promotion_evidence_sha256,
        "release_id": release_id,
        "release_python_sha256": certification["releasePythonSha256"],
        "release_spec_sha256": spec_digest[7:],
        "schema_version": SCHEMA_VERSION,
        "test_catalog_sha256": bindings["test_catalog_sha256"],
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Validate the strict current manifest shape used by both CLIs."""

    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        _fail("manifest_schema_invalid")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION
        or RELEASE_ID.fullmatch(str(manifest.get("release_id") or "")) is None
    ):
        _fail("manifest_schema_invalid")
    for key in MANIFEST_KEYS - {
        "schema_version",
        "evidence_schema_version",
        "release_id",
    }:
        if (
            not isinstance(manifest.get(key), str)
            or HEX_64.fullmatch(manifest[key]) is None
        ):
            _fail("manifest_schema_invalid")
    return manifest


def generate_manifest(
    *,
    release_id: str,
    spec_path: Path,
    promotion_evidence_path: Path,
    source_root: Path,
) -> dict[str, Any]:
    """Verify promotion evidence and derive its exact campaign manifest."""

    try:
        evidence = verify_evidence_file(
            spec_path=spec_path,
            release_id=release_id,
            evidence_path=promotion_evidence_path,
        )
    except ReleaseError as exc:
        raise ManifestError("promotion_evidence_verification_failed") from exc
    raw_digest = _sha256_file(promotion_evidence_path)
    return manifest_from_verified_promotion(
        evidence,
        release_id=release_id,
        promotion_evidence_sha256=raw_digest,
        bindings=source_bindings(source_root),
    )


def _write_new_private(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute() or SAFE_OUTPUT_NAME.fullmatch(path.name) is None:
        _fail("output_path_invalid")
    try:
        parent = path.parent.resolve(strict=True)
        metadata = parent.stat()
    except OSError:
        _fail("output_parent_invalid")
    if (
        parent != path.parent.absolute()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        _fail("output_parent_invalid")
    body = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True).encode("utf-8")
        + b"\n"
    )
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    temporary = f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        try:
            os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _fail("output_exists")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=directory,
        )
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                _fail("output_write_failed")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary,
            path.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        os.fsync(directory)
    except FileExistsError:
        _fail("output_exists")
    except ManifestError:
        raise
    except OSError:
        _fail("output_write_failed")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
        os.close(directory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate-exact-local-gates-manifest",
        description="Generate a digest-bound manifest for exact local gate certification.",
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--promotion-evidence", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = generate_manifest(
            release_id=args.release_id,
            spec_path=args.spec,
            promotion_evidence_path=args.promotion_evidence,
            source_root=args.source_root,
        )
        _write_new_private(args.output, manifest)
    except ManifestError as exc:
        print(f"manifest_status=rejected error_code={exc.code}", file=sys.stderr)
        return 1
    except Exception:
        print("manifest_status=rejected error_code=internal-error", file=sys.stderr)
        return 1
    print("manifest_status=generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
