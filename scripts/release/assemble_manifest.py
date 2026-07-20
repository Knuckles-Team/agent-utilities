#!/usr/bin/env python3
"""Assemble and externally sign an exact, evidence-backed release manifest."""

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
from typing import Any

import yaml
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release import check_compatibility as compatibility  # noqa: E402

_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")


class AssemblyError(ValueError):
    """Release inputs are incomplete, unbound, or unsafe."""


def _mapping(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise AssemblyError("release input must be a regular file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AssemblyError("release input must be a regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        maximum = 16 * 1024 * 1024
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size == 0
            or metadata.st_size > maximum
        ):
            raise AssemblyError("release input must be a bounded unaliased regular file")
        before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        payload = bytearray()
        while len(payload) <= maximum:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if before != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(payload) != metadata.st_size:
            raise AssemblyError("release input changed while it was read")
        try:
            path_metadata = path.stat(follow_symlinks=False)
        except OSError:
            raise AssemblyError("release input changed while it was read") from None
        if (path_metadata.st_dev, path_metadata.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise AssemblyError("release input changed while it was read")
    finally:
        os.close(descriptor)
    value = yaml.safe_load(bytes(payload))
    if not isinstance(value, dict):
        raise AssemblyError("release input must be a mapping")
    return value


def _exact(
    value: dict[str, Any],
    required: set[str],
    *,
    optional: set[str] = frozenset(),
    field: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        raise AssemblyError(
            f"{field} keys are not exact; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _validate_component(
    name: str,
    component: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    version_text = str(component.get("version") or "")
    try:
        version = Version(version_text)
    except InvalidVersion as exc:
        raise AssemblyError(f"component {name} has an invalid version") from exc
    expected_version = compatibility._exact_version(
        expected.get("version"), f"{name}.version"
    )
    if version_text != expected_version or str(version) != version_text:
        raise AssemblyError(f"component {name} version is not the current matrix version")
    if component.get("kind") != expected.get("artifactKind"):
        raise AssemblyError(f"component {name} artifact kind differs from the matrix")
    artifact = str(component.get("artifact") or "")
    digest = compatibility._digest(component.get("digest"), f"{name}.digest")
    expected_artifact = f"{component.get('kind')}:{name}@{digest}"
    if artifact != expected_artifact or "latest" in artifact.casefold():
        raise AssemblyError(f"component {name} artifact is not digest pinned")
    capabilities = component.get("capabilities")
    if (
        not isinstance(capabilities, list)
        or len(capabilities) != len(set(capabilities))
        or not all(isinstance(value, str) and value for value in capabilities)
    ):
        raise AssemblyError(f"component {name} capabilities are invalid")
    missing = set(expected.get("requiredCapabilities") or ()) - set(capabilities)
    if missing:
        raise AssemblyError(f"component {name} lacks required capabilities")
    if "exactEntries" in expected and int(component.get("entryCount") or 0) != int(
        expected["exactEntries"]
    ):
        raise AssemblyError(f"component {name} entry count differs from the matrix")


def assemble(
    assembly: dict[str, Any],
    matrix: dict[str, Any],
    *,
    matrix_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build an unsigned manifest only from opened, digest-bound local evidence."""

    compatibility.validate_compatibility_matrix(matrix)
    _exact(
        assembly,
        {
            "apiVersion",
            "kind",
            "releaseId",
            "sourceFreezeEvidence",
            "configuration",
            "migrationPlan",
            "certifications",
            "components",
        },
        field="release assembly",
    )
    if (
        assembly.get("apiVersion") != "graphos.io/v1"
        or assembly.get("kind") != "ReleaseAssembly"
    ):
        raise AssemblyError("unsupported release assembly apiVersion/kind")
    release_id = str(assembly.get("releaseId") or "")
    if not re.fullmatch(r"release-[a-z0-9][a-z0-9.-]{2,63}", release_id):
        raise AssemblyError("releaseId must be an opaque release identifier")
    certifications = assembly.get("certifications")
    if not isinstance(certifications, dict) or set(certifications) != {
        "connectorLiveCertificationLedger",
        "prebundledSkillValidationMatrix",
        "skillValidationDeployment",
        "skillValidationLifecycleEvidence",
        "exactArtifactClosureEvidence",
        "ociVulnerabilityScanEvidence",
    }:
        raise AssemblyError("certification evidence catalog is not exact")
    configuration_ref = str(assembly.get("configuration") or "")
    migration_ref = str(assembly.get("migrationPlan") or "")
    source_freeze_ref = str(assembly.get("sourceFreezeEvidence") or "")
    declarations = assembly.get("components")
    protected_references = [source_freeze_ref, configuration_ref, migration_ref]
    protected_references.extend(str(value) for value in certifications.values())
    if isinstance(declarations, dict):
        for declaration in declarations.values():
            if not isinstance(declaration, dict):
                continue
            evidence = declaration.get("evidence")
            if isinstance(evidence, dict):
                protected_references.extend(str(value) for value in evidence.values())
    output_identity = output_path.resolve(strict=False)
    if any(
        (output_path.parent / reference).resolve(strict=False) == output_identity
        for reference in protected_references
    ):
        raise AssemblyError("release output must not alias release evidence")
    configuration = compatibility._evidence_bytes(
        output_path, configuration_ref, "configuration"
    )
    migration = compatibility._evidence_bytes(output_path, migration_ref, "migrationPlan")
    source_freeze = compatibility._evidence_bytes(
        output_path,
        source_freeze_ref,
        "sourceFreezeEvidence",
        maximum=compatibility._MAX_COMPONENT_SOURCE_BYTES,
    )
    source_freeze_authority = compatibility.validate_source_freeze_evidence(
        source_freeze
    )
    matrix_digest = compatibility.file_digest(matrix_path)
    configuration_document = compatibility._json_evidence(
        configuration, "release configuration"
    )
    migration_document = compatibility._json_evidence(
        migration, "release migration plan"
    )
    compatibility.validate_release_configuration(
        configuration_document,
        release_id=release_id,
        matrix=matrix,
        matrix_digest=matrix_digest,
    )
    certification_digests = {
        name: _digest(
            compatibility._evidence_bytes(output_path, reference, f"certification.{name}")
        )
        for name, reference in sorted(certifications.items())
    }
    expected_names = compatibility._component_names(matrix)
    if not isinstance(declarations, dict) or set(declarations) != set(expected_names):
        raise AssemblyError("release component set differs from the matrix")
    components: dict[str, Any] = {}
    component_source_evidence: dict[str, dict[str, Any]] = {}
    for name in expected_names:
        declaration = declarations[name]
        if not isinstance(declaration, dict):
            raise AssemblyError(f"component {name} must be a mapping")
        _exact(
            declaration,
            {
                "version",
                "kind",
                "artifact",
                "digest",
                "evidence",
                "signatureVerifierEnv",
                "capabilities",
            },
            optional={"entryCount"},
            field=f"assembly component {name}",
        )
        evidence = declaration.get("evidence")
        if not isinstance(evidence, dict):
            raise AssemblyError(f"component {name} evidence is required")
        _exact(
            evidence,
            {"source", "sbom", "provenance", "signatureBundle"},
            field=f"assembly component {name} evidence",
        )
        verifier_env = str(declaration.get("signatureVerifierEnv") or "")
        if not _ENV_NAME.fullmatch(verifier_env):
            raise AssemblyError(f"component {name} verifier environment name is invalid")
        component_base = {
            "version": str(declaration["version"]),
            "kind": str(declaration["kind"]),
            "artifact": str(declaration["artifact"]),
            "digest": str(declaration["digest"]),
            "capabilities": sorted(declaration["capabilities"]),
            "evidence": {key: str(value) for key, value in evidence.items()},
            "signatureVerifierEnv": verifier_env,
        }
        if "entryCount" in declaration:
            component_base["entryCount"] = int(declaration["entryCount"])
        _validate_component(name, component_base, matrix["components"][name])
        inspected = compatibility._inspect_component_evidence(
            name,
            component_base,
            output_path,
        )
        component_source_evidence[name] = inspected["sourceEvidence"]
        component = {
            key: value
            for key, value in component_base.items()
            if key != "signatureVerifierEnv"
        }
        component.update(
            {
                "sourceDigest": inspected["sourceDigest"],
                "sbomDigest": inspected["sbomDigest"],
                "provenanceDigest": inspected["provenanceDigest"],
                "signature": {
                    "bundleDigest": inspected["bundleDigest"],
                    "verifierEnv": verifier_env,
                },
            }
        )
        _validate_component(name, component, matrix["components"][name])
        components[name] = component
    compatibility._validate_single_source_freeze(
        component_source_evidence,
        source_freeze_authority,
    )
    index_migrations = components["index-migrations"]
    compatibility.validate_release_migration_plan(
        migration_document,
        release_id=release_id,
        matrix=matrix,
        matrix_digest=matrix_digest,
        index_migration_catalog_digest=str(index_migrations["digest"]),
    )
    if migration_document["indexMigrationCount"] != index_migrations.get(
        "entryCount"
    ):
        raise AssemblyError(
            "release migration plan entry count differs from the index catalog"
        )
    manifest = {
        "apiVersion": "graphos.io/v1",
        "kind": "ReleaseManifest",
        "manifestState": "unsigned-local-binder",
        "releaseId": release_id,
        "matrixDigest": matrix_digest,
        "sourceFreezeEvidenceDigest": source_freeze_authority["evidenceDigest"],
        "configurationDigest": _digest(configuration),
        "protocolSchemas": dict(matrix["protocol"]["schemas"]),
        "components": components,
        "migrationPlanDigest": _digest(migration),
        "certificationDigests": certification_digests,
        "exactGateEvidence": compatibility.exact_gate_evidence(
            components,
            certification_digests,
        ),
        "evidence": {
            "sourceFreezeEvidence": source_freeze_ref,
            "configuration": configuration_ref,
            "migrationPlan": migration_ref,
            "certifications": {
                key: str(value) for key, value in certifications.items()
            },
        },
    }
    compatibility.verify_release_manifest(
        manifest,
        matrix,
        matrix_path=matrix_path,
        manifest_path=output_path,
        verify_signatures=False,
        require_manifest_signature=False,
    )
    return manifest


def _external_command(env_name: str, payload: bytes) -> dict[str, Any]:
    if not _ENV_NAME.fullmatch(env_name):
        raise AssemblyError("external command environment name is invalid")
    raw = os.environ.get(env_name, "")
    if not raw:
        raise AssemblyError("external signing command is absent")
    try:
        command = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssemblyError("external command must be a JSON argv array") from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise AssemblyError("external command must be a JSON argv array")
    returncode, stdout, stderr = compatibility._bounded_adapter(
        command,
        payload,
        maximum=1024 * 1024,
    )
    if returncode != 0:
        raise AssemblyError(
            "external signing failed; output_digest="
            + hashlib.sha256(stdout + stderr).hexdigest()
        )
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AssemblyError("external signer returned non-JSON") from exc
    if not isinstance(response, dict):
        raise AssemblyError("external signer returned a non-object")
    return response


def sign(
    unsigned: dict[str, Any],
    matrix: dict[str, Any],
    *,
    matrix_path: Path,
    manifest_path: Path,
    signer_env: str,
    verifier_env: str,
) -> dict[str, Any]:
    """Attach only a signature returned by the configured external signer."""

    if "signature" in unsigned:
        raise AssemblyError("release manifest is already signed")
    compatibility.verify_release_manifest(
        unsigned,
        matrix,
        matrix_path=matrix_path,
        manifest_path=manifest_path,
        verify_signatures=True,
        require_manifest_signature=False,
    )
    ledger_reference = unsigned["evidence"]["certifications"][
        "connectorLiveCertificationLedger"
    ]
    compatibility.validate_connector_ledger(
        compatibility._evidence_bytes(
            manifest_path,
            ledger_reference,
            "certification.connectorLiveCertificationLedger",
        )
    )
    signable = {**unsigned, "manifestState": "signed-release"}
    subject_digest = compatibility.canonical_digest(signable)
    payload = json.dumps(signable, sort_keys=True, separators=(",", ":")).encode()
    response = _external_command(signer_env, payload)
    _exact(
        response,
        {"scheme", "subjectDigest", "bundleDigest", "signerIdentityDigest", "signature"},
        field="external signer response",
    )
    if response.get("subjectDigest") != subject_digest:
        raise AssemblyError("external signer did not bind the release manifest")
    signed = {
        **signable,
        "signature": {
            "scheme": str(response["scheme"]),
            "subjectDigest": subject_digest,
            "bundleDigest": str(response["bundleDigest"]),
            "signerIdentityDigest": str(response["signerIdentityDigest"]),
            "value": str(response["signature"]),
            "verifierEnv": verifier_env,
        },
    }
    compatibility.verify_release_manifest(
        signed,
        matrix,
        matrix_path=matrix_path,
        manifest_path=manifest_path,
        verify_signatures=False,
    )
    return signed


def _write(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory = os.open(path.parent.resolve(), directory_flags)
    except OSError as exc:
        raise AssemblyError("release output directory is unavailable") from exc
    try:
        try:
            existing = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1
        ):
            raise AssemblyError("release output must be an unaliased regular file")
        temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise AssemblyError("release output write failed")
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        try:
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
            )
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=directory)
            except FileNotFoundError:
                pass
    finally:
        os.close(directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="assemble-graphos-release")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--assembly", type=Path, required=True)
    assemble_parser.add_argument("--matrix", type=Path, required=True)
    assemble_parser.add_argument("--output", type=Path, required=True)
    sign_parser = subparsers.add_parser("sign")
    sign_parser.add_argument("--input", type=Path, required=True)
    sign_parser.add_argument("--matrix", type=Path, required=True)
    sign_parser.add_argument("--output", type=Path, required=True)
    sign_parser.add_argument("--signer-env", required=True)
    sign_parser.add_argument("--verifier-env", required=True)
    args = parser.parse_args(argv)
    try:
        protected_cli_inputs = (
            [args.assembly, args.matrix]
            if args.operation == "assemble"
            else [args.input, args.matrix]
        )
        output_identity = args.output.resolve(strict=False)
        if any(
            path.resolve(strict=False) == output_identity
            for path in protected_cli_inputs
        ):
            raise AssemblyError("release output must not alias an input")
        matrix = _mapping(args.matrix)
        if args.operation == "assemble":
            result = assemble(
                _mapping(args.assembly),
                matrix,
                matrix_path=args.matrix,
                output_path=args.output,
            )
        else:
            if args.input.parent.resolve() != args.output.parent.resolve():
                raise AssemblyError("signed and unsigned manifests must share an evidence root")
            result = sign(
                _mapping(args.input),
                matrix,
                matrix_path=args.matrix,
                manifest_path=args.output,
                signer_env=args.signer_env,
                verifier_env=args.verifier_env,
            )
        _write(args.output, result)
    except Exception as exc:  # noqa: BLE001 - privacy-safe release boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, "operation": args.operation}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
