#!/usr/bin/env python3
"""Generate a deterministic eight-component release assembly from local evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import stat
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release import assemble_manifest, check_compatibility

_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9.-]{1,127}$")
_REFERENCE = re.compile(
    r"^(?=.{1,256}$)[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,127}){0,15}$"
)
_MAX_DECLARATION_BYTES = 1024 * 1024


class ReleaseAssemblyError(ValueError):
    """The release assembly is incomplete, unsafe, or inconsistent."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _load_regular(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ReleaseAssemblyError("component declarations must not be symlinks")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseAssemblyError("component declaration is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size == 0
            or metadata.st_size > _MAX_DECLARATION_BYTES
        ):
            raise ReleaseAssemblyError("component declaration violates its size boundary")
        before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        payload = bytearray()
        while len(payload) <= _MAX_DECLARATION_BYTES:
            chunk = os.read(
                descriptor,
                min(65_536, _MAX_DECLARATION_BYTES + 1 - len(payload)),
            )
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
            raise ReleaseAssemblyError("component declaration changed while it was read")
        try:
            path_metadata = path.stat(follow_symlinks=False)
        except OSError:
            raise ReleaseAssemblyError(
                "component declaration changed while it was read"
            ) from None
        if (path_metadata.st_dev, path_metadata.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise ReleaseAssemblyError(
                "component declaration changed while it was read"
            )
    finally:
        os.close(descriptor)
    try:
        value = json.loads(bytes(payload))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseAssemblyError("component declaration must be JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseAssemblyError("component declaration must be a JSON object")
    return value


def _load_matrix(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ReleaseAssemblyError("compatibility matrix must not be a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseAssemblyError("compatibility matrix is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size == 0
            or metadata.st_size > _MAX_DECLARATION_BYTES
        ):
            raise ReleaseAssemblyError("compatibility matrix violates its size boundary")
        before = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        payload = bytearray()
        while len(payload) <= _MAX_DECLARATION_BYTES:
            chunk = os.read(
                descriptor,
                min(65_536, _MAX_DECLARATION_BYTES + 1 - len(payload)),
            )
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
            raise ReleaseAssemblyError("compatibility matrix changed while it was read")
        try:
            path_metadata = path.stat(follow_symlinks=False)
        except OSError:
            raise ReleaseAssemblyError(
                "compatibility matrix changed while it was read"
            ) from None
        if (path_metadata.st_dev, path_metadata.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise ReleaseAssemblyError(
                "compatibility matrix changed while it was read"
            )
    finally:
        os.close(descriptor)
    try:
        value = yaml.safe_load(bytes(payload))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReleaseAssemblyError("compatibility matrix is unavailable") from exc
    if not isinstance(value, dict):
        raise ReleaseAssemblyError("compatibility matrix must be a mapping")
    return value


def _reference(value: str, field: str) -> str:
    if (
        not _REFERENCE.fullmatch(value)
        or value.startswith("/")
        or ".." in Path(value).parts
        or "://" in value
    ):
        raise ReleaseAssemblyError(f"{field} must be a release-relative reference")
    return value


def generate(
    *,
    release_id: str,
    matrix_path: Path,
    output_path: Path,
    source_freeze_evidence: str,
    configuration: str,
    migration_plan: str,
    connector_ledger: str,
    skill_validation_matrix: str,
    skill_validation_deployment: str,
    skill_validation_lifecycle_evidence: str,
    exact_artifact_closure: str,
    oci_vulnerability_scan: str,
    component_files: dict[str, Path],
) -> dict[str, Any]:
    """Load all component declarations, validate their evidence, and return assembly."""

    matrix = _load_matrix(matrix_path)
    check_compatibility.validate_compatibility_matrix(matrix)
    expected = check_compatibility._component_names(matrix)
    if set(component_files) != set(expected):
        raise ReleaseAssemblyError("component declaration set differs from the matrix")
    components = {name: _load_regular(component_files[name]) for name in expected}
    assembly = {
        "apiVersion": "graphos.io/v1",
        "kind": "ReleaseAssembly",
        "releaseId": release_id,
        "sourceFreezeEvidence": _reference(
            source_freeze_evidence, "sourceFreezeEvidence"
        ),
        "configuration": _reference(configuration, "configuration"),
        "migrationPlan": _reference(migration_plan, "migrationPlan"),
        "certifications": {
            "connectorLiveCertificationLedger": _reference(
                connector_ledger, "connectorLiveCertificationLedger"
            ),
            "prebundledSkillValidationMatrix": _reference(
                skill_validation_matrix, "prebundledSkillValidationMatrix"
            ),
            "skillValidationDeployment": _reference(
                skill_validation_deployment,
                "skillValidationDeployment",
            ),
            "skillValidationLifecycleEvidence": _reference(
                skill_validation_lifecycle_evidence,
                "skillValidationLifecycleEvidence",
            ),
            "exactArtifactClosureEvidence": _reference(
                exact_artifact_closure, "exactArtifactClosureEvidence"
            ),
            "ociVulnerabilityScanEvidence": _reference(
                oci_vulnerability_scan, "ociVulnerabilityScanEvidence"
            ),
        },
        "components": components,
    }
    referenced = {
        source_freeze_evidence,
        configuration,
        migration_plan,
        connector_ledger,
        skill_validation_matrix,
        skill_validation_deployment,
        skill_validation_lifecycle_evidence,
        exact_artifact_closure,
        oci_vulnerability_scan,
        *(
            str(reference)
            for declaration in components.values()
            if isinstance(declaration, dict)
            for reference in (
                declaration.get("evidence", {}).values()
                if isinstance(declaration.get("evidence"), dict)
                else ()
            )
        ),
    }
    output_absolute = output_path.resolve(strict=False)
    protected_paths = {matrix_path.resolve(), *(path.resolve() for path in component_files.values())}
    protected_paths.update((output_path.parent / reference).resolve() for reference in referenced)
    if output_absolute in protected_paths or (
        output_path.exists() and output_path.resolve() in protected_paths
    ):
        raise ReleaseAssemblyError("release assembly output must not alias an input")
    # The existing assembler is the canonical semantic verifier.  This call opens
    # every referenced evidence file and checks all eight matrix declarations.
    assemble_manifest.assemble(
        assembly,
        matrix,
        matrix_path=matrix_path,
        output_path=output_path,
    )
    return assembly


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory = os.open(path.parent.resolve(), directory_flags)
    except OSError as exc:
        raise ReleaseAssemblyError("release assembly output is unavailable") from exc
    try:
        try:
            existing = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1
        ):
            raise ReleaseAssemblyError(
                "release assembly output must be an unaliased regular file"
            )
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
                    raise ReleaseAssemblyError("release assembly output write failed")
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


def _component(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not _COMPONENT.fullmatch(name) or not path:
        raise argparse.ArgumentTypeError("component must be NAME=PATH")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-freeze-evidence", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--migration-plan", required=True)
    parser.add_argument("--connector-ledger", required=True)
    parser.add_argument("--skill-validation-matrix", required=True)
    parser.add_argument("--skill-validation-deployment", required=True)
    parser.add_argument("--skill-validation-lifecycle-evidence", required=True)
    parser.add_argument("--exact-artifact-closure", required=True)
    parser.add_argument("--oci-vulnerability-scan", required=True)
    parser.add_argument("--component", action="append", type=_component, default=[])
    args = parser.parse_args()
    components: dict[str, Path] = {}
    for name, path in args.component:
        if name in components:
            parser.error(f"duplicate component declaration: {name}")
        components[name] = path
    try:
        assembly = generate(
            release_id=args.release_id,
            matrix_path=args.matrix,
            output_path=args.output,
            source_freeze_evidence=args.source_freeze_evidence,
            configuration=args.configuration,
            migration_plan=args.migration_plan,
            connector_ledger=args.connector_ledger,
            skill_validation_matrix=args.skill_validation_matrix,
            skill_validation_deployment=args.skill_validation_deployment,
            skill_validation_lifecycle_evidence=(
                args.skill_validation_lifecycle_evidence
            ),
            exact_artifact_closure=args.exact_artifact_closure,
            oci_vulnerability_scan=args.oci_vulnerability_scan,
            component_files=components,
        )
        write(args.output, assembly)
    except Exception as exc:  # noqa: BLE001 - privacy-safe release boundary
        print(json.dumps({"error": type(exc).__name__, "ok": False}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
