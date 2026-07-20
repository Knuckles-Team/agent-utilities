#!/usr/bin/env python3
"""Assemble one deterministic exact-local release input directory.

The source is a flat wheel-only directory.  The output is a new private directory
containing ``exact-local-release.json`` and a closed ``wheelhouse/`` with its
canonical hash lock.  Every wheel, root extra, dependency closure, native engine
artifact, and materializing tool is bound before no-overwrite atomic publication.
Status output contains no filesystem, host, user, endpoint, or command-output data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)

from scripts.release.materialize_component_wheelhouse import (
    _PROFILES,
    _directory_descriptor,
    _private_parent,
    _publish_noreplace,
    _reject_source_output_overlap,
    _remove_stage,
    _write_regular,
    render_component_lock,
    select_component_closure,
)
from scripts.release.promote_local_release import (
    _BASENAME,
    _COMMANDS,
    _DOCTOR_CHECKS,
    _MAX_RELEASE_FILES,
    _MAX_WHEEL_MEMBER_BYTES,
    _MAX_WHEEL_TOTAL_BYTES,
    _MAX_WHEELHOUSE_BYTES,
    _RELEASE_ID,
    BoundExecutable,
    LockedRequirement,
    ReleaseError,
    WheelArtifact,
    _bind_executable,
    _copy_regular,
    _hash_regular,
    _inspect_wheel,
    _installer_environment,
    _invoke_bounded,
    _sha256,
    load_spec,
    validate_wheelhouse,
)

_SPEC_FILE = "exact-local-release.json"
_LOCK_FILE = "release-requirements.txt"
_WHEELHOUSE_DIRECTORY = "wheelhouse"
_REQUIRED_UV_VERSION = "0.11.7"
_ENGINE_SERVER = re.compile(r"^[^/]+\.data/scripts/epistemic-graph-server$")
_NUMERIC_EXTENSION = re.compile(r"^epistemic_graph/numeric(?:\.[A-Za-z0-9_]+)*\.so$")


@dataclass(frozen=True)
class ToolchainIdentity:
    python_version: str
    python_digest: str
    uv_version: str
    uv_digest: str


def _scan_source_wheels(source: Path) -> dict[str, WheelArtifact]:
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise ReleaseError("assembly-source-unavailable") from exc
    if source.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ReleaseError("assembly-source-invalid")
    entries: list[os.DirEntry[str]] = []
    iterator = None
    try:
        iterator = os.scandir(source)
        for entry in iterator:
            if len(entries) >= _MAX_RELEASE_FILES:
                raise ReleaseError("assembly-wheel-count-limit")
            entries.append(entry)
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError("assembly-source-unavailable") from exc
    finally:
        if iterator is not None:
            iterator.close()
    entries.sort(key=lambda entry: entry.name)
    if not entries:
        raise ReleaseError("assembly-source-empty")

    wheels: dict[str, WheelArtifact] = {}
    total_bytes = 0
    compatible_tags = set(sys_tags())
    for entry in entries:
        if (
            not _BASENAME.fullmatch(entry.name)
            or not entry.name.endswith(".whl")
            or entry.is_symlink()
        ):
            raise ReleaseError("assembly-source-entry-invalid")
        try:
            entry_metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise ReleaseError("assembly-source-entry-invalid") from exc
        if not stat.S_ISREG(entry_metadata.st_mode):
            raise ReleaseError("assembly-source-entry-invalid")
        total_bytes += entry_metadata.st_size
        if total_bytes > _MAX_WHEELHOUSE_BYTES:
            raise ReleaseError("assembly-wheelhouse-byte-limit")
        path = Path(entry.path)
        digest, _size = _hash_regular(
            path,
            limit=_MAX_WHEEL_TOTAL_BYTES,
            code="assembly-wheel-unreadable",
        )
        try:
            filename_name, filename_version, _build, tags = parse_wheel_filename(
                entry.name
            )
        except InvalidWheelFilename as exc:
            raise ReleaseError("assembly-wheel-filename-invalid") from exc
        if tags.isdisjoint(compatible_tags):
            raise ReleaseError("assembly-wheel-platform-mismatch")
        name = canonicalize_name(filename_name)
        if name in wheels:
            raise ReleaseError("assembly-wheel-duplicate")
        (
            metadata_name,
            metadata_version,
            record_entries,
            member_count,
            uncompressed_bytes,
            generated_scripts,
        ) = _inspect_wheel(path, digest=digest)
        if metadata_name != name or metadata_version != str(filename_version):
            raise ReleaseError("assembly-wheel-identity-mismatch")
        wheels[name] = WheelArtifact(
            name=name,
            version=metadata_version,
            filename=entry.name,
            digest=digest,
            path=path,
            record_entries=record_entries,
            member_count=member_count,
            uncompressed_bytes=uncompressed_bytes,
            generated_scripts=generated_scripts,
        )
    if not set(_PROFILES) <= set(wheels):
        raise ReleaseError("assembly-root-wheel-missing")
    return wheels


def _unified_lock(wheels: dict[str, WheelArtifact]) -> dict[str, LockedRequirement]:
    locked: dict[str, LockedRequirement] = {}
    for name, artifact in wheels.items():
        profile = _PROFILES.get(name)
        locked[name] = LockedRequirement(
            name=name,
            version=artifact.version,
            extras=profile.extras if profile is not None else frozenset(),
            digest=artifact.digest,
        )
    reachable: set[str] = set()
    for component in _PROFILES:
        reachable.update(select_component_closure(component, locked, wheels))
    if reachable != set(wheels):
        raise ReleaseError("assembly-wheelhouse-not-minimal")
    return locked


def _member_digest(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    if info.file_size <= 0 or info.file_size > _MAX_WHEEL_MEMBER_BYTES:
        raise ReleaseError("assembly-native-artifact-invalid")
    hasher = hashlib.sha256()
    consumed = 0
    try:
        with archive.open(info, "r") as member:
            while chunk := member.read(1024 * 1024):
                consumed += len(chunk)
                if consumed > _MAX_WHEEL_MEMBER_BYTES:
                    raise ReleaseError("assembly-native-artifact-invalid")
                hasher.update(chunk)
    except ReleaseError:
        raise
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ReleaseError("assembly-native-artifact-invalid") from exc
    if consumed != info.file_size:
        raise ReleaseError("assembly-native-artifact-invalid")
    return "sha256:" + hasher.hexdigest()


def _native_artifacts(engine: WheelArtifact) -> dict[str, str]:
    before, _size = _hash_regular(
        engine.path,
        limit=_MAX_WHEEL_TOTAL_BYTES,
        code="assembly-engine-wheel-unreadable",
    )
    if before != engine.digest:
        raise ReleaseError("assembly-engine-wheel-digest-mismatch")
    try:
        with zipfile.ZipFile(engine.path) as archive:
            servers = [
                info
                for info in archive.infolist()
                if _ENGINE_SERVER.fullmatch(info.filename)
            ]
            numeric = [
                info
                for info in archive.infolist()
                if _NUMERIC_EXTENSION.fullmatch(info.filename)
            ]
            if len(servers) != 1 or len(numeric) != 1:
                raise ReleaseError("assembly-native-artifact-inventory-invalid")
            result = {
                "epistemic-graph-server": _member_digest(archive, servers[0]),
                "epistemic-graph-numeric": _member_digest(archive, numeric[0]),
            }
    except ReleaseError:
        raise
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseError("assembly-engine-wheel-unreadable") from exc
    after, _size = _hash_regular(
        engine.path,
        limit=_MAX_WHEEL_TOTAL_BYTES,
        code="assembly-engine-wheel-unreadable",
    )
    if before != after:
        raise ReleaseError("assembly-engine-wheel-source-changed")
    return result


def _identity_output(
    executable: BoundExecutable,
    arguments: list[str],
    *,
    role: str,
) -> str:
    result = _invoke_bounded(
        [os.fspath(executable.proc_path), *arguments],
        cwd=Path("/"),
        environment=_installer_environment(),
        timeout_seconds=30,
        role=role,
        max_output_bytes=1024,
    )
    try:
        output = result.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ReleaseError(f"assembly-{role}-identity-invalid") from exc
    if result.return_code != 0 or not output:
        raise ReleaseError(f"assembly-{role}-identity-invalid")
    return output


def _toolchain_identity(uv: str) -> ToolchainIdentity:
    if sys.version_info[:2] != (3, 12):
        raise ReleaseError("assembly-python-version-unsupported")
    python_tool: BoundExecutable | None = None
    uv_tool: BoundExecutable | None = None
    try:
        python_tool, python_digest = _bind_executable(
            sys.executable,
            code="assembly-python-unavailable",
        )
        uv_tool, uv_digest = _bind_executable(uv, code="assembly-uv-unavailable")
        python_version = _identity_output(
            python_tool,
            ["-I", "-c", "import platform;print(platform.python_version())"],
            role="python",
        )
        if python_version != platform.python_version():
            raise ReleaseError("assembly-python-identity-mismatch")
        uv_tokens = _identity_output(uv_tool, ["--version"], role="uv").split()
        if (
            len(uv_tokens) < 2
            or uv_tokens[0] != "uv"
            or uv_tokens[1] != _REQUIRED_UV_VERSION
        ):
            raise ReleaseError("assembly-uv-identity-mismatch")
        return ToolchainIdentity(
            python_version=python_version,
            python_digest=python_digest,
            uv_version=uv_tokens[1],
            uv_digest=uv_digest,
        )
    finally:
        for tool in (python_tool, uv_tool):
            if tool is not None:
                os.close(tool.descriptor)


def _spec_payload(
    *,
    release_id: str,
    wheels: dict[str, WheelArtifact],
    lock_payload: bytes,
    native_artifacts: dict[str, str],
    toolchain: ToolchainIdentity,
) -> bytes:
    packages = {
        name: {
            "version": wheels[name].version,
            "wheel": wheels[name].filename,
            "sha256": wheels[name].digest,
        }
        for name in sorted(_PROFILES)
    }
    spec = {
        "apiVersion": "graphos.io/v2",
        "kind": "ExactLocalRelease",
        "releaseId": release_id,
        "requirements": {"file": _LOCK_FILE, "sha256": _sha256(lock_payload)},
        "packages": packages,
        "nativeArtifacts": native_artifacts,
        "toolchain": {
            "python": {
                "version": toolchain.python_version,
                "sha256": toolchain.python_digest,
            },
            "uv": {
                "version": toolchain.uv_version,
                "sha256": toolchain.uv_digest,
            },
        },
        "commands": {
            "canary": {
                "entryPoint": _COMMANDS["canary"],
                "arguments": ["--json"],
                "timeoutSeconds": 30,
            },
            "doctor": {
                "entryPoint": _COMMANDS["doctor"],
                "arguments": ["--json", "--live", "--only", *_DOCTOR_CHECKS],
                "timeoutSeconds": 120,
            },
        },
    }
    return (json.dumps(spec, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _stage_release_inputs(
    stage: Path,
    *,
    release_id: str,
    wheels: dict[str, WheelArtifact],
    locked: dict[str, LockedRequirement],
    native_artifacts: dict[str, str],
    toolchain: ToolchainIdentity,
) -> tuple[bytes, bytes]:
    wheelhouse = stage / _WHEELHOUSE_DIRECTORY
    try:
        wheelhouse.mkdir(mode=0o700)
    except OSError as exc:
        raise ReleaseError("assembly-wheelhouse-stage-failed") from exc
    for name in sorted(wheels):
        artifact = wheels[name]
        _copy_regular(
            artifact.path,
            wheelhouse / artifact.filename,
            digest=artifact.digest,
        )
    extras = {name: requirement.extras for name, requirement in locked.items()}
    lock_payload = render_component_lock(extras, locked)
    _write_regular(
        wheelhouse / _LOCK_FILE,
        lock_payload,
        code="assembly-lock-write-failed",
    )
    spec_payload = _spec_payload(
        release_id=release_id,
        wheels=wheels,
        lock_payload=lock_payload,
        native_artifacts=native_artifacts,
        toolchain=toolchain,
    )
    _write_regular(
        stage / _SPEC_FILE,
        spec_payload,
        code="assembly-spec-write-failed",
    )
    return lock_payload, spec_payload


def _verify_staged_release(stage: Path, *, release_id: str) -> int:
    try:
        entries = {path.name for path in stage.iterdir()}
    except OSError as exc:
        raise ReleaseError("assembly-stage-unreadable") from exc
    if entries != {_SPEC_FILE, _WHEELHOUSE_DIRECTORY}:
        raise ReleaseError("assembly-stage-content-mismatch")
    spec_path = stage / _SPEC_FILE
    wheelhouse = stage / _WHEELHOUSE_DIRECTORY
    if spec_path.is_symlink() or wheelhouse.is_symlink():
        raise ReleaseError("assembly-stage-content-mismatch")
    spec = load_spec(spec_path, release_id=release_id)
    locked, wheels, _lock_payload = validate_wheelhouse(wheelhouse, spec)
    reachable: set[str] = set()
    for component in _PROFILES:
        reachable.update(select_component_closure(component, locked, wheels))
    if reachable != set(locked):
        raise ReleaseError("assembly-stage-wheelhouse-not-minimal")
    if _native_artifacts(wheels["epistemic-graph"]) != spec.native_artifacts:
        raise ReleaseError("assembly-stage-native-artifact-mismatch")
    for directory in (wheelhouse, stage):
        descriptor = _directory_descriptor(directory, code="assembly-stage-invalid")
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return len(locked)


def assemble_exact_local_release(
    *,
    release_id: str,
    source: Path,
    uv: str,
    output: Path,
) -> dict[str, object]:
    """Assemble and atomically publish one exact local release input directory."""

    if not _RELEASE_ID.fullmatch(release_id):
        raise ReleaseError("assembly-release-id-invalid")
    wheels = _scan_source_wheels(source)
    locked = _unified_lock(wheels)
    native_artifacts = _native_artifacts(wheels["epistemic-graph"])
    toolchain = _toolchain_identity(uv)
    _reject_source_output_overlap(source, output, prefix="assembly")
    parent_fd = _private_parent(output, prefix="assembly")
    stage_name = f".{output.name}.stage-{os.getpid()}-{os.urandom(12).hex()}"
    stage = output.parent / stage_name
    published = False
    lock_payload = b""
    spec_payload = b""
    package_count = 0
    try:
        try:
            os.mkdir(stage_name, mode=0o700, dir_fd=parent_fd)
        except OSError as exc:
            raise ReleaseError("assembly-stage-create-failed") from exc
        lock_payload, spec_payload = _stage_release_inputs(
            stage,
            release_id=release_id,
            wheels=wheels,
            locked=locked,
            native_artifacts=native_artifacts,
            toolchain=toolchain,
        )
        package_count = _verify_staged_release(stage, release_id=release_id)
        _publish_noreplace(
            parent_fd,
            stage_name,
            output.name,
            prefix="assembly",
        )
        published = True
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            raise ReleaseError("assembly-publication-uncertain") from exc
    finally:
        os.close(parent_fd)
        if not published:
            _remove_stage(stage)
    return {
        "apiVersion": "graphos.io/v1",
        "kind": "ExactLocalReleaseAssembly",
        "releaseId": release_id,
        "packageCount": package_count,
        "requirementsFile": _LOCK_FILE,
        "requirementsSha256": _sha256(lock_payload),
        "specFile": _SPEC_FILE,
        "specSha256": _sha256(spec_payload),
        "pythonVersion": toolchain.python_version,
        "uvVersion": toolchain.uv_version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--source-wheelhouse", type=Path, required=True)
    parser.add_argument("--uv", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = assemble_exact_local_release(
            release_id=arguments.release_id,
            source=arguments.source_wheelhouse,
            uv=arguments.uv,
            output=arguments.output,
        )
    except ReleaseError as exc:
        print(f"exact-local-assembly: failed ({exc.code})", file=sys.stderr)
        return 1
    except OSError:
        print("exact-local-assembly: failed (assembly-io-failure)", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
