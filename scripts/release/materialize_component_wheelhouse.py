#!/usr/bin/env python3
"""Derive one minimal exact-component wheelhouse from a unified release closure.

The source wheelhouse is first verified by the exact local release promoter.  This
tool then evaluates wheel metadata and selected extras for one current component,
copies only the reachable wheels, writes a deterministic hash-locked requirements
file, verifies the staged result, and publishes it without overwriting an existing
directory.  Its status output contains no filesystem or environment identity.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import secrets
import shutil
import stat
import sys
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from typing import Any

from packaging.markers import (
    UndefinedComparison,
    UndefinedEnvironmentName,
    default_environment,
)
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from scripts.release.promote_local_release import (
    _BASENAME,
    _MAX_DEPENDENCY_CONTEXTS,
    _MAX_LOCK_BYTES,
    _MAX_WHEEL_TOTAL_BYTES,
    LockedRequirement,
    ReleaseError,
    WheelArtifact,
    _copy_regular,
    _hash_regular,
    _is_top_level_dist_info_member,
    _read_regular,
    _sha256,
    load_spec,
    parse_requirements_lock,
    validate_wheelhouse,
)

_RENAME_NOREPLACE = 1


@dataclass(frozen=True)
class ComponentProfile:
    root: str
    extras: frozenset[str]
    requirements_file: str


_PROFILES = {
    "epistemic-graph": ComponentProfile(
        root="epistemic-graph",
        extras=frozenset({"full"}),
        requirements_file="epistemic-graph-requirements.txt",
    ),
    "agent-utilities": ComponentProfile(
        root="agent-utilities",
        extras=frozenset({"serving"}),
        requirements_file="release-requirements.txt",
    ),
    "langfuse-agent": ComponentProfile(
        root="langfuse-agent",
        extras=frozenset({"mcp"}),
        requirements_file="langfuse-agent-requirements.txt",
    ),
}


def _metadata_for_wheel(artifact: WheelArtifact) -> Any:
    """Reopen and bind one wheel METADATA payload to the validated artifact."""

    before, _size = _hash_regular(
        artifact.path,
        limit=_MAX_WHEEL_TOTAL_BYTES,
        code="component-wheel-unreadable",
    )
    if before != artifact.digest:
        raise ReleaseError("component-wheel-digest-mismatch")
    try:
        with zipfile.ZipFile(artifact.path) as archive:
            candidates = [
                info
                for info in archive.infolist()
                if _is_top_level_dist_info_member(info.filename, "METADATA")
            ]
            if len(candidates) != 1 or candidates[0].file_size > _MAX_LOCK_BYTES:
                raise ReleaseError("component-wheel-metadata-invalid")
            metadata = BytesParser(policy=email_policy).parsebytes(
                archive.read(candidates[0])
            )
    except ReleaseError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ReleaseError("component-wheel-unreadable") from exc
    after, _size = _hash_regular(
        artifact.path,
        limit=_MAX_WHEEL_TOTAL_BYTES,
        code="component-wheel-unreadable",
    )
    if before != after:
        raise ReleaseError("component-wheel-source-changed")
    if (
        canonicalize_name(str(metadata.get("Name") or "")) != artifact.name
        or str(metadata.get("Version") or "") != artifact.version
    ):
        raise ReleaseError("component-wheel-identity-mismatch")
    return metadata


def select_component_closure(
    component: str,
    locked: dict[str, LockedRequirement],
    wheels: dict[str, WheelArtifact],
) -> dict[str, frozenset[str]]:
    """Return the minimal reachable packages and active extras for one profile."""

    profile = _PROFILES.get(component)
    if profile is None:
        raise ReleaseError("component-profile-unsupported")
    if set(locked) != set(wheels):
        raise ReleaseError("component-source-closure-mismatch")

    parsed: dict[str, tuple[Requirement, ...]] = {}
    provided: dict[str, frozenset[str]] = {}

    def load_metadata(name: str) -> tuple[Requirement, ...]:
        cached = parsed.get(name)
        if cached is not None:
            return cached
        artifact = wheels.get(name)
        if artifact is None:
            raise ReleaseError("component-dependency-missing")
        metadata = _metadata_for_wheel(artifact)
        requirements: list[Requirement] = []
        for value in metadata.get_all("Requires-Dist") or ():
            try:
                requirement = Requirement(value)
            except InvalidRequirement as exc:
                raise ReleaseError("component-requirement-invalid") from exc
            if requirement.url is not None:
                raise ReleaseError("component-direct-requirement-forbidden")
            requirements.append(requirement)
        parsed[name] = tuple(requirements)
        provided[name] = frozenset(
            canonicalize_name(value)
            for value in metadata.get_all("Provides-Extra") or ()
        )
        return parsed[name]

    contexts: list[tuple[str, str]] = []
    queued: set[tuple[str, str]] = set()
    active_extras: dict[str, set[str]] = {}

    def activate(name: str, extras: Iterable[str] = ()) -> None:
        if name not in locked or name not in wheels:
            raise ReleaseError("component-dependency-missing")
        load_metadata(name)
        requested = {canonicalize_name(extra) for extra in extras}
        if not requested <= provided[name]:
            raise ReleaseError("component-extra-unavailable")
        current = active_extras.setdefault(name, set())
        base = (name, "")
        if base not in queued:
            queued.add(base)
            contexts.append(base)
        for extra in sorted(requested - current):
            current.add(extra)
            context = (name, extra)
            if context not in queued:
                queued.add(context)
                contexts.append(context)

    activate(profile.root, profile.extras)
    environment = default_environment()
    processed_contexts: set[tuple[str, str]] = set()
    processed_edges: set[tuple[str, str]] = set()
    cursor = 0
    while cursor < len(contexts):
        if len(contexts) > _MAX_DEPENDENCY_CONTEXTS:
            raise ReleaseError("component-closure-too-large")
        name, extra = contexts[cursor]
        cursor += 1
        context = (name, extra)
        if context in processed_contexts:
            continue
        processed_contexts.add(context)
        marker_environment = dict(environment)
        marker_environment["extra"] = extra
        for requirement in load_metadata(name):
            if requirement.marker is not None:
                try:
                    enabled = requirement.marker.evaluate(marker_environment)
                except (UndefinedComparison, UndefinedEnvironmentName) as exc:
                    raise ReleaseError("component-marker-invalid") from exc
                if not enabled:
                    continue
            edge = (name, str(requirement))
            if edge in processed_edges:
                continue
            processed_edges.add(edge)
            dependency = canonicalize_name(requirement.name)
            selected = locked.get(dependency)
            if selected is None:
                raise ReleaseError("component-dependency-missing")
            if Version(selected.version) not in requirement.specifier:
                raise ReleaseError("component-dependency-version-mismatch")
            activate(dependency, requirement.extras)

    return {name: frozenset(extras) for name, extras in sorted(active_extras.items())}


def render_component_lock(
    closure: dict[str, frozenset[str]],
    locked: dict[str, LockedRequirement],
) -> bytes:
    """Render the canonical one-line-per-wheel hash lock."""

    lines: list[str] = []
    for name in sorted(closure):
        requirement = locked.get(name)
        if requirement is None:
            raise ReleaseError("component-dependency-missing")
        extras = closure[name]
        rendered_extras = ""
        if extras:
            rendered_extras = "[" + ",".join(sorted(extras)) + "]"
        if not requirement.digest.startswith("sha256:"):
            raise ReleaseError("component-wheel-digest-invalid")
        lines.append(
            f"{name}{rendered_extras}=={requirement.version} "
            f"--hash={requirement.digest}\n"
        )
    if not lines:
        raise ReleaseError("component-closure-empty")
    return "".join(lines).encode("utf-8")


def _write_regular(
    path: Path,
    payload: bytes,
    *,
    code: str = "component-lock-write-failed",
) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ReleaseError(code)
            view = view[written:]
        os.fsync(descriptor)
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError(code) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _directory_descriptor(path: Path, *, code: str) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise ReleaseError(code) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ReleaseError(code)
    return descriptor


def _private_parent(path: Path, *, prefix: str = "component") -> int:
    if not path.is_absolute() or not _BASENAME.fullmatch(path.name):
        raise ReleaseError(f"{prefix}-output-invalid")
    descriptor = _directory_descriptor(
        path.parent,
        code=f"{prefix}-parent-invalid",
    )
    metadata = os.fstat(descriptor)
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        os.close(descriptor)
        raise ReleaseError(f"{prefix}-parent-not-private")
    try:
        os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise ReleaseError(f"{prefix}-output-invalid") from exc
    os.close(descriptor)
    raise ReleaseError(f"{prefix}-output-exists")


def _reject_source_output_overlap(
    source: Path,
    output: Path,
    *,
    prefix: str = "component",
) -> None:
    try:
        source_root = source.resolve(strict=True)
        output_parent = output.parent.resolve(strict=True)
    except OSError as exc:
        raise ReleaseError(f"{prefix}-input-output-invalid") from exc
    try:
        output_parent.relative_to(source_root)
    except ValueError:
        return
    raise ReleaseError(f"{prefix}-input-output-overlap")


def _publish_noreplace(
    parent_fd: int,
    source_name: str,
    output_name: str,
    *,
    prefix: str = "component",
) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise ReleaseError(f"{prefix}-atomic-publish-unavailable") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(output_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise ReleaseError(f"{prefix}-output-exists")
    if error in {errno.ENOSYS, errno.EINVAL}:
        raise ReleaseError(f"{prefix}-atomic-publish-unavailable")
    raise ReleaseError(f"{prefix}-publish-failed")


def _remove_stage(stage: Path) -> None:
    try:
        metadata = stage.lstat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if stage.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        return
    shutil.rmtree(stage, ignore_errors=True)


def _verify_stage(
    stage: Path,
    *,
    component: str,
    requirements_file: str,
    requirements_payload: bytes,
    closure: dict[str, frozenset[str]],
    locked: dict[str, LockedRequirement],
    wheels: dict[str, WheelArtifact],
) -> None:
    try:
        entries = sorted(stage.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        raise ReleaseError("component-stage-unreadable") from exc
    expected = {requirements_file} | {wheels[name].filename for name in closure}
    if {path.name for path in entries} != expected:
        raise ReleaseError("component-stage-content-mismatch")
    for path in entries:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ReleaseError("component-stage-entry-invalid") from exc
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ReleaseError("component-stage-entry-invalid")
    actual_payload = _read_regular(
        stage / requirements_file,
        limit=_MAX_LOCK_BYTES,
        code="component-lock-unreadable",
    )
    if actual_payload != requirements_payload:
        raise ReleaseError("component-lock-content-mismatch")
    staged_lock = parse_requirements_lock(actual_payload)
    if set(staged_lock) != set(closure):
        raise ReleaseError("component-lock-closure-mismatch")
    staged_wheels: dict[str, WheelArtifact] = {}
    for name, extras in closure.items():
        source_requirement = locked[name]
        staged_requirement = staged_lock.get(name)
        if (
            staged_requirement is None
            or staged_requirement.version != source_requirement.version
            or staged_requirement.digest != source_requirement.digest
            or staged_requirement.extras != extras
        ):
            raise ReleaseError("component-lock-entry-mismatch")
        source_artifact = wheels[name]
        staged_path = stage / source_artifact.filename
        digest, _size = _hash_regular(
            staged_path,
            limit=_MAX_WHEEL_TOTAL_BYTES,
            code="component-stage-wheel-unreadable",
        )
        if digest != source_artifact.digest:
            raise ReleaseError("component-stage-wheel-digest-mismatch")
        staged_wheels[name] = WheelArtifact(
            name=source_artifact.name,
            version=source_artifact.version,
            filename=source_artifact.filename,
            digest=source_artifact.digest,
            path=staged_path,
            record_entries=source_artifact.record_entries,
            member_count=source_artifact.member_count,
            uncompressed_bytes=source_artifact.uncompressed_bytes,
            generated_scripts=source_artifact.generated_scripts,
        )
    if select_component_closure(component, staged_lock, staged_wheels) != closure:
        raise ReleaseError("component-stage-closure-mismatch")


def materialize_component_wheelhouse(
    *,
    component: str,
    release_id: str,
    spec_path: Path,
    source: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate, derive, verify, and atomically publish one component closure."""

    profile = _PROFILES.get(component)
    if profile is None:
        raise ReleaseError("component-profile-unsupported")
    spec = load_spec(spec_path, release_id=release_id)
    locked, wheels, _source_lock = validate_wheelhouse(source, spec)
    closure = select_component_closure(component, locked, wheels)
    requirements_payload = render_component_lock(closure, locked)

    _reject_source_output_overlap(source, output)
    parent_fd = _private_parent(output)
    stage_name = f".{output.name}.stage-{secrets.token_hex(12)}"
    stage = output.parent / stage_name
    published = False
    try:
        try:
            os.mkdir(stage_name, mode=0o700, dir_fd=parent_fd)
        except OSError as exc:
            raise ReleaseError("component-stage-create-failed") from exc
        for name in sorted(closure):
            artifact = wheels[name]
            _copy_regular(
                artifact.path,
                stage / artifact.filename,
                digest=artifact.digest,
            )
        _write_regular(stage / profile.requirements_file, requirements_payload)
        _verify_stage(
            stage,
            component=component,
            requirements_file=profile.requirements_file,
            requirements_payload=requirements_payload,
            closure=closure,
            locked=locked,
            wheels=wheels,
        )
        stage_fd = _directory_descriptor(stage, code="component-stage-invalid")
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        _publish_noreplace(parent_fd, stage_name, output.name)
        published = True
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            raise ReleaseError("component-publication-uncertain") from exc
    finally:
        os.close(parent_fd)
        if not published:
            _remove_stage(stage)

    return {
        "apiVersion": "graphos.io/v1",
        "kind": "ComponentWheelhouseMaterialization",
        "component": component,
        "packageCount": len(closure),
        "requirementsFile": profile.requirements_file,
        "requirementsSha256": _sha256(requirements_payload),
        "sourceSpecSha256": spec.digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=tuple(_PROFILES), required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--source-wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = materialize_component_wheelhouse(
            component=arguments.component,
            release_id=arguments.release_id,
            spec_path=arguments.spec,
            source=arguments.source_wheelhouse,
            output=arguments.output,
        )
    except ReleaseError as exc:
        print(f"component-wheelhouse: failed ({exc.code})", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
