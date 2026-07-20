#!/usr/bin/env python3
"""Assemble and atomically promote one exact local GraphOS release.

This is the only local release path.  It consumes a flat, closed wheelhouse and a
strict JSON specification, installs a complete hash-locked closure without index
access, validates installed metadata and native artifacts without importing them,
then atomically replaces the ``current`` symlink.  A failed canary or doctor proof
restores the previous target automatically.  Evidence contains no filesystem,
host, user, endpoint, credential, or command-output material.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import json
import os
import platform
import re
import secrets
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised by the platform contract test
    fcntl = None  # type: ignore[assignment]

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import sys_tags
from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)
from packaging.version import InvalidVersion, Version

_DIGEST = re.compile(r"^sha256:[a-f0-9]{64}$")
_RELEASE_ID = re.compile(r"^release-[a-z0-9][a-z0-9.-]{2,63}$")
_BASENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,254}$")
_HASH_TOKEN = re.compile(r"^--hash=sha256:([a-f0-9]{64})$")
_REQUIRED_PACKAGES = (
    "agent-utilities",
    "epistemic-graph",
    "langfuse-agent",
)
_NATIVE_ARTIFACTS = (
    "epistemic-graph-server",
    "epistemic-graph-numeric",
)
_COMMANDS = {
    "canary": "graph-os-release-canary",
    "doctor": "agent-utilities-doctor",
}
_COMMAND_MODULES = {
    "canary": "agent_utilities.deployment.release_canary",
    "doctor": "agent_utilities.deployment.doctor",
}
_DOCTOR_CHECKS = (
    "engine_request_context",
    "engine",
    "langfuse",
    "native_optimizer",
    "skills",
)
_PROCESS_EXECUTABLES = {
    "agent-utilities",
    "epistemic-graph-server",
    "graph-os",
    "graph-os-daemon",
}
_PROCESS_MODULES = {
    "agent_utilities",
    "agent_utilities.__main__",
    "agent_utilities.gateway.daemon",
    "agent_utilities.mcp.kg_server",
    "epistemic_graph.server",
}
_MAX_SPEC_BYTES = 128 * 1024
_MAX_LOCK_BYTES = 8 * 1024 * 1024
_MAX_WHEEL_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
_MAX_WHEEL_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
_MAX_WHEEL_MEMBERS = 200_000
_MAX_DEPENDENCY_CONTEXTS = 200_000
_MAX_COMMAND_OUTPUT_BYTES = 2 * 1024 * 1024
_MAX_INSTALL_SECONDS = 900
_MAX_RELEASE_FILES = 400_000
_MAX_RELEASE_BYTES = 16 * 1024 * 1024 * 1024
_MAX_WHEELHOUSE_BYTES = 12 * 1024 * 1024 * 1024
_JOURNAL_NAME = ".exact-local-activation.json"
_JOURNAL_VERSION = 1
_SIGNER_ENV = "EXACT_LOCAL_EVIDENCE_SIGNER_COMMAND"
_VERIFIER_ENV = "EXACT_LOCAL_EVIDENCE_VERIFIER_COMMAND"
_UNSAFE_PROCESS_ENV = {
    "BASH_ENV",
    "ENV",
    "GCONV_PATH",
    "IFS",
    "NODE_OPTIONS",
    "PERL5OPT",
    "PYTHONBREAKPOINT",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONWARNINGS",
    "RUBYOPT",
    "VIRTUAL_ENV",
}


class ReleaseError(RuntimeError):
    """A fail-closed release rejection with a path-free stable code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CommandProofError(ReleaseError):
    """A canary/doctor rejection carrying only bounded aggregate evidence."""

    def __init__(self, code: str, proof: dict[str, Any]) -> None:
        self.proof = proof
        super().__init__(code)


class EvidencePublicationUncertain(ReleaseError):
    """Evidence was linked but its parent-directory fsync did not complete."""


@dataclass(frozen=True)
class PackagePin:
    name: str
    version: str
    wheel: str
    digest: str


@dataclass(frozen=True)
class CommandSpec:
    entry_point: str
    arguments: tuple[str, ...]
    timeout_seconds: int


@dataclass(frozen=True)
class ToolPin:
    version: str
    digest: str


@dataclass(frozen=True)
class ReleaseSpec:
    release_id: str
    requirements_file: str
    requirements_digest: str
    packages: dict[str, PackagePin]
    native_artifacts: dict[str, str]
    toolchain: dict[str, ToolPin]
    commands: dict[str, CommandSpec]
    digest: str


@dataclass(frozen=True)
class LockedRequirement:
    name: str
    version: str
    extras: frozenset[str]
    digest: str


@dataclass(frozen=True)
class WheelArtifact:
    name: str
    version: str
    filename: str
    digest: str
    path: Path
    record_entries: dict[str, tuple[str, int]]
    member_count: int
    uncompressed_bytes: int
    generated_scripts: frozenset[str]


@dataclass(frozen=True)
class BoundExecutable:
    """An executable held open so later execution cannot follow a replaced path."""

    descriptor: int
    proc_path: Path
    source_path: Path


@dataclass(frozen=True)
class CommandResult:
    return_code: int
    output_digest: str
    stdout: bytes


def _exact_mapping(
    value: Any,
    *,
    required: set[str],
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != required:
        raise ReleaseError(f"invalid-{field}")
    return value


def _json_without_duplicates(payload: bytes) -> Any:
    def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ReleaseError("duplicate-json-key")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=_pairs)
    except ReleaseError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError("invalid-json") from exc


def _read_regular(path: Path, *, limit: int, code: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseError(code) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ReleaseError(code)
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > limit:
            raise ReleaseError(code)
        return payload
    finally:
        os.close(descriptor)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _hash_regular(path: Path, *, limit: int, code: str) -> tuple[str, int]:
    """Hash a bounded regular file without loading the artifact into memory."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseError(code) from exc
    hasher = hashlib.sha256()
    consumed = 0
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise ReleaseError(code)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > limit:
                raise ReleaseError(code)
            hasher.update(chunk)
        if consumed != metadata.st_size:
            raise ReleaseError(code)
    finally:
        os.close(descriptor)
    return "sha256:" + hasher.hexdigest(), consumed


def _require_supported_platform() -> None:
    """Require the one current release platform and its descriptor primitives."""

    seal_names = (
        "F_ADD_SEALS",
        "F_GET_SEALS",
        "F_SEAL_GROW",
        "F_SEAL_SEAL",
        "F_SEAL_SHRINK",
        "F_SEAL_WRITE",
    )
    if (
        os.name != "posix"
        or sys.platform != "linux"
        or fcntl is None
        or not hasattr(os, "memfd_create")
        or not hasattr(os, "MFD_CLOEXEC")
        or not hasattr(os, "MFD_ALLOW_SEALING")
        or not all(hasattr(fcntl, name) for name in seal_names)
        or not Path("/proc/self/fd").is_dir()
    ):
        raise ReleaseError("unsupported-platform")


def _fd_path(descriptor: int) -> Path:
    """Return a stable path to a descriptor held by this promoter process."""

    try:
        os.fstat(descriptor)
    except OSError as exc:
        raise ReleaseError("descriptor-binding-invalid") from exc
    return Path("/proc") / str(os.getpid()) / "fd" / str(descriptor)


def _digest_value(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ReleaseError(code)
    return value


def _version_value(value: Any, code: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ReleaseError(code)
    try:
        parsed = Version(value)
    except InvalidVersion as exc:
        raise ReleaseError(code) from exc
    if str(parsed) != value:
        raise ReleaseError(code)
    return value


def _basename(value: Any, *, suffix: str, code: str) -> str:
    if not isinstance(value, str):
        raise ReleaseError(code)
    rendered = value
    if (
        not _BASENAME.fullmatch(rendered)
        or Path(rendered).name != rendered
        or not rendered.endswith(suffix)
    ):
        raise ReleaseError(code)
    return rendered


def _validate_command(role: str, value: Any) -> CommandSpec:
    command = _exact_mapping(
        value,
        required={"entryPoint", "arguments", "timeoutSeconds"},
        field=f"{role}-command",
    )
    entry_point = str(command["entryPoint"] or "")
    if entry_point != _COMMANDS[role]:
        raise ReleaseError(f"invalid-{role}-entry-point")
    arguments = command["arguments"]
    if not isinstance(arguments, list) or not all(
        isinstance(item, str) and item for item in arguments
    ):
        raise ReleaseError(f"invalid-{role}-arguments")
    if role == "canary":
        if arguments != ["--json"]:
            raise ReleaseError("invalid-canary-arguments")
    else:
        expected_live = ["--json", "--live", "--only", *_DOCTOR_CHECKS]
        if arguments != expected_live:
            raise ReleaseError("invalid-doctor-arguments")
    timeout = command["timeoutSeconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or not 1 <= timeout <= 300
    ):
        raise ReleaseError(f"invalid-{role}-timeout")
    return CommandSpec(entry_point, tuple(arguments), timeout)


def load_spec(path: Path, *, release_id: str) -> ReleaseSpec:
    """Load the exact release declaration without accepting extension keys."""

    payload = _read_regular(path, limit=_MAX_SPEC_BYTES, code="unreadable-release-spec")
    root = _exact_mapping(
        _json_without_duplicates(payload),
        required={
            "apiVersion",
            "kind",
            "releaseId",
            "requirements",
            "packages",
            "nativeArtifacts",
            "toolchain",
            "commands",
        },
        field="release-spec",
    )
    if root["apiVersion"] != "graphos.io/v2" or root["kind"] != "ExactLocalRelease":
        raise ReleaseError("unsupported-release-spec")
    declared_id = root["releaseId"]
    if not isinstance(declared_id, str):
        raise ReleaseError("release-id-mismatch")
    if not _RELEASE_ID.fullmatch(release_id) or declared_id != release_id:
        raise ReleaseError("release-id-mismatch")
    requirements = _exact_mapping(
        root["requirements"],
        required={"file", "sha256"},
        field="requirements",
    )
    requirements_file = _basename(
        requirements["file"], suffix=".txt", code="invalid-requirements-file"
    )
    requirements_digest = _digest_value(
        requirements["sha256"], "invalid-requirements-digest"
    )
    declarations = _exact_mapping(
        root["packages"], required=set(_REQUIRED_PACKAGES), field="packages"
    )
    packages: dict[str, PackagePin] = {}
    for name in _REQUIRED_PACKAGES:
        declaration = _exact_mapping(
            declarations[name],
            required={"version", "wheel", "sha256"},
            field=f"{name}-package",
        )
        packages[name] = PackagePin(
            name=name,
            version=_version_value(declaration["version"], f"invalid-{name}-version"),
            wheel=_basename(
                declaration["wheel"], suffix=".whl", code=f"invalid-{name}-wheel"
            ),
            digest=_digest_value(declaration["sha256"], f"invalid-{name}-digest"),
        )
    native = _exact_mapping(
        root["nativeArtifacts"],
        required=set(_NATIVE_ARTIFACTS),
        field="native-artifacts",
    )
    native_artifacts = {
        name: _digest_value(native[name], f"invalid-{name}-digest")
        for name in _NATIVE_ARTIFACTS
    }
    toolchain_raw = _exact_mapping(
        root["toolchain"], required={"python", "uv"}, field="toolchain"
    )
    toolchain: dict[str, ToolPin] = {}
    for name in ("python", "uv"):
        declaration = _exact_mapping(
            toolchain_raw[name],
            required={"version", "sha256"},
            field=f"{name}-toolchain",
        )
        toolchain[name] = ToolPin(
            version=_version_value(
                declaration["version"], f"invalid-{name}-tool-version"
            ),
            digest=_digest_value(declaration["sha256"], f"invalid-{name}-tool-digest"),
        )
    commands_raw = _exact_mapping(
        root["commands"], required=set(_COMMANDS), field="commands"
    )
    commands = {role: _validate_command(role, commands_raw[role]) for role in _COMMANDS}
    return ReleaseSpec(
        release_id=release_id,
        requirements_file=requirements_file,
        requirements_digest=requirements_digest,
        packages=packages,
        native_artifacts=native_artifacts,
        toolchain=toolchain,
        commands=commands,
        digest=_sha256(payload),
    )


def _logical_lock_lines(text: str) -> Iterable[str]:
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            raise ReleaseError("inline-lock-comment")
        continued = line.endswith("\\")
        fragment = line[:-1].rstrip() if continued else line
        pending = f"{pending} {fragment}".strip()
        if not continued:
            yield pending
            pending = ""
    if pending:
        raise ReleaseError("unterminated-lock-line")


def parse_requirements_lock(payload: bytes) -> dict[str, LockedRequirement]:
    """Accept only exact package-name requirements with one SHA-256 artifact."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseError("invalid-requirements-encoding") from exc
    locked: dict[str, LockedRequirement] = {}
    for line in _logical_lock_lines(text):
        tokens = line.split()
        if len(tokens) != 2 or not _HASH_TOKEN.fullmatch(tokens[1]):
            raise ReleaseError("requirements-not-exactly-hash-locked")
        declaration = tokens[0]
        if any(value in declaration for value in ("@", "/", "\\", "://", ";")):
            raise ReleaseError("direct-requirement-forbidden")
        try:
            requirement = Requirement(declaration)
        except InvalidRequirement as exc:
            raise ReleaseError("invalid-locked-requirement") from exc
        specifiers = list(requirement.specifier)
        if (
            requirement.url is not None
            or requirement.marker is not None
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
            or "*" in specifiers[0].version
        ):
            raise ReleaseError("locked-requirement-not-exact")
        name = canonicalize_name(requirement.name)
        version = _version_value(specifiers[0].version, "invalid-locked-version")
        if name in locked:
            raise ReleaseError("duplicate-locked-package")
        digest_match = _HASH_TOKEN.fullmatch(tokens[1])
        if digest_match is None:  # guarded above; keeps the type boundary explicit
            raise ReleaseError("invalid-locked-digest")
        locked[name] = LockedRequirement(
            name=name,
            version=version,
            extras=frozenset(canonicalize_name(extra) for extra in requirement.extras),
            digest="sha256:" + digest_match.group(1),
        )
    if not locked:
        raise ReleaseError("empty-requirements-lock")
    return locked


def _wheel_member_is_safe(info: zipfile.ZipInfo) -> bool:
    name = info.filename
    relative = PurePosixPath(name)
    if (
        not name
        or len(name.encode("utf-8")) > 4096
        or "\\" in name
        or "\x00" in name
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) > 64
        or any(
            not part or any(ord(char) < 32 for char in part) for part in relative.parts
        )
    ):
        return False
    mode = (info.external_attr >> 16) & 0xFFFF
    kind = stat.S_IFMT(mode)
    return kind in {0, stat.S_IFREG, stat.S_IFDIR} and not (
        info.is_dir() and kind == stat.S_IFREG
    )


def _is_top_level_dist_info_member(name: str, member: str) -> bool:
    """Match one wheel-owned ``.dist-info`` member, never vendored metadata."""

    if name.count("/") != 1:
        return False
    distribution, candidate = name.split("/", 1)
    return (
        candidate == member
        and distribution.endswith(".dist-info")
        and distribution != ".dist-info"
    )


def _wheel_record_entries(
    archive: zipfile.ZipFile,
    *,
    record_name: str,
    info_by_name: dict[str, zipfile.ZipInfo],
) -> dict[str, tuple[str, int]]:
    """Verify the wheel RECORD against every archived regular member."""

    try:
        rows = list(csv.reader(archive.read(record_name).decode("utf-8").splitlines()))
    except (UnicodeDecodeError, csv.Error, KeyError) as exc:
        raise ReleaseError("invalid-wheel-record") from exc
    entries: dict[str, tuple[str, int]] = {}
    for row in rows:
        if len(row) != 3 or row[0] in entries:
            raise ReleaseError("invalid-wheel-record")
        relative = PurePosixPath(row[0])
        if (
            not row[0]
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in row[0]
            or "\x00" in row[0]
        ):
            raise ReleaseError("invalid-wheel-record")
        if row[0] == record_name:
            if row[1] or row[2]:
                raise ReleaseError("wheel-record-self-hash-present")
            entries[row[0]] = ("", 0)
            continue
        info = info_by_name.get(row[0])
        if info is None or info.is_dir() or not row[1].startswith("sha256="):
            raise ReleaseError("wheel-record-member-mismatch")
        if not row[2].isdigit() or int(row[2]) != info.file_size:
            raise ReleaseError("wheel-record-member-mismatch")
        hasher = hashlib.sha256()
        with archive.open(info, "r") as member:
            while chunk := member.read(1024 * 1024):
                hasher.update(chunk)
        encoded = base64.urlsafe_b64encode(hasher.digest()).rstrip(b"=").decode()
        expected = "sha256=" + encoded
        if row[1] != expected:
            raise ReleaseError("wheel-record-member-mismatch")
        entries[row[0]] = (row[1], int(row[2]))
    archived_files = {name for name, info in info_by_name.items() if not info.is_dir()}
    if set(entries) != archived_files:
        raise ReleaseError("wheel-record-coverage-mismatch")
    return entries


def _wheel_generated_scripts(
    archive: zipfile.ZipFile, info_by_name: dict[str, zipfile.ZipInfo]
) -> frozenset[str]:
    candidates = [
        name for name in info_by_name if name.endswith(".dist-info/entry_points.txt")
    ]
    if not candidates:
        return frozenset()
    if len(candidates) != 1 or info_by_name[candidates[0]].file_size > _MAX_LOCK_BYTES:
        raise ReleaseError("invalid-wheel-entry-points")
    try:
        payload = archive.read(candidates[0]).decode("utf-8")
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.optionxform = str
        parser.read_string(payload)
    except (UnicodeDecodeError, configparser.Error) as exc:
        raise ReleaseError("invalid-wheel-entry-points") from exc
    scripts: set[str] = set()
    for section in ("console_scripts", "gui_scripts"):
        if not parser.has_section(section):
            continue
        for name, target in parser.items(section):
            if (
                not _BASENAME.fullmatch(name)
                or Path(name).name != name
                or not target.strip()
            ):
                raise ReleaseError("invalid-wheel-entry-points")
            scripts.add(name)
    return frozenset(scripts)


def _inspect_wheel(
    path: Path, *, digest: str
) -> tuple[str, str, dict[str, tuple[str, int]], int, int, frozenset[str]]:
    """Validate archive paths/types and bind filename identity to METADATA."""

    try:
        filename_name, filename_version, _build, _tags = parse_wheel_filename(path.name)
    except InvalidWheelFilename as exc:
        raise ReleaseError("invalid-wheel-filename") from exc
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if not infos or len(infos) > _MAX_WHEEL_MEMBERS:
                raise ReleaseError("wheel-member-count")
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or len(names) != len(
                set(map(str.casefold, names))
            ):
                raise ReleaseError("duplicate-wheel-member")
            expanded_names = set(names)
            for name in names:
                parts = PurePosixPath(name).parts
                for index in range(1, len(parts)):
                    expanded_names.add("/".join(parts[:index]) + "/")
                    if len(expanded_names) > _MAX_RELEASE_FILES:
                        raise ReleaseError("release-file-count-limit")
            total = 0
            for info in infos:
                if not _wheel_member_is_safe(info) or info.flag_bits & 0x1:
                    raise ReleaseError("unsafe-wheel-member")
                if info.file_size > _MAX_WHEEL_MEMBER_BYTES:
                    raise ReleaseError("oversized-wheel-member")
                total += info.file_size
                if total > _MAX_WHEEL_TOTAL_BYTES:
                    raise ReleaseError("oversized-wheel")
                if info.compress_size and info.file_size > info.compress_size * 1000:
                    raise ReleaseError("wheel-compression-ratio")
            if any(
                PurePosixPath(name).name.casefold() == "direct_url.json"
                for name in names
            ):
                raise ReleaseError("wheel-direct-url-record")
            metadata_names = [
                name
                for name in names
                if _is_top_level_dist_info_member(name, "METADATA")
            ]
            record_names = [
                name for name in names if _is_top_level_dist_info_member(name, "RECORD")
            ]
            wheel_names = [
                name for name in names if _is_top_level_dist_info_member(name, "WHEEL")
            ]
            if not (
                len(metadata_names) == 1
                and len(record_names) == 1
                and len(wheel_names) == 1
            ):
                raise ReleaseError("incomplete-wheel-metadata")
            info_by_name = {info.filename: info for info in infos}
            if any(
                info_by_name[name].file_size > _MAX_LOCK_BYTES
                for name in (*metadata_names, *record_names, *wheel_names)
            ):
                raise ReleaseError("oversized-wheel-metadata")
            metadata = BytesParser(policy=email_policy).parsebytes(
                archive.read(metadata_names[0])
            )
            record_entries = _wheel_record_entries(
                archive,
                record_name=record_names[0],
                info_by_name=info_by_name,
            )
            generated_scripts = _wheel_generated_scripts(archive, info_by_name)
    except ReleaseError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ReleaseError("unreadable-wheel") from exc
    metadata_name = canonicalize_name(str(metadata.get("Name") or ""))
    metadata_version = _version_value(
        metadata.get("Version"), "invalid-wheel-metadata-version"
    )
    if (
        metadata_name != canonicalize_name(filename_name)
        or Version(metadata_version) != filename_version
    ):
        raise ReleaseError("wheel-identity-mismatch")
    if (
        _hash_regular(path, limit=_MAX_WHEEL_TOTAL_BYTES, code="unreadable-wheel")[0]
        != digest
    ):
        raise ReleaseError("wheel-digest-mismatch")
    return (
        metadata_name,
        metadata_version,
        record_entries,
        len(expanded_names),
        total,
        generated_scripts,
    )


def validate_wheelhouse(
    wheelhouse: Path, spec: ReleaseSpec
) -> tuple[dict[str, LockedRequirement], dict[str, WheelArtifact], bytes]:
    """Prove a flat, one-wheel-per-lock-entry, index-independent closure."""

    try:
        metadata = wheelhouse.lstat()
    except OSError as exc:
        raise ReleaseError("unavailable-wheelhouse") from exc
    if wheelhouse.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ReleaseError("unsafe-wheelhouse")
    entries: list[os.DirEntry[str]] = []
    iterator: Any = None
    try:
        iterator = os.scandir(wheelhouse)
        for entry in iterator:
            if len(entries) >= _MAX_RELEASE_FILES:
                raise ReleaseError("release-file-count-limit")
            entries.append(entry)
        entries.sort(key=lambda entry: entry.name)
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError("unavailable-wheelhouse") from exc
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    expected_names = {spec.requirements_file}
    wheel_paths: list[Path] = []
    wheelhouse_bytes = 0
    for entry in entries:
        if not _BASENAME.fullmatch(entry.name) or entry.is_symlink():
            raise ReleaseError("unsafe-wheelhouse-entry")
        entry_metadata = entry.stat(follow_symlinks=False)
        if not stat.S_ISREG(entry_metadata.st_mode):
            raise ReleaseError("non-regular-wheelhouse-entry")
        if entry.name == spec.requirements_file:
            expected_names.discard(entry.name)
        elif entry.name.endswith(".whl"):
            wheel_paths.append(Path(entry.path))
            wheelhouse_bytes += entry_metadata.st_size
            if wheelhouse_bytes > _MAX_WHEELHOUSE_BYTES:
                raise ReleaseError("wheelhouse-byte-budget-exceeded")
        else:
            raise ReleaseError("unexpected-wheelhouse-entry")
    if expected_names or not wheel_paths:
        raise ReleaseError("incomplete-wheelhouse")
    lock_payload = _read_regular(
        wheelhouse / spec.requirements_file,
        limit=_MAX_LOCK_BYTES,
        code="unreadable-requirements-lock",
    )
    if _sha256(lock_payload) != spec.requirements_digest:
        raise ReleaseError("requirements-digest-mismatch")
    locked = parse_requirements_lock(lock_payload)
    wheels: dict[str, WheelArtifact] = {}
    aggregate_member_count = 0
    aggregate_uncompressed_bytes = 0
    compatible_tags = set(sys_tags())
    for path in wheel_paths:
        digest, _size = _hash_regular(
            path, limit=_MAX_WHEEL_TOTAL_BYTES, code="unreadable-wheel"
        )
        try:
            filename_name, filename_version, _build, tags = parse_wheel_filename(
                path.name
            )
        except InvalidWheelFilename as exc:
            raise ReleaseError("invalid-wheel-filename") from exc
        name = canonicalize_name(filename_name)
        if tags.isdisjoint(compatible_tags):
            raise ReleaseError("wheel-platform-mismatch")
        requirement = locked.get(name)
        if requirement is None:
            raise ReleaseError("unlocked-wheel")
        if name in wheels:
            raise ReleaseError("multiple-wheels-for-package")
        if (
            Version(requirement.version) != filename_version
            or requirement.digest != digest
        ):
            raise ReleaseError("wheel-lock-mismatch")
        (
            metadata_name,
            metadata_version,
            record_entries,
            member_count,
            uncompressed_bytes,
            generated_scripts,
        ) = _inspect_wheel(path, digest=digest)
        aggregate_member_count += member_count
        aggregate_uncompressed_bytes += uncompressed_bytes
        if aggregate_member_count > _MAX_RELEASE_FILES:
            raise ReleaseError("release-file-count-limit")
        if (
            aggregate_uncompressed_bytes > _MAX_RELEASE_BYTES
            or aggregate_uncompressed_bytes + wheelhouse_bytes > _MAX_RELEASE_BYTES
        ):
            raise ReleaseError("release-byte-budget-exceeded")
        if metadata_name != name or metadata_version != requirement.version:
            raise ReleaseError("wheel-metadata-lock-mismatch")
        wheels[name] = WheelArtifact(
            name=name,
            version=requirement.version,
            filename=path.name,
            digest=digest,
            path=path,
            record_entries=record_entries,
            member_count=member_count,
            uncompressed_bytes=uncompressed_bytes,
            generated_scripts=generated_scripts,
        )
    if set(wheels) != set(locked):
        raise ReleaseError("wheelhouse-not-closed")
    for name, pin in spec.packages.items():
        requirement = locked.get(name)
        artifact = wheels.get(name)
        if requirement is None or artifact is None:
            raise ReleaseError("required-package-absent")
        if (
            requirement.version != pin.version
            or artifact.filename != pin.wheel
            or artifact.digest != pin.digest
        ):
            raise ReleaseError("required-package-mismatch")
    if locked["agent-utilities"].extras != frozenset({"serving"}):
        raise ReleaseError("agent-utilities-extras-mismatch")
    if locked["epistemic-graph"].extras != frozenset({"full"}):
        raise ReleaseError("epistemic-graph-extras-mismatch")
    if locked["langfuse-agent"].extras != frozenset({"mcp"}):
        raise ReleaseError("langfuse-agent-extras-mismatch")
    return locked, wheels, lock_payload


def _copy_regular(source: Path, destination: Path, *, digest: str) -> None:
    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    destination_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        source_fd = os.open(source, source_flags)
        destination_fd = os.open(destination, destination_flags, 0o400)
    except OSError as exc:
        if source_fd is not None:
            os.close(source_fd)
        raise ReleaseError("wheelhouse-staging-failed") from exc
    hasher = hashlib.sha256()
    try:
        source_metadata = os.fstat(source_fd)
        if not stat.S_ISREG(source_metadata.st_mode):
            raise ReleaseError("wheelhouse-source-changed")
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise ReleaseError("wheelhouse-staging-failed")
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        if (
            after.st_dev != source_metadata.st_dev
            or after.st_ino != source_metadata.st_ino
            or after.st_size != source_metadata.st_size
            or after.st_mtime_ns != source_metadata.st_mtime_ns
        ):
            raise ReleaseError("wheelhouse-source-changed")
    finally:
        os.close(source_fd)
        os.close(destination_fd)
    if "sha256:" + hasher.hexdigest() != digest:
        raise ReleaseError("staged-artifact-digest-mismatch")


def _invoke_bounded(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: int,
    role: str,
    input_payload: bytes | None = None,
    max_output_bytes: int = _MAX_COMMAND_OUTPUT_BYTES,
) -> CommandResult:
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    input_view = memoryview(input_payload or b"")

    def _terminate_group() -> None:
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass

    def _terminate_and_reap() -> None:
        _terminate_group()
        assert process is not None
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise ReleaseError(f"{role}-termination-failed") from exc

    try:
        try:
            process = subprocess.Popen(  # noqa: S603 - argv is strict and shell-free
                command,
                cwd=cwd,
                env=environment,
                stdin=(
                    subprocess.PIPE if input_payload is not None else subprocess.DEVNULL
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise ReleaseError(f"{role}-launch-failed") from exc
        if process.stdout is None or process.stderr is None:
            raise ReleaseError(f"{role}-launch-failed")
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, data=name)
        if process.stdin is not None:
            if input_view:
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, data="stdin")
            else:
                process.stdin.close()
        deadline = time.monotonic() + timeout_seconds
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_and_reap()
                raise ReleaseError(f"{role}-timeout")
            for key, _mask in selector.select(min(remaining, 0.25)):
                if key.data == "stdin":
                    try:
                        written = os.write(key.fileobj.fileno(), input_view[:65_536])
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        written = 0
                    if written <= 0:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        input_view = memoryview(b"")
                        continue
                    input_view = input_view[written:]
                    if not input_view:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                try:
                    chunk = os.read(key.fileobj.fileno(), 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                buffers[str(key.data)].extend(chunk)
                if sum(map(len, buffers.values())) > max_output_bytes:
                    _terminate_and_reap()
                    raise ReleaseError(f"{role}-output-invalid")
        try:
            return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            _terminate_and_reap()
            raise ReleaseError(f"{role}-timeout") from exc
        stdout = bytes(buffers["stdout"])
        stderr = bytes(buffers["stderr"])
        return CommandResult(
            return_code=return_code,
            output_digest=_sha256(stdout + b"\x00" + stderr),
            stdout=stdout,
        )
    finally:
        selector.close()
        if process is not None:
            # The direct child may have exited while descendants remain in its
            # session. Always kill the process group, even after wait() reaped
            # the leader, so no signer/proof descendant can survive this boundary.
            _terminate_group()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass


def _installer_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if (
            name.startswith(("PIP_", "UV_", "LD_", "DYLD_"))
            or name in _UNSAFE_PROCESS_ENV
        ):
            environment.pop(name, None)
    environment.update(
        {
            "PIP_NO_INDEX": "1",
            "PIP_REQUIRE_HASHES": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "UV_NO_CACHE": "1",
            "UV_NO_INDEX": "1",
            "UV_OFFLINE": "1",
        }
    )
    return environment


def _runtime_environment(runtime: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if (
            name.startswith(("PIP_", "UV_", "LD_", "DYLD_"))
            or name in _UNSAFE_PROCESS_ENV
        ):
            environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["VIRTUAL_ENV"] = os.fspath(runtime)
    environment["PATH"] = os.pathsep.join(
        (os.fspath(runtime / "bin"), environment.get("PATH", ""))
    )
    return environment


def _resolve_executable(value: str, *, code: str = "installer-unavailable") -> Path:
    candidate = shutil.which(value) if os.sep not in value else value
    if not candidate:
        raise ReleaseError(code)
    path = Path(candidate)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReleaseError(code) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or not os.access(path, os.X_OK)
    ):
        raise ReleaseError(code)
    return path


def _bind_executable(value: str, *, code: str) -> tuple[BoundExecutable, str]:
    """Copy one verified executable into a sealed anonymous Linux file."""

    _require_supported_platform()
    assert fcntl is not None
    path = _resolve_executable(value, code=code)
    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    source_fd: int | None = None
    bound_fd: int | None = None
    try:
        source_fd = os.open(path, source_flags)
        source_before = os.fstat(source_fd)
        by_path = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(source_before.st_mode)
            or source_before.st_dev != by_path.st_dev
            or source_before.st_ino != by_path.st_ino
            or not source_before.st_mode & 0o111
            or source_before.st_size > _MAX_WHEEL_MEMBER_BYTES
        ):
            raise ReleaseError(code)
        flags = os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
        bound_fd = os.memfd_create("exact-local-tool", flags)
        hasher = hashlib.sha256()
        consumed = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            consumed += len(chunk)
            if consumed > _MAX_WHEEL_MEMBER_BYTES:
                raise ReleaseError(code)
            hasher.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(bound_fd, view)
                if written <= 0:
                    raise ReleaseError(code)
                view = view[written:]
        source_after = os.fstat(source_fd)
        if (
            consumed != source_before.st_size
            or source_after.st_dev != source_before.st_dev
            or source_after.st_ino != source_before.st_ino
            or source_after.st_size != source_before.st_size
            or source_after.st_mtime_ns != source_before.st_mtime_ns
            or source_after.st_ctime_ns != source_before.st_ctime_ns
        ):
            raise ReleaseError(code)
        os.fchmod(bound_fd, 0o500)
        os.lseek(bound_fd, 0, os.SEEK_SET)
        seals = (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
        fcntl.fcntl(bound_fd, fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(bound_fd, fcntl.F_GET_SEALS) & seals != seals:
            raise ReleaseError(code)
        bound = BoundExecutable(bound_fd, _fd_path(bound_fd), path.resolve(strict=True))
        bound_fd = None
        return bound, "sha256:" + hasher.hexdigest()
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError(code) from exc
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if bound_fd is not None:
            os.close(bound_fd)


def _verify_toolchain(spec: ReleaseSpec) -> tuple[BoundExecutable, BoundExecutable]:
    """Bind immutable copies of both release-materializing executables."""

    python_tool: BoundExecutable | None = None
    uv_tool: BoundExecutable | None = None
    try:
        python_tool, python_digest = _bind_executable(
            sys.executable, code="python-unavailable"
        )
        if (
            platform.python_version() != spec.toolchain["python"].version
            or python_digest != spec.toolchain["python"].digest
        ):
            raise ReleaseError("python-toolchain-mismatch")
        uv_tool, uv_digest = _bind_executable("uv", code="installer-unavailable")
        if uv_digest != spec.toolchain["uv"].digest:
            raise ReleaseError("uv-toolchain-mismatch")
        version_result = _invoke_bounded(
            [os.fspath(uv_tool.proc_path), "--version"],
            cwd=Path("/"),
            environment=_installer_environment(),
            timeout_seconds=30,
            role="uv-identity",
        )
        try:
            tokens = version_result.stdout.decode("ascii").strip().split()
        except UnicodeDecodeError as exc:
            raise ReleaseError("uv-toolchain-mismatch") from exc
        if (
            version_result.return_code != 0
            or len(tokens) < 2
            or tokens[0] != "uv"
            or tokens[1] != spec.toolchain["uv"].version
        ):
            raise ReleaseError("uv-toolchain-mismatch")
        return python_tool, uv_tool
    except Exception:
        for tool in (python_tool, uv_tool):
            if tool is not None:
                os.close(tool.descriptor)
        raise


def _stage_wheelhouse(
    release_root: Path,
    spec: ReleaseSpec,
    wheels: dict[str, WheelArtifact],
    lock_payload: bytes,
) -> Path:
    staged = release_root / ".wheelhouse"
    try:
        staged.mkdir(mode=0o700)
    except OSError as exc:
        raise ReleaseError("wheelhouse-staging-failed") from exc
    lock_destination = staged / spec.requirements_file
    try:
        lock_destination.write_bytes(lock_payload)
        os.chmod(lock_destination, 0o400)
    except OSError as exc:
        raise ReleaseError("wheelhouse-staging-failed") from exc
    if (
        _sha256(
            _read_regular(
                lock_destination,
                limit=_MAX_LOCK_BYTES,
                code="staged-requirements-invalid",
            )
        )
        != spec.requirements_digest
    ):
        raise ReleaseError("staged-requirements-invalid")
    for artifact in wheels.values():
        _copy_regular(
            artifact.path,
            staged / artifact.filename,
            digest=artifact.digest,
        )
    return staged


def _remove_staged_wheelhouse(staged: Path) -> None:
    try:
        staged_metadata = staged.lstat()
        if staged.is_symlink() or not stat.S_ISDIR(staged_metadata.st_mode):
            raise ReleaseError("staged-wheelhouse-mutated")
        entries = list(os.scandir(staged))
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise ReleaseError("staged-wheelhouse-mutated")
            os.unlink(entry.path)
        os.rmdir(staged)
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError("staged-wheelhouse-cleanup-failed") from exc


def _normalize_venv_metadata(
    runtime: Path,
    *,
    persistent_runtime: Path,
    python_tool: BoundExecutable,
) -> None:
    """Replace transient descriptor paths emitted by stdlib venv metadata."""

    transient_runtime = os.fspath(runtime).encode()
    persistent = os.fspath(persistent_runtime).encode()
    transient_python = os.fspath(python_tool.proc_path).encode()
    transient_prefix = f"/proc/{os.getpid()}/fd/".encode()
    source_python = os.fspath(python_tool.source_path).encode()
    config = runtime / "pyvenv.cfg"
    activation_files = (
        runtime / "bin" / "activate",
        runtime / "bin" / "activate.csh",
        runtime / "bin" / "activate.fish",
        runtime / "bin" / "Activate.ps1",
    )
    try:
        lines = _read_regular(
            config, limit=_MAX_LOCK_BYTES, code="runtime-metadata-invalid"
        ).splitlines(keepends=True)
        normalized_lines: list[bytes] = []
        for line in lines:
            lowered = line.casefold()
            newline = b"\n" if line.endswith(b"\n") else b""
            if lowered.startswith(b"home = "):
                line = (
                    b"home = "
                    + os.fspath(python_tool.source_path.parent).encode()
                    + newline
                )
            elif lowered.startswith(b"executable = "):
                line = b"executable = " + source_python + newline
            else:
                line = line.replace(transient_python, source_python).replace(
                    transient_runtime, persistent
                )
            normalized_lines.append(line)
        config.write_bytes(b"".join(normalized_lines))
        for path in activation_files:
            if not path.exists():
                continue
            payload = _read_regular(
                path, limit=_MAX_LOCK_BYTES, code="runtime-metadata-invalid"
            )
            path.write_bytes(payload.replace(transient_runtime, persistent))
        for path in (config, *activation_files):
            if not path.exists():
                continue
            payload = _read_regular(
                path, limit=_MAX_LOCK_BYTES, code="runtime-metadata-invalid"
            )
            if transient_prefix in payload or b"/memfd:exact-local-tool" in payload:
                raise ReleaseError("transient-runtime-reference")
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError("runtime-metadata-invalid") from exc


def _reject_transient_command_references(runtime: Path) -> None:
    """Ensure installed launchers do not depend on this promoter process."""

    transient_prefix = f"/proc/{os.getpid()}/fd/".encode()
    try:
        for path in (runtime / "bin").iterdir():
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_LOCK_BYTES:
                continue
            payload = _read_regular(
                path, limit=_MAX_LOCK_BYTES, code="release-command-invalid"
            )
            if transient_prefix in payload or b"/memfd:exact-local-tool" in payload:
                raise ReleaseError("transient-runtime-reference")
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError("release-command-invalid") from exc


def _create_runtime(
    release_root: Path,
    staged_wheelhouse: Path,
    spec: ReleaseSpec,
    *,
    persistent_release_root: Path,
    python_tool: BoundExecutable,
    uv_tool: BoundExecutable,
) -> dict[str, Any]:
    runtime = release_root / "runtime"
    environment = _installer_environment()
    venv_result = _invoke_bounded(
        [
            os.fspath(python_tool.proc_path),
            "-I",
            "-m",
            "venv",
            "--copies",
            "--without-pip",
            os.fspath(runtime),
        ],
        cwd=release_root,
        environment=environment,
        timeout_seconds=120,
        role="venv",
    )
    if venv_result.return_code != 0:
        raise ReleaseError("venv-creation-failed")
    lib64 = runtime / "lib64"
    if lib64.is_symlink():
        try:
            if os.readlink(lib64) != "lib":
                raise ReleaseError("unexpected-venv-symlink")
            lib64.unlink()
        except OSError as exc:
            raise ReleaseError("unexpected-venv-symlink") from exc
    elif lib64.exists():
        raise ReleaseError("unexpected-venv-lib64")
    _normalize_venv_metadata(
        runtime,
        persistent_runtime=persistent_release_root / "runtime",
        python_tool=python_tool,
    )
    runtime_python = runtime / "bin" / "python"
    runtime_python_digest, _ = _hash_regular(
        runtime_python,
        limit=_MAX_WHEEL_MEMBER_BYTES,
        code="runtime-python-invalid",
    )
    if runtime_python_digest != spec.toolchain["python"].digest:
        raise ReleaseError("runtime-python-digest-mismatch")
    baseline = _content_snapshot(runtime)
    install_result = _invoke_bounded(
        [
            os.fspath(uv_tool.proc_path),
            "--no-cache",
            "pip",
            "install",
            "--python",
            os.fspath(runtime / "bin" / "python"),
            "--offline",
            "--no-index",
            "--find-links",
            os.fspath(staged_wheelhouse),
            "--require-hashes",
            "--no-deps",
            "--link-mode",
            "copy",
            "--requirement",
            os.fspath(staged_wheelhouse / spec.requirements_file),
        ],
        cwd=release_root,
        environment=environment,
        timeout_seconds=_MAX_INSTALL_SECONDS,
        role="install",
    )
    if install_result.return_code != 0:
        raise ReleaseError("offline-install-failed")
    _verify_content_snapshot(runtime, baseline)
    _reject_transient_command_references(runtime)
    return {
        "venv": {
            "status": "passed",
            "exitCode": venv_result.return_code,
            "outputDigest": _sha256(
                _canonical_json({"role": "venv", "exitCode": venv_result.return_code})
            ),
        },
        "install": {
            "status": "passed",
            "exitCode": install_result.return_code,
            "outputDigest": _sha256(
                _canonical_json(
                    {"role": "install", "exitCode": install_result.return_code}
                )
            ),
        },
    }


def _scan_regular_tree(root: Path) -> list[Path]:
    regular: list[Path] = []
    casefolded_paths: set[str] = set()
    total_bytes = 0
    try:
        root_resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ReleaseError("release-tree-unavailable") from exc
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        for name in [*names, *files]:
            if any(ord(character) < 32 for character in name) or "\\" in name:
                raise ReleaseError("unsafe-installed-path")
            candidate = directory_path / name
            try:
                metadata = candidate.lstat()
            except OSError as exc:
                raise ReleaseError("release-tree-race") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ReleaseError("installed-symlink")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ReleaseError("installed-special-file")
            if metadata.st_nlink != 1:
                raise ReleaseError("installed-hardlink")
            total_bytes += metadata.st_size
            if len(regular) + 1 > _MAX_RELEASE_FILES:
                raise ReleaseError("release-file-count-limit")
            if total_bytes > _MAX_RELEASE_BYTES:
                raise ReleaseError("release-byte-budget-exceeded")
            try:
                resolved = candidate.resolve(strict=True)
                relative = resolved.relative_to(root_resolved).as_posix().casefold()
            except (OSError, ValueError) as exc:
                raise ReleaseError("installed-path-escape") from exc
            if relative in casefolded_paths:
                raise ReleaseError("installed-path-case-collision")
            casefolded_paths.add(relative)
            if candidate.name.casefold() == "direct_url.json":
                raise ReleaseError("installed-direct-url-record")
            regular.append(candidate)
    return regular


def _content_snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    """Bind existing regular files before an installer is allowed to run."""

    resolved_root = root.resolve(strict=True)
    result: dict[str, tuple[int, int, str]] = {}
    for path in _scan_regular_tree(root):
        metadata = path.lstat()
        relative = path.resolve(strict=True).relative_to(resolved_root).as_posix()
        digest, size = _hash_regular(
            path, limit=_MAX_WHEEL_MEMBER_BYTES, code="runtime-baseline-invalid"
        )
        result[relative] = (stat.S_IMODE(metadata.st_mode), size, digest)
    return result


def _verify_content_snapshot(
    root: Path, expected: dict[str, tuple[int, int, str]]
) -> None:
    current = _content_snapshot(root)
    for relative, identity in expected.items():
        if current.get(relative) != identity:
            raise ReleaseError("runtime-baseline-mutated")


def _seal_release_tree(root: Path) -> None:
    """Remove write authority from the completed candidate before activation."""

    regular = _scan_regular_tree(root)
    directories: list[Path] = []
    for directory, names, _files in os.walk(root, topdown=False, followlinks=False):
        directory_path = Path(directory)
        directories.extend(directory_path / name for name in names)
        directories.append(directory_path)
    try:
        for path in regular:
            mode = path.lstat().st_mode
            os.chmod(path, 0o555 if mode & 0o111 else 0o444, follow_symlinks=False)
        for directory in directories:
            is_descriptor_root = directory == root
            metadata = (
                directory.stat(follow_symlinks=True)
                if is_descriptor_root
                else directory.lstat()
            )
            if (not is_descriptor_root and directory.is_symlink()) or not stat.S_ISDIR(
                metadata.st_mode
            ):
                raise ReleaseError("release-seal-failed")
            os.chmod(directory, 0o555, follow_symlinks=is_descriptor_root)
    except OSError as exc:
        raise ReleaseError("release-seal-failed") from exc


def _verify_release_sealed(root: Path) -> None:
    for path in _scan_regular_tree(root):
        if stat.S_IMODE(path.lstat().st_mode) & 0o222:
            raise ReleaseError("release-tree-writable")
    for directory, names, _files in os.walk(root, topdown=True, followlinks=False):
        candidates = [Path(directory), *(Path(directory) / name for name in names)]
        for candidate in candidates:
            metadata = (
                candidate.stat(follow_symlinks=True)
                if candidate == root
                else candidate.lstat()
            )
            if (
                stat.S_ISDIR(metadata.st_mode)
                and stat.S_IMODE(metadata.st_mode) & 0o222
            ):
                raise ReleaseError("release-tree-writable")


def _installed_agent_tree_identity(
    *,
    installed_record: dict[str, tuple[str, int]],
    site_packages: Path,
    release_root: Path,
) -> tuple[str, int]:
    # Keep this byte-for-byte identical to the exact-local campaign identity.
    digest = hashlib.sha256(b"agent-utilities-installed-release-v2\0")
    count = 0
    for name in sorted(installed_record):
        if not (
            name.startswith("agent_utilities/")
            or ".dist-info/" in name
            or name.endswith("/graph-os")
        ):
            continue
        path = _resolve_record_path(site_packages, release_root, name)
        content_digest, size = _hash_regular(
            path, limit=_MAX_WHEEL_MEMBER_BYTES, code="installed-file-invalid"
        )
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_digest.removeprefix("sha256:").encode("ascii"))
        digest.update(b"\0")
        count += 1
    if count < 10:
        raise ReleaseError("release-distribution-incomplete")
    return digest.hexdigest(), count


def _installed_closure_identity(
    *,
    distributions: dict[str, tuple[str, Path, Any]],
    installed_records: dict[str, dict[str, tuple[str, int]]],
    site_packages: Path,
    release_root: Path,
) -> str:
    """Produce the exact-local campaign's path-independent closure identity."""

    digest = hashlib.sha256(b"agent-utilities-installed-closure-v1\0")
    for distribution_name, (version, _distribution, _metadata) in sorted(
        distributions.items()
    ):
        digest.update(distribution_name.encode("utf-8"))
        digest.update(b"==")
        digest.update(version.encode("utf-8"))
        digest.update(b"\0")
        for name in sorted(installed_records[distribution_name]):
            path = _resolve_record_path(site_packages, release_root, name)
            content_digest, size = _hash_regular(
                path,
                limit=_MAX_WHEEL_MEMBER_BYTES,
                code="installed-file-invalid",
            )
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\0")
            digest.update(content_digest.removeprefix("sha256:").encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _certification_artifacts(
    release_root: Path,
    *,
    agent_utilities_sha256: str,
    agent_utilities_file_count: int,
    distribution_closure_sha256: str,
) -> dict[str, Any]:
    runtime_bin = release_root / "runtime" / "bin"
    paths = {
        "releasePythonSha256": runtime_bin / "python",
        "graphosSha256": runtime_bin / "graph-os",
        "engineSha256": runtime_bin / "epistemic-graph-server",
    }
    result: dict[str, Any] = {
        "agentUtilitiesSha256": agent_utilities_sha256,
        "agentUtilitiesFileCount": agent_utilities_file_count,
        "distributionClosureSha256": distribution_closure_sha256,
    }
    for field, path in paths.items():
        digest, _ = _hash_regular(
            path, limit=_MAX_WHEEL_MEMBER_BYTES, code="certification-artifact-invalid"
        )
        result[field] = digest.removeprefix("sha256:")
    return result


def attest_installed_release(release_root: Path) -> dict[str, Any]:
    """Recompute the sealed installed-release identities without build inputs.

    Runtime certification no longer has the wheelhouse that promotion used, so
    this verifier derives every identity from installed ``RECORD`` ownership and
    the exact runtime executables.  Matching the signed promotion identities then
    proves that the active Python tree is the promoted release, not merely a copy
    of its generic console launcher.
    """

    try:
        root_metadata = release_root.lstat()
    except OSError as exc:
        raise ReleaseError("installed-attestation-root-invalid") from exc
    if (
        not release_root.is_absolute()
        or release_root.is_symlink()
        or not stat.S_ISDIR(root_metadata.st_mode)
    ):
        raise ReleaseError("installed-attestation-root-invalid")
    _verify_release_sealed(release_root)
    all_regular = _scan_regular_tree(release_root)
    runtime = release_root / "runtime"
    site_packages = _site_packages(runtime)
    distributions: dict[str, tuple[str, Path, Any]] = {}
    for distribution in site_packages.glob("*.dist-info"):
        if distribution.is_symlink() or not distribution.is_dir():
            raise ReleaseError("installed-attestation-dist-info-invalid")
        payload = _read_regular(
            distribution / "METADATA",
            limit=_MAX_LOCK_BYTES,
            code="installed-attestation-metadata-invalid",
        )
        metadata = BytesParser(policy=email_policy).parsebytes(payload)
        name = canonicalize_name(str(metadata.get("Name") or ""))
        version = _version_value(
            metadata.get("Version"), "installed-attestation-version-invalid"
        )
        if not name or name in distributions:
            raise ReleaseError("installed-attestation-distribution-invalid")
        distributions[name] = (version, distribution, metadata)
    if "agent-utilities" not in distributions:
        raise ReleaseError("installed-attestation-agent-utilities-missing")

    recorded: set[Path] = set()
    installed_records: dict[str, dict[str, tuple[str, int]]] = {}
    for name, (_version, distribution, _metadata) in distributions.items():
        entries, installed_record = _verify_record(
            distribution,
            site_packages=site_packages,
            release_root=release_root,
        )
        if recorded & entries:
            raise ReleaseError("installed-attestation-record-overlap")
        recorded.update(entries)
        installed_records[name] = installed_record
    site_packages_resolved = site_packages.resolve(strict=True)
    site_regular = {
        path.resolve(strict=True)
        for path in all_regular
        if site_packages in path.parents
    }
    if site_regular != {
        path for path in recorded if site_packages_resolved in path.parents
    }:
        raise ReleaseError("installed-attestation-record-coverage-invalid")

    agent_digest, agent_file_count = _installed_agent_tree_identity(
        installed_record=installed_records["agent-utilities"],
        site_packages=site_packages,
        release_root=release_root,
    )
    closure_digest = _installed_closure_identity(
        distributions=distributions,
        installed_records=installed_records,
        site_packages=site_packages,
        release_root=release_root,
    )
    result = _certification_artifacts(
        release_root,
        agent_utilities_sha256=agent_digest,
        agent_utilities_file_count=agent_file_count,
        distribution_closure_sha256=closure_digest,
    )
    _verify_release_sealed(release_root)
    return result


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, int, int, int, int, int]]:
    """Capture metadata for every validated entry without retaining absolute paths."""

    _scan_regular_tree(root)
    resolved_root = root.resolve(strict=True)
    root_metadata = root.stat(follow_symlinks=True)
    snapshot: dict[str, tuple[int, int, int, int, int, int, int]] = {
        ".": (
            stat.S_IFMT(root_metadata.st_mode),
            stat.S_IMODE(root_metadata.st_mode),
            root_metadata.st_size,
            root_metadata.st_mtime_ns,
            root_metadata.st_ctime_ns,
            root_metadata.st_dev,
            root_metadata.st_ino,
        )
    }
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        for name in [*names, *files]:
            candidate = directory_path / name
            metadata = candidate.lstat()
            relative = (
                candidate.resolve(strict=True).relative_to(resolved_root).as_posix()
            )
            snapshot[relative] = (
                stat.S_IFMT(metadata.st_mode),
                stat.S_IMODE(metadata.st_mode),
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
                metadata.st_dev,
                metadata.st_ino,
            )
    return snapshot


def _site_packages(runtime: Path) -> Path:
    candidates = [
        path
        for path in (runtime / "lib").glob("python*/site-packages")
        if path.is_dir() and not path.is_symlink()
    ]
    if len(candidates) != 1:
        raise ReleaseError("site-packages-layout-mismatch")
    return candidates[0]


def _file_hash(path: Path) -> tuple[str, int]:
    digest, size = _hash_regular(
        path, limit=_MAX_WHEEL_MEMBER_BYTES, code="installed-file-invalid"
    )
    encoded = base64.urlsafe_b64encode(bytes.fromhex(digest.removeprefix("sha256:")))
    return "sha256=" + encoded.rstrip(b"=").decode("ascii"), size


def _resolve_record_path(site_packages: Path, release_root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if not value or "\\" in value or "\x00" in value or relative.is_absolute():
        raise ReleaseError("unsafe-record-path")
    candidate = site_packages.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(release_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ReleaseError("record-path-escape") from exc
    return resolved


def _verify_record(
    distribution: Path,
    *,
    site_packages: Path,
    release_root: Path,
) -> tuple[set[Path], dict[str, tuple[str, int]]]:
    record = distribution / "RECORD"
    payload = _read_regular(record, limit=_MAX_LOCK_BYTES, code="record-unavailable")
    try:
        rows = list(csv.reader(payload.decode("utf-8").splitlines()))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ReleaseError("invalid-record") from exc
    if not rows:
        raise ReleaseError("empty-record")
    observed: set[str] = set()
    resolved_paths: set[Path] = set()
    identities: dict[str, tuple[str, int]] = {}
    record_resolved = record.resolve(strict=True)
    for row in rows:
        if len(row) != 3 or row[0] in observed:
            raise ReleaseError("invalid-record-row")
        observed.add(row[0])
        candidate = _resolve_record_path(site_packages, release_root, row[0])
        metadata = candidate.lstat()
        if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ReleaseError("record-non-regular-file")
        resolved_paths.add(candidate)
        digest, size = row[1], row[2]
        if candidate == record_resolved:
            if digest or size:
                raise ReleaseError("record-self-hash-present")
            identities[row[0]] = ("", 0)
            continue
        if not digest.startswith("sha256=") or not size.isdigit():
            raise ReleaseError("record-hash-missing")
        expected_digest, expected_size = _file_hash(candidate)
        if digest != expected_digest or int(size) != expected_size:
            raise ReleaseError("record-content-mismatch")
        identities[row[0]] = (digest, int(size))
    if record_resolved not in resolved_paths:
        raise ReleaseError("record-self-entry-missing")
    return resolved_paths, identities


def _verify_installed_record_matches_wheel(
    *,
    artifact: WheelArtifact,
    installed: dict[str, tuple[str, int]],
    site_packages: Path,
    release_root: Path,
) -> None:
    """Reject installed bytes that are not bound by the locked wheel RECORD."""

    expected = artifact.record_entries
    relocated_names: set[str] = set()
    for name, identity in expected.items():
        match = re.match(r"^[^/]+\.data/(purelib|platlib|scripts)/(.*)$", name)
        installed_name = name
        if match is not None:
            scheme, suffix = match.groups()
            if not suffix:
                raise ReleaseError("wheel-record-layout-mismatch")
            if scheme == "scripts":
                relocated = "../../../bin/" + PurePosixPath(suffix).name
                if relocated not in installed:
                    raise ReleaseError("installed-wheel-record-mismatch")
                relocated_names.add(relocated)
                continue
            installed_name = suffix
            relocated_names.add(installed_name)
        elif ".data/" in name:
            raise ReleaseError("unsupported-wheel-data-scheme")
        if installed.get(installed_name) != identity:
            raise ReleaseError("installed-wheel-record-mismatch")
    generated_names = {"../../../bin/" + name for name in artifact.generated_scripts}
    if not generated_names <= set(installed):
        raise ReleaseError("installed-wheel-record-mismatch")
    extras = set(installed) - set(expected) - relocated_names
    distribution_prefixes = {
        name.rsplit("/", 1)[0]
        for name in expected
        if name.endswith(".dist-info/RECORD")
    }
    if len(distribution_prefixes) != 1:
        raise ReleaseError("wheel-record-layout-mismatch")
    distribution_prefix = next(iter(distribution_prefixes))
    for name in extras:
        if name in generated_names:
            continue
        if name not in {
            f"{distribution_prefix}/INSTALLER",
            f"{distribution_prefix}/REQUESTED",
        }:
            raise ReleaseError("installed-file-not-bound-to-wheel")
        candidate = _resolve_record_path(site_packages, release_root, name)
        payload = _read_regular(candidate, limit=64, code="installer-marker-invalid")
        if name.endswith("/INSTALLER") and payload not in {b"uv\n", b"uv"}:
            raise ReleaseError("installer-marker-invalid")
        if name.endswith("/REQUESTED") and payload:
            raise ReleaseError("installer-marker-invalid")


def _metadata_requirement_ready(
    metadata: Any,
    *,
    dependency: str,
    extra: str,
    selected_version: str,
) -> bool:
    for value in metadata.get_all("Requires-Dist") or ():
        try:
            requirement = Requirement(value)
        except InvalidRequirement:
            continue
        if (
            canonicalize_name(requirement.name) == dependency
            and extra in {canonicalize_name(item) for item in requirement.extras}
            and Version(selected_version) in requirement.specifier
        ):
            return True
    return False


def _verify_dependency_closure(
    distributions: dict[str, tuple[str, Path, Any]],
    locked: dict[str, LockedRequirement],
) -> int:
    """Prove every reachable metadata edge and reject unrelated lock entries."""

    parsed: dict[str, list[Requirement]] = {}
    provided: dict[str, set[str]] = {}
    for name, (_version, _distribution, metadata) in distributions.items():
        requirements: list[Requirement] = []
        for value in metadata.get_all("Requires-Dist") or ():
            try:
                requirement = Requirement(value)
            except InvalidRequirement as exc:
                raise ReleaseError("installed-requirement-invalid") from exc
            if requirement.url is not None:
                raise ReleaseError("installed-direct-requirement")
            requirements.append(requirement)
        parsed[name] = requirements
        provided[name] = {
            canonicalize_name(item) for item in metadata.get_all("Provides-Extra") or ()
        }

    contexts: list[tuple[str, str]] = []
    queued_contexts: set[tuple[str, str]] = set()
    reachable = set(_REQUIRED_PACKAGES)
    active_extras: dict[str, set[str]] = {name: set() for name in locked}

    def _activate(name: str, extras: Iterable[str] = ()) -> None:
        base_context = (name, "")
        if base_context not in queued_contexts:
            queued_contexts.add(base_context)
            contexts.append(base_context)
        requested = {canonicalize_name(extra) for extra in extras}
        if not requested <= provided[name]:
            raise ReleaseError("installed-extra-metadata-mismatch")
        for extra in sorted(requested - active_extras[name]):
            active_extras[name].add(extra)
            context = (name, extra)
            if context not in queued_contexts:
                queued_contexts.add(context)
                contexts.append(context)

    for name in _REQUIRED_PACKAGES:
        if name not in distributions:
            raise ReleaseError("required-package-absent")
        _activate(name, locked[name].extras)

    environment = default_environment()
    processed_contexts: set[tuple[str, str]] = set()
    processed_edges: set[tuple[str, str]] = set()
    cursor = 0
    while cursor < len(contexts):
        if len(contexts) > _MAX_DEPENDENCY_CONTEXTS:
            raise ReleaseError("dependency-closure-too-large")
        name, extra = contexts[cursor]
        cursor += 1
        context = (name, extra)
        if context in processed_contexts:
            continue
        processed_contexts.add(context)
        marker_environment = dict(environment)
        marker_environment["extra"] = extra
        for requirement in parsed[name]:
            if requirement.marker is not None:
                try:
                    active = requirement.marker.evaluate(marker_environment)
                except Exception as exc:  # noqa: BLE001 - malformed marker is terminal
                    raise ReleaseError("installed-marker-invalid") from exc
                if not active:
                    continue
            edge = (name, str(requirement))
            if edge in processed_edges:
                continue
            processed_edges.add(edge)
            dependency = canonicalize_name(requirement.name)
            locked_dependency = locked.get(dependency)
            if locked_dependency is None or dependency not in distributions:
                raise ReleaseError("dependency-closure-incomplete")
            if Version(locked_dependency.version) not in requirement.specifier:
                raise ReleaseError("dependency-version-mismatch")
            first_reach = dependency not in reachable
            reachable.add(dependency)
            requested_extras = {canonicalize_name(item) for item in requirement.extras}
            if first_reach:
                _activate(dependency, requested_extras)
            else:
                _activate(dependency, requested_extras - active_extras[dependency])
    if reachable != set(locked):
        raise ReleaseError("dependency-closure-not-minimal")
    if any(
        set(requirement.extras) - active_extras[name]
        for name, requirement in locked.items()
        if name not in _REQUIRED_PACKAGES
    ):
        raise ReleaseError("locked-extra-not-reachable")
    return len(processed_edges)


def verify_installed_release(
    release_root: Path,
    spec: ReleaseSpec,
    locked: dict[str, LockedRequirement],
    wheels: dict[str, WheelArtifact],
) -> dict[str, Any]:
    """Verify versions, RECORD coverage, native digests, and binary shape."""

    runtime = release_root / "runtime"
    all_regular = _scan_regular_tree(release_root)
    site_packages = _site_packages(runtime)
    distributions: dict[str, tuple[str, Path, Any]] = {}
    for distribution in site_packages.glob("*.dist-info"):
        if distribution.is_symlink() or not distribution.is_dir():
            raise ReleaseError("invalid-dist-info")
        payload = _read_regular(
            distribution / "METADATA",
            limit=_MAX_LOCK_BYTES,
            code="metadata-unavailable",
        )
        metadata = BytesParser(policy=email_policy).parsebytes(payload)
        name = canonicalize_name(str(metadata.get("Name") or ""))
        version = _version_value(metadata.get("Version"), "invalid-installed-version")
        if not name or name in distributions:
            raise ReleaseError("duplicate-installed-distribution")
        distributions[name] = (version, distribution, metadata)
    if set(distributions) != set(locked):
        raise ReleaseError("installed-closure-mismatch")
    dependency_edge_count = _verify_dependency_closure(distributions, locked)
    recorded: set[Path] = set()
    installed_records: dict[str, dict[str, tuple[str, int]]] = {}
    for name, requirement in locked.items():
        version, distribution, _metadata = distributions[name]
        if version != requirement.version:
            raise ReleaseError("installed-version-mismatch")
        entries, installed_record = _verify_record(
            distribution,
            site_packages=site_packages,
            release_root=release_root,
        )
        _verify_installed_record_matches_wheel(
            artifact=wheels[name],
            installed=installed_record,
            site_packages=site_packages,
            release_root=release_root,
        )
        installed_records[name] = installed_record
        if recorded & entries:
            raise ReleaseError("record-ownership-overlap")
        recorded.update(entries)
    site_packages_resolved = site_packages.resolve(strict=True)
    site_regular = {
        path.resolve(strict=True)
        for path in all_regular
        if site_packages in path.parents
    }
    if site_regular != {
        path for path in recorded if site_packages_resolved in path.parents
    }:
        raise ReleaseError("unrecorded-site-package-file")
    baseline_bin = {
        "activate",
        "activate.csh",
        "activate.fish",
        "Activate.ps1",
        "python",
        "python3",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
    }
    runtime_bin = runtime / "bin"
    baseline_paths = {
        path.resolve(strict=True)
        for name in baseline_bin
        if (path := runtime_bin / name).exists()
    }
    baseline_paths.update(
        path.resolve(strict=True)
        for path in (runtime / "pyvenv.cfg", runtime / ".gitignore")
        if path.exists()
    )
    if recorded & baseline_paths:
        raise ReleaseError("record-overwrites-runtime-baseline")
    for path in all_regular:
        if path.parent == runtime_bin and path.name not in baseline_bin:
            if path.resolve(strict=True) not in recorded:
                raise ReleaseError("unrecorded-runtime-command")
    for name, pin in spec.packages.items():
        if distributions[name][0] != pin.version:
            raise ReleaseError("required-installed-version-mismatch")
    agent_metadata = distributions["agent-utilities"][2]
    if not _metadata_requirement_ready(
        agent_metadata,
        dependency="epistemic-graph",
        extra="full",
        selected_version=spec.packages["epistemic-graph"].version,
    ):
        raise ReleaseError("agent-engine-metadata-mismatch")
    if not _metadata_requirement_ready(
        agent_metadata,
        dependency="langfuse-agent",
        extra="mcp",
        selected_version=spec.packages["langfuse-agent"].version,
    ):
        raise ReleaseError("agent-langfuse-metadata-mismatch")
    engine_metadata = distributions["epistemic-graph"][2]
    provided_extras = {
        canonicalize_name(item)
        for item in engine_metadata.get_all("Provides-Extra") or ()
    }
    if "full" not in provided_extras or "numeric" not in provided_extras:
        raise ReleaseError("engine-extra-metadata-mismatch")
    engine_binary = runtime_bin / "epistemic-graph-server"
    try:
        binary_metadata = engine_binary.lstat()
    except OSError as exc:
        raise ReleaseError("engine-binary-missing") from exc
    if (
        engine_binary.is_symlink()
        or not stat.S_ISREG(binary_metadata.st_mode)
        or not binary_metadata.st_mode & 0o111
    ):
        raise ReleaseError("engine-binary-invalid")
    binary_digest, _binary_size = _hash_regular(
        engine_binary,
        limit=_MAX_WHEEL_MEMBER_BYTES,
        code="engine-binary-invalid",
    )
    if binary_digest != spec.native_artifacts["epistemic-graph-server"]:
        raise ReleaseError("engine-binary-digest-mismatch")
    numeric_candidates = [
        path
        for path in (site_packages / "epistemic_graph").glob("numeric*.so")
        if path.is_file() and not path.is_symlink()
    ]
    if len(numeric_candidates) != 1:
        raise ReleaseError("numeric-extension-layout-mismatch")
    numeric_digest, _numeric_size = _hash_regular(
        numeric_candidates[0],
        limit=_MAX_WHEEL_MEMBER_BYTES,
        code="numeric-extension-invalid",
    )
    if numeric_digest != spec.native_artifacts["epistemic-graph-numeric"]:
        raise ReleaseError("numeric-extension-digest-mismatch")
    for entry_point in _COMMANDS.values():
        command = runtime_bin / entry_point
        try:
            command_metadata = command.lstat()
        except OSError as exc:
            raise ReleaseError("release-command-missing") from exc
        if (
            command.is_symlink()
            or not stat.S_ISREG(command_metadata.st_mode)
            or not command_metadata.st_mode & 0o111
        ):
            raise ReleaseError("release-command-invalid")
    agent_utilities_sha256, agent_utilities_file_count = _installed_agent_tree_identity(
        installed_record=installed_records["agent-utilities"],
        site_packages=site_packages,
        release_root=release_root,
    )
    distribution_closure_sha256 = _installed_closure_identity(
        distributions=distributions,
        installed_records=installed_records,
        site_packages=site_packages,
        release_root=release_root,
    )
    return {
        "distributionCount": len(distributions),
        "recordVerified": True,
        "directUrlRecordCount": 0,
        "symlinkCount": 0,
        "specialFileCount": 0,
        "nativeArtifactCount": len(_NATIVE_ARTIFACTS),
        "dependencyEdgeCount": dependency_edge_count,
        "agentUtilitiesSha256": agent_utilities_sha256,
        "agentUtilitiesFileCount": agent_utilities_file_count,
        "distributionClosureSha256": distribution_closure_sha256,
    }


def _process_identity(pid: str) -> bool:
    process = Path("/proc") / pid
    try:
        executable = Path(os.readlink(process / "exe")).name
    except OSError:
        executable = ""
    try:
        with (process / "cmdline").open("rb") as command_file:
            command_line = command_file.read(65_536).split(b"\x00")
        arguments = [item.decode("utf-8", "ignore") for item in command_line if item]
    except OSError:
        arguments = []
    candidates = {executable}
    if arguments:
        candidates.add(Path(arguments[0]).name)
    if len(arguments) > 1 and Path(arguments[0]).name.startswith("python"):
        candidates.add(Path(arguments[1]).name)
    if candidates & _PROCESS_EXECUTABLES:
        return True
    for index, argument in enumerate(arguments[:-1]):
        if argument == "-m" and arguments[index + 1] in _PROCESS_MODULES:
            return True
    if any(
        argument.startswith(
            ("agent_utilities.gateway:", "agent_utilities.mcp.kg_server:")
        )
        for argument in arguments
    ):
        return True
    return False


def running_graph_process_count() -> int:
    """Return an aggregate count without exposing process IDs or command lines."""

    own_pid = str(os.getpid())
    try:
        pids = [entry.name for entry in os.scandir("/proc") if entry.name.isdigit()]
    except OSError as exc:
        raise ReleaseError("process-gate-unavailable") from exc
    return sum(pid != own_pid and _process_identity(pid) for pid in pids)


def _open_releases_root(path: Path) -> int:
    if not path.is_absolute():
        raise ReleaseError("releases-root-must-be-absolute")
    try:
        metadata = path.lstat()
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise ReleaseError("releases-root-unavailable") from exc
    if (
        canonical != path
        or path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ReleaseError("unsafe-releases-root")
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise ReleaseError("releases-root-unavailable") from exc


def _assert_root_binding(path: Path, root_fd: int) -> None:
    try:
        by_path = path.stat(follow_symlinks=False)
        by_fd = os.fstat(root_fd)
    except OSError as exc:
        raise ReleaseError("releases-root-changed") from exc
    if (
        path.is_symlink()
        or by_path.st_dev != by_fd.st_dev
        or by_path.st_ino != by_fd.st_ino
        or by_path.st_uid != os.geteuid()
        or stat.S_IMODE(by_path.st_mode) & 0o077
    ):
        raise ReleaseError("releases-root-changed")


def _open_candidate(root_fd: int, release_id: str) -> int:
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(release_id, flags, dir_fd=root_fd)
        metadata = os.fstat(descriptor)
    except OSError as exc:
        raise ReleaseError("release-stage-unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        os.close(descriptor)
        raise ReleaseError("release-stage-unavailable")
    return descriptor


def _assert_candidate_binding(
    releases_root: Path,
    release_id: str,
    candidate_fd: int,
) -> None:
    try:
        by_path = (releases_root / release_id).stat(follow_symlinks=False)
        by_fd = os.fstat(candidate_fd)
    except OSError as exc:
        raise ReleaseError("release-stage-changed") from exc
    if (
        not stat.S_ISDIR(by_path.st_mode)
        or by_path.st_uid != os.geteuid()
        or by_path.st_dev != by_fd.st_dev
        or by_path.st_ino != by_fd.st_ino
    ):
        raise ReleaseError("release-stage-changed")


def _validate_evidence_destination(
    path: Path,
    *,
    spec_path: Path,
    wheelhouse: Path,
    releases_root: Path,
) -> bool:
    """Validate alias/collision safety and report a recoverable existing file."""

    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ReleaseError("evidence-path-must-be-absolute")
    exists = False
    try:
        destination_metadata = path.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ReleaseError("evidence-destination-invalid") from exc
    else:
        if (
            path.is_symlink()
            or not stat.S_ISREG(destination_metadata.st_mode)
            or destination_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(destination_metadata.st_mode) & 0o077
        ):
            raise ReleaseError("evidence-destination-invalid")
        exists = True
    try:
        parent = path.parent.resolve(strict=True)
        parent_metadata = path.parent.lstat()
        spec_resolved = spec_path.resolve(strict=True)
        wheelhouse_resolved = wheelhouse.resolve(strict=True)
        releases_resolved = releases_root.resolve(strict=True)
    except OSError as exc:
        raise ReleaseError("evidence-parent-unavailable") from exc
    if (
        path.parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o077
    ):
        raise ReleaseError("unsafe-evidence-parent")
    candidate = parent / path.name
    if candidate != path:
        raise ReleaseError("evidence-path-must-be-canonical")
    if candidate == spec_resolved:
        raise ReleaseError("evidence-input-collision")
    for protected in (wheelhouse_resolved, releases_resolved):
        if candidate == protected or candidate.is_relative_to(protected):
            raise ReleaseError("evidence-input-collision")
    return exists


def _fsync_evidence_parent(path: Path) -> None:
    """Durably bind an already-published evidence name before journal removal."""

    parent = path.parent
    descriptor: int | None = None
    try:
        by_path = parent.stat(follow_symlinks=False)
        descriptor = os.open(
            parent,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        by_fd = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(by_fd.st_mode)
            or by_fd.st_dev != by_path.st_dev
            or by_fd.st_ino != by_path.st_ino
            or by_fd.st_uid != os.geteuid()
            or stat.S_IMODE(by_fd.st_mode) & 0o077
        ):
            raise ReleaseError("unsafe-evidence-parent")
        os.fsync(descriptor)
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError("evidence-write-failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _current_target(root_fd: int) -> str | None:
    try:
        metadata = os.stat("current", dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReleaseError("current-target-invalid") from exc
    if not stat.S_ISLNK(metadata.st_mode):
        raise ReleaseError("current-is-not-symlink")
    try:
        target = os.readlink("current", dir_fd=root_fd)
    except OSError as exc:
        raise ReleaseError("current-target-invalid") from exc
    if not _RELEASE_ID.fullmatch(target):
        raise ReleaseError("current-target-invalid")
    try:
        target_metadata = os.stat(target, dir_fd=root_fd, follow_symlinks=False)
    except OSError as exc:
        raise ReleaseError("current-target-invalid") from exc
    if (
        not stat.S_ISDIR(target_metadata.st_mode)
        or target_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(target_metadata.st_mode) & 0o222
    ):
        raise ReleaseError("current-target-invalid")
    return target


def _atomic_current_replace(root_fd: int, target: str) -> None:
    temporary = f".current-{secrets.token_hex(12)}"
    try:
        os.symlink(target, temporary, dir_fd=root_fd)
        os.replace(
            temporary,
            "current",
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        os.fsync(root_fd)
    except OSError as exc:
        try:
            os.unlink(temporary, dir_fd=root_fd)
        except OSError:
            pass
        raise ReleaseError("atomic-current-replace-failed") from exc


def _atomic_current_clear(root_fd: int) -> None:
    try:
        os.unlink("current", dir_fd=root_fd)
        os.fsync(root_fd)
    except OSError as exc:
        raise ReleaseError("atomic-current-clear-failed") from exc


def _journal_payload(
    spec: ReleaseSpec,
    *,
    previous: str | None,
    phase: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in {"prepared", "activated", "committed"}:
        raise ReleaseError("activation-journal-invalid")
    return {
        "version": _JOURNAL_VERSION,
        "releaseId": spec.release_id,
        "specDigest": spec.digest,
        "previousTarget": previous,
        "phase": phase,
        "evidence": evidence,
    }


def _write_journal(root_fd: int, value: dict[str, Any]) -> None:
    payload = _canonical_json(value) + b"\n"
    if len(payload) > 512 * 1024:
        raise ReleaseError("activation-journal-invalid")
    temporary = f".activation-{secrets.token_hex(12)}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=root_fd,
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ReleaseError("activation-journal-write-failed")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary,
            _JOURNAL_NAME,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        os.fsync(root_fd)
    except ReleaseError:
        raise
    except OSError as exc:
        raise ReleaseError("activation-journal-write-failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=root_fd)
        except OSError:
            pass


def _load_journal(root_fd: int) -> dict[str, Any] | None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(_JOURNAL_NAME, flags, dir_fd=root_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReleaseError("activation-journal-invalid") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_size > 512 * 1024
        ):
            raise ReleaseError("activation-journal-invalid")
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, 512 * 1024 + 1 - observed_size))
            if not chunk:
                break
            chunks.append(chunk)
            observed_size += len(chunk)
            if observed_size > 512 * 1024:
                raise ReleaseError("activation-journal-invalid")
        after = os.fstat(descriptor)
        if (
            observed_size != metadata.st_size
            or after.st_size != metadata.st_size
            or after.st_mtime_ns != metadata.st_mtime_ns
            or after.st_ctime_ns != metadata.st_ctime_ns
        ):
            raise ReleaseError("activation-journal-invalid")
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    value = _json_without_duplicates(payload)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "version",
            "releaseId",
            "specDigest",
            "previousTarget",
            "phase",
            "evidence",
        }
        or value.get("version") != _JOURNAL_VERSION
        or not isinstance(value.get("releaseId"), str)
        or _RELEASE_ID.fullmatch(value["releaseId"]) is None
        or not isinstance(value.get("specDigest"), str)
        or _DIGEST.fullmatch(value["specDigest"]) is None
        or value.get("phase") not in {"prepared", "activated", "committed"}
        or (
            value.get("previousTarget") is not None
            and (
                not isinstance(value["previousTarget"], str)
                or _RELEASE_ID.fullmatch(value["previousTarget"]) is None
            )
        )
    ):
        raise ReleaseError("activation-journal-invalid")
    return value


def _remove_journal(root_fd: int) -> None:
    try:
        os.unlink(_JOURNAL_NAME, dir_fd=root_fd)
        os.fsync(root_fd)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ReleaseError("activation-journal-remove-failed") from exc


def _recover_activation(
    root_fd: int,
    *,
    spec: ReleaseSpec,
    evidence_path: Path,
) -> tuple[str, dict[str, Any] | None]:
    journal = _load_journal(root_fd)
    if journal is None:
        return "none", None
    if journal["releaseId"] != spec.release_id or journal["specDigest"] != spec.digest:
        raise ReleaseError("activation-journal-release-mismatch")
    current = _current_target(root_fd)
    previous = journal["previousTarget"]
    if journal["phase"] == "committed":
        committed = journal.get("evidence")
        if current != spec.release_id or not isinstance(committed, dict):
            raise ReleaseError("activation-journal-invalid")
        _verify_signed_evidence(committed, spec)
        if evidence_path.exists():
            payload = _read_regular(
                evidence_path,
                limit=1024 * 1024,
                code="release-evidence-unreadable",
            )
            published = _json_without_duplicates(payload)
            if published != committed or not isinstance(published, dict):
                raise ReleaseError("activation-evidence-conflict")
            _verify_signed_evidence(published, spec)
            _fsync_evidence_parent(evidence_path)
        else:
            _write_evidence(evidence_path, committed, spec=spec)
        _remove_journal(root_fd)
        return "committed", committed
    if current == spec.release_id:
        if previous is None:
            _atomic_current_clear(root_fd)
        else:
            _atomic_current_replace(root_fd, previous)
    elif current != previous:
        raise ReleaseError("activation-recovery-conflict")
    _remove_journal(root_fd)
    return "rolled-back", None


def _command_proof(
    release_root: Path,
    spec: CommandSpec,
    *,
    role: str,
) -> dict[str, Any]:
    runtime = release_root / "runtime"
    executable = runtime / "bin" / "python"
    result = _invoke_bounded(
        [
            os.fspath(executable),
            "-I",
            "-m",
            _COMMAND_MODULES[role],
            *spec.arguments,
        ],
        cwd=release_root,
        environment=_runtime_environment(runtime),
        timeout_seconds=spec.timeout_seconds,
        role=role,
    )
    proof = {
        "status": "failed",
        "exitCode": result.return_code,
        "outputDigest": _sha256(
            _canonical_json({"role": role, "exitCode": result.return_code})
        ),
    }
    try:
        report = _json_without_duplicates(result.stdout)
    except ReleaseError as exc:
        raise CommandProofError(f"{role}-non-json-output", proof) from exc
    if not isinstance(report, dict):
        raise CommandProofError(f"{role}-invalid-report", proof)
    expected_status = "passed" if role == "canary" else None
    canary_checks = report.get("checks") if role == "canary" else None
    valid_canary = role != "canary" or (
        isinstance(canary_checks, dict)
        and set(canary_checks)
        == {"entry_points", "engine_binary", "numeric_kernel", "langfuse_agent"}
        and all(value is True for value in canary_checks.values())
    )
    doctor_checks = report.get("checks") if role == "doctor" else None
    valid_doctor = (
        role == "doctor"
        and report.get("status") == "healthy"
        and isinstance(doctor_checks, list)
        and len(doctor_checks) == len(_DOCTOR_CHECKS)
        and {item.get("name") for item in doctor_checks if isinstance(item, dict)}
        == set(_DOCTOR_CHECKS)
        and all(
            isinstance(item, dict) and item.get("status") == "ok"
            for item in doctor_checks
        )
    )
    if (
        result.return_code != 0
        or (expected_status is not None and report.get("status") != expected_status)
        or (role == "canary" and report.get("privacySafe") is not True)
        or not valid_canary
        or (role == "doctor" and not valid_doctor)
    ):
        raise CommandProofError(f"{role}-failed", proof)
    proof["status"] = "passed"
    summary = {
        "role": role,
        "exitCode": result.return_code,
        "status": report.get("status"),
        "checks": sorted(
            ((name, value) for name, value in canary_checks.items())
            if role == "canary" and isinstance(canary_checks, dict)
            else (
                (item["name"], item["status"])
                for item in doctor_checks
                if isinstance(item, dict)
            )
        ),
    }
    proof["outputDigest"] = _sha256(_canonical_json(summary))
    return proof


def _base_evidence(spec: ReleaseSpec) -> dict[str, Any]:
    return {
        "apiVersion": "graphos.io/v2",
        "kind": "ExactLocalReleaseEvidence",
        "releaseId": spec.release_id,
        "status": "assembling",
        "specDigest": spec.digest,
        "requirementsDigest": spec.requirements_digest,
        "packages": {
            name: {
                "version": pin.version,
                "artifactDigest": pin.digest,
            }
            for name, pin in spec.packages.items()
        },
        "nativeArtifacts": dict(spec.native_artifacts),
        "toolchain": {
            name: {"version": pin.version, "artifactDigest": pin.digest}
            for name, pin in spec.toolchain.items()
        },
        "certificationArtifacts": None,
        "closure": {
            "distributionCount": 0,
            "recordVerified": False,
            "directUrlRecordCount": 0,
            "symlinkCount": 0,
            "specialFileCount": 0,
            "nativeArtifactCount": 0,
            "dependencyEdgeCount": 0,
            "releaseTreeEntryCount": 0,
            "immutableAfterProof": False,
        },
        "processGate": {
            "beforePromotion": None,
            "afterVerification": None,
        },
        "activation": {
            "method": "atomic-symlink-replace",
            "hadPreviousRelease": False,
            "rollback": "not-required",
        },
        "commands": {},
        "errorCode": None,
        "privacySafe": True,
    }


def _assert_path_free_evidence(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = {
            "path",
            "location",
            "endpoint",
            "host",
            "user",
            "command",
            "argv",
            "url",
            "uri",
        }
        for key, item in value.items():
            lowered = key.casefold()
            if lowered in forbidden or lowered.endswith(
                (
                    "path",
                    "location",
                    "endpoint",
                    "host",
                    "user",
                    "command",
                    "argv",
                    "url",
                    "uri",
                )
            ):
                raise ReleaseError("evidence-field-not-path-free")
            _assert_path_free_evidence(item)
    elif isinstance(value, list):
        for item in value:
            _assert_path_free_evidence(item)
    elif isinstance(value, str):
        lowered = value.casefold()
        if (
            value.startswith(("/", "~/"))
            or re.match(r"^[a-zA-Z]:[\\/]", value)
            or "file://" in lowered
            or "/home/" in lowered
            or "/users/" in lowered
            or "/mnt/" in lowered
            or "\\users\\" in lowered
        ):
            raise ReleaseError("evidence-value-not-path-free")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _external_json(environment_name: str, payload: bytes) -> dict[str, Any]:
    raw = os.environ.get(environment_name)
    try:
        command = json.loads(raw or "")
    except json.JSONDecodeError as exc:
        raise ReleaseError("evidence-signing-unavailable") from exc
    if (
        not isinstance(command, list)
        or not 1 <= len(command) <= 32
        or any(
            not isinstance(item, str)
            or not item
            or len(item.encode("utf-8")) > 4096
            or "\x00" in item
            for item in command
        )
    ):
        raise ReleaseError("evidence-signing-unavailable")
    try:
        executable = _resolve_executable(
            command[0], code="evidence-signing-unavailable"
        )
        completed = _invoke_bounded(
            [os.fspath(executable), *command[1:]],
            cwd=Path("/"),
            environment=_installer_environment(),
            timeout_seconds=60,
            role="evidence-signing",
            input_payload=payload,
            max_output_bytes=128 * 1024,
        )
    except ReleaseError as exc:
        raise ReleaseError("evidence-signing-failed") from exc
    if completed.return_code != 0:
        raise ReleaseError("evidence-signing-failed")
    try:
        value = _json_without_duplicates(completed.stdout)
    except ReleaseError as exc:
        raise ReleaseError("evidence-signing-failed") from exc
    if not isinstance(value, dict):
        raise ReleaseError("evidence-signing-failed")
    return value


def _evidence_unsigned(evidence: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evidence.items() if key != "signature"}


def _validate_evidence_semantics(evidence: dict[str, Any], spec: ReleaseSpec) -> None:
    _assert_path_free_evidence(evidence)
    if set(evidence) != {
        "apiVersion",
        "kind",
        "releaseId",
        "status",
        "specDigest",
        "requirementsDigest",
        "packages",
        "nativeArtifacts",
        "toolchain",
        "certificationArtifacts",
        "closure",
        "processGate",
        "activation",
        "commands",
        "errorCode",
        "privacySafe",
    }:
        raise ReleaseError("evidence-structure-invalid")
    if (
        evidence.get("apiVersion") != "graphos.io/v2"
        or evidence.get("kind") != "ExactLocalReleaseEvidence"
        or evidence.get("releaseId") != spec.release_id
        or evidence.get("specDigest") != spec.digest
        or evidence.get("requirementsDigest") != spec.requirements_digest
        or evidence.get("privacySafe") is not True
    ):
        raise ReleaseError("evidence-spec-binding-invalid")
    packages = evidence.get("packages")
    expected_packages = {
        name: {"version": pin.version, "artifactDigest": pin.digest}
        for name, pin in spec.packages.items()
    }
    expected_toolchain = {
        name: {"version": pin.version, "artifactDigest": pin.digest}
        for name, pin in spec.toolchain.items()
    }
    if (
        packages != expected_packages
        or evidence.get("nativeArtifacts") != spec.native_artifacts
        or evidence.get("toolchain") != expected_toolchain
    ):
        raise ReleaseError("evidence-artifact-binding-invalid")
    closure = evidence.get("closure")
    process_gate = evidence.get("processGate")
    activation = evidence.get("activation")
    commands = evidence.get("commands")
    certification = evidence.get("certificationArtifacts")
    if (
        not isinstance(closure, dict)
        or set(closure)
        != {
            "distributionCount",
            "recordVerified",
            "directUrlRecordCount",
            "symlinkCount",
            "specialFileCount",
            "nativeArtifactCount",
            "dependencyEdgeCount",
            "releaseTreeEntryCount",
            "immutableAfterProof",
        }
        or any(
            isinstance(closure.get(field), bool)
            or not isinstance(closure.get(field), int)
            or closure[field] < 0
            for field in (
                "distributionCount",
                "directUrlRecordCount",
                "symlinkCount",
                "specialFileCount",
                "nativeArtifactCount",
                "dependencyEdgeCount",
                "releaseTreeEntryCount",
            )
        )
        or not isinstance(closure.get("recordVerified"), bool)
        or not isinstance(closure.get("immutableAfterProof"), bool)
        or not isinstance(process_gate, dict)
        or set(process_gate) != {"beforePromotion", "afterVerification"}
        or any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in process_gate.values()
        )
        or not isinstance(activation, dict)
        or set(activation) != {"method", "hadPreviousRelease", "rollback"}
        or activation.get("method") != "atomic-symlink-replace"
        or not isinstance(activation.get("hadPreviousRelease"), bool)
        or activation.get("rollback") not in {"not-required", "completed", "failed"}
        or not isinstance(commands, dict)
        or not set(commands).issubset({"venv", "install", "canary", "doctor"})
    ):
        raise ReleaseError("evidence-structure-invalid")
    for proof in commands.values():
        if (
            not isinstance(proof, dict)
            or set(proof) != {"status", "exitCode", "outputDigest"}
            or proof.get("status") not in {"passed", "failed"}
            or isinstance(proof.get("exitCode"), bool)
            or not isinstance(proof.get("exitCode"), int)
            or not -255 <= proof["exitCode"] <= 255
            or not isinstance(proof.get("outputDigest"), str)
            or _DIGEST.fullmatch(proof["outputDigest"]) is None
        ):
            raise ReleaseError("evidence-command-invalid")
    if certification is not None and (
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
        or isinstance(certification.get("agentUtilitiesFileCount"), bool)
        or not isinstance(certification.get("agentUtilitiesFileCount"), int)
        or certification["agentUtilitiesFileCount"] < 10
        or any(
            not isinstance(certification.get(field), str)
            or re.fullmatch(r"[a-f0-9]{64}", certification[field]) is None
            for field in (
                "agentUtilitiesSha256",
                "distributionClosureSha256",
                "releasePythonSha256",
                "graphosSha256",
                "engineSha256",
            )
        )
    ):
        raise ReleaseError("evidence-certification-invalid")
    status_value = evidence.get("status")
    if status_value not in {"promoted", "rejected", "rolled-back", "rollback-failed"}:
        raise ReleaseError("evidence-status-invalid")
    error_code = evidence.get("errorCode")
    if status_value != "promoted" and (
        not isinstance(error_code, str)
        or re.fullmatch(r"[a-z][a-z0-9-]{0,127}", error_code) is None
    ):
        raise ReleaseError("evidence-error-code-invalid")
    expected_rollback = {
        "promoted": "not-required",
        "rejected": "not-required",
        "rolled-back": "completed",
        "rollback-failed": "failed",
    }[status_value]
    if activation["rollback"] != expected_rollback:
        raise ReleaseError("evidence-activation-invalid")
    if status_value == "promoted":
        if not isinstance(certification, dict):
            raise ReleaseError("promoted-evidence-incomplete")
        if (
            not isinstance(closure.get("distributionCount"), int)
            or closure["distributionCount"] < 1
            or closure.get("recordVerified") is not True
            or closure.get("directUrlRecordCount") != 0
            or closure.get("symlinkCount") != 0
            or closure.get("specialFileCount") != 0
            or closure.get("nativeArtifactCount") != 2
            or not isinstance(closure.get("dependencyEdgeCount"), int)
            or closure.get("dependencyEdgeCount") < 1
            or not isinstance(closure.get("releaseTreeEntryCount"), int)
            or closure.get("releaseTreeEntryCount") < 1
            or closure.get("immutableAfterProof") is not True
            or process_gate != {"beforePromotion": 0, "afterVerification": 0}
            or activation.get("rollback") != "not-required"
            or evidence.get("errorCode") is not None
            or set(commands) != {"venv", "install", "canary", "doctor"}
            or any(
                proof
                != {
                    "status": "passed",
                    "exitCode": 0,
                    "outputDigest": proof.get("outputDigest"),
                }
                or not isinstance(proof.get("outputDigest"), str)
                or not _DIGEST.fullmatch(proof["outputDigest"])
                for proof in commands.values()
            )
            or set(certification)
            != {
                "agentUtilitiesSha256",
                "agentUtilitiesFileCount",
                "distributionClosureSha256",
                "releasePythonSha256",
                "graphosSha256",
                "engineSha256",
            }
            or not isinstance(certification.get("agentUtilitiesFileCount"), int)
            or certification.get("agentUtilitiesFileCount") < 10
            or any(
                not isinstance(certification.get(field), str)
                or re.fullmatch(r"[a-f0-9]{64}", certification[field]) is None
                for field in (
                    "agentUtilitiesSha256",
                    "distributionClosureSha256",
                    "releasePythonSha256",
                    "graphosSha256",
                    "engineSha256",
                )
            )
        ):
            raise ReleaseError("promoted-evidence-invariant-failed")


def _validate_signature_shape(signature: Any) -> dict[str, str]:
    if (
        not isinstance(signature, dict)
        or set(signature) != {"algorithm", "keyId", "signature", "subjectDigest"}
        or signature.get("algorithm")
        not in {"ed25519", "ecdsa-p256-sha256", "rsa-pss-sha256"}
        or not isinstance(signature.get("keyId"), str)
        or re.fullmatch(r"key:[a-f0-9]{64}", signature["keyId"]) is None
        or not isinstance(signature.get("signature"), str)
        or re.fullmatch(r"[A-Za-z0-9_-]{43,4096}", signature["signature"]) is None
        or not isinstance(signature.get("subjectDigest"), str)
        or _DIGEST.fullmatch(signature["subjectDigest"]) is None
    ):
        raise ReleaseError("evidence-signature-invalid")
    return signature


def _sign_evidence(evidence: dict[str, Any], spec: ReleaseSpec) -> dict[str, Any]:
    unsigned = _evidence_unsigned(evidence)
    _validate_evidence_semantics(unsigned, spec)
    subject_digest = _sha256(_canonical_json(unsigned))
    signature = _validate_signature_shape(
        _external_json(_SIGNER_ENV, _canonical_json(unsigned))
    )
    if signature["subjectDigest"] != subject_digest:
        raise ReleaseError("evidence-signature-invalid")
    signed = {**unsigned, "signature": signature}
    response = _external_json(_VERIFIER_ENV, _canonical_json(signed))
    if response != {
        "verified": True,
        "subjectDigest": subject_digest,
        "keyId": signature["keyId"],
    }:
        raise ReleaseError("evidence-verification-failed")
    return signed


def _verify_signed_evidence(evidence: dict[str, Any], spec: ReleaseSpec) -> None:
    if set(evidence) != {
        "apiVersion",
        "kind",
        "releaseId",
        "status",
        "specDigest",
        "requirementsDigest",
        "packages",
        "nativeArtifacts",
        "toolchain",
        "certificationArtifacts",
        "closure",
        "processGate",
        "activation",
        "commands",
        "errorCode",
        "privacySafe",
        "signature",
    }:
        raise ReleaseError("evidence-structure-invalid")
    signature = _validate_signature_shape(evidence.get("signature"))
    unsigned = _evidence_unsigned(evidence)
    _validate_evidence_semantics(unsigned, spec)
    subject_digest = _sha256(_canonical_json(unsigned))
    if signature.get("subjectDigest") != subject_digest:
        raise ReleaseError("evidence-signature-invalid")
    response = _external_json(_VERIFIER_ENV, _canonical_json(evidence))
    if response != {
        "verified": True,
        "subjectDigest": subject_digest,
        "keyId": signature.get("keyId"),
    }:
        raise ReleaseError("evidence-verification-failed")


def _write_evidence(path: Path, evidence: dict[str, Any], *, spec: ReleaseSpec) -> None:
    _verify_signed_evidence(evidence, spec)
    _assert_path_free_evidence(evidence)
    payload = _canonical_json(evidence) + b"\n"
    parent = path.parent
    try:
        metadata = parent.lstat()
    except OSError as exc:
        raise ReleaseError("evidence-parent-unavailable") from exc
    if (
        not path.is_absolute()
        or parent.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ReleaseError("unsafe-evidence-parent")
    temporary_name = f".release-evidence-{secrets.token_hex(12)}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd: int | None = None
    destination_linked = False
    try:
        parent_fd = os.open(
            parent,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_parent = os.fstat(parent_fd)
        if (
            opened_parent.st_dev != metadata.st_dev
            or opened_parent.st_ino != metadata.st_ino
        ):
            raise ReleaseError("unsafe-evidence-parent")
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_fd)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ReleaseError("evidence-write-failed")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        destination_linked = True
        os.unlink(temporary_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except ReleaseError:
        try:
            if parent_fd is not None:
                os.unlink(temporary_name, dir_fd=parent_fd)
        except OSError:
            pass
        raise
    except OSError as exc:
        try:
            if parent_fd is not None:
                os.unlink(temporary_name, dir_fd=parent_fd)
        except OSError:
            pass
        if destination_linked:
            raise EvidencePublicationUncertain(
                "evidence-publication-uncertain"
            ) from exc
        if isinstance(exc, FileExistsError):
            raise ReleaseError("evidence-destination-must-be-new") from exc
        raise ReleaseError("evidence-write-failed") from exc
    finally:
        if parent_fd is not None:
            os.close(parent_fd)


def promote(
    *,
    spec_path: Path,
    release_id: str,
    wheelhouse: Path,
    releases_root: Path,
    evidence_path: Path,
) -> tuple[int, dict[str, Any]]:
    """Assemble, validate, promote, prove, and if necessary roll back one release."""

    _require_supported_platform()
    assert fcntl is not None
    evidence_already_exists = _validate_evidence_destination(
        evidence_path,
        spec_path=spec_path,
        wheelhouse=wheelhouse,
        releases_root=releases_root,
    )
    spec = load_spec(spec_path, release_id=release_id)
    evidence = _base_evidence(spec)
    root_fd: int | None = None
    candidate_fd: int | None = None
    candidate_root: Path | None = None
    python_tool: BoundExecutable | None = None
    uv_tool: BoundExecutable | None = None
    activated = False
    committed = False
    previous: str | None = None
    try:
        root_fd = _open_releases_root(releases_root)
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ReleaseError("release-promotion-busy") from exc
        _assert_root_binding(releases_root, root_fd)
        recovery, recovered_evidence = _recover_activation(
            root_fd, spec=spec, evidence_path=evidence_path
        )
        if recovery == "committed" and recovered_evidence is not None:
            return 0, recovered_evidence
        if recovery == "rolled-back":
            raise ReleaseError("interrupted-promotion-recovered")
        if evidence_already_exists:
            raise ReleaseError("evidence-destination-must-be-new")
        python_tool, uv_tool = _verify_toolchain(spec)
        locked, wheels, lock_payload = validate_wheelhouse(wheelhouse, spec)
        if len(wheels) != len(locked):
            raise ReleaseError("installed-closure-mismatch")
        evidence["closure"]["distributionCount"] = len(locked)
        try:
            os.mkdir(spec.release_id, mode=0o700, dir_fd=root_fd)
        except FileExistsError as exc:
            raise ReleaseError("release-id-already-exists") from exc
        except OSError as exc:
            raise ReleaseError("release-stage-create-failed") from exc
        candidate_fd = _open_candidate(root_fd, spec.release_id)
        release_root = _fd_path(candidate_fd)
        candidate_root = release_root
        _assert_root_binding(releases_root, root_fd)
        _assert_candidate_binding(releases_root, spec.release_id, candidate_fd)
        if os.fstat(root_fd).st_dev != os.fstat(candidate_fd).st_dev:
            raise ReleaseError("release-stage-filesystem-mismatch")
        marker = release_root / ".incomplete"
        marker.write_text("exact-local-release\n", encoding="ascii")
        os.chmod(marker, 0o600)
        staged_wheelhouse = _stage_wheelhouse(release_root, spec, wheels, lock_payload)
        evidence["commands"].update(
            _create_runtime(
                release_root,
                staged_wheelhouse,
                spec,
                persistent_release_root=releases_root / spec.release_id,
                python_tool=python_tool,
                uv_tool=uv_tool,
            )
        )
        os.close(python_tool.descriptor)
        os.close(uv_tool.descriptor)
        python_tool = None
        uv_tool = None
        verification = verify_installed_release(release_root, spec, locked, wheels)
        verified_distribution_count = verification.pop("distributionCount")
        if verified_distribution_count != evidence["closure"]["distributionCount"]:
            raise ReleaseError("installed-closure-mismatch")
        agent_utilities_sha256 = verification.pop("agentUtilitiesSha256")
        agent_utilities_file_count = verification.pop("agentUtilitiesFileCount")
        distribution_closure_sha256 = verification.pop("distributionClosureSha256")
        evidence["closure"].update(verification)
        _remove_staged_wheelhouse(staged_wheelhouse)
        marker.unlink()
        _scan_regular_tree(release_root)
        evidence["certificationArtifacts"] = _certification_artifacts(
            release_root,
            agent_utilities_sha256=agent_utilities_sha256,
            agent_utilities_file_count=agent_utilities_file_count,
            distribution_closure_sha256=distribution_closure_sha256,
        )
        _seal_release_tree(release_root)
        _verify_release_sealed(release_root)
        _assert_candidate_binding(releases_root, spec.release_id, candidate_fd)
        release_snapshot = _tree_snapshot(release_root)
        evidence["closure"]["releaseTreeEntryCount"] = len(release_snapshot)
        before = running_graph_process_count()
        evidence["processGate"]["beforePromotion"] = before
        if before:
            raise ReleaseError("graph-processes-running")
        _assert_root_binding(releases_root, root_fd)
        previous = _current_target(root_fd)
        evidence["activation"]["hadPreviousRelease"] = previous is not None
        _write_journal(
            root_fd,
            _journal_payload(spec, previous=previous, phase="prepared"),
        )
        try:
            _atomic_current_replace(root_fd, spec.release_id)
        except ReleaseError:
            # replace(2) may have succeeded before a following directory fsync
            # reported failure. Bind the observed state so the outer fault path
            # rolls it back instead of deleting the only recovery journal.
            if _current_target(root_fd) == spec.release_id:
                activated = True
            raise
        activated = True
        _write_journal(
            root_fd,
            _journal_payload(spec, previous=previous, phase="activated"),
        )
        for role in ("canary", "doctor"):
            try:
                evidence["commands"][role] = _command_proof(
                    release_root, spec.commands[role], role=role
                )
            except CommandProofError as exc:
                evidence["commands"][role] = exc.proof
                raise
        after = running_graph_process_count()
        evidence["processGate"]["afterVerification"] = after
        if after:
            raise ReleaseError("graph-process-leaked")
        _assert_root_binding(releases_root, root_fd)
        _assert_candidate_binding(releases_root, spec.release_id, candidate_fd)
        if _current_target(root_fd) != spec.release_id:
            raise ReleaseError("current-target-changed-during-proof")
        if _tree_snapshot(release_root) != release_snapshot:
            raise ReleaseError("release-tree-mutated-during-proof")
        _verify_release_sealed(release_root)
        evidence["closure"]["immutableAfterProof"] = True
        evidence["status"] = "promoted"
        evidence["errorCode"] = None
        signed_evidence = _sign_evidence(evidence, spec)
        _write_journal(
            root_fd,
            _journal_payload(
                spec,
                previous=previous,
                phase="committed",
                evidence=signed_evidence,
            ),
        )
        try:
            _write_evidence(evidence_path, signed_evidence, spec=spec)
        except EvidencePublicationUncertain:
            # The committed journal is the durable source of truth. Keep the
            # candidate active and let same-release recovery fsync or recreate
            # the exact signed evidence before removing that journal.
            committed = True
            return 0, signed_evidence
        except ReleaseError:
            _write_journal(
                root_fd,
                _journal_payload(spec, previous=previous, phase="activated"),
            )
            raise
        committed = True
        try:
            _remove_journal(root_fd)
        except ReleaseError:
            # The signed evidence and committed journal agree. A same-release
            # recovery invocation can remove the journal without reactivation.
            pass
        return 0, signed_evidence
    except Exception as failure:  # noqa: BLE001 - rollback must cover every boundary
        exc = (
            failure
            if isinstance(failure, ReleaseError)
            else ReleaseError("internal-error")
        )
        if activated and evidence["processGate"]["afterVerification"] is None:
            try:
                evidence["processGate"]["afterVerification"] = (
                    running_graph_process_count()
                )
            except ReleaseError:
                pass
        if activated and root_fd is not None and not committed:
            try:
                if _current_target(root_fd) != spec.release_id:
                    raise ReleaseError("current-target-changed-during-proof")
                if previous is None:
                    _atomic_current_clear(root_fd)
                else:
                    _atomic_current_replace(root_fd, previous)
                evidence["activation"]["rollback"] = "completed"
                evidence["status"] = "rolled-back"
                _remove_journal(root_fd)
            except ReleaseError:
                evidence["activation"]["rollback"] = "failed"
                evidence["status"] = "rollback-failed"
        else:
            evidence["status"] = "promoted" if committed else "rejected"
            if root_fd is not None and not committed:
                try:
                    journal = _load_journal(root_fd)
                    if (
                        journal is not None
                        and journal.get("releaseId") == spec.release_id
                        and journal.get("phase") == "prepared"
                    ):
                        current = _current_target(root_fd)
                        if current == spec.release_id:
                            if previous is None:
                                _atomic_current_clear(root_fd)
                            else:
                                _atomic_current_replace(root_fd, previous)
                            evidence["activation"]["rollback"] = "completed"
                            evidence["status"] = "rolled-back"
                            _remove_journal(root_fd)
                        elif current == previous:
                            _remove_journal(root_fd)
                except ReleaseError:
                    pass
        evidence["errorCode"] = exc.code
        if candidate_root is not None:
            staged_copy = candidate_root / ".wheelhouse"
            try:
                _remove_staged_wheelhouse(staged_copy)
            except (FileNotFoundError, ReleaseError):
                pass
        try:
            if not committed:
                signed_failure = _sign_evidence(evidence, spec)
                _write_evidence(evidence_path, signed_failure, spec=spec)
                evidence = signed_failure
        except ReleaseError:
            pass
        return 1, evidence
    finally:
        for tool in (python_tool, uv_tool):
            if tool is not None:
                os.close(tool.descriptor)
        if candidate_fd is not None:
            os.close(candidate_fd)
        if root_fd is not None:
            try:
                fcntl.flock(root_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promote-local-graphos-release",
        description="Install and atomically promote one exact offline GraphOS release.",
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--releases-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    return parser


def verify_evidence_file(
    *, spec_path: Path, release_id: str, evidence_path: Path
) -> dict[str, Any]:
    """Cross-bind one signed evidence document to its exact release spec."""

    _require_supported_platform()
    spec = load_spec(spec_path, release_id=release_id)
    payload = _read_regular(
        evidence_path, limit=1024 * 1024, code="release-evidence-unreadable"
    )
    value = _json_without_duplicates(payload)
    if not isinstance(value, dict):
        raise ReleaseError("release-evidence-invalid")
    _verify_signed_evidence(value, spec)
    return value


def verify_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify-local-graphos-release-evidence",
        description="Verify signed exact-local promotion evidence against its spec.",
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evidence = verify_evidence_file(
            spec_path=args.spec,
            release_id=args.release_id,
            evidence_path=args.evidence,
        )
    except ReleaseError as exc:
        print(f"evidence_status=rejected error_code={exc.code}", file=sys.stderr)
        return 1
    except Exception:
        print("evidence_status=rejected error_code=internal-error", file=sys.stderr)
        return 1
    print(
        f"evidence_status=verified release_status={evidence['status']} error_code=none"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        status, evidence = promote(
            spec_path=args.spec,
            release_id=args.release_id,
            wheelhouse=args.wheelhouse,
            releases_root=args.releases_root,
            evidence_path=args.evidence,
        )
        print(
            f"release_status={evidence['status']} "
            f"error_code={evidence.get('errorCode') or 'none'}"
        )
        return status
    except ReleaseError as exc:
        print(f"release_status=rejected error_code={exc.code}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - never expose environment detail at the CLI
        print("release_status=rejected error_code=internal-error", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
