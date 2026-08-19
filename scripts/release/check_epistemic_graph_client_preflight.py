#!/usr/bin/env python3
"""Fail closed unless the image carries the approved epistemic-graph client.

The unified image receives the engine as a build-time wheel.  A matching
distribution version alone is not enough evidence: an old client can be
source-shadowed, or can expose a package without the native work-item CAS
surface.  This check validates the staged wheel before installation and the
actually imported client after installation.  When an installer emits a PEP
610 file URL without an archive hash, the wheel RECORD and installed
regular-file hashes provide the cryptographic binding instead of a basename
heuristic.  The installed-tree proof runs before importing the client and rejects
unproved interpreter caches while accepting only bytecode derived from verified
wheel sources; the import phase suppresses new bytecode so the early-start
preflight cannot create unproved executable state.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import marshal
import re
import stat
import sys
import sysconfig
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as default_email_policy
from pathlib import Path, PurePosixPath
from types import CodeType, ModuleType
from typing import Any, Final
from urllib.parse import unquote, urlparse

PACKAGE_NAME: Final = "epistemic-graph"
EXPECTED_VERSION: Final = "2.26.2"
REQUIRED_CAPABILITY: Final = "work_items.cas_metadata"
CAPABILITIES_MODULE: Final = "epistemic_graph.client_capabilities"
CLIENT_MODULE_PATH: Final = "epistemic_graph/client_capabilities.py"
_WHEEL_FILENAME = re.compile(
    rf"^epistemic_graph-{re.escape(EXPECTED_VERSION)}-"
    r"[A-Za-z0-9.]+-[A-Za-z0-9.]+-[A-Za-z0-9_.]+\.whl$"
)
_MAX_RECORD_BYTES: Final = 4 * 1024 * 1024
_MAX_BYTECODE_BYTES: Final = 16 * 1024 * 1024
# ``cache_from_source(..., optimization=0)`` gained an explicit ``.opt-0``
# spelling on newer interpreters, while installers still emit the historical
# no-suffix spelling.  Both spellings mean the same exact optimization level.
_SUPPORTED_OPTIMIZATIONS: Final = ((0, None), (0, 0), (1, 1), (2, 2))
_INSTALLER_METADATA: Final = frozenset(
    {"INSTALLER", "REQUESTED", "direct_url.json", "uv_cache.json"}
)
_WHEEL_DATA_SCHEMES: Final = frozenset({"purelib", "platlib", "scripts"})


class PreflightError(RuntimeError):
    """A deterministic, privacy-safe preflight rejection."""


@dataclass(frozen=True)
class WheelEvidence:
    """The identity proved by one staged wheel."""

    path: Path
    name: str
    version: str
    sha256: str


@dataclass(frozen=True)
class _WheelArchive:
    """Validated archive identities used to bind installed files."""

    members: Mapping[str, tuple[str, int]]
    dist_info: str
    record_name: str


def _regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except (OSError, ValueError):
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and not path.is_symlink()
        and metadata.st_nlink == 1
    )


def _metadata_identity(path: Path) -> tuple[str, str]:
    try:
        with zipfile.ZipFile(path) as archive:
            candidates = [
                info
                for info in archive.infolist()
                if len(info.filename.split("/")) == 2
                and info.filename.split("/")[0].endswith(".dist-info")
                and info.filename.split("/")[1] == "METADATA"
            ]
            if len(candidates) != 1 or candidates[0].file_size > 1_048_576:
                raise PreflightError("wheel-metadata-invalid")
            metadata = BytesParser(policy=default_email_policy).parsebytes(
                archive.read(candidates[0])
            )
    except PreflightError:
        raise
    except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
        raise PreflightError("wheel-unreadable") from exc

    names = metadata.get_all("Name") or []
    versions = metadata.get_all("Version") or []
    if len(names) != 1 or len(versions) != 1:
        raise PreflightError("wheel-metadata-identity-invalid")
    return str(names[0]), str(versions[0])


def inspect_wheel(path: Path) -> WheelEvidence:
    """Validate one wheel's filename and top-level distribution metadata."""

    if not _regular_file(path) or path.suffix.lower() != ".whl":
        raise PreflightError("wheel-invalid")
    if not _WHEEL_FILENAME.fullmatch(path.name):
        raise PreflightError("wheel-filename-version-mismatch")

    metadata_name, metadata_version = _metadata_identity(path)
    if metadata_name != PACKAGE_NAME or metadata_version != EXPECTED_VERSION:
        raise PreflightError("wheel-metadata-version-mismatch")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1_048_576), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PreflightError("wheel-unreadable") from exc
    return WheelEvidence(
        path=path,
        name=PACKAGE_NAME,
        version=EXPECTED_VERSION,
        sha256=digest.hexdigest(),
    )


def select_wheel(wheel_dir: Path) -> WheelEvidence:
    """Select exactly one staged wheel and prove its identity."""

    try:
        wheel_dir.lstat()
    except OSError as exc:
        raise PreflightError("wheel-directory-unavailable") from exc
    if wheel_dir.is_symlink() or not wheel_dir.is_dir():
        raise PreflightError("wheel-directory-invalid")
    try:
        resolved_dir = wheel_dir.resolve(strict=True)
    except OSError as exc:
        raise PreflightError("wheel-directory-invalid") from exc
    try:
        candidates = tuple(
            sorted(
                child
                for child in wheel_dir.iterdir()
                if child.name.lower().endswith(".whl")
            )
        )
    except OSError as exc:
        raise PreflightError("wheel-directory-unreadable") from exc
    if len(candidates) != 1:
        raise PreflightError("wheel-count-invalid")
    try:
        if candidates[0].resolve(strict=True).parent != resolved_dir:
            raise PreflightError("wheel-containment-invalid")
    except PreflightError:
        raise
    except OSError as exc:
        raise PreflightError("wheel-containment-invalid") from exc
    return inspect_wheel(candidates[0])


def _safe_member_name(raw_name: object) -> str:
    """Validate a wheel/RECORD POSIX member without normalizing escapes."""

    if not isinstance(raw_name, str) or not raw_name:
        raise PreflightError("wheel-member-path-invalid")
    try:
        if len(raw_name.encode("utf-8")) > 4096:
            raise PreflightError("wheel-member-path-invalid")
    except UnicodeError as exc:
        raise PreflightError("wheel-member-path-invalid") from exc
    if "\\" in raw_name or "\x00" in raw_name:
        raise PreflightError("wheel-member-path-invalid")
    relative = PurePosixPath(raw_name)
    if (
        relative.is_absolute()
        or not relative.parts
        or re.fullmatch(r"[A-Za-z]:", relative.parts[0])
    ):
        raise PreflightError("wheel-member-path-invalid")
    parts = raw_name.split("/")
    if raw_name.endswith("/"):
        parts = parts[:-1]
    if not parts or any(
        not part or part in {".", ".."} or any(ord(char) < 32 for char in part)
        for part in parts
    ):
        raise PreflightError("wheel-member-path-invalid")
    return "/".join(parts)


def _zip_member_kind(info: zipfile.ZipInfo) -> int:
    mode = (info.external_attr >> 16) & 0xFFFF
    kind = stat.S_IFMT(mode)
    if kind == stat.S_IFLNK:
        raise PreflightError("wheel-member-symlink")
    if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise PreflightError("wheel-member-type-invalid")
    if info.is_dir() and kind == stat.S_IFREG:
        raise PreflightError("wheel-member-type-invalid")
    if not info.is_dir() and kind == stat.S_IFDIR:
        raise PreflightError("wheel-member-type-invalid")
    return kind


def _hash_zip_member(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with archive.open(info, "r") as stream:
            while chunk := stream.read(1_048_576):
                digest.update(chunk)
                size += len(chunk)
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise PreflightError("wheel-unreadable") from exc
    return (
        "sha256="
        + base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii"),
        size,
    )


def _hash_regular_member(path: Path) -> tuple[str, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PreflightError("installed-file-missing") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise PreflightError("installed-path-symlink")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise PreflightError("installed-file-not-regular")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1_048_576):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise PreflightError("installed-file-unreadable") from exc
    return (
        "sha256="
        + base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii"),
        size,
    )


def _derived_cache_targets(
    expected_targets: Mapping[Path, tuple[str, tuple[str, int]]],
) -> dict[Path, tuple[Path, int]]:
    """Map exact interpreter cache paths to the RECORD-verified source files."""

    cache_targets: dict[Path, tuple[Path, int]] = {}
    for source_target, (member_name, _identity) in expected_targets.items():
        if not member_name.endswith(".py"):
            continue
        for optimization, cache_optimization in _SUPPORTED_OPTIMIZATIONS:
            try:
                cache_target = Path(
                    importlib.util.cache_from_source(
                        str(source_target), optimization=cache_optimization
                    )
                ).resolve(strict=False)
            except (NotImplementedError, OSError, ValueError):
                continue
            previous = cache_targets.get(cache_target)
            if previous is not None and previous != (source_target, optimization):
                raise PreflightError("installed-bytecode-path-invalid")
            cache_targets[cache_target] = (source_target, optimization)
    return cache_targets


def _validate_derived_cache(
    path: Path,
    cache_targets: Mapping[Path, tuple[Path, int]],
) -> tuple[str, int]:
    """Prove a cache is exact bytecode for one installed wheel-owned source."""

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PreflightError("installed-bytecode-unavailable") from exc
    source_info = cache_targets.get(resolved)
    if source_info is None or path.suffix != ".pyc":
        raise PreflightError("installed-bytecode-path-invalid")
    source_path, optimization = source_info
    try:
        if not _regular_file(path) or not _regular_file(source_path):
            raise PreflightError("installed-bytecode-not-regular")
        payload = path.read_bytes()
        source_payload = source_path.read_bytes()
    except PreflightError:
        raise
    except OSError as exc:
        raise PreflightError("installed-bytecode-unavailable") from exc
    if len(payload) > _MAX_BYTECODE_BYTES or len(payload) < 16:
        raise PreflightError("installed-bytecode-invalid")
    if payload[:4] != importlib.util.MAGIC_NUMBER:
        raise PreflightError("installed-bytecode-invalid")

    flags = int.from_bytes(payload[4:8], "little")
    # PEP 552 defines only the hash bit and the check-source bit.  Bit 1
    # without bit 0 is invalid; all higher bits are reserved and rejected.
    if flags & ~0x03 or flags == 0x02:
        raise PreflightError("installed-bytecode-invalid")
    if flags & 0x01:
        try:
            expected_hash = importlib.util.source_hash(source_payload)
        except (AttributeError, TypeError, ValueError) as exc:
            raise PreflightError("installed-bytecode-invalid") from exc
        if payload[8:16] != expected_hash:
            raise PreflightError("installed-bytecode-source-mismatch")
    elif int.from_bytes(payload[12:16], "little") != len(source_payload):
        raise PreflightError("installed-bytecode-source-mismatch")

    try:
        stream = io.BytesIO(payload[16:])
        code = marshal.load(stream)
        if stream.read() or not isinstance(code, CodeType):
            raise PreflightError("installed-bytecode-invalid")
    except PreflightError:
        raise
    except (
        EOFError,
        ImportError,
        MemoryError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise PreflightError("installed-bytecode-invalid") from exc
    filename = code.co_filename
    if not isinstance(filename, str) or not filename:
        raise PreflightError("installed-bytecode-filename-invalid")
    try:
        expected_code = compile(
            source_payload,
            filename,
            "exec",
            optimize=optimization,
            dont_inherit=True,
        )
        # ``marshal`` is a transport format, not a canonical code-object
        # serialization: reference/interning tables may differ even when the
        # loaded CodeType is structurally identical to a fresh compilation.
        if expected_code != code:
            raise PreflightError("installed-bytecode-code-mismatch")
    except PreflightError:
        raise
    except (
        MemoryError,
        OverflowError,
        RecursionError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as exc:
        raise PreflightError("installed-bytecode-code-mismatch") from exc
    return _hash_regular_member(path)


def _read_record_payload(path: Path) -> bytes:
    try:
        metadata = path.lstat()
        if metadata.st_size > _MAX_RECORD_BYTES or not _regular_file(path):
            raise PreflightError("installed-record-invalid")
        return path.read_bytes()
    except PreflightError:
        raise
    except (OSError, UnicodeError) as exc:
        raise PreflightError("installed-record-unavailable") from exc


def _record_rows(payload: bytes, *, error: str) -> list[tuple[str, str, str]]:
    try:
        rows = list(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise PreflightError(error) from exc
    if not rows or any(len(row) != 3 for row in rows):
        raise PreflightError(error)
    return [(row[0], row[1], row[2]) for row in rows]


def _wheel_archive(path: Path) -> _WheelArchive:
    """Validate archive paths/types/RECORD and return immutable identities."""

    try:
        with zipfile.ZipFile(path) as archive:
            info_by_name: dict[str, zipfile.ZipInfo] = {}
            seen_names: set[str] = set()
            for info in archive.infolist():
                name = _safe_member_name(info.filename)
                canonical_name = name.casefold()
                if canonical_name in seen_names:
                    raise PreflightError("wheel-member-duplicate")
                seen_names.add(canonical_name)
                kind = _zip_member_kind(info)
                if kind != stat.S_IFDIR:
                    info_by_name[name] = info

            metadata_names = [
                name
                for name in info_by_name
                if len(name.split("/")) == 2
                and name.split("/", 1)[0].endswith(".dist-info")
                and name.split("/", 1)[1] == "METADATA"
            ]
            if len(metadata_names) != 1:
                raise PreflightError("wheel-metadata-invalid")
            metadata_name = metadata_names[0]
            dist_info = metadata_name.split("/", 1)[0]
            metadata_info = info_by_name[metadata_name]
            if metadata_info.file_size > 1_048_576:
                raise PreflightError("wheel-metadata-invalid")
            try:
                metadata = BytesParser(policy=default_email_policy).parsebytes(
                    archive.read(metadata_info)
                )
            except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise PreflightError("wheel-metadata-invalid") from exc
            names = metadata.get_all("Name") or []
            versions = metadata.get_all("Version") or []
            if (
                len(names) != 1
                or len(versions) != 1
                or str(names[0]) != PACKAGE_NAME
                or str(versions[0]) != EXPECTED_VERSION
            ):
                raise PreflightError("wheel-metadata-version-mismatch")

            record_name = f"{dist_info}/RECORD"
            record_info = info_by_name.get(record_name)
            if record_info is None:
                raise PreflightError("wheel-record-missing")
            if record_info.file_size > _MAX_RECORD_BYTES:
                raise PreflightError("wheel-record-invalid")
            try:
                record_payload = archive.read(record_info)
            except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise PreflightError("wheel-record-invalid") from exc
            rows = _record_rows(record_payload, error="wheel-record-invalid")
            identities: dict[str, tuple[str, int]] = {}
            seen_records: set[str] = set()
            for raw_name, digest, size in rows:
                name = _safe_member_name(raw_name)
                canonical_name = name.casefold()
                if canonical_name in seen_records:
                    raise PreflightError("wheel-record-invalid")
                seen_records.add(canonical_name)
                if name == record_name:
                    if digest or size:
                        raise PreflightError("wheel-record-self-hash-present")
                    identities[name] = ("", 0)
                    continue
                info = info_by_name.get(name)
                if info is None or info.is_dir() or not digest.startswith("sha256="):
                    raise PreflightError("wheel-record-member-mismatch")
                if not size.isdigit():
                    raise PreflightError("wheel-record-member-mismatch")
                expected = _hash_zip_member(archive, info)
                if (digest, int(size)) != expected:
                    raise PreflightError("wheel-record-member-mismatch")
                identities[name] = expected
            if set(identities) != set(info_by_name) or record_name not in identities:
                raise PreflightError("wheel-record-coverage-mismatch")
            return _WheelArchive(
                members=identities,
                dist_info=dist_info,
                record_name=record_name,
            )
    except PreflightError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise PreflightError("wheel-unreadable") from exc


def _resolved_directory(path: Path, error: str) -> Path:
    try:
        if path.is_symlink() or not path.is_dir():
            raise PreflightError(error)
        return path.resolve(strict=True)
    except PreflightError:
        raise
    except OSError as exc:
        raise PreflightError(error) from exc


def _distribution_root(distribution: Any) -> Path:
    try:
        return _resolved_directory(
            Path(distribution.locate_file("")), "client-layout-invalid"
        )
    except (AttributeError, OSError, RuntimeError, TypeError) as exc:
        raise PreflightError("client-layout-invalid") from exc


def _scripts_root() -> Path:
    configured = sysconfig.get_path("scripts")
    if not isinstance(configured, str) or not configured:
        raise PreflightError("client-scripts-layout-invalid")
    return _resolved_directory(Path(configured), "client-scripts-layout-invalid")


def _target_path(base: Path, parts: tuple[str, ...]) -> Path:
    """Resolve a member beneath an approved install root without symlink hops."""

    if base.is_symlink():
        raise PreflightError("installed-path-symlink")
    cursor = base
    for part in parts:
        cursor /= part
        try:
            if cursor.is_symlink():
                raise PreflightError("installed-path-symlink")
        except OSError as exc:
            raise PreflightError("installed-path-unavailable") from exc
    try:
        resolved_base = base.resolve(strict=True)
        resolved = cursor.resolve(strict=False)
        resolved.relative_to(resolved_base)
    except (OSError, ValueError) as exc:
        raise PreflightError("installed-path-containment-invalid") from exc
    return resolved


def _wheel_target(
    root: Path,
    scripts_root: Path | None,
    name: str,
) -> Path:
    parts = PurePosixPath(name).parts
    if len(parts) >= 3 and parts[0].endswith(".data"):
        scheme = parts[1]
        if scheme not in _WHEEL_DATA_SCHEMES:
            raise PreflightError("wheel-data-scheme-invalid")
        if scheme == "scripts":
            if scripts_root is None:
                raise PreflightError("client-scripts-layout-invalid")
            return _target_path(scripts_root, tuple(parts[2:]))
        return _target_path(root, tuple(parts[2:]))
    if parts and parts[0].endswith(".data"):
        raise PreflightError("wheel-data-scheme-invalid")
    return _target_path(root, tuple(parts))


def _record_target(
    root: Path,
    scripts_root: Path | None,
    raw_name: str,
) -> Path:
    """Resolve an installed RECORD path, allowing only controlled script moves."""

    if not raw_name or "\\" in raw_name or "\x00" in raw_name:
        raise PreflightError("installed-record-path-invalid")
    relative = PurePosixPath(raw_name)
    if relative.is_absolute() or not relative.parts:
        raise PreflightError("installed-record-path-invalid")
    parts = tuple(relative.parts)
    if any(not part or part == "." for part in parts):
        raise PreflightError("installed-record-path-invalid")
    if ".." not in parts:
        return _target_path(root, parts)
    if scripts_root is None:
        raise PreflightError("installed-record-path-invalid")
    first_real = next((index for index, part in enumerate(parts) if part != ".."), None)
    if first_real is None or ".." in parts[first_real:]:
        raise PreflightError("installed-record-path-invalid")
    try:
        candidate = root.joinpath(*parts).resolve(strict=False)
        scripts_resolved = scripts_root.resolve(strict=True)
        suffix = candidate.relative_to(scripts_resolved)
    except (OSError, ValueError) as exc:
        raise PreflightError("installed-record-path-containment-invalid") from exc
    return _target_path(scripts_root, tuple(suffix.parts))


def _scan_owned_tree(
    current: Path,
    expected_targets: set[Path],
    adaptation_targets: set[Path],
    cache_targets: Mapping[Path, tuple[Path, int]],
    accepted_cache_targets: set[Path],
) -> None:
    try:
        metadata = current.lstat()
    except OSError as exc:
        raise PreflightError("installed-file-unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise PreflightError("installed-path-symlink")
    if stat.S_ISDIR(metadata.st_mode):
        try:
            children = tuple(current.iterdir())
        except OSError as exc:
            raise PreflightError("installed-file-unavailable") from exc
        for child in children:
            _scan_owned_tree(
                child,
                expected_targets,
                adaptation_targets,
                cache_targets,
                accepted_cache_targets,
            )
        return
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise PreflightError("installed-file-not-regular")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise PreflightError("installed-file-unavailable") from exc
    if resolved in expected_targets or resolved in adaptation_targets:
        return
    if resolved in accepted_cache_targets:
        return
    if resolved in cache_targets:
        _validate_derived_cache(resolved, cache_targets)
        accepted_cache_targets.add(resolved)
        return
    raise PreflightError("installed-file-unexpected")


def _verify_installed_wheel(
    *,
    artifact: WheelEvidence,
    distribution: Any,
    require_direct_url: bool,
) -> None:
    """Bind installed files and RECORD to the exact selected wheel archive."""

    archive_layout = _wheel_archive(artifact.path)
    root = _distribution_root(distribution)
    needs_scripts = any(
        name.split("/", 2)[1] == "scripts"
        for name in archive_layout.members
        if len(name.split("/")) >= 3 and name.split("/", 1)[0].endswith(".data")
    )
    scripts_root = _scripts_root() if needs_scripts else None

    expected_targets: dict[Path, tuple[str, tuple[str, int]]] = {}
    for name, identity in archive_layout.members.items():
        target = _wheel_target(root, scripts_root, name)
        if target in expected_targets:
            raise PreflightError("wheel-target-duplicate")
        expected_targets[target] = (name, identity)

    record_target = _wheel_target(root, scripts_root, archive_layout.record_name)
    if record_target not in expected_targets:
        raise PreflightError("installed-record-layout-invalid")
    for target, (name, expected) in expected_targets.items():
        if target == record_target:
            if not _regular_file(target):
                raise PreflightError("installed-record-unavailable")
            continue
        actual = _hash_regular_member(target)
        if actual != expected:
            raise PreflightError("installed-wheel-content-mismatch")

    cache_targets = _derived_cache_targets(expected_targets)

    installed_rows = _record_rows(
        _read_record_payload(record_target), error="installed-record-invalid"
    )
    observed: dict[Path, tuple[str, int]] = {}
    accepted_cache_targets: set[Path] = set()
    for raw_name, digest, size in installed_rows:
        target = _record_target(root, scripts_root, raw_name)
        if target in observed:
            raise PreflightError("installed-record-duplicate")
        if not _regular_file(target):
            raise PreflightError("installed-record-non-regular")
        if target == record_target:
            if digest or size:
                raise PreflightError("installed-record-self-hash-present")
            observed[target] = ("", 0)
            continue
        if target.suffix == ".pyc":
            actual = _validate_derived_cache(target, cache_targets)
            accepted_cache_targets.add(target)
            if not digest and not size:
                observed[target] = actual
                continue
        if not digest.startswith("sha256=") or not size.isdigit():
            raise PreflightError("installed-record-hash-invalid")
        actual = _hash_regular_member(target)
        if actual != (digest, int(size)):
            raise PreflightError("installed-record-content-mismatch")
        observed[target] = actual

    expected_set = set(expected_targets)
    if not expected_set <= set(observed):
        raise PreflightError("installed-record-missing-member")
    for target, (_name, identity) in expected_targets.items():
        if target != record_target and observed[target] != identity:
            raise PreflightError("installed-record-wheel-mismatch")

    dist_info_root = _wheel_target(root, scripts_root, archive_layout.dist_info)
    adaptation_targets: set[Path] = set()
    for target in set(observed) - expected_set:
        if target in accepted_cache_targets:
            continue
        try:
            relative = target.relative_to(dist_info_root)
        except ValueError as exc:
            raise PreflightError("installed-file-unexpected") from exc
        if len(relative.parts) != 1 or relative.name not in _INSTALLER_METADATA:
            raise PreflightError("installed-file-unexpected")
        adaptation_targets.add(target)

    direct_url_target = dist_info_root / "direct_url.json"
    if require_direct_url and direct_url_target not in observed:
        raise PreflightError("installed-record-missing-metadata")

    owned_roots: set[Path] = set()
    for name in archive_layout.members:
        parts = PurePosixPath(name).parts
        if len(parts) >= 3 and parts[0].endswith(".data"):
            if parts[1] == "scripts":
                continue
            top = parts[2]
        else:
            top = parts[0]
        base = (
            scripts_root
            if len(parts) >= 3 and parts[0].endswith(".data") and parts[1] == "scripts"
            else root
        )
        if base is not None:
            owned_roots.add(_target_path(base, (top,)))
    for owned_root in owned_roots:
        _scan_owned_tree(
            owned_root,
            expected_set,
            adaptation_targets,
            cache_targets,
            accepted_cache_targets,
        )


def _validate_artifact_url(artifact: WheelEvidence, value: object) -> None:
    parsed = urlparse(value) if isinstance(value, str) else None
    if (
        parsed is None
        or parsed.scheme != "file"
        or parsed.netloc.lower() not in {"", "localhost"}
        or parsed.params
        or parsed.query
        or parsed.fragment
        or not parsed.path
    ):
        raise PreflightError("client-artifact-provenance-mismatch")
    try:
        url_path = Path(unquote(parsed.path))
    except (OSError, ValueError) as exc:
        raise PreflightError("client-artifact-provenance-mismatch") from exc
    if not url_path.is_absolute() or not _regular_file(url_path):
        raise PreflightError("client-artifact-provenance-mismatch")
    try:
        selected = artifact.path.resolve(strict=True)
        resolved_url = url_path.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise PreflightError("client-artifact-provenance-mismatch") from exc
    if resolved_url != selected:
        raise PreflightError("client-artifact-provenance-mismatch")


def _archive_hash_status(
    archive_info: Mapping[str, Any], artifact: WheelEvidence
) -> tuple[bool, bool]:
    """Return ``(supplied, matches)`` for PEP 610 archive hash data."""

    supplied = False
    if "hash" in archive_info:
        supplied = True
        value = archive_info["hash"]
        if value != f"sha256={artifact.sha256}":
            return supplied, False
    if "hashes" in archive_info:
        hashes = archive_info["hashes"]
        if not isinstance(hashes, Mapping):
            return supplied, False
        if hashes:
            supplied = True
            if hashes.get("sha256") != artifact.sha256:
                return supplied, False
    return supplied, True


def _assert_selected_artifact_unchanged(artifact: WheelEvidence) -> None:
    current = inspect_wheel(artifact.path)
    if current.sha256 != artifact.sha256:
        raise PreflightError("client-artifact-changed")


def _resolve_regular_module_file(module: ModuleType) -> Path:
    raw_file = getattr(module, "__file__", None)
    if not isinstance(raw_file, str) or not raw_file:
        raise PreflightError("client-module-origin-invalid")
    path = Path(raw_file)
    if not path.is_absolute() or path.is_symlink() or not _regular_file(path):
        raise PreflightError("client-module-origin-invalid")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise PreflightError("client-module-origin-invalid") from exc


def validate_installed_client(
    *,
    artifact: WheelEvidence | None = None,
    distribution_reader: Callable[[str], Any] = importlib.metadata.distribution,
    module_importer: Callable[[str], ModuleType] = importlib.import_module,
) -> Mapping[str, Any]:
    """Validate the installed client and invoke its producer-owned gate.

    The injectable readers keep the contract tests synthetic and side-effect
    free; the default readers are the real image interpreter's metadata/import
    surfaces.  Installed-tree verification completes before the capability
    import, whose bytecode writes are disabled for this early-start check.
    """

    try:
        distribution = distribution_reader(PACKAGE_NAME)
    except Exception as exc:
        raise PreflightError("client-distribution-unavailable") from exc
    if str(getattr(distribution, "version", "")) != EXPECTED_VERSION:
        raise PreflightError("client-distribution-version-mismatch")

    try:
        direct_url = distribution.read_text("direct_url.json")
    except (AttributeError, OSError, UnicodeError) as exc:
        if artifact is not None:
            raise PreflightError("client-artifact-provenance-unavailable") from exc
        direct_url = None
    if direct_url:
        try:
            direct_url_data = json.loads(direct_url)
        except (TypeError, ValueError) as exc:
            raise PreflightError("client-distribution-metadata-invalid") from exc
        if not isinstance(direct_url_data, Mapping):
            raise PreflightError("client-distribution-metadata-invalid")
    else:
        direct_url_data = {}
    if isinstance(direct_url_data, Mapping) and (
        isinstance(direct_url_data.get("dir_info"), Mapping)
        and direct_url_data["dir_info"].get("editable") is True
    ):
        raise PreflightError("client-editable-install")
    if artifact is not None:
        if not direct_url_data:
            raise PreflightError("client-artifact-provenance-unavailable")
        _validate_artifact_url(artifact, direct_url_data.get("url"))
        _assert_selected_artifact_unchanged(artifact)
        if "archive_info" in direct_url_data:
            archive_info = direct_url_data["archive_info"]
            if not isinstance(archive_info, Mapping):
                raise PreflightError("client-artifact-provenance-unavailable")
        else:
            archive_info = {}
        hash_supplied, hash_matches = _archive_hash_status(archive_info, artifact)
        if hash_supplied and not hash_matches:
            raise PreflightError("client-artifact-provenance-mismatch")
        # PEP 610 hashes authenticate the selected archive, but do not prove
        # what the installer left importable in site-packages.  Always bind
        # the installed RECORD/tree to that exact archive as well.
        _verify_installed_wheel(
            artifact=artifact,
            distribution=distribution,
            require_direct_url=True,
        )

    try:
        expected_path = Path(distribution.locate_file(CLIENT_MODULE_PATH))
        if (
            not expected_path.is_absolute()
            or expected_path.is_symlink()
            or not _regular_file(expected_path)
        ):
            raise PreflightError("client-distribution-layout-invalid")
        expected_file = expected_path.resolve(strict=True)
    except PreflightError:
        raise
    except (AttributeError, OSError, RuntimeError, TypeError) as exc:
        raise PreflightError("client-distribution-layout-invalid") from exc
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        try:
            module = module_importer(CAPABILITIES_MODULE)
        except Exception as exc:
            raise PreflightError("client-capabilities-module-unavailable") from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    actual_file = _resolve_regular_module_file(module)
    if actual_file != expected_file:
        raise PreflightError("client-source-shadowed")

    capability = getattr(
        module, "WORK_ITEM_METADATA_CAS_CAPABILITY", REQUIRED_CAPABILITY
    )
    if capability != REQUIRED_CAPABILITY:
        raise PreflightError("client-capability-unknown")
    require = getattr(module, "require_client_capabilities", None)
    if not callable(require):
        raise PreflightError("client-capability-gate-unavailable")
    try:
        manifest = require((capability,))
    except Exception as exc:
        raise PreflightError("client-capability-rejected") from exc
    if not isinstance(manifest, Mapping):
        raise PreflightError("client-capability-manifest-invalid")
    if (
        manifest.get("package") != PACKAGE_NAME
        or manifest.get("package_version") != EXPECTED_VERSION
    ):
        raise PreflightError("client-capability-identity-mismatch")
    capabilities = manifest.get("capabilities")
    if (
        not isinstance(capabilities, Mapping)
        or capabilities.get(REQUIRED_CAPABILITY) is not True
    ):
        raise PreflightError("client-capability-unavailable")
    return manifest


def run_preflight(wheel_dir: Path, *, require_installed: bool = False) -> WheelEvidence:
    """Validate the staged artifact and, optionally, the installed client."""

    evidence = select_wheel(wheel_dir)
    if require_installed:
        validate_installed_client(artifact=evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument(
        "--require-installed",
        action="store_true",
        help="also validate the imported installed client capability contract",
    )
    parser.add_argument(
        "--print-wheel-basename",
        action="store_true",
        help="print only the validated wheel basename for an install command",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.require_installed and args.print_wheel_basename:
        _parser().error("--require-installed and --print-wheel-basename are exclusive")
    try:
        evidence = run_preflight(
            args.wheel_dir,
            require_installed=args.require_installed,
        )
    except PreflightError as exc:
        print(f"epistemic-graph client preflight failed: {exc}", file=sys.stderr)
        return 1
    if args.print_wheel_basename:
        print(evidence.path.name)
    else:
        print(
            "epistemic-graph client preflight OK: "
            f"version={evidence.version} capability={REQUIRED_CAPABILITY}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
