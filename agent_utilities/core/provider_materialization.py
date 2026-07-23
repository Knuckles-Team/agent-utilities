"""Current-only, content-addressed provider materialization primitives.

Provider assets are untrusted data until this module has verified their filesystem
shape and bounded their size.  Active materializations are immutable generations;
an atomically replaced, path-free marker selects the current generation.  Version 1
markers are deliberately not accepted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MANAGED_PROVIDER_MARKER = ".agent-utilities-managed.json"
MANAGED_PROVIDER_GENERATIONS = ".generations"
MANAGED_PROVIDER_SCHEMA_VERSION = 2
MANAGED_PROVIDER_LEGS: frozenset[str] = frozenset({"skills", "prompts", "ontologies"})

MAX_MARKER_BYTES = 4096
MAX_PROVIDER_FILES = 10_000
MAX_PROVIDER_FILE_BYTES = 16 * 1024 * 1024
MAX_PROVIDER_BYTES = 128 * 1024 * 1024
MAX_PROVIDER_NAME_BYTES = 128

_MARKER_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "provider",
        "leg",
        "active",
        "registration_digest",
        "content_digest",
        "file_count",
        "byte_count",
    }
)
_SAFE_PROVIDER_NAME = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$", re.ASCII
)
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)
_IGNORED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


class ProviderMaterializationError(RuntimeError):
    """Base class for privacy-safe provider materialization failures."""


class ProviderAssetError(ProviderMaterializationError):
    """A source or generation violates the bounded regular-file contract."""


class EmptyProviderAssets(ProviderAssetError):
    """A registered provider has no required assets for its declared leg."""


class ProviderOwnershipConflict(ProviderMaterializationError):
    """A destination exists but is not proven installer-owned."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One verified regular file in a provider asset manifest."""

    relative_path: str
    source: Path
    size: int
    mode: int
    digest: str


@dataclass(frozen=True, slots=True)
class AssetManifest:
    """A deterministic, path-free digest over selected provider files."""

    entries: tuple[ManifestEntry, ...]
    content_digest: str
    file_count: int
    byte_count: int


@dataclass(frozen=True, slots=True)
class ManagedProviderMarker:
    """The exact closed activation record stored at a provider root."""

    provider: str
    leg: str
    active: bool
    registration_digest: str
    content_digest: str
    file_count: int
    byte_count: int
    schema_version: int = MANAGED_PROVIDER_SCHEMA_VERSION

    def payload(self) -> dict[str, str | int | bool]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "leg": self.leg,
            "active": self.active,
            "registration_digest": self.registration_digest,
            "content_digest": self.content_digest,
            "file_count": self.file_count,
            "byte_count": self.byte_count,
        }


def is_safe_provider_name(value: str) -> bool:
    """Return whether *value* is one portable, bounded filesystem component."""

    if not isinstance(value, str) or not value:
        return False
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return False
    if len(encoded) > MAX_PROVIDER_NAME_BYTES:
        return False
    if not _SAFE_PROVIDER_NAME.fullmatch(value) or value in {".", ".."}:
        return False
    return value.upper() not in _WINDOWS_RESERVED


def _require_digest(value: str, field: str) -> str:
    if not isinstance(value, str) or _HEX_DIGEST.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def registration_digest(payload: dict[str, str]) -> str:
    """Hash a closed registration identity without storing module or host details."""

    if not payload or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in payload.items()
    ):
        raise ValueError("registration identity must contain non-empty string fields")
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(rendered).hexdigest()


def _empty_manifest_digest() -> str:
    return hashlib.sha256(b"agent-utilities-provider-manifest-v2\0").hexdigest()


EMPTY_MANIFEST_DIGEST = _empty_manifest_digest()


def marker_for_manifest(
    *, provider: str, leg: str, registration: str, manifest: AssetManifest
) -> ManagedProviderMarker:
    """Create an active marker for a fully verified manifest."""

    if manifest.file_count <= 0:
        raise EmptyProviderAssets("active provider materialization cannot be empty")
    return ManagedProviderMarker(
        provider=provider,
        leg=leg,
        active=True,
        registration_digest=registration,
        content_digest=manifest.content_digest,
        file_count=manifest.file_count,
        byte_count=manifest.byte_count,
    )


def inactive_marker(
    *, provider: str, leg: str, registration: str
) -> ManagedProviderMarker:
    """Create a current registration marker that deliberately activates no assets."""

    return ManagedProviderMarker(
        provider=provider,
        leg=leg,
        active=False,
        registration_digest=registration,
        content_digest=EMPTY_MANIFEST_DIGEST,
        file_count=0,
        byte_count=0,
    )


def _validate_marker(marker: ManagedProviderMarker) -> None:
    if marker.schema_version != MANAGED_PROVIDER_SCHEMA_VERSION:
        raise ValueError("provider marker schema is unsupported")
    if not is_safe_provider_name(marker.provider):
        raise ValueError("provider name must be one safe filesystem component")
    if marker.leg not in MANAGED_PROVIDER_LEGS:
        raise ValueError("provider materialization leg is unsupported")
    _require_digest(marker.registration_digest, "registration_digest")
    _require_digest(marker.content_digest, "content_digest")
    if type(marker.active) is not bool:  # bool is intentionally exact here
        raise ValueError("active must be a boolean")
    if type(marker.file_count) is not int or marker.file_count < 0:
        raise ValueError("file_count must be a non-negative integer")
    if type(marker.byte_count) is not int or marker.byte_count < 0:
        raise ValueError("byte_count must be a non-negative integer")
    if marker.file_count > MAX_PROVIDER_FILES or marker.byte_count > MAX_PROVIDER_BYTES:
        raise ValueError("provider marker exceeds materialization bounds")
    if marker.active and marker.file_count == 0:
        raise ValueError("active provider marker cannot be empty")
    if not marker.active and (
        marker.file_count != 0
        or marker.byte_count != 0
        or marker.content_digest != EMPTY_MANIFEST_DIGEST
    ):
        raise ValueError("inactive provider marker must describe the empty manifest")


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate provider marker key")
        value[key] = item
    return value


def _safe_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and not _is_linklike(path)


def _is_linklike(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction is not None and is_junction())


def marker_path_exists(root: Path) -> bool:
    """Return whether a marker filesystem entry exists, without following links."""

    try:
        (root / MANAGED_PROVIDER_MARKER).lstat()
    except OSError:
        return False
    return True


def read_managed_provider_marker(
    root: Path,
    *,
    provider: str | None = None,
    leg: str | None = None,
) -> ManagedProviderMarker | None:
    """Read a bounded v2 marker without following root or marker symlinks."""

    if not _safe_directory(root):
        return None
    marker_path = root / MANAGED_PROVIDER_MARKER
    try:
        info = marker_path.lstat()
        if not stat.S_ISREG(info.st_mode) or _is_linklike(marker_path):
            return None
        if info.st_size <= 0 or info.st_size > MAX_MARKER_BYTES:
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(marker_path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size > MAX_MARKER_BYTES:
                return None
            raw_bytes = os.read(descriptor, MAX_MARKER_BYTES + 1)
        finally:
            os.close(descriptor)
        if len(raw_bytes) > MAX_MARKER_BYTES:
            return None
        raw = json.loads(
            raw_bytes.decode("utf-8"), object_pairs_hook=_pairs_without_duplicates
        )
        if not isinstance(raw, dict) or set(raw) != _MARKER_KEYS:
            return None
        if type(raw.get("schema_version")) is not int:
            return None
        marker = ManagedProviderMarker(**raw)
        _validate_marker(marker)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if provider is not None and marker.provider != provider:
        return None
    if leg is not None and marker.leg != leg:
        return None
    return marker


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability on platforms that permit directory fsync."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def write_managed_provider_marker(root: Path, marker: ManagedProviderMarker) -> None:
    """Atomically activate a closed marker after its generation is durable."""

    _validate_marker(marker)
    if not root.exists():
        root.mkdir(parents=True, mode=0o700)
    if not _safe_directory(root):
        raise ProviderOwnershipConflict("provider root is not a regular directory")
    rendered = (
        json.dumps(marker.payload(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    if len(rendered) > MAX_MARKER_BYTES:
        raise ValueError("provider marker exceeds maximum size")
    destination = root / MANAGED_PROVIDER_MARKER
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{MANAGED_PROVIDER_MARKER}.", dir=root
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(root)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _selected_for_leg(relative: PurePosixPath, leg: str) -> bool:
    if leg == "data":
        return relative.suffix.lower() not in _IGNORED_SUFFIXES
    if leg == "skills":
        return relative.suffix.lower() not in _IGNORED_SUFFIXES
    if leg == "prompts":
        return len(relative.parts) == 1 and relative.suffix.lower() == ".json"
    if leg == "ontologies":
        return (len(relative.parts) == 1 and relative.suffix.lower() == ".ttl") or (
            len(relative.parts) == 2
            and relative.parts[0] == "shapes"
            and relative.suffix.lower() == ".ttl"
        )
    raise ValueError("provider materialization leg is unsupported")


def _read_regular_file(path: Path, *, maximum: int) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ProviderAssetError("provider asset is not a regular file")
        if info.st_size > maximum:
            raise ProviderAssetError("provider asset exceeds the per-file bound")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise ProviderAssetError("provider asset exceeds the per-file bound")
            chunks.append(chunk)
        if total != info.st_size:
            raise ProviderAssetError("provider asset changed during validation")
        return b"".join(chunks), info
    finally:
        os.close(descriptor)


def _manifest_digest(entries: Iterable[ManifestEntry]) -> str:
    digest = hashlib.sha256(b"agent-utilities-provider-manifest-v2\0")
    for entry in entries:
        digest.update(entry.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry.mode).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry.size).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(entry.digest))
    return digest.hexdigest()


def _bounded_regular_files(root: Path) -> Iterable[Path]:
    """Yield regular files without first materializing an unbounded directory list."""

    pending = [root]
    tree_entry_count = 0
    while pending:
        directory = pending.pop()
        try:
            scanner = os.scandir(directory)
        except OSError as exc:
            raise ProviderAssetError("provider directory cannot be inspected") from exc
        with scanner:
            for directory_entry in scanner:
                tree_entry_count += 1
                if tree_entry_count > MAX_PROVIDER_FILES:
                    raise ProviderAssetError(
                        "provider source exceeds the tree-entry bound"
                    )
                path = Path(directory_entry.path)
                try:
                    info = directory_entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise ProviderAssetError(
                        "provider asset cannot be inspected"
                    ) from exc
                if _is_linklike(path):
                    raise ProviderAssetError(
                        "provider source contains a linked or special entry"
                    )
                if stat.S_ISDIR(info.st_mode):
                    if directory_entry.name not in _IGNORED_DIRS:
                        pending.append(path)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise ProviderAssetError(
                        "provider source contains a linked or special entry"
                    )
                yield path


def build_asset_manifest(
    root: Path,
    *,
    leg: str,
    allowed_relative_paths: frozenset[str] | None = None,
) -> AssetManifest:
    """Validate and hash one provider source/generation without following links."""

    if leg not in MANAGED_PROVIDER_LEGS and leg != "data":
        raise ValueError("provider materialization leg is unsupported")
    if not _safe_directory(root):
        raise ProviderAssetError("provider asset root is not a regular directory")
    canonical_root = root.resolve(strict=True)
    entries: list[ManifestEntry] = []
    required_asset_seen = False
    byte_count = 0

    for path in _bounded_regular_files(canonical_root):
        if path.name == MANAGED_PROVIDER_MARKER:
            raise ProviderAssetError("provider source contains the reserved marker")
        relative = PurePosixPath(path.relative_to(canonical_root).as_posix())
        if not _selected_for_leg(relative, leg):
            continue
        if (
            allowed_relative_paths is not None
            and relative.as_posix() not in allowed_relative_paths
        ):
            raise ProviderAssetError(
                "provider asset is not owned by the registering distribution"
            )
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(canonical_root):
            raise ProviderAssetError("provider asset escapes its source root")
        content, opened = _read_regular_file(path, maximum=MAX_PROVIDER_FILE_BYTES)
        size = len(content)
        byte_count += size
        if byte_count > MAX_PROVIDER_BYTES:
            raise ProviderAssetError("provider assets exceed the total byte bound")
        mode = stat.S_IMODE(opened.st_mode) & 0o777
        entries.append(
            ManifestEntry(
                relative_path=relative.as_posix(),
                source=path,
                size=size,
                mode=mode,
                digest=hashlib.sha256(content).hexdigest(),
            )
        )
        if leg == "skills" and relative.name == "SKILL.md":
            required_asset_seen = True
        elif leg == "prompts":
            required_asset_seen = True
        elif leg == "ontologies" and len(relative.parts) == 1:
            required_asset_seen = True
        elif leg == "data":
            required_asset_seen = True

    entries.sort(key=lambda item: item.relative_path)
    if not entries or not required_asset_seen:
        raise EmptyProviderAssets(
            "provider ships no required assets for its declared leg"
        )
    frozen = tuple(entries)
    return AssetManifest(
        entries=frozen,
        content_digest=_manifest_digest(frozen),
        file_count=len(frozen),
        byte_count=byte_count,
    )


def copy_manifest(manifest: AssetManifest, destination: Path) -> None:
    """Copy one verified manifest as regular bytes, then verify the staged result."""

    if not _safe_directory(destination):
        raise ProviderOwnershipConflict("provider staging destination is unsafe")
    if os.name != "nt" and stat.S_IMODE(destination.lstat().st_mode) & 0o077:
        raise ProviderOwnershipConflict("provider staging destination is not private")
    try:
        if next(destination.iterdir(), None) is not None:
            raise ProviderOwnershipConflict("provider staging destination is not empty")
    except OSError as exc:
        raise ProviderOwnershipConflict(
            "provider staging destination cannot be inspected"
        ) from exc
    for entry in manifest.entries:
        relative = PurePosixPath(entry.relative_path)
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        content, opened = _read_regular_file(
            entry.source, maximum=MAX_PROVIDER_FILE_BYTES
        )
        if (
            len(content) != entry.size
            or hashlib.sha256(content).hexdigest() != entry.digest
            or (stat.S_IMODE(opened.st_mode) & 0o777) != entry.mode
        ):
            raise ProviderAssetError("provider asset changed during materialization")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(target, flags, entry.mode or 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(target, entry.mode)
        finally:
            os.close(descriptor)
    staged = build_asset_manifest(destination, leg=_infer_leg(manifest, destination))
    if (
        staged.content_digest != manifest.content_digest
        or staged.file_count != manifest.file_count
        or staged.byte_count != manifest.byte_count
    ):
        raise ProviderAssetError("staged provider manifest does not match its source")
    _fsync_tree(destination)


def _infer_leg(manifest: AssetManifest, destination: Path) -> str:
    """Infer the already-validated leg from its selected file shape."""

    paths = [PurePosixPath(entry.relative_path) for entry in manifest.entries]
    if any(path.name == "SKILL.md" for path in paths):
        return "skills"
    if all(len(path.parts) == 1 and path.suffix.lower() == ".json" for path in paths):
        return "prompts"
    if all(path.suffix.lower() == ".ttl" for path in paths):
        return "ontologies"
    raise ProviderAssetError("provider manifest leg cannot be inferred")


def _fsync_tree(root: Path) -> None:
    for directory, _, _ in os.walk(root, topdown=False, followlinks=False):
        _fsync_directory(Path(directory))


def resolve_managed_generation(
    root: Path,
    *,
    provider: str,
    leg: str,
    registration: str,
    source_manifest: AssetManifest,
) -> Path | None:
    """Return a complete current generation, or ``None`` on any mismatch."""

    marker = read_managed_provider_marker(root, provider=provider, leg=leg)
    if marker is None or not marker.active:
        return None
    if (
        marker.registration_digest != registration
        or marker.content_digest != source_manifest.content_digest
        or marker.file_count != source_manifest.file_count
        or marker.byte_count != source_manifest.byte_count
    ):
        return None
    generation = root / MANAGED_PROVIDER_GENERATIONS / marker.content_digest
    try:
        materialized = build_asset_manifest(generation, leg=leg)
    except (OSError, ProviderAssetError, ValueError):
        return None
    if (
        materialized.content_digest != marker.content_digest
        or materialized.file_count != marker.file_count
        or materialized.byte_count != marker.byte_count
    ):
        return None
    return generation
