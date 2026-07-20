#!/usr/bin/env python3
"""Generate deterministic, privacy-safe evidence for one exact release component.

The generator opens the real local artifact, binds it to a frozen source manifest,
derives a CycloneDX inventory from a mandatory closed wheelhouse for OCI subjects,
emits an in-toto statement, and accepts signatures only from an external JSON-argv
adapter.  Local
filesystem locations are inputs only; no path, endpoint, or signer identity is copied
into retained evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import struct
import sys
import tarfile
import zipfile
from collections.abc import Iterable
from contextlib import contextmanager
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release import check_compatibility

_NAME = re.compile(r"^[a-z0-9][a-z0-9.-]{1,127}$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
_DIGEST = re.compile(r"^sha256:(?!0{64}$)[a-f0-9]{64}$")
_SCHEME = re.compile(r"^[a-z0-9][a-z0-9+._-]{1,63}$")
_SIGNATURE = re.compile(r"^[A-Za-z0-9+/_=-]{16,16384}$")
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
_MAX_INPUT_BYTES = 64 * 1024 * 1024
_MAX_SIGNATURE_BYTES = 1024 * 1024
_MAX_WHEELS = 4096
_MAX_WHEEL_BYTES = 512 * 1024 * 1024
_MAX_WHEEL_MEMBERS = 16_384
_MAX_WHEEL_CENTRAL_DIRECTORY_BYTES = 16 * 1024 * 1024
_MAX_OCI_ENTRIES = 65_536
_MAX_LAYER_ENTRIES = 262_144
_MAX_LAYER_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
_MAX_OCI_JSON_BYTES = 16 * 1024 * 1024
_MAX_INSTALLED_METADATA_BYTES = 1024 * 1024
_BOOTSTRAP_DISTRIBUTIONS = {"pip", "setuptools", "wheel"}
_OCI_INDEX = "application/vnd.oci.image.index.v1+json"
_OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
_OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
_OCI_LAYERS = {
    "application/vnd.oci.image.layer.v1.tar",
    "application/vnd.oci.image.layer.v1.tar+gzip",
}


class ComponentEvidenceError(ValueError):
    """An evidence input is unsafe, ambiguous, or not bound to the artifact."""


def _metadata_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@contextmanager
def _regular_descriptor(path: Path, *, maximum: int) -> Any:
    if path.is_symlink():
        raise ComponentEvidenceError("evidence inputs must not be symlinks")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ComponentEvidenceError("evidence input is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size == 0
            or metadata.st_size > maximum
        ):
            raise ComponentEvidenceError("evidence input violates its size boundary")
        before = _metadata_identity(metadata)
        yield descriptor, metadata
        after = os.fstat(descriptor)
        if _metadata_identity(after) != before:
            raise ComponentEvidenceError("evidence input changed while it was read")
        try:
            path_metadata = path.stat(follow_symlinks=False)
        except OSError:
            raise ComponentEvidenceError("evidence input changed while it was read") from None
        if (path_metadata.st_dev, path_metadata.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            raise ComponentEvidenceError("evidence input changed while it was read")
    finally:
        os.close(descriptor)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _open_regular(path: Path, *, maximum: int, nonempty: bool = True) -> bytes:
    with _regular_descriptor(path, maximum=maximum) as (descriptor, metadata):
        payload = bytearray()
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            payload.extend(chunk)
            remaining -= len(chunk)
        if len(payload) > maximum or len(payload) != metadata.st_size:
            raise ComponentEvidenceError("evidence input violates its size boundary")
        if nonempty and not payload:
            raise ComponentEvidenceError("evidence input violates its size boundary")
        return bytes(payload)


def _file_digest(path: Path, *, maximum: int) -> tuple[str, int]:
    with _regular_descriptor(path, maximum=maximum) as (descriptor, metadata):
        digest = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum:
                raise ComponentEvidenceError("evidence input violates its size boundary")
            digest.update(chunk)
        if observed != metadata.st_size:
            raise ComponentEvidenceError("evidence input changed while it was read")
        return "sha256:" + digest.hexdigest(), observed


def _safe_json(path: Path, *, maximum: int) -> dict[str, Any]:
    try:
        value = json.loads(_open_regular(path, maximum=maximum))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComponentEvidenceError("evidence input must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ComponentEvidenceError("evidence input must be a JSON object")
    return value


def _source_freeze(path: Path) -> tuple[str, str]:
    payload = _open_regular(path, maximum=_MAX_INPUT_BYTES)
    try:
        authority = check_compatibility.validate_source_freeze_evidence(payload)
    except check_compatibility.CompatibilityError as exc:
        raise ComponentEvidenceError(str(exc)) from exc
    return authority["snapshotDigest"], authority["evidenceDigest"]


def _relative(path: Path, release_root: Path) -> str:
    try:
        value = path.resolve().relative_to(release_root.resolve()).as_posix()
    except ValueError as exc:
        raise ComponentEvidenceError("evidence output must remain under the release root") from exc
    if not value or value.startswith("../"):
        raise ComponentEvidenceError("evidence output reference is invalid")
    return value


def _write(path: Path, payload: bytes, *, release_root: Path) -> None:
    root = release_root.resolve()
    try:
        relative = path.absolute().relative_to(root)
    except ValueError as exc:
        raise ComponentEvidenceError("evidence output must remain under the release root") from exc
    if not relative.parts or relative.name in {"", ".", ".."}:
        raise ComponentEvidenceError("evidence output reference is invalid")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_descriptor = os.open(root, directory_flags)
    except OSError as exc:
        raise ComponentEvidenceError("evidence output is unavailable") from exc
    directory_descriptor = root_descriptor
    try:
        for part in relative.parts[:-1]:
            if part in {"", ".", ".."}:
                raise ComponentEvidenceError("evidence output reference is invalid")
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            try:
                child = os.open(part, directory_flags, dir_fd=directory_descriptor)
            except OSError as exc:
                raise ComponentEvidenceError(
                    "evidence output directories must not be symlinks"
                ) from exc
            if directory_descriptor != root_descriptor:
                os.close(directory_descriptor)
            directory_descriptor = child
        try:
            existing = os.stat(
                relative.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and (
            not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1
        ):
            raise ComponentEvidenceError("evidence output must be an unaliased regular file")
        temporary_name = f".{relative.name}.{secrets.token_hex(16)}.tmp"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=directory_descriptor,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise ComponentEvidenceError("evidence output write failed")
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
        try:
            os.replace(
                temporary_name,
                relative.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
            )
            os.fsync(directory_descriptor)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=directory_descriptor)
            except FileNotFoundError:
                pass
    finally:
        if directory_descriptor != root_descriptor:
            os.close(directory_descriptor)
        os.close(root_descriptor)


def _wheel_metadata(path: Path) -> tuple[str, str, str]:
    try:
        parsed_name, parsed_version, _, _ = parse_wheel_filename(path.name)
    except (InvalidVersion, ValueError) as exc:
        raise ComponentEvidenceError("wheelhouse contains an invalid wheel filename") from exc
    with _regular_descriptor(path, maximum=_MAX_WHEEL_BYTES) as (descriptor, file_metadata):
        tail_size = min(file_metadata.st_size, 65_557)
        os.lseek(descriptor, file_metadata.st_size - tail_size, os.SEEK_SET)
        tail = os.read(descriptor, tail_size)
        offset = tail.rfind(b"PK\x05\x06")
        if offset < 0 or len(tail) - offset < 22:
            raise ComponentEvidenceError("wheel central directory is invalid")
        try:
            (
                _signature,
                disk_number,
                directory_disk,
                disk_entries,
                total_entries,
                directory_size,
                directory_offset,
                comment_size,
            ) = struct.unpack("<4s4H2LH", tail[offset : offset + 22])
        except struct.error as exc:
            raise ComponentEvidenceError("wheel central directory is invalid") from exc
        if (
            disk_number != 0
            or directory_disk != 0
            or disk_entries != total_entries
            or total_entries == 0
            or total_entries > _MAX_WHEEL_MEMBERS
            or total_entries == 0xFFFF
            or directory_size > _MAX_WHEEL_CENTRAL_DIRECTORY_BYTES
            or directory_size == 0xFFFFFFFF
            or directory_offset == 0xFFFFFFFF
            or directory_offset + directory_size > file_metadata.st_size
            or offset + 22 + comment_size != len(tail)
        ):
            raise ComponentEvidenceError("wheel central directory violates its boundary")
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed += len(chunk)
            digest.update(chunk)
        if observed != file_metadata.st_size:
            raise ComponentEvidenceError("wheel changed while it was read")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            try:
                with zipfile.ZipFile(stream) as archive:
                    if len(archive.infolist()) != total_entries:
                        raise ComponentEvidenceError("wheel member count differs")
                    metadata_names = [
                        name
                        for name in archive.namelist()
                        if name.endswith(".dist-info/METADATA")
                    ]
                    if len(metadata_names) != 1:
                        raise ComponentEvidenceError(
                            "wheel does not contain exactly one METADATA file"
                        )
                    info = archive.getinfo(metadata_names[0])
                    if info.file_size > _MAX_INSTALLED_METADATA_BYTES:
                        raise ComponentEvidenceError(
                            "wheel METADATA exceeds its size boundary"
                        )
                    metadata = BytesParser().parsebytes(archive.read(info))
            except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
                raise ComponentEvidenceError("wheel archive is invalid") from exc
    name = canonicalize_name(str(metadata.get("Name") or ""))
    version = str(metadata.get("Version") or "")
    if name != canonicalize_name(str(parsed_name)) or version != str(parsed_version):
        raise ComponentEvidenceError("wheel filename and METADATA identity differ")
    return name, version, "sha256:" + digest.hexdigest()


def _wheel_components(wheelhouse: Path | None) -> list[dict[str, Any]]:
    if wheelhouse is None:
        return []
    if wheelhouse.is_symlink() or not wheelhouse.is_dir():
        raise ComponentEvidenceError("wheelhouse must be a regular directory")
    wheels: list[Path] = []
    try:
        with os.scandir(wheelhouse) as entries:
            for entry in entries:
                wheels.append(Path(entry.path))
                if len(wheels) > _MAX_WHEELS:
                    raise ComponentEvidenceError(
                        "wheelhouse entry count violates its boundary"
                    )
    except OSError as exc:
        raise ComponentEvidenceError("wheelhouse is unavailable") from exc
    wheels.sort(key=lambda item: item.name.casefold())
    if not wheels:
        raise ComponentEvidenceError("wheelhouse entry count violates its boundary")
    if any(item.suffix != ".whl" for item in wheels):
        raise ComponentEvidenceError("closed wheelhouse may contain only wheels")
    seen: set[str] = set()
    components: list[dict[str, Any]] = []
    for wheel in wheels:
        if wheel.is_symlink() or not wheel.is_file():
            raise ComponentEvidenceError("wheelhouse entries must be regular files")
        name, version, digest = _wheel_metadata(wheel)
        if name in seen:
            raise ComponentEvidenceError("wheelhouse contains duplicate distributions")
        seen.add(name)
        purl = f"pkg:pypi/{name}@{version}"
        components.append(
            {
                "type": "library",
                "bom-ref": purl,
                "name": name,
                "version": version,
                "purl": purl,
                "hashes": [{"alg": "SHA-256", "content": digest.removeprefix("sha256:")}],
            }
        )
    return components


def _safe_archive_name(value: str, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ComponentEvidenceError(f"{field} contains an unsafe path")
    return path


def _read_tar_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    *,
    maximum: int,
) -> bytes:
    if not member.isfile() or member.size < 0 or member.size > maximum:
        raise ComponentEvidenceError("OCI archive member violates its size boundary")
    stream = archive.extractfile(member)
    if stream is None:
        raise ComponentEvidenceError("OCI archive member is unavailable")
    payload = bytearray()
    while len(payload) <= maximum:
        chunk = stream.read(min(1024 * 1024, maximum + 1 - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) != member.size or len(payload) > maximum:
        raise ComponentEvidenceError("OCI archive member violates its size boundary")
    return bytes(payload)


def _oci_descriptor(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ComponentEvidenceError(f"{field} descriptor is invalid")
    allowed = {"mediaType", "digest", "size", "annotations", "platform", "artifactType"}
    if not {"mediaType", "digest", "size"}.issubset(value) or not set(value).issubset(
        allowed
    ):
        raise ComponentEvidenceError(f"{field} descriptor is not exact")
    digest = str(value.get("digest") or "")
    if (
        _DIGEST.fullmatch(digest) is None
        or not isinstance(value.get("size"), int)
        or not 0 < value["size"] <= _MAX_ARTIFACT_BYTES
        or not isinstance(value.get("mediaType"), str)
    ):
        raise ComponentEvidenceError(f"{field} descriptor is invalid")
    for optional in ("annotations", "platform"):
        if optional in value and not isinstance(value[optional], dict):
            raise ComponentEvidenceError(f"{field} descriptor is invalid")
    if "artifactType" in value and not isinstance(value["artifactType"], str):
        raise ComponentEvidenceError(f"{field} descriptor is invalid")
    return value


def _layer_inventory(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    media_type: str,
    inventory: dict[str, tuple[str, str]],
) -> None:
    if media_type not in _OCI_LAYERS:
        raise ComponentEvidenceError("OCI layer compression is unsupported")
    outer = archive.extractfile(member)
    if outer is None:
        raise ComponentEvidenceError("OCI layer is unavailable")
    try:
        mode = "r|gz" if media_type.endswith("+gzip") else "r|"
        with tarfile.open(fileobj=outer, mode=mode) as layer:
            entry_count = 0
            observed_size = 0
            for entry in layer:
                entry_count += 1
                observed_size += max(entry.size, 0)
                if (
                    entry_count > _MAX_LAYER_ENTRIES
                    or observed_size > _MAX_LAYER_UNCOMPRESSED_BYTES
                ):
                    raise ComponentEvidenceError("OCI layer violates its expansion boundary")
                path = _safe_archive_name(entry.name, "OCI layer")
                parts = tuple(part for part in path.parts if part != ".")
                if not parts:
                    continue
                parent = "/".join(parts[:-1])
                basename = parts[-1]
                if basename == ".wh..wh..opq":
                    prefix = parent + "/" if parent else ""
                    for known in tuple(inventory):
                        if known.startswith(prefix):
                            inventory.pop(known, None)
                    continue
                if basename.startswith(".wh."):
                    target = "/".join((*parts[:-1], basename.removeprefix(".wh.")))
                    inventory.pop(target, None)
                    prefix = target + "/"
                    for known in tuple(inventory):
                        if known.startswith(prefix):
                            inventory.pop(known, None)
                    continue
                normalized = "/".join(parts)
                match = re.fullmatch(
                    r"usr/local/lib/python3\.12/site-packages/"
                    r"([^/]+\.dist-info)/METADATA",
                    normalized,
                )
                if match is None:
                    continue
                if not entry.isfile() or entry.size > _MAX_INSTALLED_METADATA_BYTES:
                    inventory.pop(normalized, None)
                    continue
                extracted = layer.extractfile(entry)
                if extracted is None:
                    raise ComponentEvidenceError("installed distribution metadata is unavailable")
                metadata_payload = extracted.read(_MAX_INSTALLED_METADATA_BYTES + 1)
                if len(metadata_payload) != entry.size or len(metadata_payload) > _MAX_INSTALLED_METADATA_BYTES:
                    raise ComponentEvidenceError(
                        "installed distribution metadata violates its size boundary"
                    )
                metadata = BytesParser().parsebytes(metadata_payload)
                name = canonicalize_name(str(metadata.get("Name") or ""))
                version = str(metadata.get("Version") or "")
                if not name or not version:
                    raise ComponentEvidenceError("installed distribution metadata is invalid")
                inventory[normalized] = (name, version)
    except (OSError, tarfile.TarError) as exc:
        raise ComponentEvidenceError("OCI layer archive is invalid") from exc


def _oci_archive_identity(
    path: Path,
    expected_distributions: dict[str, str],
) -> tuple[str, str]:
    with _regular_descriptor(path, maximum=_MAX_ARTIFACT_BYTES) as (
        descriptor,
        file_metadata,
    ):
        archive_hash = hashlib.sha256()
        observed = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            archive_hash.update(chunk)
            observed += len(chunk)
        if observed != file_metadata.st_size:
            raise ComponentEvidenceError("OCI archive changed while it was read")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            try:
                with tarfile.open(fileobj=stream, mode="r:*") as archive:
                    members: dict[str, tarfile.TarInfo] = {}
                    blob_members: dict[str, tarfile.TarInfo] = {}
                    entry_count = 0
                    for member in archive:
                        entry_count += 1
                        if entry_count > _MAX_OCI_ENTRIES:
                            raise ComponentEvidenceError(
                                "OCI archive entry count violates its boundary"
                            )
                        member_path = _safe_archive_name(member.name, "OCI archive")
                        normalized = member_path.as_posix()
                        if normalized in members:
                            raise ComponentEvidenceError("OCI archive contains duplicate paths")
                        members[normalized] = member
                        if member.isdir():
                            continue
                        if not member.isfile() or member.issparse():
                            raise ComponentEvidenceError(
                                "OCI archive may contain only directories and regular files"
                            )
                        if normalized in {"oci-layout", "index.json"}:
                            continue
                        blob_match = re.fullmatch(r"blobs/sha256/([a-f0-9]{64})", normalized)
                        if blob_match is None:
                            raise ComponentEvidenceError("OCI archive contains a foreign path")
                        blob_members["sha256:" + blob_match.group(1)] = member
                    if "oci-layout" not in members or "index.json" not in members:
                        raise ComponentEvidenceError("OCI archive layout metadata is missing")
                    layout = json.loads(
                        _read_tar_member(
                            archive,
                            members["oci-layout"],
                            maximum=_MAX_OCI_JSON_BYTES,
                        )
                    )
                    if layout != {"imageLayoutVersion": "1.0.0"}:
                        raise ComponentEvidenceError("OCI layout version is unsupported")
                    index = json.loads(
                        _read_tar_member(
                            archive,
                            members["index.json"],
                            maximum=_MAX_OCI_JSON_BYTES,
                        )
                    )
                    if (
                        not isinstance(index, dict)
                        or not {"schemaVersion", "manifests"}.issubset(index)
                        or not set(index).issubset(
                            {"schemaVersion", "mediaType", "manifests", "annotations"}
                        )
                        or index.get("schemaVersion") != 2
                        or (
                            "mediaType" in index and index.get("mediaType") != _OCI_INDEX
                        )
                        or not isinstance(index.get("manifests"), list)
                        or len(index["manifests"]) != 1
                    ):
                        raise ComponentEvidenceError("OCI root index is not exact")
                    root = _oci_descriptor(index["manifests"][0], "OCI root")
                    visited: set[str] = set()
                    referenced_blobs: set[str] = set()
                    image_manifests: list[dict[str, Any]] = []

                    def blob_payload(item: dict[str, Any], field: str) -> bytes:
                        digest = str(item["digest"])
                        member = blob_members.get(digest)
                        if member is None or member.size != item["size"]:
                            raise ComponentEvidenceError(f"{field} blob is unavailable")
                        referenced_blobs.add(digest)
                        payload = _read_tar_member(
                            archive,
                            member,
                            maximum=_MAX_OCI_JSON_BYTES,
                        )
                        if _sha256_bytes(payload) != digest:
                            raise ComponentEvidenceError(f"{field} blob digest differs")
                        return payload

                    def walk(item: dict[str, Any], field: str) -> None:
                        digest = str(item["digest"])
                        if digest in visited:
                            raise ComponentEvidenceError("OCI descriptor graph contains a cycle")
                        visited.add(digest)
                        media_type = str(item["mediaType"])
                        try:
                            value = json.loads(blob_payload(item, field))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise ComponentEvidenceError(f"{field} JSON is invalid") from exc
                        if not isinstance(value, dict) or value.get("schemaVersion") != 2:
                            raise ComponentEvidenceError(f"{field} document is invalid")
                        if media_type == _OCI_INDEX:
                            children = value.get("manifests")
                            if (
                                not set(value).issubset(
                                    {
                                        "schemaVersion",
                                        "mediaType",
                                        "manifests",
                                        "annotations",
                                    }
                                )
                                or (
                                    "mediaType" in value
                                    and value.get("mediaType") != _OCI_INDEX
                                )
                                or not isinstance(children, list)
                                or not children
                                or len(children) > 64
                            ):
                                raise ComponentEvidenceError("OCI image index is invalid")
                            for position, child in enumerate(children):
                                walk(
                                    _oci_descriptor(child, f"{field}.manifests[{position}]"),
                                    f"{field}.manifests[{position}]",
                                )
                            return
                        if media_type != _OCI_MANIFEST:
                            raise ComponentEvidenceError("OCI root is not an image subject")
                        if (
                            not {"config", "layers"}.issubset(value)
                            or not set(value).issubset(
                                {
                                    "schemaVersion",
                                    "mediaType",
                                    "config",
                                    "layers",
                                    "annotations",
                                }
                            )
                            or (
                                "mediaType" in value
                                and value.get("mediaType") != _OCI_MANIFEST
                            )
                        ):
                            raise ComponentEvidenceError("OCI image manifest is not exact")
                        config = _oci_descriptor(value["config"], f"{field}.config")
                        if config["mediaType"] != _OCI_CONFIG:
                            raise ComponentEvidenceError("OCI image config media type is invalid")
                        try:
                            config_value = json.loads(blob_payload(config, f"{field}.config"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise ComponentEvidenceError("OCI image config is invalid") from exc
                        if not isinstance(config_value, dict):
                            raise ComponentEvidenceError("OCI image config is invalid")
                        layers = value.get("layers")
                        if not isinstance(layers, list) or not layers:
                            raise ComponentEvidenceError("OCI image has no layers")
                        image_manifests.append(
                            {
                                "layers": [
                                    _oci_descriptor(layer, f"{field}.layers[{position}]")
                                    for position, layer in enumerate(layers)
                                ]
                            }
                        )

                    walk(root, "OCI root")
                    if not image_manifests:
                        raise ComponentEvidenceError("OCI archive has no image manifests")
                    for image in image_manifests:
                        installed_paths: dict[str, tuple[str, str]] = {}
                        for layer in image["layers"]:
                            digest = str(layer["digest"])
                            member = blob_members.get(digest)
                            if member is None or member.size != layer["size"]:
                                raise ComponentEvidenceError("OCI layer blob is unavailable")
                            referenced_blobs.add(digest)
                            layer_hash = hashlib.sha256()
                            layer_stream = archive.extractfile(member)
                            if layer_stream is None:
                                raise ComponentEvidenceError("OCI layer blob is unavailable")
                            while True:
                                chunk = layer_stream.read(1024 * 1024)
                                if not chunk:
                                    break
                                layer_hash.update(chunk)
                            if "sha256:" + layer_hash.hexdigest() != digest:
                                raise ComponentEvidenceError("OCI layer digest differs")
                            _layer_inventory(
                                archive,
                                member,
                                str(layer["mediaType"]),
                                installed_paths,
                            )
                        installed: dict[str, str] = {}
                        for distribution, version in installed_paths.values():
                            if distribution in _BOOTSTRAP_DISTRIBUTIONS:
                                continue
                            if distribution in installed and installed[distribution] != version:
                                raise ComponentEvidenceError(
                                    "OCI image contains duplicate installed distributions"
                                )
                            installed[distribution] = version
                        if installed != expected_distributions:
                            raise ComponentEvidenceError(
                                "OCI installed distributions differ from the closed wheelhouse"
                            )
                    if referenced_blobs != set(blob_members):
                        raise ComponentEvidenceError(
                            "OCI archive contains unreferenced blob material"
                        )
                    return str(root["digest"]), "sha256:" + archive_hash.hexdigest()
            except (OSError, tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                if isinstance(exc, ComponentEvidenceError):
                    raise
                raise ComponentEvidenceError("OCI layout archive is invalid") from exc


def _signature_bundle(
    value: dict[str, Any],
    artifact_digest: str,
    subject_digest: str,
) -> dict[str, str]:
    expected = {
        "schema",
        "scheme",
        "subjectDigest",
        "artifactDigest",
        "signature",
        "verificationMaterialDigest",
        "signerIdentityDigest",
    }
    if set(value) != expected or value.get("schema") != "graphos-external-signature/2":
        raise ComponentEvidenceError("signature adapter returned an unsupported bundle")
    if (
        value.get("subjectDigest") != subject_digest
        or value.get("artifactDigest") != artifact_digest
    ):
        raise ComponentEvidenceError("signature bundle is not bound to the exact subject")
    if not _SCHEME.fullmatch(str(value.get("scheme") or "")):
        raise ComponentEvidenceError("signature scheme is invalid")
    if not _SIGNATURE.fullmatch(str(value.get("signature") or "")):
        raise ComponentEvidenceError("signature value is invalid")
    for field in (
        "subjectDigest",
        "artifactDigest",
        "verificationMaterialDigest",
        "signerIdentityDigest",
    ):
        if not _DIGEST.fullmatch(str(value.get(field) or "")):
            raise ComponentEvidenceError(f"signature {field} is invalid")
    return {field: str(value[field]) for field in sorted(expected)}


def _external_json(env_name: str, payload: bytes) -> dict[str, Any]:
    if not _ENV_NAME.fullmatch(env_name):
        raise ComponentEvidenceError("external command environment name is invalid")
    raw = os.environ.get(env_name, "")
    if not raw:
        raise ComponentEvidenceError("external signing command is absent")
    try:
        command = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ComponentEvidenceError("external command must be a JSON argv array") from exc
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise ComponentEvidenceError("external command must be a JSON argv array")
    try:
        returncode, stdout, stderr = check_compatibility._bounded_adapter(
            command,
            payload,
            maximum=_MAX_SIGNATURE_BYTES,
        )
    except check_compatibility.CompatibilityError as exc:
        raise ComponentEvidenceError("external signing command failed safely") from exc
    if returncode != 0:
        output_digest = hashlib.sha256(stdout + stderr).hexdigest()
        raise ComponentEvidenceError(
            f"external signing failed; output_digest={output_digest}"
        )
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ComponentEvidenceError("external signer returned non-JSON") from exc
    if not isinstance(value, dict):
        raise ComponentEvidenceError("external signer returned a non-object")
    return value


def generate(
    *,
    name: str,
    version: str,
    kind: str,
    artifact_path: Path,
    source_manifest: Path,
    output_dir: Path,
    release_root: Path,
    verifier_env: str,
    capabilities: Iterable[str] = (),
    entry_count: int | None = None,
    wheelhouse: Path | None = None,
    signature_bundle_path: Path | None = None,
    signer_env: str | None = None,
    verify_signature: bool = False,
) -> dict[str, Any]:
    """Generate all evidence files and return an assembly component declaration."""

    if not _NAME.fullmatch(name):
        raise ComponentEvidenceError("component name is invalid")
    try:
        parsed_version = Version(version)
    except InvalidVersion as exc:
        raise ComponentEvidenceError("component version is invalid") from exc
    if str(parsed_version) != version:
        raise ComponentEvidenceError("component version is not canonical")
    if kind not in {"oci", "catalog"}:
        raise ComponentEvidenceError("component kind is invalid")
    if not _ENV_NAME.fullmatch(verifier_env):
        raise ComponentEvidenceError("signature verifier environment name is invalid")
    capabilities_list = sorted(set(capabilities))
    if any(not _NAME.fullmatch(value) for value in capabilities_list):
        raise ComponentEvidenceError("component capability is invalid")
    if entry_count is not None and entry_count < 1:
        raise ComponentEvidenceError("component entry count is invalid")
    if (signature_bundle_path is None) == (signer_env is None):
        raise ComponentEvidenceError(
            "provide exactly one external signer environment or signature bundle"
        )

    source_snapshot_digest, source_evidence_digest = _source_freeze(source_manifest)
    wheel_components = _wheel_components(wheelhouse)
    if kind == "oci":
        if wheelhouse is None or not wheel_components:
            raise ComponentEvidenceError("OCI evidence requires a closed wheelhouse")
        expected_distributions = {
            str(item["name"]): str(item["version"]) for item in wheel_components
        }
        artifact_digest, artifact_input_digest = _oci_archive_identity(
            artifact_path,
            expected_distributions,
        )
        artifact_format = "oci-layout-archive"
    else:
        artifact_digest, _ = _file_digest(
            artifact_path,
            maximum=_MAX_ARTIFACT_BYTES,
        )
        artifact_input_digest = artifact_digest
        artifact_format = "opaque-catalog"
    root = release_root.resolve()
    destination = output_dir.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ComponentEvidenceError("component evidence must remain under release root") from exc

    source = {
        "apiVersion": "graphos.io/v1",
        "kind": "ComponentSourceEvidence",
        "component": name,
        "version": version,
        "artifactFormat": artifact_format,
        "artifactDigest": artifact_digest,
        "artifactInputDigest": artifact_input_digest,
        "sourceSnapshotDigest": source_snapshot_digest,
        "sourceEvidenceDigest": source_evidence_digest,
    }
    package_type = "pypi" if kind == "oci" else "generic"
    purl = f"pkg:{package_type}/{name}@{version}"
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application" if kind == "oci" else "data",
                "bom-ref": purl,
                "name": name,
                "version": version,
                "purl": purl,
                "hashes": [
                    {"alg": "SHA-256", "content": artifact_digest.removeprefix("sha256:")}
                ],
            }
        },
        "components": wheel_components,
    }
    provenance = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {"name": name, "digest": {"sha256": artifact_digest.removeprefix("sha256:")}}
        ],
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://graphos.invalid/build/exact-local/v1",
                "externalParameters": {},
                "internalParameters": {},
                "resolvedDependencies": [
                    {
                        "uri": "urn:graphos:source-freeze",
                        "digest": {
                            "sha256": source_snapshot_digest.removeprefix("sha256:")
                        },
                    }
                ],
            },
            "runDetails": {
                "builder": {"id": "https://graphos.invalid/builders/exact-local/v1"},
                "byproducts": [],
            },
        },
    }
    source_payload = _canonical(source)
    sbom_payload = _canonical(sbom)
    provenance_payload = _canonical(provenance)
    paths = {
        "source": destination / "source.json",
        "sbom": destination / "sbom.cyclonedx.json",
        "provenance": destination / "provenance.intoto.json",
        "signatureBundle": destination / "signature-bundle.json",
    }
    input_paths = [artifact_path, source_manifest]
    if signature_bundle_path is not None:
        input_paths.append(signature_bundle_path)
    if wheelhouse is not None:
        try:
            if destination == wheelhouse.resolve() or destination.is_relative_to(
                wheelhouse.resolve()
            ):
                raise ComponentEvidenceError(
                    "component evidence must not be written into the wheelhouse"
                )
        except OSError as exc:
            raise ComponentEvidenceError("wheelhouse is unavailable") from exc
    input_identities = {path.resolve() for path in input_paths}
    if any(path.resolve() in input_identities for path in paths.values()):
        raise ComponentEvidenceError("component evidence outputs must not alias inputs")
    evidence_references = {key: _relative(path, root) for key, path in paths.items()}
    component_subject: dict[str, Any] = {
        "version": version,
        "kind": kind,
        "artifact": f"{kind}:{name}@{artifact_digest}",
        "digest": artifact_digest,
        "sourceDigest": _sha256_bytes(source_payload),
        "sbomDigest": _sha256_bytes(sbom_payload),
        "provenanceDigest": _sha256_bytes(provenance_payload),
        "signatureVerifierEnv": verifier_env,
        "capabilities": capabilities_list,
        "evidence": evidence_references,
    }
    if entry_count is not None:
        component_subject["entryCount"] = entry_count
    signer_subject = check_compatibility.component_signing_subject(
        name,
        component_subject,
    )
    signer_subject_digest = _sha256_bytes(signer_subject)
    raw_bundle = (
        _safe_json(signature_bundle_path, maximum=_MAX_SIGNATURE_BYTES)
        if signature_bundle_path is not None
        else _external_json(str(signer_env), signer_subject)
    )
    bundle = _signature_bundle(
        raw_bundle,
        artifact_digest,
        signer_subject_digest,
    )
    bundle_payload = _canonical(bundle)
    for path, payload in (
        (paths["source"], source_payload),
        (paths["sbom"], sbom_payload),
        (paths["provenance"], provenance_payload),
        (paths["signatureBundle"], bundle_payload),
    ):
        _write(path, payload, release_root=root)

    declaration: dict[str, Any] = {
        "version": version,
        "kind": kind,
        "artifact": component_subject["artifact"],
        "digest": artifact_digest,
        "evidence": evidence_references,
        "signatureVerifierEnv": verifier_env,
        "capabilities": capabilities_list,
    }
    if entry_count is not None:
        declaration["entryCount"] = entry_count

    component = {
        **component_subject,
        "signature": {
            "bundleDigest": _sha256_bytes(bundle_payload),
            "verifierEnv": verifier_env,
        },
    }
    component.pop("signatureVerifierEnv", None)
    manifest_path = root / "release-manifest.json"
    check_compatibility._validate_component_evidence(name, component, manifest_path)
    if verify_signature:
        encoded = check_compatibility._validate_component_evidence(
            name, component, manifest_path
        )
        check_compatibility._verify_signature(name, component, encoded)
    return declaration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--kind", choices=("oci", "catalog"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier-env", required=True)
    parser.add_argument("--signer-env")
    parser.add_argument("--signature-bundle", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--capability", action="append", default=[])
    parser.add_argument("--entry-count", type=int)
    parser.add_argument("--verify-signature", action="store_true")
    args = parser.parse_args()
    try:
        root = args.release_root.resolve()
        output = args.output.absolute()
        output.relative_to(root)
        reserved = {
            (args.output_dir.resolve() / name).absolute()
            for name in (
                "source.json",
                "sbom.cyclonedx.json",
                "provenance.intoto.json",
                "signature-bundle.json",
            )
        }
        input_paths = {args.artifact.resolve(), args.source_manifest.resolve()}
        if args.signature_bundle is not None:
            input_paths.add(args.signature_bundle.resolve())
        if output in reserved or args.output.resolve() in input_paths:
            raise ComponentEvidenceError(
                "component declaration output must not alias evidence or inputs"
            )
        declaration = generate(
            name=args.name,
            version=args.version,
            kind=args.kind,
            artifact_path=args.artifact,
            source_manifest=args.source_manifest,
            output_dir=args.output_dir,
            release_root=args.release_root,
            verifier_env=args.verifier_env,
            capabilities=args.capability,
            entry_count=args.entry_count,
            wheelhouse=args.wheelhouse,
            signature_bundle_path=args.signature_bundle,
            signer_env=args.signer_env,
            verify_signature=args.verify_signature,
        )
        _write(args.output, _canonical(declaration), release_root=args.release_root)
    except Exception as exc:  # noqa: BLE001 - privacy-safe release boundary
        print(json.dumps({"error": type(exc).__name__, "ok": False}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
