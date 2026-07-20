#!/usr/bin/env python3
"""Export one exact local image as a verified, private OCI-layout archive.

The container runtime is an argv-only materializer.  It writes the archive to a
caller-owned descriptor; this module independently validates the OCI descriptor
graph and publishes the result without an overwrite window.  Image references,
container-runtime diagnostics, and filesystem locations never enter status output.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import resource
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

_DEFAULT_CONTAINER_CLI = "podman"
_EXPORT_TIMEOUT_SECONDS = 30 * 60
_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
_MAX_ENTRIES = 65_536
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_DESCRIPTOR_DEPTH = 16
_MAX_DESCRIPTORS = 65_536
_MAX_METADATA_STRING = 65_536

_DIGEST = re.compile(r"^sha256:(?!0{64}$)[a-f0-9]{64}$")
_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_IMAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,447}$")
_BLOB_PATH = re.compile(r"^blobs/sha256/([a-f0-9]{64})$")
_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HOST_USER_PATH = re.compile(
    r"(?i)(?:[a-z]:[\\/](?:users|documents and settings)[\\/][^\\/\s]+"
    r"|/mnt/[a-z]/users/[^/\s]+|/users/[^/\s]+)"
)
_EMAIL = re.compile(r"(?i)(?<![a-z0-9._%+-])[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}")
_CREDENTIAL_URI = re.compile(r"(?i)[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@")
_SENSITIVE_KEY = re.compile(
    r"(?i)(?:^|[_.-])(?:api[_.-]?key|credential|password|passwd|private[_.-]?key|secret|token)(?:$|[_.-])"
)

_OCI_INDEX = "application/vnd.oci.image.index.v1+json"
_OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
_OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
_OCI_LAYERS = {
    "application/vnd.oci.image.layer.v1.tar",
    "application/vnd.oci.image.layer.v1.tar+gzip",
}


class OciLayoutExportError(ValueError):
    """A stable, privacy-safe export rejection."""

    def __init__(self, code: str) -> None:
        if _ERROR_CODE.fullmatch(code) is None:
            code = "unexpected_failure"
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class OciLayoutIdentity:
    """Path-free identity of a verified OCI-layout archive."""

    root_digest: str
    archive_sha256: str
    byte_size: int
    image_manifest_count: int


def _reject(code: str) -> None:
    raise OciLayoutExportError(code)


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_image_reference(value: str) -> str | None:
    if (
        not value
        or len(value) > 512
        or not value.isascii()
        or _CONTROL.search(value)
        or any(character.isspace() for character in value)
    ):
        _reject("image_reference_not_exact")
    if _DIGEST.fullmatch(value):
        return None
    name, separator, digest = value.rpartition("@")
    if (
        separator != "@"
        or _DIGEST.fullmatch(digest) is None
        or _IMAGE_NAME.fullmatch(name) is None
        or ".." in name
        or "//" in name
        or name.endswith(":")
    ):
        _reject("image_reference_not_exact")
    return digest


def _resolve_container_cli(value: str) -> tuple[Path, int, tuple[int, int, int, int]]:
    if not value or len(value) > 4096 or _CONTROL.search(value):
        _reject("container_cli_invalid")
    supplied = Path(value)
    if supplied.is_absolute():
        candidate = supplied
    elif supplied.name == value and value not in {".", ".."}:
        discovered = shutil.which(value)
        if discovered is None:
            _reject("container_cli_unavailable")
        candidate = Path(discovered)
    else:
        _reject("container_cli_invalid")
    try:
        resolved = candidate.resolve(strict=True)
        descriptor = os.open(
            resolved,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
    except OSError:
        _reject("container_cli_unavailable")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o111 == 0:
        os.close(descriptor)
        _reject("container_cli_invalid")
    identity = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    return resolved, descriptor, identity


def _revalidate_container_cli(
    path: Path, descriptor: int, expected: tuple[int, int, int, int]
) -> None:
    try:
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = path.stat(follow_symlinks=False)
    except OSError:
        _reject("container_cli_changed")
    observed = (
        descriptor_metadata.st_dev,
        descriptor_metadata.st_ino,
        descriptor_metadata.st_mtime_ns,
        descriptor_metadata.st_ctime_ns,
    )
    if (
        observed != expected
        or (path_metadata.st_dev, path_metadata.st_ino) != expected[:2]
    ):
        _reject("container_cli_changed")


def _open_private_parent(parent: Path) -> int:
    if (
        not parent.is_absolute()
        or parent.anchor != "/"
        or _CONTROL.search(str(parent))
        or any(part in {"", ".", ".."} for part in parent.parts[1:])
    ):
        _reject("output_parent_invalid")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open("/", flags | nofollow)
        for component in parent.parts[1:]:
            next_descriptor = os.open(
                component,
                flags | nofollow,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        metadata = os.fstat(descriptor)
    except OSError:
        with contextlib.suppress(UnboundLocalError, OSError):
            os.close(descriptor)
        _reject("output_parent_invalid")
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        os.close(descriptor)
        _reject("output_parent_not_private")
    return descriptor


def _safe_member_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        _reject("archive_path_unsafe")
    normalized = path.as_posix()
    if value.rstrip("/") != normalized:
        _reject("archive_path_noncanonical")
    return normalized


def _read_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo, maximum: int
) -> bytes:
    if not member.isfile() or member.size < 0 or member.size > maximum:
        _reject("archive_member_size_invalid")
    stream = archive.extractfile(member)
    if stream is None:
        _reject("archive_member_unavailable")
    payload = bytearray()
    while len(payload) <= maximum:
        chunk = stream.read(min(1024 * 1024, maximum + 1 - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) != member.size or len(payload) > maximum:
        _reject("archive_member_size_invalid")
    return bytes(payload)


def _json_member(
    archive: tarfile.TarFile, member: tarfile.TarInfo, code: str
) -> dict[str, Any]:
    try:
        value = json.loads(_read_member(archive, member, _MAX_JSON_BYTES))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        _reject(code)
    if not isinstance(value, dict):
        _reject(code)
    return value


def _assert_metadata_private(value: Any, *, depth: int = 0) -> None:
    if depth > 64:
        _reject("archive_metadata_privacy_violation")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _reject("archive_metadata_privacy_violation")
            if _SENSITIVE_KEY.search(key) and item not in (None, "", [], {}):
                _reject("archive_metadata_privacy_violation")
            _assert_metadata_private(key, depth=depth + 1)
            _assert_metadata_private(item, depth=depth + 1)
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and "=" in item:
                key, item_value = item.split("=", 1)
                if _SENSITIVE_KEY.search(key) and item_value:
                    _reject("archive_metadata_privacy_violation")
            _assert_metadata_private(item, depth=depth + 1)
        return
    if isinstance(value, str) and (
        len(value) > _MAX_METADATA_STRING
        or _CONTROL.search(value)
        or _HOST_USER_PATH.search(value)
        or _EMAIL.search(value)
        or _CREDENTIAL_URI.search(value)
    ):
        _reject("archive_metadata_privacy_violation")


def _descriptor(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        _reject("archive_descriptor_invalid")
    allowed = {"mediaType", "digest", "size", "annotations", "platform", "artifactType"}
    if not {"mediaType", "digest", "size"}.issubset(value) or not set(value).issubset(
        allowed
    ):
        _reject("archive_descriptor_invalid")
    if (
        not isinstance(value.get("mediaType"), str)
        or _DIGEST.fullmatch(str(value.get("digest") or "")) is None
        or not isinstance(value.get("size"), int)
        or not 0 < value["size"] <= _MAX_ARCHIVE_BYTES
    ):
        _reject("archive_descriptor_invalid")
    for key in ("annotations", "platform"):
        if key in value and not isinstance(value[key], dict):
            _reject("archive_descriptor_invalid")
    if "artifactType" in value and not isinstance(value["artifactType"], str):
        _reject("archive_descriptor_invalid")
    _assert_metadata_private(value.get("annotations", {}))
    return value


def _hash_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    stream = archive.extractfile(member)
    if stream is None:
        _reject("archive_blob_unavailable")
    digest = hashlib.sha256()
    observed = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        observed += len(chunk)
        if observed > _MAX_ARCHIVE_BYTES:
            _reject("archive_blob_size_invalid")
        digest.update(chunk)
    if observed != member.size:
        _reject("archive_blob_size_invalid")
    return "sha256:" + digest.hexdigest()


def _archive_hash(stream: BinaryIO, expected_size: int) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    observed = 0
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        observed += len(chunk)
        if observed > _MAX_ARCHIVE_BYTES:
            _reject("archive_size_invalid")
        digest.update(chunk)
    if observed != expected_size:
        _reject("archive_changed")
    return "sha256:" + digest.hexdigest()


def validate_oci_layout(descriptor: int) -> OciLayoutIdentity:
    """Validate an open OCI archive without trusting its filename or producer."""

    try:
        before = os.fstat(descriptor)
    except OSError:
        _reject("archive_unavailable")
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= _MAX_ARCHIVE_BYTES
    ):
        _reject("archive_size_invalid")
    with os.fdopen(os.dup(descriptor), "rb") as archive_stream:
        archive_digest = _archive_hash(archive_stream, before.st_size)
        archive_stream.seek(0)
        try:
            with tarfile.open(fileobj=archive_stream, mode="r:*") as archive:
                members: dict[str, tarfile.TarInfo] = {}
                blobs: dict[str, tarfile.TarInfo] = {}
                for position, member in enumerate(archive, start=1):
                    if position > _MAX_ENTRIES:
                        _reject("archive_entry_count_invalid")
                    normalized = _safe_member_name(member.name)
                    if normalized in members:
                        _reject("archive_duplicate_path")
                    if (
                        member.uid != 0
                        or member.gid != 0
                        or member.uname not in {"", "root"}
                        or member.gname not in {"", "root"}
                    ):
                        _reject("archive_header_privacy_violation")
                    _assert_metadata_private(member.pax_headers)
                    members[normalized] = member
                    if member.isdir():
                        if normalized not in {"blobs", "blobs/sha256"}:
                            _reject("archive_foreign_path")
                        continue
                    if not member.isfile() or member.issparse():
                        _reject("archive_member_type_invalid")
                    if normalized in {"oci-layout", "index.json"}:
                        continue
                    match = _BLOB_PATH.fullmatch(normalized)
                    if match is None:
                        _reject("archive_foreign_path")
                    blobs["sha256:" + match.group(1)] = member
                if "oci-layout" not in members or "index.json" not in members:
                    _reject("archive_layout_missing")

                layout = _json_member(
                    archive, members["oci-layout"], "archive_layout_invalid"
                )
                if layout != {"imageLayoutVersion": "1.0.0"}:
                    _reject("archive_layout_invalid")
                index = _json_member(
                    archive, members["index.json"], "archive_index_invalid"
                )
                if (
                    not {"schemaVersion", "manifests"}.issubset(index)
                    or not set(index).issubset(
                        {"schemaVersion", "mediaType", "manifests", "annotations"}
                    )
                    or index.get("schemaVersion") != 2
                    or ("mediaType" in index and index.get("mediaType") != _OCI_INDEX)
                    or not isinstance(index.get("manifests"), list)
                    or len(index["manifests"]) != 1
                ):
                    _reject("archive_index_invalid")
                _assert_metadata_private(index.get("annotations", {}))
                root = _descriptor(index["manifests"][0])
                referenced: set[str] = set()
                visiting: set[str] = set()
                visited: set[str] = set()
                descriptor_count = 0
                image_manifest_count = 0

                def blob_payload(item: dict[str, Any], *, maximum: int) -> bytes:
                    digest = str(item["digest"])
                    member = blobs.get(digest)
                    if member is None or member.size != item["size"]:
                        _reject("archive_blob_unavailable")
                    referenced.add(digest)
                    if maximum < member.size:
                        _reject("archive_blob_size_invalid")
                    payload = _read_member(archive, member, maximum)
                    if "sha256:" + hashlib.sha256(payload).hexdigest() != digest:
                        _reject("archive_blob_digest_mismatch")
                    return payload

                def walk(item: dict[str, Any], depth: int) -> None:
                    nonlocal descriptor_count, image_manifest_count
                    descriptor_count += 1
                    if (
                        descriptor_count > _MAX_DESCRIPTORS
                        or depth > _MAX_DESCRIPTOR_DEPTH
                    ):
                        _reject("archive_descriptor_graph_invalid")
                    digest = str(item["digest"])
                    if digest in visiting or digest in visited:
                        _reject("archive_descriptor_graph_invalid")
                    visiting.add(digest)
                    try:
                        payload = blob_payload(item, maximum=_MAX_JSON_BYTES)
                        try:
                            document = json.loads(payload)
                        except (
                            UnicodeDecodeError,
                            json.JSONDecodeError,
                            RecursionError,
                        ):
                            _reject("archive_descriptor_json_invalid")
                        if (
                            not isinstance(document, dict)
                            or document.get("schemaVersion") != 2
                        ):
                            _reject("archive_descriptor_json_invalid")
                        _assert_metadata_private(document.get("annotations", {}))
                        media_type = str(item["mediaType"])
                        if media_type == _OCI_INDEX:
                            children = document.get("manifests")
                            if (
                                not set(document).issubset(
                                    {
                                        "schemaVersion",
                                        "mediaType",
                                        "manifests",
                                        "annotations",
                                    }
                                )
                                or (
                                    "mediaType" in document
                                    and document.get("mediaType") != _OCI_INDEX
                                )
                                or not isinstance(children, list)
                                or not children
                                or len(children) > 64
                            ):
                                _reject("archive_index_invalid")
                            for child in children:
                                walk(_descriptor(child), depth + 1)
                            return
                        if media_type != _OCI_MANIFEST:
                            _reject("archive_subject_invalid")
                        if (
                            not {"config", "layers"}.issubset(document)
                            or not set(document).issubset(
                                {
                                    "schemaVersion",
                                    "mediaType",
                                    "config",
                                    "layers",
                                    "annotations",
                                }
                            )
                            or (
                                "mediaType" in document
                                and document.get("mediaType") != _OCI_MANIFEST
                            )
                        ):
                            _reject("archive_manifest_invalid")
                        config = _descriptor(document["config"])
                        if config["mediaType"] != _OCI_CONFIG:
                            _reject("archive_config_invalid")
                        try:
                            config_document = json.loads(
                                blob_payload(config, maximum=_MAX_JSON_BYTES)
                            )
                        except (
                            UnicodeDecodeError,
                            json.JSONDecodeError,
                            RecursionError,
                        ):
                            _reject("archive_config_invalid")
                        if not isinstance(config_document, dict):
                            _reject("archive_config_invalid")
                        _assert_metadata_private(config_document)
                        layers = document.get("layers")
                        if not isinstance(layers, list) or not layers:
                            _reject("archive_layers_invalid")
                        for raw_layer in layers:
                            layer = _descriptor(raw_layer)
                            if layer["mediaType"] not in _OCI_LAYERS:
                                _reject("archive_layer_media_type_invalid")
                            layer_digest = str(layer["digest"])
                            layer_member = blobs.get(layer_digest)
                            if (
                                layer_member is None
                                or layer_member.size != layer["size"]
                            ):
                                _reject("archive_blob_unavailable")
                            referenced.add(layer_digest)
                            if _hash_member(archive, layer_member) != layer_digest:
                                _reject("archive_blob_digest_mismatch")
                        image_manifest_count += 1
                    finally:
                        visiting.discard(digest)
                        visited.add(digest)

                walk(root, 0)
                if image_manifest_count == 0:
                    _reject("archive_subject_invalid")
                if referenced != set(blobs):
                    _reject("archive_unreferenced_blob")
        except (OSError, tarfile.TarError, EOFError):
            _reject("archive_tar_invalid")
    try:
        after = os.fstat(descriptor)
    except OSError:
        _reject("archive_changed")
    if _file_identity(after) != _file_identity(before):
        _reject("archive_changed")
    return OciLayoutIdentity(
        root_digest=str(root["digest"]),
        archive_sha256=archive_digest,
        byte_size=before.st_size,
        image_manifest_count=image_manifest_count,
    )


def _child_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    limit = _MAX_ARCHIVE_BYTES
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    if soft != resource.RLIM_INFINITY:
        limit = min(limit, max(soft, 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))


def _terminate(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


def _run_container_export(
    cli_path: Path,
    cli_descriptor: int,
    image_reference: str,
    archive_descriptor: int,
) -> None:
    executable = f"/proc/self/fd/{cli_descriptor}"
    if not Path(executable).exists():
        _reject("exact_executable_boundary_unavailable")
    argv = [str(cli_path), "save", "--format", "oci-archive", image_reference]
    try:
        process = subprocess.Popen(
            argv,
            executable=executable,
            stdin=subprocess.DEVNULL,
            stdout=archive_descriptor,
            stderr=subprocess.DEVNULL,
            shell=False,
            close_fds=True,
            pass_fds=(cli_descriptor,),
            start_new_session=True,
            preexec_fn=_child_limits,
        )
        try:
            return_code = process.wait(timeout=_EXPORT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _terminate(process)
            _reject("container_export_timeout")
    except OciLayoutExportError:
        raise
    except Exception:
        _reject("container_export_unavailable")
    if return_code != 0:
        _reject("container_export_failed")


def export_oci_layout(
    *, image_reference: str, output: Path, container_cli: str
) -> OciLayoutIdentity:
    """Export and no-replace publish one digest-addressed local image."""

    expected_root = _validate_image_reference(image_reference)
    if (
        not output.is_absolute()
        or output.anchor != "/"
        or _CONTROL.search(str(output))
        or _OUTPUT_NAME.fullmatch(output.name) is None
        or any(part in {"", ".", ".."} for part in output.parts[1:])
    ):
        _reject("output_destination_invalid")
    parent_descriptor = _open_private_parent(output.parent)
    cli_descriptor: int | None = None
    archive_descriptor: int | None = None
    temporary = f".{output.name}.{secrets.token_hex(8)}.tmp"
    try:
        try:
            os.stat(output.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError:
            _reject("output_destination_invalid")
        else:
            _reject("output_exists")
        cli_path, cli_descriptor, cli_identity = _resolve_container_cli(container_cli)
        try:
            archive_descriptor = os.open(
                temporary,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            os.fchmod(archive_descriptor, 0o600)
        except OSError:
            _reject("temporary_output_unavailable")
        _run_container_export(
            cli_path,
            cli_descriptor,
            image_reference,
            archive_descriptor,
        )
        os.fsync(archive_descriptor)
        identity = validate_oci_layout(archive_descriptor)
        if expected_root is not None and identity.root_digest != expected_root:
            _reject("exported_root_digest_mismatch")
        _revalidate_container_cli(cli_path, cli_descriptor, cli_identity)
        linked = False
        try:
            os.link(
                temporary,
                output.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            linked = True
            os.unlink(temporary, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
        except FileExistsError:
            _reject("output_exists")
        except OSError:
            if linked:
                with contextlib.suppress(OSError):
                    os.unlink(output.name, dir_fd=parent_descriptor)
            _reject("output_publish_failed")
        try:
            published = os.stat(
                output.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            current = os.fstat(archive_descriptor)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(output.name, dir_fd=parent_descriptor)
            _reject("output_publish_verification_failed")
        if (
            not stat.S_ISREG(published.st_mode)
            or (published.st_dev, published.st_ino) != (current.st_dev, current.st_ino)
            or published.st_nlink != 1
            or stat.S_IMODE(published.st_mode) != 0o600
        ):
            with contextlib.suppress(OSError):
                os.unlink(output.name, dir_fd=parent_descriptor)
            _reject("output_publish_verification_failed")
        return identity
    finally:
        if archive_descriptor is not None:
            os.close(archive_descriptor)
        if cli_descriptor is not None:
            os.close(cli_descriptor)
        with contextlib.suppress(FileNotFoundError, OSError):
            os.unlink(temporary, dir_fd=parent_descriptor)
        os.close(parent_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        required=True,
        help="Local sha256 image ID or name pinned with @sha256:<digest>.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--container-cli",
        default=_DEFAULT_CONTAINER_CLI,
        help="Compatible container CLI executable (default: podman).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        identity = export_oci_layout(
            image_reference=args.image,
            output=args.output,
            container_cli=args.container_cli,
        )
    except OciLayoutExportError as exc:
        print(
            json.dumps(
                {
                    "errorCode": exc.code,
                    "schema": "oci-layout-export-status/1",
                    "status": "rejected",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    except Exception:
        print(
            '{"errorCode":"unexpected_failure",'
            '"schema":"oci-layout-export-status/1","status":"rejected"}',
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "archiveSha256": identity.archive_sha256,
                "byteSize": identity.byte_size,
                "imageManifestCount": identity.image_manifest_count,
                "rootDigest": identity.root_digest,
                "schema": "oci-layout-export-status/1",
                "status": "passed",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
