"""Fail-closed authority proof for loadgen deployment manifests.

The workspace snapshot may contain deployment-shaped files that are not owned by
any Git repository.  Such files are not a deployable source authority.  This
module accepts a manifest only when the file is a regular tracked file in a
repository whose remote, commit, and exact bytes all match the reviewed pins.
The workspace registry is part of the authority proof and must also register
the service; an unregistered ``services/loadgen`` copy therefore cannot qualify.

The checker is local and side-effect free: Git is queried without a shell or
network, and failures expose stable codes rather than paths or command output.
The returned authority digest is safe to carry in runtime evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml

_DIGEST = re.compile(r"^sha256:(?!0{64}$)[a-f0-9]{64}$")
_REVISION = re.compile(r"^[a-f0-9]{40}$")
_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_TEXT_BYTES = 4096
_GIT_TIMEOUT_SECONDS = 5


class LoadgenSourceAuthorityError(RuntimeError):
    """Stable, path-free failure from the loadgen source-authority gate."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _text(value: object, code: str) -> str:
    if not isinstance(value, str):
        raise LoadgenSourceAuthorityError(code)
    text = value.strip()
    if (
        not text
        or len(text.encode("utf-8")) > _MAX_TEXT_BYTES
        or any(character in text for character in "\x00\r\n")
    ):
        raise LoadgenSourceAuthorityError(code)
    return text


def _repository_url(value: object) -> str:
    text = _text(value, "source_repository_invalid")
    if "://" in text:
        parsed = urlsplit(text)
        if parsed.scheme not in {"http", "https", "ssh"} or not parsed.netloc:
            raise LoadgenSourceAuthorityError("source_repository_invalid")
        if parsed.username or parsed.password or not parsed.path:
            raise LoadgenSourceAuthorityError("source_repository_invalid")
    elif re.fullmatch(r"[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:.+", text) is None:
        raise LoadgenSourceAuthorityError("source_repository_invalid")
    return text.rstrip("/")


def _revision(value: object) -> str:
    text = _text(value, "source_revision_invalid").casefold()
    if _REVISION.fullmatch(text) is None:
        raise LoadgenSourceAuthorityError("source_revision_invalid")
    return text


def _digest(value: object, code: str) -> str:
    text = _text(value, code).casefold()
    if _DIGEST.fullmatch(text) is None:
        raise LoadgenSourceAuthorityError(code)
    return text


def source_authority_digest(
    repository: str,
    revision: str,
    manifest_digest: str,
) -> str:
    """Return the deterministic opaque identity of one checked source."""

    payload = {
        "manifest_digest": _digest(manifest_digest, "source_manifest_digest_invalid"),
        "repository": _repository_url(repository),
        "revision": _revision(revision),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _run_git(root: Path, arguments: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LoadgenSourceAuthorityError("source_git_unavailable") from exc
    if completed.returncode != 0:
        raise LoadgenSourceAuthorityError("source_git_authority_missing")
    return completed.stdout.strip()


def _git_root(start: Path) -> Path:
    raw = _run_git(start, ("rev-parse", "--show-toplevel"))
    if not raw:
        raise LoadgenSourceAuthorityError("source_git_authority_missing")
    root = Path(raw)
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise LoadgenSourceAuthorityError("source_git_authority_missing") from exc
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink():
        raise LoadgenSourceAuthorityError("source_git_authority_missing")
    return root.resolve(strict=True)


def _assert_workspace_registration(
    workspace_manifest: Path,
    *,
    service_name: str,
    repository: str,
) -> None:
    try:
        metadata = workspace_manifest.lstat()
        if (
            workspace_manifest.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > _MAX_FILE_BYTES
        ):
            raise LoadgenSourceAuthorityError("workspace_registry_invalid")
        payload = yaml.safe_load(workspace_manifest.read_text(encoding="utf-8"))
    except LoadgenSourceAuthorityError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LoadgenSourceAuthorityError("workspace_registry_invalid") from exc

    # Only the canonical ``services.items`` registry is authoritative.  A
    # recursively discovered ``name: loadgen`` object elsewhere in the YAML
    # would let an unrelated metadata block satisfy the source proof.
    if not isinstance(payload, Mapping):
        raise LoadgenSourceAuthorityError("workspace_registry_invalid")
    services = payload.get("services")
    if not isinstance(services, Mapping):
        raise LoadgenSourceAuthorityError("workspace_registry_invalid")
    items = services.get("items")
    if not isinstance(items, list) or len(items) > 4096:
        raise LoadgenSourceAuthorityError("workspace_registry_invalid")
    matches: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise LoadgenSourceAuthorityError("workspace_registry_invalid")
        if item.get("name") == service_name and "url" in item:
            matches.append(_repository_url(item["url"]))
    if len(matches) != 1 or matches[0] != _repository_url(repository):
        raise LoadgenSourceAuthorityError("source_service_not_registered")


@dataclass(frozen=True)
class LoadgenSourceAuthority:
    """Privacy-safe identity of a tracked loadgen source manifest."""

    repository: str
    revision: str
    manifest_digest: str
    authority_digest: str

    def as_report(self) -> dict[str, str]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "manifest_digest": self.manifest_digest,
            "authority_digest": self.authority_digest,
        }


def verify_tracked_manifest(
    manifest_path: Path,
    *,
    repository: str,
    revision: str,
    manifest_digest: str,
    workspace_manifest: Path,
    service_name: str = "loadgen",
) -> LoadgenSourceAuthority:
    """Verify one manifest is tracked by the exact pinned source authority."""

    if not manifest_path.is_absolute():
        raise LoadgenSourceAuthorityError("source_manifest_path_invalid")
    try:
        metadata = manifest_path.lstat()
        if (
            manifest_path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > _MAX_FILE_BYTES
        ):
            raise LoadgenSourceAuthorityError("source_manifest_invalid")
        payload = manifest_path.read_bytes()
        if len(payload) > _MAX_FILE_BYTES:
            raise LoadgenSourceAuthorityError("source_manifest_invalid")
    except LoadgenSourceAuthorityError:
        raise
    except OSError as exc:
        raise LoadgenSourceAuthorityError("source_manifest_unavailable") from exc

    expected_repository = _repository_url(repository)
    expected_revision = _revision(revision)
    expected_manifest_digest = _digest(
        manifest_digest, "source_manifest_digest_invalid"
    )
    actual_manifest_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual_manifest_digest != expected_manifest_digest:
        raise LoadgenSourceAuthorityError("source_manifest_digest_mismatch")

    root = _git_root(manifest_path.parent)
    try:
        relative = manifest_path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise LoadgenSourceAuthorityError("source_manifest_outside_repository") from exc
    relative_text = relative.as_posix()
    tracked = _run_git(root, ("ls-files", "--error-unmatch", "--", relative_text))
    if tracked != relative_text:
        raise LoadgenSourceAuthorityError("source_manifest_untracked")
    actual_repository = _repository_url(
        _run_git(root, ("config", "--get", "remote.origin.url"))
    )
    if actual_repository != expected_repository:
        raise LoadgenSourceAuthorityError("source_repository_mismatch")
    actual_revision = _run_git(root, ("rev-parse", "HEAD")).casefold()
    if actual_revision != expected_revision:
        raise LoadgenSourceAuthorityError("source_revision_mismatch")
    _assert_workspace_registration(
        workspace_manifest,
        service_name=service_name,
        repository=expected_repository,
    )
    return LoadgenSourceAuthority(
        repository=expected_repository,
        revision=expected_revision,
        manifest_digest=expected_manifest_digest,
        authority_digest=source_authority_digest(
            expected_repository,
            expected_revision,
            expected_manifest_digest,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that the loadgen manifest has a tracked Git authority."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest-digest", required=True)
    parser.add_argument("--workspace-manifest", type=Path, required=True)
    parser.add_argument("--service-name", default="loadgen")
    args = parser.parse_args(argv)
    try:
        authority = verify_tracked_manifest(
            args.manifest,
            repository=args.repository,
            revision=args.revision,
            manifest_digest=args.manifest_digest,
            workspace_manifest=args.workspace_manifest,
            service_name=args.service_name,
        )
    except LoadgenSourceAuthorityError as exc:
        print(json.dumps({"ok": False, "error": exc.code}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {"ok": True, "authority": authority.as_report()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
