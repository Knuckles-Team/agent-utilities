#!/usr/bin/env python3
"""Inventory and integrity-check every committed Python, Rust, and pnpm lock.

The fleet inventory reads lock files directly and never invokes a resolver,
package manager, build backend, or project code, and it makes no network
requests.  Every registry artifact must carry a verifiable digest: Python
artifacts use HTTPS with a SHA-256 hash, Cargo packages an HTTPS registry with a
64-character checksum, and pnpm packages a SHA-512 integrity value.  No absolute
checkout path is included in diagnostics.

Vulnerability lookups against external advisory services are deliberately not
part of this gate: blocking gates must behave identically offline, locally and
in CI.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import stat
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from check_fleet_supply_chain import resolve_snapshot_repositories


class AuditError(RuntimeError):
    """A lock inventory is missing, malformed, or unverifiable."""


MAX_LOCK_BYTES = 64 * 1024 * 1024
MAX_COORDINATES = 30_000
MAX_REPOSITORIES = 1_000
MAX_DISCOVERY_DEPTH = 5
PACKAGE_RE = re.compile(r"^[A-Za-z0-9@][A-Za-z0-9@/._+-]{0,254}$")
VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.+_-]{0,127}$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CARGO_CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")
NPM_INTEGRITY_RE = re.compile(r"^sha512-[A-Za-z0-9+/]+={0,2}$")
SKIP_DIRECTORIES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".worktrees",
        "__pycache__",
        "dist",
        "node_modules",
        "target",
        "vendor",
    }
)


@dataclass(frozen=True, order=True)
class Coordinate:
    ecosystem: str
    name: str
    version: str


@dataclass(frozen=True)
class Inventory:
    repository: pathlib.Path
    label: str
    coordinates: frozenset[Coordinate]


def _normalise_name(ecosystem: str, name: str) -> str:
    value = name.strip().casefold()
    if ecosystem == "PyPI":
        return re.sub(r"[-_.]+", "-", value)
    return value


def _bounded_text(path: pathlib.Path) -> str:
    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_LOCK_BYTES or path.is_symlink():
            raise AuditError("dependency lock is unavailable or exceeds its safe bound")
        return path.read_text(encoding="utf-8")
    except AuditError:
        raise
    except (OSError, UnicodeError):
        raise AuditError("dependency lock is unreadable") from None


def _coordinate(ecosystem: str, name: Any, version: Any) -> Coordinate | None:
    if not isinstance(name, str) or not isinstance(version, str):
        return None
    normalised = _normalise_name(ecosystem, name)
    if not PACKAGE_RE.fullmatch(normalised) or not VERSION_RE.fullmatch(version):
        raise AuditError("dependency lock contains an invalid package identity")
    return Coordinate(ecosystem, normalised, version)


def parse_uv_lock(path: pathlib.Path) -> frozenset[Coordinate]:
    try:
        document = tomllib.loads(_bounded_text(path))
    except tomllib.TOMLDecodeError:
        raise AuditError("uv dependency lock is invalid") from None
    packages = document.get("package")
    if not isinstance(packages, list):
        raise AuditError("uv dependency lock package inventory is invalid")
    coordinates: set[Coordinate] = set()
    for item in packages:
        if not isinstance(item, dict):
            raise AuditError("uv dependency lock package inventory is invalid")
        source = item.get("source")
        # Local/editable and Git packages are not PyPI release coordinates.
        if not isinstance(source, dict) or "registry" not in source:
            continue
        registry = source.get("registry")
        if not isinstance(registry, str) or not registry.startswith("https://"):
            raise AuditError("uv dependency lock contains an insecure registry source")
        artifacts: list[Any] = []
        if "sdist" in item:
            artifacts.append(item["sdist"])
        wheels = item.get("wheels", [])
        if not isinstance(wheels, list):
            raise AuditError("uv dependency lock artifact inventory is invalid")
        artifacts.extend(wheels)
        if not artifacts:
            raise AuditError(
                "uv dependency lock registry package has no hashed artifact"
            )
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise AuditError("uv dependency lock artifact inventory is invalid")
            url, digest = artifact.get("url"), artifact.get("hash")
            if (
                not isinstance(url, str)
                or not url.startswith("https://")
                or not isinstance(digest, str)
                or SHA256_RE.fullmatch(digest) is None
            ):
                raise AuditError("uv dependency lock contains an unverified artifact")
        coordinate = _coordinate("PyPI", item.get("name"), item.get("version"))
        if coordinate:
            coordinates.add(coordinate)
    if not coordinates:
        raise AuditError("uv dependency lock contains no auditable registry packages")
    return frozenset(coordinates)


def parse_cargo_lock(path: pathlib.Path) -> frozenset[Coordinate]:
    try:
        document = tomllib.loads(_bounded_text(path))
    except tomllib.TOMLDecodeError:
        raise AuditError("Cargo dependency lock is invalid") from None
    packages = document.get("package")
    if not isinstance(packages, list):
        raise AuditError("Cargo dependency lock package inventory is invalid")
    coordinates: set[Coordinate] = set()
    for item in packages:
        if not isinstance(item, dict):
            raise AuditError("Cargo dependency lock package inventory is invalid")
        source = item.get("source")
        if not isinstance(source, str) or not source.startswith("registry+"):
            continue
        checksum = item.get("checksum")
        if (
            not source.startswith("registry+https://")
            or not isinstance(checksum, str)
            or CARGO_CHECKSUM_RE.fullmatch(checksum) is None
        ):
            raise AuditError(
                "Cargo dependency lock contains an unverified registry package"
            )
        coordinate = _coordinate("crates.io", item.get("name"), item.get("version"))
        if coordinate:
            coordinates.add(coordinate)
    if not coordinates:
        raise AuditError(
            "Cargo dependency lock contains no auditable registry packages"
        )
    return frozenset(coordinates)


def _pnpm_key(value: str) -> tuple[str, str] | None:
    value = value.strip().strip("\"'").split("(", 1)[0]
    separator = value.rfind("@")
    if separator <= 0:
        return None
    name, version = value[:separator], value[separator + 1 :]
    if name.startswith("/"):
        name = name[1:]
    if not name or not VERSION_RE.fullmatch(version):
        return None
    return name, version


def parse_pnpm_lock(path: pathlib.Path) -> frozenset[Coordinate]:
    coordinates: set[Coordinate] = set()
    in_packages = False
    pending: tuple[str, str] | None = None
    pending_integrity = False

    def finish_pending() -> None:
        nonlocal pending, pending_integrity
        if pending is not None:
            if not pending_integrity:
                raise AuditError("pnpm dependency lock contains an unverified package")
            coordinate = _coordinate("npm", pending[0], pending[1])
            if coordinate:
                coordinates.add(coordinate)
        pending = None
        pending_integrity = False

    for line in _bounded_text(path).splitlines():
        if line == "packages:":
            in_packages = True
            continue
        if not in_packages:
            continue
        if line and not line[0].isspace():
            finish_pending()
            break
        match = re.match(r"^  (?!\s)(.+):\s*$", line)
        if match:
            finish_pending()
            parsed = _pnpm_key(match.group(1))
            if parsed is not None:
                pending = parsed
            continue
        if pending is not None:
            integrity_match = re.search(r"\bintegrity:\s*([^,}\s]+)", line)
            if integrity_match:
                if NPM_INTEGRITY_RE.fullmatch(integrity_match.group(1)) is None:
                    raise AuditError(
                        "pnpm dependency lock contains an invalid integrity digest"
                    )
                pending_integrity = True
    finish_pending()
    if not coordinates:
        raise AuditError("pnpm dependency lock contains no auditable registry packages")
    return frozenset(coordinates)


def _discover_repositories(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    root = root.resolve()
    if (root / ".git").exists():
        return (root,)
    repositories: list[pathlib.Path] = []
    pending: list[tuple[pathlib.Path, int]] = [(root, 0)]
    while pending:
        current_path, depth = pending.pop()
        if (current_path / ".git").exists():
            repositories.append(current_path)
            if len(repositories) > MAX_REPOSITORIES:
                raise AuditError("repository discovery exceeds the safe bound")
            continue
        if depth >= MAX_DISCOVERY_DEPTH:
            continue
        try:
            children = sorted(
                (
                    pathlib.Path(entry.path)
                    for entry in os.scandir(current_path)
                    if entry.name not in SKIP_DIRECTORIES
                    and entry.is_dir(follow_symlinks=False)
                ),
                reverse=True,
            )
        except OSError:
            raise AuditError("repository discovery is unavailable") from None
        pending.extend((child, depth + 1) for child in children)
    return tuple(sorted(repositories))


def _label(repository: pathlib.Path, root: pathlib.Path) -> str:
    try:
        relative = repository.relative_to(root)
    except ValueError:
        return repository.name
    return repository.name if relative.as_posix() == "." else relative.as_posix()


def inventory(
    root: pathlib.Path,
    *,
    repositories: tuple[pathlib.Path, ...] | None = None,
) -> tuple[Inventory, ...]:
    inventories: list[Inventory] = []
    selected = _discover_repositories(root) if repositories is None else repositories
    for repository in selected:
        coordinates: set[Coordinate] = set()
        for filename, parser in (
            ("uv.lock", parse_uv_lock),
            ("Cargo.lock", parse_cargo_lock),
            ("pnpm-lock.yaml", parse_pnpm_lock),
        ):
            path = repository / filename
            if path.is_file():
                coordinates.update(parser(path))
        if coordinates:
            inventories.append(
                Inventory(repository, _label(repository, root), frozenset(coordinates))
            )
    total = len({coordinate for item in inventories for coordinate in item.coordinates})
    if total == 0:
        raise AuditError("no supported dependency locks were discovered")
    if total > MAX_COORDINATES:
        raise AuditError("dependency inventory exceeds the safe bound")
    return tuple(inventories)


def _single_source_snapshot(root: pathlib.Path) -> tuple[pathlib.Path, ...]:
    """Validate one explicitly selected no-Git repository source root."""

    try:
        metadata = root.lstat()
    except OSError:
        raise AuditError("source snapshot root is unavailable") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise AuditError("source snapshot root must be a directory")
    locks = tuple(root / name for name in ("uv.lock", "Cargo.lock", "pnpm-lock.yaml"))
    if not any(path.is_file() and not path.is_symlink() for path in locks):
        raise AuditError("source snapshot contains no supported dependency lock")
    return (root,)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="accepted for compatibility; inventory is the only mode",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--single-source-snapshot",
        action="store_true",
        help="treat the explicit root as one validated no-Git repository",
    )
    mode.add_argument(
        "--source-snapshot-root",
        help="inspect exactly the workspace-declared direct provider source roots",
    )
    parser.add_argument(
        "--snapshot-workspace",
        help="authoritative repository-manager workspace for provider snapshot mode",
    )
    arguments = parser.parse_args(argv)
    provider_snapshot = arguments.source_snapshot_root is not None
    if provider_snapshot != (arguments.snapshot_workspace is not None):
        parser.error(
            "--source-snapshot-root and --snapshot-workspace must be used together"
        )
    root = pathlib.Path(arguments.source_snapshot_root or arguments.root).resolve()
    try:
        repositories: tuple[pathlib.Path, ...] | None = None
        if arguments.single_source_snapshot:
            repositories = _single_source_snapshot(root)
        elif provider_snapshot:
            root, repositories = resolve_snapshot_repositories(
                root, pathlib.Path(arguments.snapshot_workspace)
            )
        inventories = inventory(root, repositories=repositories)
        coordinates = tuple(
            sorted(
                {coordinate for item in inventories for coordinate in item.coordinates}
            )
        )
    except AuditError as error:
        print(f"dependency-inventory: FAILED - {error}", file=sys.stderr)
        return 2
    by_ecosystem: dict[str, int] = defaultdict(int)
    for coordinate in coordinates:
        by_ecosystem[coordinate.ecosystem] += 1
    summary = ", ".join(
        f"{ecosystem}={count}" for ecosystem, count in sorted(by_ecosystem.items())
    )
    print(
        f"dependency-inventory: clean ({len(inventories)} repositories, "
        f"{len(coordinates)} unique coordinates; {summary})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
