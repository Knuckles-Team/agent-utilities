#!/usr/bin/python
"""Transactional XDG materialization for skills, prompts, and ontologies.

Every provider root contains immutable content-addressed generations.  A bounded v2
marker is the atomic activation pointer.  Installer locks serialize writers; readers
validate the current registration, source manifest, marker, and generation before
using XDG content.  Unmarked destinations are operator-owned and are never replaced.
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
import shutil
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agent_utilities.core.paths import ontology_dir, skills_dir, unified_prompts_dir
from agent_utilities.core.provider_materialization import (
    MANAGED_PROVIDER_GENERATIONS,
    MANAGED_PROVIDER_LEGS,
    MAX_PROVIDER_FILES,
    AssetManifest,
    EmptyProviderAssets,
    ProviderAssetError,
    ProviderMaterializationError,
    ProviderOwnershipConflict,
    build_asset_manifest,
    copy_manifest,
    inactive_marker,
    is_safe_provider_name,
    marker_for_manifest,
    read_managed_provider_marker,
    registration_digest,
    write_managed_provider_marker,
)
from agent_utilities.core.providers import (
    ONTOLOGY_PROVIDER_GROUP,
    PROMPT_PROVIDER_GROUP,
    SKILL_PROVIDER_GROUP,
    ProviderRegistration,
    provider_registrations,
)

logger = logging.getLogger(__name__)

OWN_PROVIDER = "agent-utilities"
_LOCK_FILE = ".agent-utilities-materialization.lock"
_MAX_MANAGED_TREE_ENTRIES = MAX_PROVIDER_FILES * 8


def unified_skills_dir() -> Path:
    return skills_dir()


def unified_ontologies_dir() -> Path:
    return ontology_dir()


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


@contextlib.contextmanager
def _materialization_lock(root: Path) -> Iterator[None]:
    """Serialize materialization writers with a non-following local lock file."""

    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not _safe_directory(root):
        raise ProviderOwnershipConflict(
            "materialization root is not a regular directory"
        )
    lock_path = root / _LOCK_FILE
    try:
        lock_info = lock_path.lstat()
    except FileNotFoundError:
        lock_info = None
    if lock_info is not None and (
        _is_linklike(lock_path) or not stat.S_ISREG(lock_info.st_mode)
    ):
        raise ProviderOwnershipConflict("materialization lock is unsafe")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ProviderOwnershipConflict(
                "materialization lock is not a regular file"
            )
        opened_path = lock_path.lstat()
        if (
            _is_linklike(lock_path)
            or opened_path.st_dev != info.st_dev
            or opened_path.st_ino != info.st_ino
        ):
            raise ProviderOwnershipConflict("materialization lock changed during open")
        if os.name == "nt":  # pragma: no cover - exercised by Windows CI
            import msvcrt

            if info.st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":  # pragma: no cover - exercised by Windows CI
            import msvcrt

            try:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)


def _assert_non_overlapping(source: Path, destination: Path) -> None:
    source_root = source.resolve(strict=True)
    destination_root = destination.resolve(strict=False)
    if source_root == destination_root:
        raise ProviderOwnershipConflict("provider source overlaps its destination")
    if destination_root.is_relative_to(source_root) or source_root.is_relative_to(
        destination_root
    ):
        raise ProviderOwnershipConflict("provider source overlaps its destination")


def _provider_root(root: Path, provider: str, leg: str) -> tuple[Path, bool]:
    if not is_safe_provider_name(provider):
        raise ProviderOwnershipConflict("provider destination name is unsafe")
    if leg not in MANAGED_PROVIDER_LEGS:
        raise ProviderOwnershipConflict("provider materialization leg is unsupported")
    destination = root / provider
    created = False
    if destination.exists():
        if not _safe_directory(destination):
            raise ProviderOwnershipConflict(
                "provider destination is not a regular directory"
            )
        marker = read_managed_provider_marker(destination, provider=provider, leg=leg)
        if marker is None:
            raise ProviderOwnershipConflict("provider destination is operator-owned")
    else:
        try:
            destination.lstat()
        except OSError:
            pass
        else:
            raise ProviderOwnershipConflict("provider destination is an unsafe link")
        destination.mkdir(mode=0o700)
        created = True
    generations = destination / MANAGED_PROVIDER_GENERATIONS
    if generations.exists():
        if not _safe_directory(generations):
            raise ProviderOwnershipConflict("provider generation root is unsafe")
    else:
        generations.mkdir(mode=0o700)
    return destination, created


def _safe_generated_tree(path: Path) -> bool:
    if not _safe_directory(path):
        return False
    pending = [path]
    entry_count = 0
    while pending:
        directory = pending.pop()
        try:
            scanner = os.scandir(directory)
        except OSError:
            return False
        with scanner:
            for directory_entry in scanner:
                entry_count += 1
                if entry_count > _MAX_MANAGED_TREE_ENTRIES:
                    return False
                child = Path(directory_entry.path)
                try:
                    info = directory_entry.stat(follow_symlinks=False)
                except OSError:
                    return False
                if _is_linklike(child):
                    return False
                if stat.S_ISDIR(info.st_mode):
                    pending.append(child)
                elif not stat.S_ISREG(info.st_mode):
                    return False
    return True


def _remove_generated_tree(path: Path) -> None:
    if not _safe_generated_tree(path):
        raise ProviderOwnershipConflict("managed provider tree is unsafe to remove")
    shutil.rmtree(path)


def _materialize_provider(
    *,
    root: Path,
    provider: str,
    leg: str,
    registration: str,
    source: Path,
    manifest: AssetManifest,
) -> int:
    """Stage one immutable generation and atomically activate its v2 marker."""

    _assert_non_overlapping(source, root / provider)
    destination, created = _provider_root(root, provider, leg)
    generations = destination / MANAGED_PROVIDER_GENERATIONS
    generation = generations / manifest.content_digest
    stage: Path | None = None
    try:
        if generation.exists():
            try:
                existing = build_asset_manifest(generation, leg=leg)
            except (OSError, ProviderAssetError, ValueError):
                existing = None
            if existing != manifest:
                # The manifest dataclass contains source paths, so compare only its
                # path-free summary before deciding whether an immutable generation
                # can be reused.
                matching = existing is not None and (
                    existing.content_digest == manifest.content_digest
                    and existing.file_count == manifest.file_count
                    and existing.byte_count == manifest.byte_count
                )
                if not matching:
                    quarantine = generations / f".invalid-{secrets.token_hex(8)}"
                    os.replace(generation, quarantine)
                    _remove_generated_tree(quarantine)
        if not generation.exists():
            stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=generations))
            copy_manifest(manifest, stage)
            os.replace(stage, generation)
            stage = None
            # Make the immutable generation rename durable before publishing its
            # marker as the active reader view.
            from agent_utilities.core.provider_materialization import _fsync_directory

            _fsync_directory(generations)
        write_managed_provider_marker(
            destination,
            marker_for_manifest(
                provider=provider,
                leg=leg,
                registration=registration,
                manifest=manifest,
            ),
        )
    except BaseException:
        if stage is not None and stage.exists() and _safe_generated_tree(stage):
            shutil.rmtree(stage)
        if created and not (destination / ".agent-utilities-managed.json").exists():
            try:
                _remove_generated_tree(destination)
            except ProviderMaterializationError:
                pass
        raise
    return manifest.file_count


def _deactivate_provider(
    *, root: Path, provider: str, leg: str, registration: str
) -> None:
    destination, _created = _provider_root(root, provider, leg)
    write_managed_provider_marker(
        destination,
        inactive_marker(provider=provider, leg=leg, registration=registration),
    )


def _prune_removed_managed(root: Path, *, leg: str, registered: set[str]) -> int:
    if not _safe_directory(root):
        return 0
    removed = 0
    for child in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if child.name.startswith(".") or child.name in registered:
            continue
        if not _safe_directory(child):
            continue
        marker = read_managed_provider_marker(child, provider=child.name, leg=leg)
        if marker is None:
            continue
        _remove_generated_tree(child)
        removed += 1
    return removed


def _own_source(leg: str) -> tuple[Path, str, AssetManifest]:
    from agent_utilities._version import __version__

    package_root = Path(__file__).resolve().parent.parent
    source = {
        "skills": package_root / "skills",
        "prompts": package_root / "prompts",
        "ontologies": package_root / "knowledge_graph",
    }[leg]
    target = {
        "skills": "agent_utilities.skills",
        "prompts": "agent_utilities.prompts",
        "ontologies": "agent_utilities.knowledge_graph",
    }[leg]
    digest = registration_digest(
        {
            "group": {
                "skills": SKILL_PROVIDER_GROUP,
                "prompts": PROMPT_PROVIDER_GROUP,
                "ontologies": ONTOLOGY_PROVIDER_GROUP,
            }[leg],
            "provider": OWN_PROVIDER,
            "target": target,
            "owner": OWN_PROVIDER,
            "version": __version__,
        }
    )
    return source, digest, build_asset_manifest(source, leg=leg)


def own_provider_asset(leg: str) -> tuple[Path, str, AssetManifest]:
    """Return the trusted hub source using the same manifest contract as providers."""

    return _own_source(leg)


def _install_registration(
    *, root: Path, leg: str, registration: ProviderRegistration
) -> tuple[int, bool]:
    if registration.source_root is None:
        _deactivate_provider(
            root=root,
            provider=registration.name,
            leg=leg,
            registration=registration.digest,
        )
        return 0, False
    try:
        manifest = build_asset_manifest(
            registration.source_root,
            leg=leg,
            allowed_relative_paths=registration.owned_paths,
        )
    except EmptyProviderAssets:
        _deactivate_provider(
            root=root,
            provider=registration.name,
            leg=leg,
            registration=registration.digest,
        )
        return 0, False
    count = _materialize_provider(
        root=root,
        provider=registration.name,
        leg=leg,
        registration=registration.digest,
        source=registration.source_root,
        manifest=manifest,
    )
    return count, True


def install_unified() -> dict[str, Any]:
    """Reconcile all current provider legs without returning local filesystem data."""

    legs = {
        "skills": (SKILL_PROVIDER_GROUP, unified_skills_dir()),
        "prompts": (PROMPT_PROVIDER_GROUP, unified_prompts_dir()),
        "ontologies": (ONTOLOGY_PROVIDER_GROUP, unified_ontologies_dir()),
    }
    registrations: dict[str, tuple[ProviderRegistration, ...]] = {}
    # Duplicate/case-fold conflicts fail before the first mutation.
    for leg, (group, _root) in legs.items():
        registrations[leg] = provider_registrations(group)

    result: dict[str, Any] = {
        "skills": {"providers": 0, "files": 0, "failed": 0},
        "prompts": {"providers": 0, "files": 0, "failed": 0},
        "ontologies": {"providers": 0, "files": 0, "failed": 0},
        "pruned": {},
        "path_free": True,
    }
    for leg, (_group, root) in legs.items():
        current = registrations[leg]
        names = {item.name for item in current}
        names.add(OWN_PROVIDER)
        with _materialization_lock(root):
            result["pruned"][leg] = _prune_removed_managed(
                root, leg=leg, registered=names
            )
            for item in current:
                if item.name == OWN_PROVIDER:
                    continue
                try:
                    count, active = _install_registration(
                        root=root, leg=leg, registration=item
                    )
                    result[leg]["providers"] += int(active)
                    result[leg]["files"] += count
                    result[leg]["failed"] += int(not active)
                except (
                    OSError,
                    shutil.Error,
                    ProviderMaterializationError,
                    ValueError,
                ) as exc:
                    result[leg]["failed"] += 1
                    logger.warning(
                        "Provider materialization failed (exception_type=%s)",
                        type(exc).__name__,
                    )
            try:
                source, digest, manifest = _own_source(leg)
                count = _materialize_provider(
                    root=root,
                    provider=OWN_PROVIDER,
                    leg=leg,
                    registration=digest,
                    source=source,
                    manifest=manifest,
                )
                result[leg]["providers"] += 1
                result[leg]["files"] += count
            except (
                OSError,
                shutil.Error,
                ProviderMaterializationError,
                ValueError,
            ) as exc:
                result[leg]["failed"] += 1
                logger.warning(
                    "Hub materialization failed (exception_type=%s)", type(exc).__name__
                )
    return result


__all__ = [
    "OWN_PROVIDER",
    "install_unified",
    "own_provider_asset",
    "unified_ontologies_dir",
    "unified_prompts_dir",
    "unified_skills_dir",
]
