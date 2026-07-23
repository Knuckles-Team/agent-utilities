"""Deterministic, privacy-safe catalogs for source-controlled release assets."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from agent_utilities.skills import BUNDLED_SKILLS

_MAX_SKILL_FILES = 4_096
_MAX_SKILL_FILE_BYTES = 16 * 1024 * 1024
_MAX_SKILL_TREE_BYTES = 64 * 1024 * 1024
_TRANSIENT_DIRECTORIES = frozenset({"__pycache__"})

# Top-level entries under ``agent_utilities/skills/`` that are never
# themselves an installable skill (no ``SKILL.md`` at that level), so
# ``_skill_directories`` must not require one there: ``skill_graphs`` is a
# KG-ingestion reference corpus (mirrors ``providers.py``'s
# ``_SKILL_GRAPH_SEGMENTS``/``is_skill_graph_reference_path``), ``workflows``
# is a container for nested workflow-type skills (each nested dir, e.g.
# ``workflows/agent-os-genesis/``, has its own ``SKILL.md``), and
# ``fleet_harness`` is the skill-validation harness's own support code, not a
# skill.
_NON_SKILL_STRUCTURAL_DIRECTORIES = frozenset(
    {"skill_graphs", "workflows", "fleet_harness"}
)
_TRANSIENT_SUFFIXES = frozenset({".pyc", ".pyo"})


class ReleaseCatalogError(ValueError):
    """Raised when release catalog input is incomplete or unsafe."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical retained-catalog representation."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def content_digest(payload: bytes) -> str:
    """Return the release digest spelling used by every generated catalog."""

    return "sha256:" + hashlib.sha256(payload).hexdigest()


def canonical_value_digest(value: Any) -> str:
    """Digest a JSON value without binding the retained file terminator."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return content_digest(payload)


def write_catalog(path: Path, payload: bytes, *, prefix: str) -> None:
    """Atomically replace a regular catalog without following symlinks."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        metadata = None
    except OSError as exc:
        raise ReleaseCatalogError("catalog_output_uninspectable") from exc
    if metadata is not None and (
        stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode)
    ):
        raise ReleaseCatalogError("catalog_output_not_regular")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent_metadata = path.parent.lstat()
        if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(
            parent_metadata.st_mode
        ):
            raise ReleaseCatalogError("catalog_output_parent_not_directory")
    except OSError as exc:
        raise ReleaseCatalogError("catalog_output_parent_uninspectable") from exc

    descriptor, temporary_name = tempfile.mkstemp(prefix=prefix, dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_skill_names(skill_names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(sorted(skill_names))
    if (
        len(names) != 10
        or len(names) != len(set(names))
        or any(
            not name
            or PurePosixPath(name).name != name
            or "\\" in name
            or name in {".", ".."}
            for name in names
        )
    ):
        raise ReleaseCatalogError("prebundled_skill_membership_invalid")
    return names


def _skill_directories(skills_root: Path) -> tuple[str, ...]:
    try:
        root_metadata = skills_root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode
        ):
            raise ReleaseCatalogError("prebundled_skill_root_not_directory")
        children = tuple(skills_root.iterdir())
    except OSError as exc:
        raise ReleaseCatalogError("prebundled_skill_root_uninspectable") from exc

    names: list[str] = []
    for child in children:
        try:
            metadata = child.lstat()
        except OSError as exc:
            raise ReleaseCatalogError("prebundled_skill_entry_uninspectable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ReleaseCatalogError("prebundled_skill_symlink_rejected")
        if not stat.S_ISDIR(metadata.st_mode):
            continue
        if (
            child.name in _TRANSIENT_DIRECTORIES
            or child.name in _NON_SKILL_STRUCTURAL_DIRECTORIES
        ):
            continue
        try:
            skill_metadata = (child / "SKILL.md").lstat()
        except FileNotFoundError as exc:
            raise ReleaseCatalogError(
                "prebundled_skill_entry_missing_contract"
            ) from exc
        except OSError as exc:
            raise ReleaseCatalogError("prebundled_skill_entry_uninspectable") from exc
        if stat.S_ISLNK(skill_metadata.st_mode) or not stat.S_ISREG(
            skill_metadata.st_mode
        ):
            raise ReleaseCatalogError("prebundled_skill_contract_not_regular")
        names.append(child.name)
    return tuple(sorted(names))


def _skill_files(skill_root: Path) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    total_bytes = 0
    pending = [skill_root]
    while pending:
        directory = pending.pop()
        try:
            children = tuple(directory.iterdir())
        except OSError as exc:
            raise ReleaseCatalogError("prebundled_skill_tree_uninspectable") from exc
        for child in sorted(children, key=lambda item: item.name, reverse=True):
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise ReleaseCatalogError(
                    "prebundled_skill_tree_uninspectable"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ReleaseCatalogError("prebundled_skill_symlink_rejected")
            if stat.S_ISDIR(metadata.st_mode):
                if child.name not in _TRANSIENT_DIRECTORIES:
                    pending.append(child)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ReleaseCatalogError("prebundled_skill_special_file_rejected")
            if child.suffix in _TRANSIENT_SUFFIXES:
                continue
            if metadata.st_size > _MAX_SKILL_FILE_BYTES:
                raise ReleaseCatalogError("prebundled_skill_file_too_large")
            total_bytes += metadata.st_size
            if total_bytes > _MAX_SKILL_TREE_BYTES:
                raise ReleaseCatalogError("prebundled_skill_tree_too_large")
            try:
                payload = child.read_bytes()
            except OSError as exc:
                raise ReleaseCatalogError("prebundled_skill_file_unreadable") from exc
            if len(payload) != metadata.st_size:
                raise ReleaseCatalogError("prebundled_skill_file_changed_during_read")
            relative = child.relative_to(skill_root).as_posix()
            if (
                not relative
                or not PurePosixPath(relative).parts
                or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts
                or "\\" in relative
            ):
                raise ReleaseCatalogError("prebundled_skill_file_name_invalid")
            files.append({"name": relative, "digest": content_digest(payload)})
            if len(files) > _MAX_SKILL_FILES:
                raise ReleaseCatalogError("prebundled_skill_file_count_exceeded")
    files.sort(key=lambda item: item["name"])
    if not any(item["name"] == "SKILL.md" for item in files):
        raise ReleaseCatalogError("prebundled_skill_contract_missing")
    return files


def prebundled_skill_catalog(
    skills_root: Path,
    *,
    skill_names: Iterable[str] = BUNDLED_SKILLS,
) -> dict[str, Any]:
    """Build the exact, content-addressed ten-skill release catalog."""

    names = _safe_skill_names(skill_names)
    if _skill_directories(skills_root) != names:
        raise ReleaseCatalogError("prebundled_skill_membership_not_exact")

    entries: list[dict[str, Any]] = []
    for name in names:
        files = _skill_files(skills_root / name)
        entries.append(
            {
                "skill": name,
                "treeDigest": canonical_value_digest(files),
                "files": files,
            }
        )
    return {
        "apiVersion": "graphos.io/v1",
        "kind": "PrebundledSkillCatalog",
        "catalogVersion": 1,
        "membershipDigest": canonical_value_digest(list(names)),
        "entryCount": len(entries),
        "entries": entries,
    }


def prebundled_skill_catalog_bytes(
    skills_root: Path,
    *,
    skill_names: Iterable[str] = BUNDLED_SKILLS,
) -> bytes:
    """Render the exact retained bytes for the ten-skill catalog."""

    return canonical_json_bytes(
        prebundled_skill_catalog(skills_root, skill_names=skill_names)
    )


def prebundled_skill_catalog_digest(
    skills_root: Path,
    *,
    skill_names: Iterable[str] = BUNDLED_SKILLS,
) -> str:
    """Bind runtime validation evidence to the exact retained catalog bytes."""

    return content_digest(
        prebundled_skill_catalog_bytes(skills_root, skill_names=skill_names)
    )
