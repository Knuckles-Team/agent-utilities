#!/usr/bin/env python3
"""Run uv against the canonical ecosystem workspace from an external worktree.

Git worktrees intentionally live below the XDG state directory, outside the uv
workspace.  A workspace member copied there still contains ``workspace = true``
sources, so invoking uv directly cannot resolve those sources.  This launcher
materializes a generated symlink view of the canonical workspace in XDG state,
replaces only the agent-utilities member with the current worktree, and points uv
at that view.  No source is copied and the committed workspace manifest and lock
remain authoritative.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

PROJECT_NAME = "agent-utilities"
_SHADOW_MARKER = ".agent-utilities-worktree.json"


def _git_output(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository_root(start: Path) -> Path:
    """Return the current worktree root."""
    return Path(_git_output(start, "rev-parse", "--show-toplevel")).resolve()


def canonical_repository(worktree: Path) -> Path:
    """Return the primary checkout that owns the worktree's common git dir."""
    common_dir = Path(
        _git_output(
            worktree,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
    ).resolve()
    if common_dir.name != ".git":
        raise RuntimeError(f"unsupported git common directory: {common_dir}")
    return common_dir.parent


def _workspace_config(root: Path) -> dict[str, Any] | None:
    manifest = root / "pyproject.toml"
    if not manifest.is_file():
        return None
    with manifest.open("rb") as handle:
        document = tomllib.load(handle)
    workspace = document.get("tool", {}).get("uv", {}).get("workspace")
    return workspace if isinstance(workspace, dict) else None


def workspace_root(canonical: Path) -> Path:
    """Find the nearest uv workspace that contains the canonical checkout."""
    for candidate in (canonical, *canonical.parents):
        config = _workspace_config(candidate)
        if config is None:
            continue
        try:
            canonical.relative_to(candidate)
        except ValueError:
            continue
        if canonical in _workspace_members(candidate, config):
            return candidate
    raise RuntimeError(
        f"no uv workspace containing canonical repository {canonical} was found"
    )


def _expanded_paths(root: Path, patterns: list[str]) -> set[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        for match in glob.glob(str(root / pattern)):
            path = Path(match)
            if path.is_dir() and (path / "pyproject.toml").is_file():
                paths.add(path.resolve())
    return paths


def _workspace_members(root: Path, config: dict[str, Any]) -> set[Path]:
    raw_members = config.get("members", [])
    raw_excludes = config.get("exclude", [])
    if not isinstance(raw_members, list) or not all(
        isinstance(item, str) for item in raw_members
    ):
        raise RuntimeError("tool.uv.workspace.members must be a list of paths")
    if not isinstance(raw_excludes, list) or not all(
        isinstance(item, str) for item in raw_excludes
    ):
        raise RuntimeError("tool.uv.workspace.exclude must be a list of paths")
    members = _expanded_paths(root, raw_members)
    return members - _expanded_paths(root, raw_excludes)


def _safe_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink shadow path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def _refresh_managed_copy(
    copy: Path,
    canonical: Path,
    *,
    previously_managed: bool,
    label: str,
) -> None:
    copy.parent.mkdir(parents=True, exist_ok=True)
    if copy.is_symlink():
        copy.unlink()
    elif copy.exists() and not previously_managed:
        raise RuntimeError(f"refusing to replace unmanaged shadow {label}: {copy}")
    temporary = copy.with_name(f".{copy.name}.tmp")
    shutil.copyfile(canonical, temporary)
    temporary.replace(copy)


def _user_state_root() -> Path:
    configured = os.environ.get("XDG_STATE_HOME")
    root = Path(configured) if configured else Path.home() / ".local" / "state"
    state = root / PROJECT_NAME
    state.mkdir(parents=True, exist_ok=True)
    return state


def shadow_workspace(
    worktree: Path,
    canonical: Path,
    workspace: Path,
    *,
    state_root: Path | None = None,
) -> Path:
    """Materialize and return a generated workspace view for ``worktree``."""
    config = _workspace_config(workspace)
    if config is None:
        raise RuntimeError(f"{workspace} is not a uv workspace")
    members = _workspace_members(workspace, config)
    if canonical not in members:
        raise RuntimeError(f"{canonical} is not a member of {workspace}")

    state = state_root or _user_state_root()
    identity = hashlib.sha256(str(worktree).encode()).hexdigest()[:16]
    shadow = state / "uv-workspaces" / identity
    shadow.mkdir(parents=True, exist_ok=True)

    desired_links: dict[str, Path] = {}
    python_version = workspace / ".python-version"
    if python_version.is_file():
        desired_links[".python-version"] = python_version

    for member in members:
        relative = member.relative_to(workspace).as_posix()
        desired_links[relative] = worktree if member == canonical else member

    marker_path = shadow / _SHADOW_MARKER
    previous: dict[str, str] = {}
    if marker_path.is_file():
        loaded = json.loads(marker_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            previous = {
                key: value
                for key, value in loaded.items()
                if isinstance(key, str) and isinstance(value, str)
            }
    for relative in previous.keys() - desired_links.keys():
        stale = shadow / relative
        if stale.is_symlink():
            stale.unlink()

    for relative, target in desired_links.items():
        _safe_symlink(shadow / relative, target)
    canonical_manifest = workspace / "pyproject.toml"
    canonical_lock = workspace / "uv.lock"
    _refresh_managed_copy(
        shadow / "pyproject.toml",
        canonical_manifest,
        previously_managed="pyproject.toml" in previous,
        label="manifest",
    )
    _refresh_managed_copy(
        shadow / "uv.lock",
        canonical_lock,
        previously_managed="uv.lock" in previous,
        label="lock",
    )
    desired_links["pyproject.toml"] = canonical_manifest
    desired_links["uv.lock"] = canonical_lock

    marker_path.write_text(
        json.dumps(
            {relative: str(target) for relative, target in desired_links.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return shadow


def uv_invocation(
    arguments: list[str],
    *,
    worktree: Path,
    shadow: Path,
) -> tuple[list[str], dict[str, str]]:
    """Build the uv command and isolated environment for this worktree."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed or visible on PATH")
    if not arguments:
        raise RuntimeError("an uv subcommand is required")
    command = [uv, "--project", str(shadow)]
    tail = [argument for argument in arguments[1:] if argument != "--locked"]
    if arguments[0] in {"run", "sync"}:
        command.extend([arguments[0], "--locked", "--package", PROJECT_NAME, *tail])
    elif arguments[0] == "lock":
        command.extend(["lock", "--locked", *tail])
    else:
        command.extend(arguments)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["UV_PROJECT_ENVIRONMENT"] = str(worktree / ".venv")
    return command, environment


def doctor_payload(
    worktree: Path, canonical: Path, workspace: Path, shadow: Path
) -> dict:
    """Return bounded evidence that the shadow executes the worktree member."""
    member = shadow / canonical.relative_to(workspace)
    manifest = shadow / "pyproject.toml"
    lock = shadow / "uv.lock"
    canonical_manifest = workspace / "pyproject.toml"
    canonical_lock = workspace / "uv.lock"
    return {
        "status": "ok",
        "external_worktree": worktree != canonical,
        "worktree": str(worktree),
        "canonical_repository": str(canonical),
        "workspace_root": str(workspace),
        "shadow_workspace": str(shadow),
        "member_resolves_to_worktree": member.resolve() == worktree,
        "manifest_is_generated_copy": not manifest.is_symlink(),
        "manifest_matches_canonical": manifest.read_bytes()
        == canonical_manifest.read_bytes(),
        "lock_is_generated_copy": not lock.is_symlink(),
        "lock_matches_canonical": lock.read_bytes() == canonical_lock.read_bytes(),
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_uv(
    command: list[str],
    *,
    worktree: Path,
    environment: dict[str, str],
    workspace: Path,
    shadow: Path,
) -> int:
    """Execute uv and prove neither authoritative nor generated inputs changed."""
    protected = (
        workspace / "pyproject.toml",
        workspace / "uv.lock",
        shadow / "pyproject.toml",
        shadow / "uv.lock",
    )
    before = {path: _digest(path) for path in protected}
    result = subprocess.run(
        command,
        cwd=worktree,
        env=environment,
        check=False,
    )
    changed = [str(path) for path in protected if _digest(path) != before[path]]
    if changed:
        raise RuntimeError(
            "uv changed a lock-governed workspace input: " + ", ".join(changed)
        )
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run uv with exact workspace sources from an XDG git worktree."
    )
    parser.add_argument(
        "uv_arguments",
        nargs=argparse.REMAINDER,
        help="uv subcommand and arguments, or 'doctor'",
    )
    namespace = parser.parse_args(argv)

    worktree = repository_root(Path.cwd())
    canonical = canonical_repository(worktree)
    workspace = workspace_root(canonical)
    shadow = shadow_workspace(worktree, canonical, workspace)

    if namespace.uv_arguments == ["doctor"]:
        payload = doctor_payload(worktree, canonical, workspace, shadow)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return (
            0
            if all(
                payload[key]
                for key in (
                    "member_resolves_to_worktree",
                    "manifest_is_generated_copy",
                    "manifest_matches_canonical",
                    "lock_is_generated_copy",
                    "lock_matches_canonical",
                )
            )
            else 1
        )

    command, environment = uv_invocation(
        namespace.uv_arguments,
        worktree=worktree,
        shadow=shadow,
    )
    return run_uv(
        command,
        worktree=worktree,
        environment=environment,
        workspace=workspace,
        shadow=shadow,
    )


if __name__ == "__main__":
    raise SystemExit(main())
