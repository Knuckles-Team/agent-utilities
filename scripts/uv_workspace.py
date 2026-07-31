#!/usr/bin/env python3
"""Run uv against the canonical ecosystem workspace from an external worktree.

Git worktrees intentionally live below the XDG state directory, outside the uv
workspace.  A workspace member copied there still contains ``workspace = true``
sources, so invoking uv directly cannot resolve those sources.  This launcher
materializes a generated symlink view of the canonical workspace in XDG state,
replaces only the agent-utilities member with the current worktree, and points uv
at that view.  No source is copied and the committed workspace manifest and lock
remain authoritative.  Epistemic-graph native builds use one explicit per-user
artifact cache outside those generated shadows so concurrent worktrees share the
same content-addressed wheel.

The virtualenv is PARTITIONED by dependency selection
(CONCEPT:AU-OS.governance.lane-partitioned-resources).  One worktree serves many
concurrent invocations that ask for *different* dependency sets — the pre-commit
``pytest`` hook asks for ``--all-extras`` while the documented
``uv_workspace.py run pytest`` asks for none — and ``uv run`` re-synchronises the
environment named by ``UV_PROJECT_ENVIRONMENT`` on every invocation.  Pointing
every selection at one ``<worktree>/.venv`` therefore let those invocations
rewrite one environment underneath each other: a measured 44-package environment
where a 700-package one was expected, and imports failing from inside a venv that
was mid-installation.  Because uv releases the environment lock *before* it execs
the child, the corrupted window covers the whole test run, and a test process that
samples it reports a dependency that is present in the lock as absent from the
world — a false "environment-blocked" verdict, the one disposition that ends
inquiry rather than inviting the next reader to check.

Two properties close that structurally, with no caller change:

* **Partition.** Each distinct selection resolves to its own environment
  directory, so two selections can no longer contend at all.
* **Sync before exec.** The environment is synchronised by an explicit ``uv sync``
  and the child is then run with ``uv run --no-sync``, so an invocation never
  mutates an environment while another invocation's process is running in it.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

PROJECT_NAME = "agent-utilities"
_SHADOW_MARKER = ".agent-utilities-worktree.json"
_NATIVE_ARTIFACT_CACHE_ENV = "EPISTEMIC_GRAPH_NATIVE_ARTIFACT_CACHE"
_ENVIRONMENT_DIRNAME = ".venv"
_ENVIRONMENT_MARKER = ".uv-workspace-selection.json"

# uv flags that change WHICH distributions are resolved into the environment, and
# therefore which environment an invocation may safely share.  ``True`` marks a
# flag that consumes the following token as its value.
_SELECTION_FLAGS: dict[str, bool] = {
    "--extra": True,
    "--no-extra": True,
    "--all-extras": False,
    "--no-all-extras": False,
    "--group": True,
    "--no-group": True,
    "--only-group": True,
    "--default-groups": True,
    "--all-groups": False,
    "--no-default-groups": False,
    "--dev": False,
    "--no-dev": False,
    "--only-dev": False,
    "--no-editable": False,
    "--python": True,
    "-p": True,
}

# Recognised uv flags that do NOT change the resolved distribution set, so two
# invocations differing only in these may share one environment.
_NEUTRAL_FLAGS: dict[str, bool] = {
    "--locked": False,
    "--frozen": False,
    "--no-sync": False,
    "--refresh": False,
    "--no-cache": False,
    "--offline": False,
    "--native-tls": False,
    "--quiet": False,
    "-q": False,
    "--verbose": False,
    "-v": False,
}

# The one selection that keeps the conventional ``<worktree>/.venv`` path.
# It is not privileged arbitrarily: it is the selection this repository's own
# pre-commit ``pytest`` gate uses, every existing worktree environment is already
# in that state (~700 distributions / ~6.3 GB each), and ``.venv/bin`` is on the
# PATH built by ``scripts/bootstrap.sh`` and both CI workflows.  Keying it like
# any other selection would have forced every worktree to rebuild the largest
# environment it already owns.
_CANONICAL_SELECTION: tuple[str, ...] = ("--all-extras",)


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
    elif link.exists():
        raise RuntimeError(f"refusing to replace non-symlink shadow path: {link}")
    # Concurrent invocations sharing this worktree's shadow race between the
    # unlink and the create, so stage a privately-named link and rename it into
    # place: os.replace over an existing symlink is atomic and never leaves the
    # path absent.
    staged = link.with_name(f".{link.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        staged.symlink_to(target, target_is_directory=target.is_dir())
        os.replace(staged, link)
    finally:
        staged.unlink(missing_ok=True)


def _refresh_managed_copy(
    copy: Path,
    canonical: Path,
    *,
    previously_managed: bool,
    label: str,
) -> None:
    copy.parent.mkdir(parents=True, exist_ok=True)
    if copy.is_symlink():
        copy.unlink(missing_ok=True)
    elif copy.exists() and not previously_managed:
        raise RuntimeError(f"refusing to replace unmanaged shadow {label}: {copy}")
    # The shadow directory is keyed by worktree, not by invocation, so the
    # concurrent invocations this launcher exists to support all refresh these
    # copies at once. A fixed temporary name made them collide: one invocation's
    # rename consumed the file another had just written, and the second died with
    # FileNotFoundError before uv ever started. The rename stays atomic; only the
    # staging name has to be private.
    temporary = copy.with_name(f".{copy.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(canonical, temporary)
        temporary.replace(copy)
    finally:
        temporary.unlink(missing_ok=True)


def _user_state_root() -> Path:
    configured = os.environ.get("XDG_STATE_HOME")
    root = Path(configured) if configured else Path.home() / ".local" / "state"
    state = root / PROJECT_NAME
    state.mkdir(parents=True, exist_ok=True)
    return state


def _native_artifact_cache_root() -> Path:
    """Return the stable per-user cache shared by every hermetic worktree."""
    return Path.home() / ".cache" / "epistemic-graph" / "native-artifacts" / "v1"


@contextmanager
def _shadow_materialization_lock(shadow: Path) -> Iterator[None]:
    """Serialize materialization of *shadow* across concurrent invocations.

    The shadow is keyed by worktree, not by invocation, so every concurrent
    invocation from one worktree materializes the SAME directory. Each guard here
    is a check-then-act — "is this a symlink", "did we write this copy" — and
    siblings interleaving between the check and the act made the guards refuse
    paths that a sibling had just made correct.

    Serializing is what keeps those guards exact. Weakening them to tolerate a
    sibling would also have made them tolerate the foreign content they exist to
    refuse. Materialization is symlinks plus two small copies, so the held window
    is milliseconds and no test execution happens inside it.
    """
    shadow.parent.mkdir(parents=True, exist_ok=True)
    lock_path = shadow.with_name(f"{shadow.name}.materialize.lock")
    handle = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)


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
    with _shadow_materialization_lock(shadow):
        return _materialize_shadow(shadow, worktree, canonical, workspace, members)


def _materialize_shadow(
    shadow: Path,
    worktree: Path,
    canonical: Path,
    workspace: Path,
    members: set[Path],
) -> Path:
    """Bring *shadow* up to date. Callers must hold the materialization lock."""
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
            stale.unlink(missing_ok=True)

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

    # Written atomically: a concurrent invocation reads this marker with
    # json.loads, and a torn read would abort it before uv ever started.
    staged_marker = marker_path.with_name(
        f".{marker_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        staged_marker.write_text(
            json.dumps(
                {relative: str(target) for relative, target in desired_links.items()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        staged_marker.replace(marker_path)
    finally:
        staged_marker.unlink(missing_ok=True)
    return shadow


def split_selection(tail: Sequence[str]) -> tuple[list[str], bool, int]:
    """Return the dependency-selecting flags leading *tail*, and whether the whole
    leading flag run was recognised, and the index at which the command begins.

    Recognition is the honest part.  Every uv flag understood here is classified
    as either selecting (part of the environment identity) or neutral; the first
    flag that is neither ends the scan and reports ``False``, so the caller can
    fall back to uv's own behaviour instead of guessing at an environment
    identity it cannot actually derive.
    """
    selection: list[str] = []
    index = 0
    while index < len(tail):
        token = tail[index]
        if token == "--" or not token.startswith("-"):
            return selection, True, index
        name, _, inline = token.partition("=")
        if name in _SELECTION_FLAGS:
            takes_value = _SELECTION_FLAGS[name]
        elif name in _NEUTRAL_FLAGS:
            takes_value = _NEUTRAL_FLAGS[name]
        else:
            return selection, False, index
        value: str | None = None
        if takes_value:
            if inline:
                value = inline
            else:
                index += 1
                if index >= len(tail):
                    return selection, False, index
                value = tail[index]
        if name in _SELECTION_FLAGS:
            selection.append(name)
            if value is not None:
                selection.append(value)
        index += 1
    return selection, True, index


def foreign_python_console_script(name: str, environment: Path) -> Path | None:
    """Return where *name* resolves if that is a Python tool OUTSIDE *environment*.

    ``uv run pytest`` does not fail when the environment lacks ``pytest``: it
    falls through to the first ``pytest`` on ``PATH`` and runs the project's tests
    under ``/usr/bin/python`` against system site-packages. Measured in this
    worktree, the same command two ways:

    ==========================  =====================  =======
    invocation                  ``sys.executable``     fastmcp
    ==========================  =====================  =======
    ``run pytest``              ``/usr/bin/python``    3.3.1
    ``run --all-extras pytest`` ``<env>/bin/python3``  4.0.0b1
    ==========================  =====================  =======

    The wrong-interpreter run still executes the project's own fail-closed guards,
    which then report truthfully about the WRONG environment — so the verdict
    looks authoritative and cites the codebase's own checks. That is what made it
    survive repeated diagnosis.

    The shebang is the discriminator, and it is why this can be a general rule
    rather than a list of blessed command names: a Python console script names a
    Python interpreter on its first line, so ``bash`` or ``find`` resolving
    outside the environment is legitimate and passes, while ``pytest``, ``ruff``
    or ``mypy`` resolving outside it is the silent-fallback bug.
    """
    if not name or os.sep in name or (os.altsep and os.altsep in name):
        # An explicit path is a deliberate choice, not a PATH fallback.
        return None
    for directory in ("bin", "Scripts"):
        if (environment / directory / name).exists():
            return None
    resolved = shutil.which(name)
    if resolved is None:
        # Nowhere at all: let uv report the missing command itself.
        return None
    path = Path(resolved)
    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"#!":
                return None
            shebang = handle.readline(512).decode("utf-8", "replace")
    except OSError:
        return None
    return path if "python" in shebang else None


def normalized_selection(selection: Sequence[str]) -> tuple[str, ...]:
    """Return *selection* as an order- and duplicate-independent identity."""
    pairs: list[str] = []
    index = 0
    while index < len(selection):
        name = selection[index]
        if _SELECTION_FLAGS.get(name) and index + 1 < len(selection):
            pairs.append(f"{name}={selection[index + 1]}")
            index += 2
        else:
            pairs.append(name)
            index += 1
    return tuple(sorted(set(pairs)))


def environment_label(selection: Sequence[str]) -> str:
    """Return the directory suffix that partitions *selection*'s environment."""
    normalized = normalized_selection(selection)
    if normalized == _CANONICAL_SELECTION:
        return ""
    if not normalized:
        return "base"
    digest = hashlib.sha256("\x00".join(normalized).encode()).hexdigest()[:12]
    return f"x{digest}"


def environment_path(worktree: Path, selection: Sequence[str]) -> Path:
    """Return the environment directory this worktree owns for *selection*."""
    label = environment_label(selection)
    name = _ENVIRONMENT_DIRNAME if not label else f"{_ENVIRONMENT_DIRNAME}-{label}"
    return worktree / name


def _describe_environment(path: Path, selection: Sequence[str]) -> None:
    """Record which selection owns *path*, so a keyed directory is self-describing.

    Called only *after* uv has built the environment, and it never creates the
    directory: uv refuses to adopt a path that already exists without a Python
    executable in it, so writing this note early would break the very first
    invocation for a new selection.
    """
    if not path.is_dir():
        return
    marker = path / _ENVIRONMENT_MARKER
    payload = {
        "selection": list(normalized_selection(selection)),
        "label": environment_label(selection),
    }
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    staged = marker.with_name(f".{marker.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        if marker.is_file() and marker.read_text(encoding="utf-8") == body:
            return
        staged.write_text(body, encoding="utf-8")
        staged.replace(marker)
    except OSError:
        # Evidence only; never fail an invocation because the note could not be
        # written.
        return
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass


class UvPlan(NamedTuple):
    """The commands, environment and evidence for one launcher invocation.

    A :class:`NamedTuple` rather than a dataclass because this module is also
    loaded standalone by file path (``importlib.util.spec_from_file_location``),
    where ``dataclasses`` cannot resolve ``cls.__module__`` back to a live module.
    """

    prepare: tuple[tuple[str, ...], ...]
    execute: tuple[str, ...]
    environment: dict[str, str]
    environment_path: Path
    selection: tuple[str, ...]
    selection_recognized: bool
    command_name: str | None


def uv_plan(
    arguments: list[str],
    *,
    worktree: Path,
    shadow: Path,
) -> UvPlan:
    """Build the partitioned, sync-before-exec plan for this worktree."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed or visible on PATH")
    if not arguments:
        raise RuntimeError("an uv subcommand is required")
    subcommand = arguments[0]
    tail = [argument for argument in arguments[1:] if argument != "--locked"]

    selection: list[str] = []
    recognized = True
    command_name: str | None = None
    if subcommand in {"run", "sync"}:
        selection, recognized, start = split_selection(tail)
        if subcommand == "run" and recognized and start < len(tail):
            command_name = tail[start] if tail[start] != "--" else None

    prepare: list[tuple[str, ...]] = []
    base = [uv, "--project", str(shadow)]
    if subcommand == "run":
        if recognized:
            # Synchronise explicitly, then exec without syncing, so this
            # invocation cannot mutate the environment a sibling process is
            # already running in.  ``--inexact`` keeps the pruning behaviour uv
            # run has always had; partitioning, not pruning, is what makes the
            # selection correct.
            prepare.append(
                (
                    *base,
                    "sync",
                    "--locked",
                    "--inexact",
                    "--package",
                    PROJECT_NAME,
                    *selection,
                )
            )
            command = [*base, "run", "--no-sync", "--locked", "--package", PROJECT_NAME]
        else:
            command = [*base, "run", "--locked", "--package", PROJECT_NAME]
        command.extend(tail)
    elif subcommand == "sync":
        command = [*base, "sync", "--locked", "--package", PROJECT_NAME, *tail]
    elif subcommand == "lock":
        command = [*base, "lock", "--locked", *tail]
    else:
        command = [*base, *arguments]

    directory = environment_path(worktree, selection)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["UV_PROJECT_ENVIRONMENT"] = str(directory)
    environment[_NATIVE_ARTIFACT_CACHE_ENV] = str(_native_artifact_cache_root())
    return UvPlan(
        prepare=tuple(prepare),
        execute=tuple(command),
        environment=environment,
        environment_path=directory,
        selection=normalized_selection(selection),
        selection_recognized=recognized,
        command_name=command_name,
    )


def _environment_evidence(worktree: Path) -> list[dict[str, Any]]:
    """Report every partitioned environment this worktree owns, and its selection."""
    evidence: list[dict[str, Any]] = []
    for directory in sorted(worktree.glob(f"{_ENVIRONMENT_DIRNAME}*")):
        if not directory.is_dir():
            continue
        marker = directory / _ENVIRONMENT_MARKER
        selection: list[str] | None = None
        if marker.is_file():
            try:
                loaded = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = None
            if isinstance(loaded, dict) and isinstance(loaded.get("selection"), list):
                selection = [str(item) for item in loaded["selection"]]
        distributions = len(
            [
                path
                for pattern in ("lib/python*/site-packages", "Lib/site-packages")
                for parent in directory.glob(pattern)
                for path in parent.glob("*.dist-info")
            ]
        )
        evidence.append(
            {
                "path": str(directory),
                "selection": selection,
                "distributions": distributions,
            }
        )
    return evidence


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
        "environments": _environment_evidence(worktree),
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
    prepare: Sequence[Sequence[str]] = (),
    environment_path: Path | None = None,
    command_name: str | None = None,
) -> int:
    """Execute uv and prove neither authoritative nor generated inputs changed.

    ``prepare`` runs first and short-circuits on failure: it is the environment
    synchronisation that must complete before the child may be exec'd against an
    environment nobody is allowed to mutate afterwards.

    ``environment_path`` and ``command_name`` enable the interpreter guard, which
    runs between the two: the environment is final by then, so the check sees
    exactly what the child would.
    """
    protected = (
        workspace / "pyproject.toml",
        workspace / "uv.lock",
        shadow / "pyproject.toml",
        shadow / "uv.lock",
    )
    before = {path: _digest(path) for path in protected}

    def _assert_unchanged() -> None:
        changed = [str(path) for path in protected if _digest(path) != before[path]]
        if changed:
            raise RuntimeError(
                "uv changed a lock-governed workspace input: " + ", ".join(changed)
            )

    for step in prepare:
        step_result = subprocess.run(
            list(step),
            cwd=worktree,
            env=environment,
            check=False,
        )
        _assert_unchanged()
        if step_result.returncode != 0:
            return step_result.returncode
    if environment_path is not None and command_name is not None:
        foreign = foreign_python_console_script(command_name, environment_path)
        if foreign is not None:
            raise RuntimeError(
                f"refusing to run {command_name!r}: it is not installed in "
                f"{environment_path}, so uv would fall through to {foreign} and "
                "execute against a DIFFERENT interpreter and site-packages. That "
                "run would still execute this project's fail-closed guards, which "
                "would then report truthfully about the wrong environment and look "
                "authoritative. Request the extras that provide "
                f"{command_name!r} (for the test suite: --all-extras), or invoke it "
                "as 'python -m' so it can only resolve inside the environment."
            )
    result = subprocess.run(
        command,
        cwd=worktree,
        env=environment,
        check=False,
    )
    _assert_unchanged()
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

    plan = uv_plan(
        namespace.uv_arguments,
        worktree=worktree,
        shadow=shadow,
    )
    returncode = run_uv(
        list(plan.execute),
        worktree=worktree,
        environment=plan.environment,
        workspace=workspace,
        shadow=shadow,
        prepare=plan.prepare,
        environment_path=plan.environment_path,
        command_name=plan.command_name,
    )
    _describe_environment(plan.environment_path, plan.selection)
    return returncode


def _cli() -> int:
    """Report a refusal as a message, not a traceback.

    Every ``RuntimeError`` this module raises is a deliberate refusal addressed to
    a human — a foreign interpreter, a mutated lock, an unmanaged shadow. A
    traceback buries that message under frames nobody needs, and this one has to
    be read to be acted on.
    """
    try:
        return main()
    except RuntimeError as error:
        print(f"uv_workspace: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
