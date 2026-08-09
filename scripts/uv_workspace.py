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
import base64
import email
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import uuid
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

PROJECT_NAME = "agent-utilities"
_SHADOW_MARKER = ".agent-utilities-worktree.json"
_NATIVE_ARTIFACT_CACHE_ENV = "EPISTEMIC_GRAPH_NATIVE_ARTIFACT_CACHE"
_ENVIRONMENT_DIRNAME = ".venv"
_ENVIRONMENT_MARKER = ".uv-workspace-selection.json"

# BUG-063: uv's own editable-build cache key includes the absolute source path,
# so every worktree of THIS repo is a distinct cache entry for the
# ``epistemic-graph`` workspace member -- a worker whose worktree has never
# synced before pays a from-scratch ``cargo rustc --profile release --features
# full --features ast-extended -C lto=thin -C codegen-units=1`` PEP 517 build
# (measured 40m49s) to reproduce a wheel byte-identical to one 14 siblings
# already built. A worker who is not touching epistemic-graph's Rust source has
# no reason to build it from source at all -- see ``_eg_fast_vendor_dir`` below.
_EG_MEMBER_NAME = "epistemic-graph"
_EG_WHEELHOUSE_ENV = "EPISTEMIC_GRAPH_WHEELHOUSE"
_EG_DEFAULT_WHEELHOUSE = Path("/home/apps/worktrees/SHARED-WHEELS")
# Escape hatch (Configuration discipline: opt-IN, because the always-on default
# below is the cheap path and this one is the expensive one). A developer who is
# genuinely changing epistemic-graph's Rust source sets this so their worktree
# builds and installs the LIVE source editable, exactly like before this fix.
_EG_SOURCE_BUILD_ENV = "EPISTEMIC_GRAPH_SOURCE_BUILD"
_EG_FASTPATH_DIRNAME = "eg-fastpath-vendor"
_EG_FASTPATH_BACKEND_MODULE = "_eg_fastpath_backend"

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


# D-ORC-33: the binding constraint on parallelism measured on this host is disk
# I/O and swap, not the subagent/lane cap -- at the peak of a 13-20 lane wave,
# load average hit ~26 on 24 cores, swap was 100% exhausted (7/7 GB), and a bare
# `git add` died with "Bus error (core dumped)" (D-ORC-9). Several lanes were
# running `uv sync` simultaneously against the SAME 112 GB shared `~/.cache/uv`
# on the SAME /home spindle; one lane's sync stalled at 0% CPU for 17+ minutes
# and never recovered, forcing it to fall back to unreplicated evidence
# (D-ORC-32). Concurrent `uv sync` invocations are correctness-safe (each
# worktree+selection already gets its own partitioned `.venv`, see
# `environment_path()`), but they are NOT free: they contend for the same
# on-disk cache and download/build bandwidth. Capping concurrent heavy syncs
# to a small pool -- independent of however many lanes are running in total --
# keeps that contention bounded without serializing it to one lane at a time
# (an exclusive lease would defeat the very partitioning this module exists
# to provide, since 20 lanes could genuinely each need a sync at once and each
# request is legitimate).
_DEPENDENCY_SYNC_POOL_CAPACITY = 4


def _dependency_sync_pool_dir() -> Path:
    """Host-wide, so the cap holds across every worktree of every repo, not
    just this one -- the shared uv cache and spindle are contended workspace
    -wide, exactly like `dependency-lock` and `epistemic-graph-daemon` in
    agent_utilities/governance/lane_resources.yaml."""
    root = _user_state_root() / "heavy-lane-pool"
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def _dependency_sync_slot() -> Iterator[None]:
    """Block until one of ``_DEPENDENCY_SYNC_POOL_CAPACITY`` slots is free.

    A flock-based counting semaphore, deliberately NOT
    ``agent_utilities.governance.lanes.hold_lease`` (which this bootstrap-time
    launcher cannot safely import: importing the very package this script
    exists to sync/prepare the environment for, before that environment
    exists, is circular -- and would risk resolving against whichever
    ``agent_utilities`` happens to be importable from the ambient interpreter,
    the same "poisoned interpreter" trap ``uv run pytest`` fell into). Each
    slot is its own lock file; mutual exclusion per slot is `flock`, capacity
    is the number of slot files tried. A heavy step WAITS for a slot rather
    than failing -- unlike the exclusive leases in lanes.py, which raise so
    the caller can defer explicitly, a sync step has nothing useful to do
    but wait its turn, and forcing every caller to hand-roll a retry loop
    would just re-implement this blocking wait one layer up.
    """
    pool_dir = _dependency_sync_pool_dir()
    acquired: tuple[int, int] | None = None
    try:
        for index in range(_DEPENDENCY_SYNC_POOL_CAPACITY):
            slot = pool_dir / f"slot-{index}.lock"
            handle = os.open(str(slot), os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(handle)
                continue
            acquired = (handle, index)
            break
        if acquired is None:
            # Every slot busy right now: block on slot 0 rather than refuse.
            # Guarantees eventual acquisition (flock queues waiters FIFO per
            # kernel) without an unbounded number of processes polling.
            slot = pool_dir / "slot-0.lock"
            handle = os.open(str(slot), os.O_CREAT | os.O_WRONLY, 0o644)
            fcntl.flock(handle, fcntl.LOCK_EX)
            acquired = (handle, 0)
        yield
    finally:
        if acquired is not None:
            handle, _index = acquired
            try:
                fcntl.flock(handle, fcntl.LOCK_UN)
            finally:
                os.close(handle)


def _environment_activity_lock_path(environment_path: Path) -> Path:
    """Lock file coordinating readers/writers of one partitioned environment.

    One file per environment directory (keyed by its own path), so distinct
    selections in the same worktree (``.venv`` vs ``.venv-base``) never
    contend with each other — only concurrent invocations that resolved to
    the SAME ``environment_path`` do.
    """
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    return environment_path.parent / f"{environment_path.name}.activity.lock"


def _acquire_environment_activity(
    environment_path: Path, *, want_sync: bool, sync_mandatory: bool = False
) -> tuple[int, bool]:
    """Readers-writer coordination for one partitioned environment (D-W2T-3).

    A second ``uv_workspace.py run`` invocation in the SAME worktree and
    selection, launched while a first invocation's long-running child
    (pytest, etc.) was still executing, used to trigger a SECOND, concurrent
    ``uv sync`` against the SAME ``.venv`` — mid-sync, uv removes and
    reinstalls entry-point scripts, and a forkserver child re-exec'ing
    ``.venv/bin/pytest`` found it briefly gone, corrupting an otherwise-clean
    baseline run (measured live: 933 failed / 477 errors, almost entirely
    contamination, confirmed by re-running the same targets in isolation
    immediately afterward — 77 passed / 0 failed). ``_dependency_sync_slot()``
    does not close this: it bounds how many syncs run *concurrently
    workspace-wide*, but releases before the child execs, so nothing stopped
    a SECOND sync of the SAME environment while the FIRST invocation's child
    was still reading it.

    One flock per environment directory arbitrates that specific race:

    * a ``uv sync`` (``want_sync=True``) tries a NON-BLOCKING exclusive lock
      first. Winning means no other invocation currently has a child running
      against this environment (no readers) and no sibling is mid-sync (no
      other writer) — safe to sync.
    * losing that race (or not wanting to sync at all) falls through to a
      BLOCKING shared-lock wait — immediate if the current holder is only
      readers (SH+SH is compatible), but genuinely blocks until a current
      WRITER's sync finishes, so this invocation is never hands a child a
      torn, mid-sync environment.
    * every invocation, writer or not, ends up holding a SHARED lock for the
      remainder of its work (see :func:`_downgrade_environment_activity`) —
      normally the child process's entire lifetime — so a LATER sibling's own
      writer attempt correctly loses the exclusive race and skips its sync
      instead of mutating a live one. Suggested fix (b) from the incident
      report: "detect an in-progress sync/run ... and skip re-syncing".

    ``sync_mandatory`` covers the DIFFERENT shape where the sync is not an
    optional pre-step this invocation could skip but the very operation the
    caller asked for (a bare ``uv sync``/``uv lock``, or a ``run`` whose
    selection this launcher could not partition — ``execute_is_heavy_sync``
    in :func:`run_uv`). There, "lose the race, read stale state instead" is
    not a valid fallback, so this BLOCKS for the exclusive lock instead of
    trying non-blocking-then-falling-back-to-shared; ``won_sync`` is always
    ``True`` when it returns.

    Returns ``(fd, won_sync)``. The caller performs the actual ``uv sync``
    subprocess itself (this function only arbitrates); once it is done, call
    :func:`_downgrade_environment_activity` before exec'ing the child, and
    always call :func:`_release_environment_activity` when finished.

    **Stated plainly.** This is a ``flock`` — filesystem-local and advisory.
    It coordinates concurrent ``uv_workspace.py`` invocations *on this one
    host* against *this one environment directory*. It provides no
    protection against a bare ``uv sync``/``pip install`` run outside this
    launcher (that residual gap is D-VI-1, deliberately not closed here) and
    no cross-host guarantee for an environment directory reachable from more
    than one machine.
    """
    lock_path = _environment_activity_lock_path(environment_path)
    handle = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o644)
    won_sync = False
    if want_sync:
        if sync_mandatory:
            fcntl.flock(handle, fcntl.LOCK_EX)  # blocking: this call IS the sync
            won_sync = True
        else:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                won_sync = True
            except OSError:
                won_sync = False
    if not won_sync:
        fcntl.flock(handle, fcntl.LOCK_SH)
    return handle, won_sync


def _downgrade_environment_activity(handle: int) -> None:
    """After a won sync completes, drop the exclusive lock to shared (reader).

    A second call with the same fd already holding SH is a harmless no-op —
    flock re-locking the same fd simply reasserts the mode.
    """
    fcntl.flock(handle, fcntl.LOCK_SH)


def _release_environment_activity(handle: int) -> None:
    """Release and close a handle from :func:`_acquire_environment_activity`."""
    try:
        fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


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


_OWN_SIBLINGS_DIRNAME = ".uv-workspace-siblings"


def has_own_tracked_lock(worktree: Path) -> bool:
    """Detect whether *worktree*'s own repository tracks its own ``uv.lock``.

    D-75-1 / D-CIP-19 / D-W3BP-4 (operator decision, 2026-08-07): gates used to
    validate every invocation against the shared, untracked, unshipped
    ecosystem-workspace lock at ``workspace/uv.lock`` -- even for a repository
    that ships its own tracked lock and is fully standalone-resolvable on its
    own. That let a workspace-wide override silently defeat a repo's own,
    correctly-raised dependency floor (the cryptography CVE fix), and made
    every gate route through machinery (the giant shared shadow) that a
    single-repo change had no reason to depend on.

    The fix is detection, not a hardcoded switch: a repository that commits
    its own ``uv.lock`` is authoritative for itself -- resolve and validate
    directly against ITS OWN tree. A repository with no tracked lock of its
    own (there is currently none among this launcher's callers, but nothing
    prevents one existing later) falls back to the shared ecosystem workspace,
    unchanged. This is a boolean fact about the tree (``git ls-files``), not a
    config flag a repo owner has to remember to set.
    """
    lock = worktree / "uv.lock"
    if not lock.is_file():
        return False
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", "uv.lock"],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip() != ""


def _own_sibling_member_names(worktree: Path) -> list[str]:
    """Return the sibling package names ``[tool.uv.workspace]`` expects under
    ``.uv-workspace-siblings/`` in *worktree*'s own, self-hosted workspace."""
    config = _workspace_config(worktree)
    if config is None:
        return []
    members = config.get("members", [])
    if not isinstance(members, list):
        return []
    names: list[str] = []
    prefix = f"{_OWN_SIBLINGS_DIRNAME}/"
    for member in members:
        if isinstance(member, str) and member.startswith(prefix):
            names.append(member[len(prefix) :])
    return names


def _truthy_env(name: str) -> bool:
    """Return whether env var *name* is set to a conventional truthy value."""

    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _eg_wheelhouse_root() -> Path:
    """Return the shared, pre-built epistemic-graph wheelhouse directory."""

    configured = os.environ.get(_EG_WHEELHOUSE_ENV)
    return Path(configured).expanduser() if configured else _EG_DEFAULT_WHEELHOUSE


def _select_eg_wheel(wheelhouse: Path) -> Path | None:
    """Return the newest staged ``epistemic_graph-*.whl`` in *wheelhouse*, or
    ``None`` when the wheelhouse is absent or empty (fast path unavailable --
    the caller falls back to today's real-source behaviour).

    "Newest" is by parsed version (the wheel filename's second ``-``-separated
    field), not mtime, so a re-staged older wheel never shadows a newer one.
    """

    if not wheelhouse.is_dir():
        return None
    candidates = sorted(wheelhouse.glob("epistemic_graph-*.whl"))
    if not candidates:
        return None

    def _version_key(wheel: Path) -> tuple[int, ...]:
        parts = wheel.name.split("-")
        if len(parts) < 2:
            return (0,)
        try:
            return tuple(int(segment) for segment in parts[1].split("."))
        except ValueError:
            return (0,)

    return max(candidates, key=_version_key)


# A tiny, dependency-free PEP 517 backend: every hook hands back the SAME
# staged wheel, verbatim, whether the frontend asked for ``build_wheel`` or
# the PEP 660 ``build_editable`` -- deliberately non-editable in effect either
# way, because an editable install of this compiled sibling is exactly what a
# short-lived worker worktree should never pay for (BUG-063).
_EG_FASTPATH_BACKEND_SOURCE = '''"""PEP 517 passthrough backend for a pre-staged epistemic-graph wheel.

Generated by agent-utilities' scripts/uv_workspace.py -- do not hand-edit,
it is regenerated whenever the staged wheel identity changes. Set
{escape_hatch_env}=1 to skip this vendor entirely and build the real,
live epistemic-graph source instead (for someone actually changing its
Rust code).
"""

from __future__ import annotations

import shutil
from pathlib import Path

_WHEEL = Path({wheel_path!r})


def _install(wheel_directory: str) -> str:
    if not _WHEEL.is_file():
        raise RuntimeError(
            f"epistemic-graph fast-path vendor wheel is missing: {{_WHEEL}}. "
            "Set {escape_hatch_env}=1 for a real source build."
        )
    destination = Path(wheel_directory) / _WHEEL.name
    shutil.copy2(_WHEEL, destination)
    return _WHEEL.name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _install(wheel_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _install(wheel_directory)


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def build_sdist(sdist_directory, config_settings=None):
    raise RuntimeError(
        "epistemic-graph fast-path vendor does not build sdists; set "
        "{escape_hatch_env}=1 for a real source build."
    )
'''


_EG_EXTRA_MARKER_RE = re.compile(
    r"^(?P<req>.+?)\s*;\s*extra\s*==\s*['\"](?P<extra>[^'\"]+)['\"]\s*$"
)


def _eg_wheel_requirements(
    wheel: Path,
) -> tuple[str, str, list[str], dict[str, list[str]]]:
    """Return ``(version, requires_python, dependencies, {extra: deps})`` parsed
    straight from *wheel*'s own ``dist-info/METADATA``.

    The vendor's declared requirements must exactly match what this specific
    wheel actually needs -- `uv sync --locked` cross-checks a workspace
    member's live declared dependencies against ``uv.lock``'s recorded graph
    for that member, so a vendor `pyproject.toml` that under- or over-declares
    (e.g. an empty ``dependencies = []``) makes uv report the lock stale and
    refuse to sync. Reading the wheel's own METADATA -- rather than copying
    epistemic-graph's live `pyproject.toml` by hand -- keeps this exactly
    right for whichever wheel is actually staged, with nothing to drift.
    """

    with zipfile.ZipFile(wheel) as archive:
        candidates = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"cannot locate a single dist-info/METADATA in {wheel}")
        raw = archive.read(candidates[0]).decode("utf-8")
    message = email.message_from_string(raw)
    version = message.get("Version", "0")
    requires_python = message.get("Requires-Python", ">=3.10")
    provides_extra = set(message.get_all("Provides-Extra") or [])
    base: list[str] = []
    by_extra: dict[str, list[str]] = {name: [] for name in provides_extra}
    for entry in message.get_all("Requires-Dist") or []:
        match = _EG_EXTRA_MARKER_RE.match(entry.strip())
        if match:
            by_extra.setdefault(match.group("extra"), []).append(
                match.group("req").strip()
            )
        else:
            base.append(entry.strip())
    return version, requires_python, base, by_extra


def _eg_wheel_record_digest(data: bytes) -> str:
    """Return one wheel RECORD-format hash field for *data* (PEP 376)."""

    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _eg_repackage_non_editable(source_wheel: Path, destination: Path) -> None:
    """Rewrite *source_wheel* (a PEP-660 EDITABLE payload -- one ``.pth`` file
    pointing at the live canonical checkout, plus the actually-expensive
    compiled server binary and native module) into a genuinely self-contained,
    non-editable wheel at *destination*: same compiled artifacts, but with the
    ``.pth`` redirect replaced by a real snapshot of the pure-Python package
    the ``.pth`` pointed at, taken once, right now, from that checkout's
    git-tracked files.

    This is the actual "non-editable" half of BUG-063's fix -- copying the
    staged editable wheel's bytes verbatim (what the fast path did before
    this) still left ``epistemic_graph.__file__`` resolving into the live
    canonical checkout via the ``.pth``, which is exactly the per-worktree-
    editable-install shape this bug exists to get worker worktrees off of.
    The compiled artifacts (the genuinely expensive 40-minute part) are
    carried through byte-for-byte; only the cheap Python glue is refreshed.
    """

    with zipfile.ZipFile(source_wheel) as source:
        pth_entries = [name for name in source.namelist() if name.endswith(".pth")]
        if len(pth_entries) != 1:
            raise RuntimeError(f"expected exactly one .pth entry in {source_wheel}")
        pth_name = pth_entries[0]
        record_entries = [
            name for name in source.namelist() if name.endswith(".dist-info/RECORD")
        ]
        if len(record_entries) != 1:
            raise RuntimeError(f"expected exactly one RECORD entry in {source_wheel}")
        record_name = record_entries[0]
        checkout = Path(source.read(pth_name).decode("utf-8").strip())
        if not checkout.is_dir():
            raise RuntimeError(
                f"{source_wheel}'s .pth points at a missing checkout: {checkout}"
            )
        tracked = subprocess.run(
            ["git", "-C", str(checkout), "ls-files", "--", "epistemic_graph"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        py_files = [
            relative
            for relative in tracked
            if not relative.endswith((".so", ".pyd", ".pyc"))
        ]

        staged = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as out:
                records: list[str] = []
                for info in source.infolist():
                    if info.filename in (pth_name, record_name):
                        continue
                    data = source.read(info.filename)
                    new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                    new_info.external_attr = info.external_attr
                    new_info.compress_type = zipfile.ZIP_DEFLATED
                    out.writestr(new_info, data)
                    records.append(
                        f"{info.filename},{_eg_wheel_record_digest(data)},{len(data)}"
                    )
                for relative in py_files:
                    data = (checkout / relative).read_bytes()
                    new_info = zipfile.ZipInfo(relative)
                    new_info.external_attr = (0o644 & 0xFFFF) << 16
                    new_info.compress_type = zipfile.ZIP_DEFLATED
                    out.writestr(new_info, data)
                    records.append(
                        f"{relative},{_eg_wheel_record_digest(data)},{len(data)}"
                    )
                records.append(f"{record_name},,")
                record_info = zipfile.ZipInfo(record_name)
                record_info.external_attr = (0o644 & 0xFFFF) << 16
                out.writestr(record_info, "\n".join(records) + "\n")
            os.replace(staged, destination)
        finally:
            staged.unlink(missing_ok=True)


def _eg_fastpath_identity(wheel: Path) -> str:
    """Return a short, content-stable key for *wheel* (path + size + mtime)."""

    stat = wheel.stat()
    raw = f"{wheel.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _eg_fastpath_vendor_dir(wheel: Path, *, state_root: Path | None = None) -> Path:
    """Materialize (once, content-keyed, shared by every worktree on the host)
    a tiny PEP 517 project that installs *wheel* verbatim, and return its
    directory.

    Keyed by the wheel's own identity, not by worktree: unlike the per-worktree
    ``.uv-workspace-siblings/epistemic-graph`` symlink this replaces the target
    of, this vendor directory is the SAME absolute path for every worktree on
    the host. That is what actually fixes BUG-063 -- uv's own editable-build
    cache is keyed by source path, so pointing every worktree's symlink at one
    shared path makes uv see one source and build (copy) it at most once per
    host, instead of once per worktree.
    """

    state = state_root or _user_state_root()
    key = _eg_fastpath_identity(wheel)
    vendor = state / _EG_FASTPATH_DIRNAME / key
    marker = vendor / ".vendor.json"
    expected = {"wheel": str(wheel.resolve())}

    def _current() -> dict[str, Any] | None:
        if not marker.is_file():
            return None
        try:
            loaded = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    if _current() == expected:
        return vendor

    lock_dir = state / _EG_FASTPATH_DIRNAME
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{key}.materialize.lock"
    handle = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if _current() == expected:
            return vendor
        vendor.mkdir(parents=True, exist_ok=True)
        frozen = vendor / wheel.name
        _eg_repackage_non_editable(wheel, frozen)
        backend_source = _EG_FASTPATH_BACKEND_SOURCE.format(
            wheel_path=str(frozen.resolve()),
            escape_hatch_env=_EG_SOURCE_BUILD_ENV,
        )
        (vendor / f"{_EG_FASTPATH_BACKEND_MODULE}.py").write_text(
            backend_source, encoding="utf-8"
        )
        _version, requires_python, base_deps, extras = _eg_wheel_requirements(wheel)
        lines = [
            "[build-system]",
            "requires = []",
            f'build-backend = "{_EG_FASTPATH_BACKEND_MODULE}"',
            'backend-path = ["."]',
            "",
            "[project]",
            f'name = "{_EG_MEMBER_NAME}"',
            # Dynamic, like the real epistemic-graph pyproject.toml (its version
            # is sourced from Cargo.toml by maturin): a STATIC version here made
            # uv treat this as a structurally different source than what
            # uv.lock recorded ("Resolving despite existing lockfile due to
            # static version: epistemic-graph (expected a dynamic version)"),
            # forcing a relock every sync despite the version value matching.
            # The real value still reaches uv -- from the METADATA inside the
            # wheel our backend hands back below, exactly like maturin does.
            'dynamic = ["version"]',
            f'requires-python = "{requires_python}"',
            "dependencies = [" + ", ".join(json.dumps(dep) for dep in base_deps) + "]",
        ]
        if extras:
            lines.append("")
            lines.append("[project.optional-dependencies]")
            for name in sorted(extras):
                rendered = ", ".join(json.dumps(dep) for dep in extras[name])
                lines.append(f"{name} = [{rendered}]")
        project = "\n".join(lines) + "\n"
        (vendor / "pyproject.toml").write_text(project, encoding="utf-8")
        marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        return vendor
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)


def _eg_sibling_target(real_checkout: Path) -> Path:
    """Return where ``.uv-workspace-siblings/epistemic-graph`` should point.

    Defaults to the fast, content-addressed vendor of a pre-staged wheel
    (BUG-063); falls back to *real_checkout* -- today's editable, build-from-
    source behaviour -- when the escape hatch is set or no staged wheel is
    available, so this is never a hard dependency on the wheelhouse existing.
    """

    if _truthy_env(_EG_SOURCE_BUILD_ENV):
        return real_checkout
    wheel = _select_eg_wheel(_eg_wheelhouse_root())
    if wheel is None:
        return real_checkout
    return _eg_fastpath_vendor_dir(wheel)


def materialize_own_siblings(worktree: Path, workspace: Path) -> None:
    """Symlink *worktree*'s ``.uv-workspace-siblings/<name>`` entries to the
    real sibling checkouts already present in the ecosystem workspace tree.

    Mirrors, at repo scope, exactly what CI's own ``actions/checkout`` steps
    materialize at the same relative paths for the same reason (see
    ``.github/workflows/security.yml`` and
    ``scripts/security/check_ci_environment_contract.py``): agent-utilities'
    own ``[tool.uv.sources]`` pins ``epistemic-graph``/``langfuse-agent`` as
    editable workspace members at ``.uv-workspace-siblings/<name>``, a path
    that does not exist in a bare checkout. On a dev machine or lane worktree
    the real sibling repos already exist in the ecosystem tree, so a symlink
    is all that is needed -- no clone, no copy.

    ``epistemic-graph`` is special-cased to the BUG-063 fast path (see
    :func:`_eg_sibling_target`): every other sibling (``langfuse-agent``)
    always symlinks straight to its real checkout, unchanged.
    """
    names = _own_sibling_member_names(worktree)
    if not names:
        return
    config = _workspace_config(workspace)
    if config is None:
        raise RuntimeError(f"{workspace} is not a uv workspace")
    members = _workspace_members(workspace, config)
    by_name = {member.name: member for member in members}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise RuntimeError(
            "cannot materialize .uv-workspace-siblings/: "
            f"{', '.join(missing)} not found among ecosystem workspace members "
            f"under {workspace}"
        )
    for name in names:
        real_checkout = by_name[name]
        target = (
            _eg_sibling_target(real_checkout)
            if name == _EG_MEMBER_NAME
            else real_checkout
        )
        _safe_symlink(worktree / _OWN_SIBLINGS_DIRNAME / name, target)


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
    execute_is_heavy_sync: bool = False


def uv_plan(
    arguments: list[str],
    *,
    worktree: Path,
    shadow: Path,
    own_lock: bool = False,
) -> UvPlan:
    """Build the partitioned, sync-before-exec plan for this worktree.

    ``shadow`` is the ``--project`` root uv resolves against: the shared
    ecosystem shadow workspace for a repo with no tracked lock of its own, or
    ``worktree`` itself for a repo whose own ``uv.lock`` is authoritative
    (:func:`has_own_tracked_lock`). The parameter keeps its name for callers,
    but it is a *resolution project root*, not necessarily a generated shadow.

    ``own_lock`` additionally passes ``--prerelease allow``, matching how this
    repo's own tracked lock is actually produced (it floors on
    ``fastmcp>=4.0.0b1``, a pre-release). uv only honours ``[tool.uv]``
    override-dependencies/prerelease-mode settings declared in the WORKSPACE
    ROOT manifest -- see the identical note in the ecosystem workspace root's
    own ``pyproject.toml`` -- and a repo resolving standalone against itself
    IS that root, so the flag has to travel with the invocation rather than
    live in a config key.
    """
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
    # ``--prerelease`` is a per-subcommand option, not a global ``uv`` flag, so
    # it has to land after the subcommand token in every command built below.
    prerelease = ["--prerelease", "allow"] if own_lock else []
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
                    *prerelease,
                    "--package",
                    PROJECT_NAME,
                    *selection,
                )
            )
            command = [
                *base,
                "run",
                "--no-sync",
                "--locked",
                *prerelease,
                "--package",
                PROJECT_NAME,
            ]
        else:
            command = [*base, "run", "--locked", *prerelease, "--package", PROJECT_NAME]
        command.extend(tail)
    elif subcommand == "sync":
        command = [
            *base,
            "sync",
            "--locked",
            *prerelease,
            "--package",
            PROJECT_NAME,
            *tail,
        ]
    elif subcommand == "lock":
        command = [*base, "lock", "--locked", *prerelease, *tail]
    else:
        command = [*base, *arguments]

    directory = environment_path(worktree, selection)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    # D-W5ALC-1 / D-GS27-2: when this launcher runs as (or under) a git hook
    # -- `git commit`/`git push` set GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE and
    # GIT_PREFIX in the hook's own subprocess environment, pointing at THIS
    # repository. Left in place, `os.environ.copy()` above carries them
    # straight into the `uv` subprocess and, from there, into every git
    # subprocess `uv` itself shells out to -- including fetching an unrelated
    # workspace-member git dependency (searxng) into its own, completely
    # different clone under ~/.cache/uv/git-v0/. That child `git fetch` then
    # inherits a GIT_DIR pointing at agent-utilities' worktree gitdir, which
    # is meaningless relative to the searxng clone's own directory, and fails
    # with "fatal: not a git repository (or any parent up to mount point /)".
    # Reproduced directly (no git hook involved): exporting GIT_DIR/
    # GIT_INDEX_FILE to this worktree's real values before invoking this
    # module standalone reproduces the identical failure; unsetting them
    # fixes it. Same class of leak as the PYTHONPATH strip above, same fix.
    for leaked in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX"):
        environment.pop(leaked, None)
    environment["UV_PROJECT_ENVIRONMENT"] = str(directory)
    environment[_NATIVE_ARTIFACT_CACHE_ENV] = str(_native_artifact_cache_root())
    # D-ORC-33: `prepare`'s own sync step is always pool-gated (see run_uv()).
    # This flag covers the OTHER shape -- no separate `prepare` step, but the
    # final `execute` command itself still resolves/downloads/writes against
    # the shared uv cache: a bare `sync`/`lock` subcommand, or a `run` whose
    # selection this launcher did not recognize (so it falls through to a
    # plain `uv run` with no `--no-sync`, which performs its own implicit
    # sync as a side effect).
    execute_is_heavy_sync = subcommand in {"sync", "lock"} or (
        subcommand == "run" and not recognized
    )
    return UvPlan(
        prepare=tuple(prepare),
        execute=tuple(command),
        environment=environment,
        environment_path=directory,
        selection=normalized_selection(selection),
        selection_recognized=recognized,
        command_name=command_name,
        execute_is_heavy_sync=execute_is_heavy_sync,
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
    worktree: Path,
    canonical: Path,
    workspace: Path,
    shadow: Path,
    *,
    own_lock: bool = False,
) -> dict:
    """Return bounded evidence of how *worktree* resolves.

    Two disjoint shapes (D-75-1/D-CIP-19/D-W3BP-4): a repo with its own
    tracked ``uv.lock`` resolves directly against itself (``own_lock=True``,
    ``shadow == worktree``) and is evidenced by its declared
    ``.uv-workspace-siblings/`` entries actually being materialized; a repo
    with no lock of its own still resolves through the generated ecosystem
    shadow and is evidenced exactly as before.
    """
    if own_lock:
        names = _own_sibling_member_names(worktree)
        siblings = {}
        for name in names:
            link = worktree / _OWN_SIBLINGS_DIRNAME / name
            siblings[name] = {
                "path": str(link),
                "is_symlink": link.is_symlink(),
                "resolves": link.exists(),
            }
        return {
            "status": "ok",
            "resolution_mode": "own_tracked_lock",
            "external_worktree": worktree != canonical,
            "worktree": str(worktree),
            "canonical_repository": str(canonical),
            "workspace_root": str(workspace),
            "project_root": str(shadow),
            "environments": _environment_evidence(worktree),
            "member_resolves_to_worktree": True,
            "siblings": siblings,
            "siblings_materialized": all(
                entry["resolves"] for entry in siblings.values()
            ),
        }
    member = shadow / canonical.relative_to(workspace)
    manifest = shadow / "pyproject.toml"
    lock = shadow / "uv.lock"
    canonical_manifest = workspace / "pyproject.toml"
    canonical_lock = workspace / "uv.lock"
    return {
        "status": "ok",
        "resolution_mode": "ecosystem_shadow",
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
    execute_is_heavy_sync: bool = False,
) -> int:
    """Execute uv and prove neither authoritative nor generated inputs changed.

    ``prepare`` runs first and short-circuits on failure: it is the environment
    synchronisation that must complete before the child may be exec'd against an
    environment nobody is allowed to mutate afterwards.

    ``environment_path`` and ``command_name`` enable the interpreter guard, which
    runs between the two: the environment is final by then, so the check sees
    exactly what the child would.

    D-ORC-33: every ``prepare`` step is a `uv sync` and always runs inside
    ``_dependency_sync_slot()`` -- it is by construction the resource-heavy
    operation this pool caps. ``execute_is_heavy_sync`` additionally pool-gates
    the final command when there was no separate ``prepare`` step but
    ``execute`` itself still syncs (bare ``uv sync``/``uv lock``, or a `run`
    whose selection this launcher did not recognise). The ordinary
    `run --no-sync ...` execute path (the actual test/build the caller asked
    for) is deliberately left OUTSIDE the pool: capping that too would cap
    general lane parallelism, not just the disk/cache contention this item
    measured.

    D-W2T-3: whenever ``environment_path`` is known, the WHOLE call (prepare
    through execute) also holds ``_acquire_environment_activity()``'s
    readers-writer lock for that one environment directory, so a sibling
    invocation against the SAME environment either serializes its sync ahead
    of us, or -- if we are already reading -- skips its own sync rather than
    mutating the environment out from under our still-running child. See that
    function's docstring for exactly what this does and does not guarantee.
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

    activity_handle: int | None = None
    won_sync = True
    if environment_path is not None:
        want_sync = bool(prepare) or execute_is_heavy_sync
        activity_handle, won_sync = _acquire_environment_activity(
            environment_path,
            want_sync=want_sync,
            sync_mandatory=execute_is_heavy_sync,
        )
    try:
        if prepare and not won_sync:
            print(
                f"uv_workspace: another invocation is already using "
                f"{environment_path} -- skipping `uv sync` for this "
                "invocation and running directly against its current state "
                "(D-W2T-3). This only coordinates uv_workspace.py "
                "invocations on this host; it is not a guarantee against a "
                "bare `uv sync`/`pip install` run outside this launcher "
                "(D-VI-1).",
                file=sys.stderr,
            )
        else:
            for step in prepare:
                with _dependency_sync_slot():
                    step_result = subprocess.run(
                        list(step),
                        cwd=worktree,
                        env=environment,
                        check=False,
                    )
                _assert_unchanged()
                if step_result.returncode != 0:
                    return step_result.returncode
        if activity_handle is not None and not execute_is_heavy_sync:
            # Sync (if any) is done; become a plain reader for the exec below
            # so a LATER sibling's sync attempt correctly loses the exclusive
            # race instead of mutating a live child's environment. Skipped
            # when `execute_is_heavy_sync` is true and we won the write lock:
            # in that shape the sync is baked into `execute` itself (a bare
            # `sync`/`lock`, or a `run` this launcher could not partition),
            # so the exclusive lock must stay held through it.
            _downgrade_environment_activity(activity_handle)
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
        if execute_is_heavy_sync:
            with _dependency_sync_slot():
                result = subprocess.run(
                    command,
                    cwd=worktree,
                    env=environment,
                    check=False,
                )
        else:
            result = subprocess.run(
                command,
                cwd=worktree,
                env=environment,
                check=False,
            )
        _assert_unchanged()
        return result.returncode
    finally:
        if activity_handle is not None:
            _release_environment_activity(activity_handle)


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
    own_lock = has_own_tracked_lock(worktree)
    if own_lock:
        # D-75-1/D-CIP-19/D-W3BP-4: this repo ships its own tracked uv.lock --
        # resolve and validate directly against ITS OWN tree, not the shared
        # ecosystem shadow. Still need the ecosystem tree for exactly one
        # thing: locating the real sibling checkouts to symlink in (the
        # ecosystem workspace is not otherwise consulted).
        materialize_own_siblings(worktree, workspace)
        shadow = worktree
    else:
        shadow = shadow_workspace(worktree, canonical, workspace)

    if namespace.uv_arguments == ["doctor"]:
        payload = doctor_payload(
            worktree, canonical, workspace, shadow, own_lock=own_lock
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        required_keys = (
            ("member_resolves_to_worktree", "siblings_materialized")
            if own_lock
            else (
                "member_resolves_to_worktree",
                "manifest_is_generated_copy",
                "manifest_matches_canonical",
                "lock_is_generated_copy",
                "lock_matches_canonical",
            )
        )
        return 0 if all(payload[key] for key in required_keys) else 1

    plan = uv_plan(
        namespace.uv_arguments,
        worktree=worktree,
        shadow=shadow,
        own_lock=own_lock,
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
        execute_is_heavy_sync=plan.execute_is_heavy_sync,
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
