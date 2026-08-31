"""Arbitration between concurrent development lanes on one shared repository.

Many agent sessions and humans work these repos at the same time, each in its own
git worktree. Every collision we have actually suffered has the same shape: **a
background or global actor mutates state a lane assumed it owned** — a sync that
resets a canonical checkout mid-pre-commit, a ``git stash`` against the one shared
``refs/stash``, a shared ``CARGO_TARGET_DIR`` that corrupts a concurrent build, a
venv swapped underneath a running test, a shared mutable YAML ledger clobbered by
a whole-file rewrite.

Writing the rules down did not work — every one of those rules was already
documented when it was violated. So this module makes the arbitration
*mechanical*: each shared resource is classified into exactly one of four classes,
and each class has one mechanism that makes the dangerous path fail loudly rather
than silently succeed.

``PARTITION``
    Stop sharing it. Derive a per-lane instance (cargo target dir, pytest
    basetemp, scratch, and a per-lane stash ref instead of ``refs/stash``).
    CONCEPT:AU-OS.governance.lane-partitioned-resources

``APPEND-ONLY``
    Stop rewriting it. Each writer appends immutable records to *its own*
    fragment; a reconciler folds the fragments into a generated canonical view.
    Distinct files never merge-conflict and no writer can clobber another's
    record. CONCEPT:AU-OS.governance.append-only-fragment-fold

``LEASE``
    Announce, then defer. A holder publishes a lease at the scope every lane
    shares; other lanes detect it and **defer instead of proceeding** (relock,
    venv swap, ``pre-commit --all-files``, a reconciliation merge).
    CONCEPT:AU-OS.governance.shared-scope-lease

``READ-ONLY``
    Forbid it. The canonical checkout is not a place agents edit, and no global
    actor may reset a tree holding work it does not own.
    CONCEPT:AU-OS.governance.canonical-checkout-immutable

The classification itself is data (``lane_resources.yaml``), not code, so adding a
resource is a one-row change: CONCEPT:AU-OS.governance.lane-arbitration-classes.

**Why the shared git directory is the arbitration scope.** Every worktree of a
repository shares one ``--git-common-dir``. That directory is (a) reachable and
identical from every lane, (b) never rewritten by a checkout, reset, or merge, and
(c) not version-controlled, so it can neither conflict nor be clobbered by a
merge. It is the only location with all three properties, which is exactly what an
arbiter needs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent_utilities.core.config import setting

# R-07: file_lock is the cross-platform chokepoint for advisory file locking
# (POSIX fcntl.flock / Windows kernel32 LockFileEx via ctypes); stdlib-only,
# no new deps.
from agent_utilities.knowledge_graph.core.file_lock import lock_exclusive, unlock

ARBITRATION_DIRNAME = "agent-lanes"
DEFAULT_LEASE_TTL_SECONDS = 1_800
MAX_LEASE_TTL_SECONDS = 86_400
RESOURCE_OUTPUT_MAX_BYTES = 16 * 1024 * 1024
RESOURCE_OUTPUT_TRUNCATE = True
_LANE_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_HOST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
HOST_INVENTORY_ENV = "AGENT_UTILITIES_HOST_INVENTORY"
_UNKNOWN_HOST_IDS = frozenset(
    {
        "",
        "?",
        "unknown",
        "none",
        "null",
        "unset",
        "invalid",
        "unresolved",
        "unidentified",
        "localhost",
        "localhost.localdomain",
        "0.0.0.0",
    }
)
_HOST_CAPABILITY_ROLES = frozenset({"heavy", "light-only", "gpu-guarded"})
_HOST_CAPABILITY_LIGHT_ONLY = "light-only"
_HOST_CAPABILITY_GPU_GUARDED = "gpu-guarded"
LIGHT_LANE_RESOURCE = "light-check"
LIGHT_LANE_COUNT = 32
LIGHT_LANE_CAPACITY = LIGHT_LANE_COUNT
_HOST_HEAVY_LEASE_NAME = "host-heavy"
_HOST_HEAVY_RESOURCES = frozenset(
    {"cpu-heavy", "memory-heavy", "gpu-heavy", "global-scanner-build"}
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeaseProfile:
    """Abstract host capability profile supplied by the runtime inventory."""

    heavy_capacity: int
    requires_verification: bool = False


@dataclass(frozen=True)
class HostCapabilitySnapshot:
    """One admission-time view of runtime identity, roles, and verification receipts."""

    identity: str
    roles: frozenset[str]
    health: str | None
    reboot_verified: bool
    nvml_verified: bool


_HOST_LEASE_PROFILES = {
    _HOST_CAPABILITY_LIGHT_ONLY: LeaseProfile(heavy_capacity=0),
    _HOST_CAPABILITY_GPU_GUARDED: LeaseProfile(
        heavy_capacity=0, requires_verification=True
    ),
}
# Structural evidence that a canonical-tree mutation is a sanctioned merge-back
# rather than a lane editing the shared checkout. Deliberately not a flag: a flag
# is something an agent can set, and every flag-shaped rule here was bypassed.
_IN_PROGRESS_HEADS = ("MERGE_HEAD", "REBASE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD")


class ArbitrationClass(StrEnum):
    """How a shared resource is arbitrated between concurrent lanes."""

    PARTITION = "partition"
    APPEND_ONLY = "append-only"
    LEASE = "lease"
    READ_ONLY = "read-only"


class HolderLiveness(StrEnum):
    """Evidence state for a lease holder; only ``STALE`` may be reclaimed."""

    LIVE = "live"
    STALE = "stale"
    UNKNOWN = "unknown"


class LaneArbitrationError(RuntimeError):
    """Base for every refusal this module raises."""


class CanonicalCheckoutError(LaneArbitrationError):
    """A lane tried to mutate the canonical (main) checkout."""


class UnownedTreeError(LaneArbitrationError):
    """A global actor tried to reset a tree holding work it does not own."""


class HostIdentityUnavailable(LaneArbitrationError):
    """Admission cannot proceed without a concrete, usable host identity."""


class LeaseUnavailable(LaneArbitrationError):
    """Another lane holds the lease; the caller must defer rather than proceed."""

    def __init__(self, name: str, holder: dict[str, Any]) -> None:
        held_by = holder.get("lane", "?")
        until = holder.get("expires_at", "?")
        super().__init__(
            f"lease {name!r} is held by lane {held_by!r} for "
            f"{holder.get('operation', 'an operation')!r} until {until} — defer, "
            "do not proceed"
        )
        self.name = name
        self.holder = holder


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------
def _git(args: list[str], cwd: Path) -> str:
    """Run a read-only git query, raising with the real stderr on failure."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise LaneArbitrationError(
            f"git {' '.join(args)} failed in {cwd}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


@dataclass(frozen=True)
class LaneScope:
    """Where a lane is, and the scope every lane of this repo shares.

    ``tree`` is the caller's working tree; ``main_tree`` is the canonical
    checkout; ``common_dir`` is the shared git directory that all worktrees —
    canonical and linked — resolve to identically.
    """

    tree: Path
    common_dir: Path
    main_tree: Path
    lane: str

    @property
    def is_canonical(self) -> bool:
        """True when this lane *is* the canonical checkout (never a safe place to edit)."""
        return self.tree == self.main_tree

    @property
    def arbitration_dir(self) -> Path:
        """Shared, unversioned, reset-immune directory holding leases and lane state."""
        return self.common_dir / ARBITRATION_DIRNAME

    @property
    def merge_in_progress(self) -> bool:
        """True while git is mid-merge/rebase/cherry-pick/revert in this tree."""
        git_dir = _git(["rev-parse", "--absolute-git-dir"], self.tree)
        return any(Path(git_dir, head).exists() for head in _IN_PROGRESS_HEADS)


def current_tree(path: Path | str | None = None) -> Path | None:
    """The git working tree containing *path* (default cwd), or ``None`` outside one.

    The non-raising probe callers use before deciding whether git-scoped
    arbitration is available at all (tests and installed-wheel runs are not).
    """
    start = Path(path).expanduser().resolve() if path else Path.cwd()
    if start.is_file():
        start = start.parent
    if not start.is_dir():
        return None
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return Path(proc.stdout.strip()).resolve()


def lane_name(path: Path | str | None = None) -> str:
    """This lane's identity, or ``"local"`` when *path* is not in a working tree."""
    return "local" if current_tree(path) is None else lane_scope(path).lane


def shared_arbitration_dir(path: Path | str | None = None) -> Path | None:
    """The arbitration directory every worktree of this repo shares, if any."""
    return None if current_tree(path) is None else lane_scope(path).arbitration_dir


def lane_scope(path: Path | str | None = None) -> LaneScope:
    """Resolve the :class:`LaneScope` containing *path* (default: the process cwd)."""
    start = Path(path).expanduser().resolve() if path else Path.cwd()
    if start.is_file():
        start = start.parent
    tree = Path(_git(["rev-parse", "--show-toplevel"], start)).resolve()
    common = Path(
        _git(["rev-parse", "--path-format=absolute", "--git-common-dir"], tree)
    )
    listing = _git(["worktree", "list", "--porcelain"], tree)
    main_line = next(
        (ln for ln in listing.splitlines() if ln.startswith("worktree ")), ""
    )
    main_tree = Path(main_line[len("worktree ") :]).resolve() if main_line else tree
    lane = "canonical" if tree == main_tree else _LANE_SAFE_RE.sub("-", tree.name)
    return LaneScope(
        tree=tree, common_dir=common.resolve(), main_tree=main_tree, lane=lane
    )


# ---------------------------------------------------------------------------
# READ-ONLY — the canonical checkout is not a workspace
# ---------------------------------------------------------------------------
def require_mutable_tree(
    path: Path | str | None = None, *, operation: str
) -> LaneScope:
    """Refuse *operation* when it would edit the canonical checkout.

    A merge/rebase/cherry-pick in progress is the one sanctioned canonical
    mutation (the merge-back at the end of a lane), and it is detected
    structurally from git's own state rather than trusted from a caller flag.
    """
    scope = lane_scope(path)
    if scope.is_canonical and not scope.merge_in_progress:
        raise CanonicalCheckoutError(
            f"refusing to {operation} in the canonical checkout {scope.tree} — a "
            "background sync may reset this tree and discard the work. Take a "
            "worktree first:\n"
            f"    git -C {scope.main_tree} worktree add <path> -b <branch> main"
        )
    return scope


def tree_has_uncommitted_work(path: Path | str) -> bool:
    """True when the working tree at *path* holds uncommitted (tracked or new) work."""
    return bool(_git(["status", "--porcelain"], Path(path).expanduser().resolve()))


def require_resettable_tree(path: Path | str, *, operation: str, owner: str) -> None:
    """Refuse a destructive tree operation unless the caller owns that tree.

    This is the predicate every *global* actor must consult before it resets,
    force-checks-out, or otherwise discards a working tree it did not create —
    a background repo sync, a venv/lock swapper, a fleet-wide cleanup job. The
    rule is deliberately blunt: **a tree with uncommitted work is never
    resettable by anyone but its own lane**, because the window in which the
    work is unrecoverable is exactly the window a lane cannot commit from
    (mid-pre-commit).
    """
    scope = lane_scope(path)
    if scope.lane == owner:
        return
    if tree_has_uncommitted_work(scope.tree):
        raise UnownedTreeError(
            f"refusing to {operation} {scope.tree}: it holds uncommitted work and "
            f"belongs to lane {scope.lane!r}, not to {owner!r}. Skip this tree."
        )


# ---------------------------------------------------------------------------
# PARTITION — stop sharing it
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PartitionedPaths:
    """Per-lane instances of resources that corrupt or serialize when shared."""

    cargo_target_dir: Path
    pytest_basetemp: Path
    scratch_dir: Path
    precommit_home: Path
    stash_ref: str


LANE_TEMP_DIRNAME = ".al"


def _lane_temp_root(scope: LaneScope) -> Path:
    """Where this lane's pytest basetemp / scratch (``TMPDIR``) live (D-ORC-16).

    Deliberately **not** under ``scope.arbitration_dir`` (the repo's shared
    ``.git`` dir): that path ran ~90-100 chars deep, which left under 10 bytes
    of ``AF_UNIX``'s 108-byte ``sun_path`` for anything a fixture appended —
    its own per-test subdirectory plus a socket filename — so *any* Unix-socket
    fixture under a lane basetemp failed deterministically on path length
    alone (measured: a 99-char basetemp produced a 158-byte socket path and a
    live-engine subprocess that could not bind it). Living inside ``.git`` had
    a second, independent cost: a supposedly hermetic ``git -C <tmp_path>
    remote get-url origin`` resolved to the **canonical repo's own origin**,
    leaking its identity into tests that believed themselves isolated.

    ``~/.al/<token>`` fixes both — short (~35-45 chars total, comfortably
    inside ``sun_path``) and outside every repo's ``.git``. ``/home`` was
    chosen over ``/tmp`` because ``/tmp`` here is a RAM-backed ``tmpfs`` that
    has already hit 100% swap under concurrent lane load; keeping pytest's
    temp-file volume on disk avoids compounding that.

    ``token`` hashes ``(common_dir, lane)``, **not** ``lane`` alone: every
    repo's canonical checkout is independently named lane ``"canonical"``, so
    hashing the lane name alone would collapse the canonical checkout of
    *every* repo in the workspace onto one path — exactly the PARTITION
    collision this module exists to prevent. This changes **where** the
    per-lane resource lives, never **whether** it is per-lane.
    """
    token = hashlib.sha1(
        f"{scope.common_dir}:{scope.lane}".encode(), usedforsecurity=False
    ).hexdigest()[:12]
    return Path.home() / LANE_TEMP_DIRNAME / token


def partitioned_paths(path: Path | str | None = None) -> PartitionedPaths:
    """Resolve this lane's private instance of every PARTITION-class resource.

    ``stash_ref`` matters most: ``refs/stash`` is a **single ref shared by every
    worktree**, so ``git stash`` in one lane is visible — and poppable — in all of
    them. A per-lane ref makes concurrent stashing safe, and is the only reason
    the blanket never-stash rule can ever be relaxed.

    ``pytest_basetemp``/``scratch_dir``/``precommit_home`` resolve under
    ``_lane_temp_root()``, a short ``~/.al/<token>`` path outside any repo's
    ``.git`` — see that function's docstring for why (D-ORC-16: AF_UNIX
    ``sun_path`` overflow + a git-identity leak that both traced to the same
    `.git`-nested location).

    ``temp_root`` is created here (``mkdir(parents=True, exist_ok=True)``),
    mirroring :func:`workspace_arbitration_dir`'s same pattern for the other
    host-wide resource root. Without it, a freshly-hashed token's directory
    does not exist yet, and pytest's own ``TempPathFactory.getbasetemp()``
    creates only ``--basetemp`` itself (``mkdir(exist_ok=True)``, no
    ``parents=True``) — so the very first ``pytest`` invocation under a new
    lane's exported ``PYTEST_ADDOPTS`` failed with a raw
    ``FileNotFoundError`` on ``<temp_root>/pytest``, one lane env call before
    any test code ran.

    ``precommit_home`` is the pre-commit store (D-ORC-37). ``PRE_COMMIT_HOME``,
    or ``XDG_CACHE_HOME/pre-commit`` when unset (verified in
    ``pre_commit/store.py``), is where
    ``staged_files_only._unstaged_changes_cleared`` writes a lane's UNSTAGED
    changes to a ``patch<epoch>-<pid>`` file before ``git checkout``-ing them
    away, restoring them by ``git apply`` in a ``finally`` (verified:
    ``pre_commit/commands/run.py`` passes ``store.directory`` — the very
    directory holding pre-commit's SQLite ``db.db`` — as that patch_dir).
    ``PRE_COMMIT_HOME`` was exported nowhere in this codebase, so every lane
    shared ONE store, producing two independent hazards: a killed/OOMed/
    power-lost pre-commit strands a lane's unstaged work as an orphaned patch
    file nobody replays (D-OB-12; see :func:`orphaned_precommit_patches`), and
    the shared ``db.db`` raised ``OperationalError: database is locked`` under
    concurrent lanes (lane-concept-docs-0801). Partitioning the store removes
    both: each lane's patches and SQLite state are invisible to every other
    lane's pre-commit. It is NOT created here — pre-commit's own ``Store``
    creates its directory, and :func:`orphaned_precommit_patches` treats a
    missing directory as the genuine "this lane has never run pre-commit".
    """
    scope = lane_scope(path)
    temp_root = _lane_temp_root(scope)
    temp_root.mkdir(parents=True, exist_ok=True)
    scratch_dir = temp_root / "scratch"
    # Unlike pytest_basetemp (pytest creates that leaf itself, exist_ok=True,
    # once its parent exists), TMPDIR has no such self-creating consumer —
    # every ordinary tempfile.mkstemp()/mkdtemp() caller assumes the
    # directory it names already exists.
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return PartitionedPaths(
        cargo_target_dir=scope.tree / "target-isolated",
        pytest_basetemp=temp_root / "pytest",
        scratch_dir=scratch_dir,
        precommit_home=temp_root / "precommit",
        stash_ref=f"refs/lane/{scope.lane}/stash",
    )


_PRECOMMIT_PATCH_RE = re.compile(r"^patch(\d+)-(\d+)$")


def is_pid_alive(pid: int) -> bool:
    """Portable liveness probe (R-07): does a process with this pid exist on
    THIS host, without sending it any real signal or otherwise touching it.

    THE chokepoint for this check — every other pid-liveness site in this
    ecosystem (:func:`_holder_liveness` below, ``cli/__init__.py``'s
    ``status()``, ``repository_manager.task_queue``'s
    ``_global_holder_alive``) should import and call this rather than
    re-implement it, for the same "route through one primitive" reason as
    :mod:`file_lock`.

    POSIX: ``os.kill(pid, 0)`` — the decades-old idiom. Signal 0 delivers no
    real signal; the kernel only validates that the target pid exists and is
    signalable by this process, raising ``ProcessLookupError``/
    ``PermissionError`` instead of actually affecting the target.

    Windows: **not** ``os.kill(pid, 0)``. CPython's Windows implementation
    of ``os.kill`` has no special case for signal 0 — unlike POSIX, 0 has no
    meaning to Win32 at all, so the call falls through to the general path
    and invokes ``TerminateProcess(handle, 0)``, which would ACTUALLY KILL
    the target process (confirmed via CPython's own issue tracker:
    https://bugs.python.org/issue14480, closed by a core dev with "0 has no
    special meaning on Windows so I'd rather not add another special case
    for posix emulation" — i.e. this is documented upstream behaviour, not
    an oversight this fix can rely on changing). Using it for a liveness
    check would be a live, safety-critical bug the moment this code ever
    runs on Windows: probing a stale/reused pid could kill an unrelated
    live process. Instead this calls the real Win32 liveness primitive,
    ``OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, ...)``: it succeeds
    (and the handle is immediately closed again) if a process with that pid
    exists, regardless of ownership — matching the POSIX
    ``PermissionError`` -> still-alive branch below — and fails if the pid
    does not correspond to a running process. Nothing is ever terminated,
    signalled, or otherwise touched; this only asks "does it exist".
    """
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Owned by another user but real: still alive.
            return True
        except OSError:
            return False
        return True

    import ctypes

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False  # ERROR_INVALID_PARAMETER et al.: no such process
    kernel32.CloseHandle(handle)
    return True


def _classify_precommit_patch(patch_path: Path, scope: LaneScope) -> dict[str, Any]:
    """Classify one ``patch<epoch>-<pid>`` file honestly (see caller's docstring)."""
    name = patch_path.name
    try:
        size: int | None = patch_path.stat().st_size
    except OSError:
        size = None
    match = _PRECOMMIT_PATCH_RE.match(name)
    if not match:
        return {
            "path": str(patch_path),
            "state": "unknown",
            "bytes": size,
            "recorded_at": None,
            "pid": None,
            "reason": (
                f"filename {name!r} does not match pre-commit's "
                "patch<epoch>-<pid> shape — cannot decide"
            ),
            "replay": f"git apply {patch_path}",
        }
    epoch, pid = int(match.group(1)), int(match.group(2))
    recorded_at = datetime.fromtimestamp(epoch, tz=UTC).isoformat()
    if is_pid_alive(pid):
        return {
            "path": str(patch_path),
            "state": "in-progress",
            "bytes": size,
            "recorded_at": recorded_at,
            "pid": pid,
            "reason": (
                f"pid {pid} is still alive — pre-commit may still be mid-run "
                "(the pid may also belong to an unrelated process that reused it)"
            ),
        }
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "apply", "--check", "--reverse", str(patch_path)],
            cwd=str(scope.tree),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return {
            "path": str(patch_path),
            "state": "unknown",
            "bytes": size,
            "recorded_at": recorded_at,
            "pid": pid,
            "reason": f"could not run `git apply --check --reverse`: {exc}",
            "replay": f"git apply {patch_path}",
        }
    if proc.returncode == 0:
        return {
            "path": str(patch_path),
            "state": "restored",
            "bytes": size,
            "recorded_at": recorded_at,
            "pid": pid,
            "reason": (
                "`git apply --check --reverse` succeeded: this patch's content "
                "is already present in the tree"
            ),
        }
    detail = (proc.stderr.strip() or proc.stdout.strip())[:300]
    return {
        "path": str(patch_path),
        "state": "ORPHANED",
        "bytes": size,
        "recorded_at": recorded_at,
        "pid": pid,
        "reason": (
            "`git apply --check --reverse` failed: this patch's content is NOT "
            f"in the tree — data-loss incident (D-OB-12). git said: {detail}"
        ),
        "replay": f"git apply {patch_path}",
    }


def orphaned_precommit_patches(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Classify every patch file in this lane's PARTITIONed pre-commit store.

    pre-commit's ``_unstaged_changes_cleared`` writes a lane's unstaged changes
    to ``<patch_dir>/patch<epoch>-<pid>``, ``git checkout``s them away, and
    restores them by ``git apply`` in a ``finally``. Crucially, **pre-commit
    never deletes a patch file, even on success** — so "a patch file exists" is
    not, by itself, evidence of anything. This function tells the difference:

    - filename doesn't parse as ``patch<epoch>-<pid>`` -> ``"unknown"``.
    - the recorded pid is still alive -> ``"in-progress"`` (note: the pid may
      have been reused by an unrelated process — the reason says so).
    - dead pid: ask git whether the patch is already applied —
      ``git apply --check --reverse <patch>`` in this lane's tree. Exit 0 means
      the content IS in the tree -> ``"restored"``. Non-zero means it is NOT ->
      ``"ORPHANED"`` — a real data-loss incident (D-OB-12).
    - git itself unusable, or any other undecidable outcome -> ``"unknown"``,
      **never** ``"restored"``: an unreadable answer must never be reported as
      clean.

    Returns newest-first. ``[]`` means the store directory genuinely does not
    exist (or holds no patch files); a directory that exists but cannot be
    listed yields one ``"unknown"`` entry naming the problem instead of a
    silent ``[]``. Never raises.
    """
    scope = lane_scope(path)
    precommit_home = partitioned_paths(scope.tree).precommit_home
    if not precommit_home.exists():
        return []
    try:
        candidates = sorted(precommit_home.glob("patch*"))
    except OSError as exc:
        return [
            {
                "path": str(precommit_home),
                "state": "unknown",
                "bytes": None,
                "recorded_at": None,
                "pid": None,
                "reason": f"could not list {precommit_home}: {exc}",
                "replay": None,
            }
        ]
    results = [_classify_precommit_patch(p, scope) for p in candidates if p.is_file()]
    results.sort(key=lambda r: r.get("recorded_at") or "", reverse=True)
    return results


CARGO_CONFIG_MARKER = "# CONCEPT:AU-OS.governance.lane-partitioned-resources"


def write_cargo_partition_config(
    path: Path | str | None = None, *, force: bool = False
) -> dict[str, Any]:
    """Make cargo PARTITION structurally **bind** instead of merely being exported.

    ``lane env`` hands back a private ``cargo_target_dir``, but nothing forced its
    use — the exact failure mode as the old ``--target-dir ./target-isolated``
    convention (D-CP-4). Cargo resolves a *relative* ``build.target-dir`` relative
    to the directory containing ``.cargo/config.toml`` — the worktree root here —
    so the **same committed file** gives every worktree of this repo its own
    isolated target dir for free: no per-lane content, no env var, no action
    needed after checkout. That is what makes this PREVENTION, not detection.

    The residual gap is cargo's own precedence: an exported ``CARGO_TARGET_DIR``
    still wins over this file. That case is *detected* loudly by
    ``scripts/check_lane_guard.py``'s cargo-target-override check, not solved here.

    Never clobbers unrelated existing cargo config (e.g. a repo's target-cpu
    notes) — refuses when ``.cargo/config.toml`` already exists with different
    content unless ``force=True``, in which case the partition block is appended.
    """
    scope = lane_scope(path)
    if not (scope.tree / "Cargo.toml").is_file():
        raise LaneArbitrationError(
            f"{scope.tree} has no Cargo.toml at its root — not a cargo project"
        )
    rel_target = partitioned_paths(scope.tree).cargo_target_dir.relative_to(scope.tree)
    config_path = scope.tree / ".cargo" / "config.toml"
    block = (
        f"{CARGO_CONFIG_MARKER} — generated by `agent-utilities lane bind-cargo`.\n"
        "# Structural PARTITION binding (D-CP-4): a RELATIVE target-dir resolves\n"
        "# relative to this file's parent directory (the worktree root), so the\n"
        "# SAME committed file gives every worktree of this repo its own isolated\n"
        "# cargo target dir for free — no per-lane content, no env var, no action\n"
        "# needed after checkout. An exported CARGO_TARGET_DIR still overrides this\n"
        "# (cargo's own precedence) — lane-guard's cargo-target-override check\n"
        "# detects that case loudly; see docs/architecture/lane-concurrency.md.\n"
        "[build]\n"
        f'target-dir = "{rel_target}"\n'
    )
    if config_path.is_file():
        existing = config_path.read_text(encoding="utf-8")
        if block in existing:
            return {
                "written": False,
                "already_present": True,
                "path": str(config_path),
            }
        if not force:
            raise LaneArbitrationError(
                f"{config_path} already exists with different content — pass "
                "force=True to append the partition block, or edit it by hand"
            )
        config_path.write_text(existing.rstrip("\n") + "\n\n" + block, encoding="utf-8")
        return {"written": True, "appended": True, "path": str(config_path)}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(block, encoding="utf-8")
    return {"written": True, "appended": False, "path": str(config_path)}


def park_worktree(
    path: Path | str | None = None, *, message: str = "lane park"
) -> dict[str, Any]:
    """Give this lane a clean tree **without** touching the shared ``refs/stash``.

    ``git stash`` is the reflex when you need a clean tree for a moment, and
    forbidding it without supplying that affordance guarantees the rule keeps
    losing — it has now been violated ten times, most recently by an actor with
    the prohibition in front of it. So this *is* the affordance: ``git stash
    create`` builds the same stash commit but writes **no ref**, we point this
    lane's own ref at it, and only then clean the tree.

    Untracked files are deliberately left in place: ``reset --hard`` does not
    remove them, so nothing has to be captured to survive, and nothing can be
    lost by a mistake in the capture.
    """
    scope = require_mutable_tree(path, operation="park the working tree")
    require_resettable_tree(scope.tree, operation="park", owner=scope.lane)
    ref = partitioned_paths(scope.tree).stash_ref
    created = _git(["stash", "create", message], scope.tree)
    if not created:
        return {"parked": False, "reason": "tree has no tracked changes", "ref": ref}
    _git(["update-ref", ref, created], scope.tree)
    _git(["reset", "--hard"], scope.tree)
    return {"parked": True, "ref": ref, "commit": created}


def unpark_worktree(path: Path | str | None = None) -> dict[str, Any]:
    """Restore what :func:`park_worktree` set aside, from this lane's own ref."""
    scope = require_mutable_tree(path, operation="unpark the working tree")
    ref = partitioned_paths(scope.tree).stash_ref
    # `rev-parse --verify` exits non-zero for a missing ref; for-each-ref exits 0
    # with empty output, so a missing park is an explained refusal, not a stack
    # trace about git.
    commit = _git(["for-each-ref", "--format=%(objectname)", ref], scope.tree)
    if not commit:
        raise LaneArbitrationError(f"nothing parked at {ref}")
    _git(["stash", "apply", commit], scope.tree)
    _git(["update-ref", "-d", ref], scope.tree)
    return {"restored": True, "ref": ref, "commit": commit}


# ---------------------------------------------------------------------------
# LEASE — announce, then defer
# ---------------------------------------------------------------------------
def _parse_inventory_roles(value: object) -> frozenset[str] | None:
    if not isinstance(value, list) or not value:
        return None
    roles: set[str] = set()
    for role in value:
        if not isinstance(role, str):
            return None
        normalized = role.strip().casefold()
        if normalized not in _HOST_CAPABILITY_ROLES:
            return None
        roles.add(normalized)
    return frozenset(roles)


def _parse_inventory_entry(
    identity: object, roles: object
) -> tuple[str, frozenset[str]] | None:
    if not isinstance(identity, str):
        return None
    candidate = identity.strip()
    if not candidate or not _HOST_ID_RE.fullmatch(candidate):
        return None
    parsed_roles = _parse_inventory_roles(roles)
    if parsed_roles is None:
        return None
    return candidate.casefold(), parsed_roles


def _host_inventory() -> dict[str, frozenset[str]]:
    """Load the operator's canonical identities and capability roles."""
    raw: str = setting(HOST_INVENTORY_ENV, "")
    if not raw.strip():
        return {}
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    inventory: dict[str, frozenset[str]] = {}
    for identity, roles in loaded.items():
        entry = _parse_inventory_entry(identity, roles)
        if entry is None or entry[0] in inventory:
            return {}
        inventory[entry[0]] = entry[1]
    return inventory


def _host_snapshot(
    host_id: str | None,
    host_health: str | None,
    reboot_verified: bool,
    nvml_verified: bool,
) -> HostCapabilitySnapshot:
    identity = resolve_host_identity(host_id)
    roles = _host_inventory().get(identity.casefold(), frozenset())
    return HostCapabilitySnapshot(
        identity=identity,
        roles=roles,
        health=host_health,
        reboot_verified=reboot_verified,
        nvml_verified=nvml_verified,
    )


def resolve_host_identity(host_id: str | None = None) -> str:
    """Return an exact runtime-inventory identity, refusing ambiguity.

    Host-scoped resources use one workspace arbitration directory per machine.
    The operator supplies the canonical identities and abstract capability roles
    through ``AGENT_UTILITIES_HOST_INVENTORY``. A missing or unknown,
    or suffixed identity is refused instead of sharing an ``unknown`` bucket.
    Omitting ``host_id`` uses the local hostname, which must be in that runtime
    inventory as an exact key.
    """
    try:
        candidate = socket.gethostname() if host_id is None else host_id
    except OSError as exc:
        raise HostIdentityUnavailable(
            "host identity is unavailable; refusing host-scoped admission"
        ) from exc
    if not isinstance(candidate, str):
        raise HostIdentityUnavailable(
            "host identity must be a non-empty hostname or operator label"
        )
    identity = candidate.strip()
    if (
        not identity
        or identity.casefold() in _UNKNOWN_HOST_IDS
        or not _HOST_ID_RE.fullmatch(identity)
    ):
        raise HostIdentityUnavailable(
            f"host identity {candidate!r} is missing or ambiguous; refusing "
            "host-scoped admission"
        )
    if identity.casefold() not in _host_inventory():
        raise HostIdentityUnavailable(
            f"host identity {identity!r} is not an exact runtime inventory key; "
            "refusing host-scoped admission"
        )
    return identity


def _host_lease_profile(roles: frozenset[str]) -> LeaseProfile | None:
    for capability, profile in _HOST_LEASE_PROFILES.items():
        if capability in roles:
            return profile
    return None


def _local_hostname() -> str | None:
    """Return the local hostname only when it is in runtime inventory."""
    try:
        return resolve_host_identity()
    except HostIdentityUnavailable:
        return None


def _legacy_local_hostname() -> str | None:
    """Return the local hostname for legacy, non-host-scoped lease records."""
    try:
        candidate = socket.gethostname()
    except OSError:
        return None
    if not isinstance(candidate, str):
        return None
    normalized = candidate.strip()
    return normalized or None


def _is_healthy(host_health: str | None) -> bool:
    return isinstance(host_health, str) and host_health.strip().casefold() == "healthy"


def _validate_verification_receipts(
    reboot_verified: object, nvml_verified: object
) -> tuple[bool, bool]:
    """Accept only real boolean verification receipts at the public boundary."""
    if type(reboot_verified) is not bool or type(nvml_verified) is not bool:
        raise LaneArbitrationError(
            "reboot_verified and nvml_verified must be exact boolean receipts"
        )
    return reboot_verified, nvml_verified


def _validate_ttl(ttl_seconds: int) -> int:
    """Require a bounded positive integer TTL for every new lease."""
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int)
        or not 0 < ttl_seconds <= MAX_LEASE_TTL_SECONDS
    ):
        raise LaneArbitrationError(
            f"ttl_seconds must be a positive integer <= {MAX_LEASE_TTL_SECONDS}"
        )
    return ttl_seconds


def _host_resource_capacity(
    name: str,
    snapshot: HostCapabilitySnapshot,
    declared: int,
) -> int:
    """Apply the conservative interim host profile to a declared capacity.

    Native resource reservations are the eventual authority.  Until they are
    available, heavy work is serialized by this external lease. A light-only
    role is denied; other inventory roles use the declared one-lane interim cap
    even when their hardware upper bounds are higher.
    """
    if name not in _HOST_HEAVY_RESOURCES:
        return declared
    profile = _host_lease_profile(snapshot.roles)
    if profile is None:
        return declared
    if (
        profile.requires_verification
        and snapshot.reboot_verified
        and snapshot.nvml_verified
    ):
        return declared
    return profile.heavy_capacity


def resource_scope(name: str) -> str:
    """Declared contention scope of *name* (``"repo"`` for anything unregistered)."""
    for rule in resource_rules():
        if rule.name == name:
            return rule.scope
    return "repo"


def workspace_arbitration_dir() -> Path:
    """Host-wide arbitration root, for resources no single repository owns.

    The shared virtualenv and lockfile are the motivating case: they are
    contended across every worktree of every repo on the host, so their lease
    cannot live inside one repository's git directory.
    """
    import platformdirs

    path = Path(platformdirs.user_state_dir("agent-utilities")) / ARBITRATION_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _lease_dir(scope: LaneScope, name: str = "") -> Path:
    """Where a lease file lives.

    Repo-scoped leases sit under the repository's shared git directory, which
    ``git status`` never reports and no commit can capture — the same property
    repository-manager relies on for its own ``.git/`` lease, so the two are
    interchangeable rather than competing.
    """
    root = (
        workspace_arbitration_dir()
        if name and resource_scope(name) == "workspace"
        else scope.arbitration_dir
    )
    path = root / "leases"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _lease_mutex(scope: LaneScope, name: str = "") -> Iterator[None]:
    """Serialize lease read-modify-write across every lane sharing this resource."""
    lock = _lease_dir(scope, name) / ".mutex"
    fd = os.open(str(lock), os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        lock_exclusive(fd)
        yield
    finally:
        try:
            unlock(fd)
        finally:
            os.close(fd)


def _validated_holder_host(holder: dict[str, Any]) -> str | None:
    recorded_value = holder.get("host_identity", holder.get("host"))
    if not isinstance(recorded_value, str) or not recorded_value.strip():
        return None
    try:
        recorded_host = resolve_host_identity(recorded_value)
    except HostIdentityUnavailable:
        return None
    legacy_host = holder.get("host")
    if legacy_host is None:
        return recorded_host
    if not isinstance(legacy_host, str):
        return None
    try:
        legacy_identity = resolve_host_identity(legacy_host)
    except HostIdentityUnavailable:
        return None
    return (
        recorded_host
        if legacy_identity.casefold() == recorded_host.casefold()
        else None
    )


def _legacy_holder_host(holder: dict[str, Any]) -> str | None:
    recorded_value = holder.get("host")
    local_host = _legacy_local_hostname()
    if (
        not isinstance(recorded_value, str)
        or local_host is None
        or recorded_value.strip().casefold() != local_host.casefold()
    ):
        return None
    return recorded_value.strip()


def _lease_expired(expires: object) -> bool | None:
    if not isinstance(expires, str) or not expires:
        return None
    try:
        return datetime.fromisoformat(expires) < datetime.now(UTC)
    except (TypeError, ValueError):
        return None


def _holder_process_liveness(holder: dict[str, Any]) -> HolderLiveness:
    expired = _lease_expired(holder.get("expires_at"))
    if expired is None:
        return HolderLiveness.UNKNOWN
    if expired:
        return HolderLiveness.STALE
    pid = holder.get("pid")
    if not isinstance(pid, int):
        return HolderLiveness.UNKNOWN
    return HolderLiveness.LIVE if is_pid_alive(pid) else HolderLiveness.STALE


def _host_holder_liveness(holder: dict[str, Any]) -> HolderLiveness:
    """Validate canonical host evidence before considering expiry or process state."""
    recorded_host = _validated_holder_host(holder)
    local_host = _local_hostname()
    if recorded_host is None or local_host is None:
        return HolderLiveness.UNKNOWN
    if recorded_host.casefold() != local_host.casefold():
        return HolderLiveness.UNKNOWN
    return _holder_process_liveness(holder)


def _legacy_holder_liveness(holder: dict[str, Any]) -> HolderLiveness:
    """Use exact local-host evidence for old non-host-scoped lease records."""
    if _legacy_holder_host(holder) is None:
        return HolderLiveness.UNKNOWN
    return _holder_process_liveness(holder)


def _holder_is_host_scoped(holder: dict[str, Any]) -> bool:
    resource = holder.get("resource")
    if not isinstance(resource, str):
        return False
    rule = _resource_rule(resource)
    return rule is not None and rule.host_scoped


def _holder_liveness(holder: dict[str, Any]) -> HolderLiveness:
    """Classify holder evidence; only a same-host stale record is reclaimable."""
    if not isinstance(holder, dict):
        return HolderLiveness.UNKNOWN
    if _holder_is_host_scoped(holder):
        return _host_holder_liveness(holder)
    return _legacy_holder_liveness(holder)


def _read_current_holder(lease_file: Path) -> dict[str, Any] | None:
    """Read one lease and reclaim it only when liveness is conclusively stale."""
    if not os.path.lexists(str(lease_file)):
        return None
    try:
        info = lease_file.lstat()
    except OSError as exc:
        return {
            "name": lease_file.stem,
            "lease_file": str(lease_file),
            "state": "unreadable",
            "error": str(exc),
        }
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        return {
            "name": lease_file.stem,
            "lease_file": str(lease_file),
            "state": "unreadable",
            "error": "lease path is not a private regular file",
        }
    holder = _read_lease_file(lease_file)
    if _holder_liveness(holder) is HolderLiveness.STALE:
        lease_file.unlink()
        return None
    return holder


def _read_lease_file(lease_file: Path) -> dict[str, Any]:
    """Read one lease without turning corrupt state into a fail-open path."""
    try:
        loaded = json.loads(lease_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "name": lease_file.stem,
            "lease_file": str(lease_file),
            "state": "unreadable",
            "error": str(exc),
        }
    if not isinstance(loaded, dict):
        return {
            "name": lease_file.stem,
            "lease_file": str(lease_file),
            "state": "unreadable",
            "error": "lease record is not a JSON object",
        }
    return loaded


def resource_log_path() -> Path:
    """The append-only, file-backed host resource admission log."""
    path = workspace_arbitration_dir() / "logs" / "resource-leases.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _append_resource_log(event: str, record: dict[str, Any]) -> None:
    """Append an admission event while holding a separate file lock."""
    path = resource_log_path()
    lock_path = path.with_name(f"{path.name}.lock")
    fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        lock_exclusive(fd)
        payload = {
            "event": event,
            "recorded_at": datetime.now(UTC).isoformat(),
            "resource": record.get("resource", record.get("name")),
            "slot": record.get("slot", 0),
            "capacity": record.get("capacity"),
            "lane": record.get("lane"),
            "operation": record.get("operation"),
            "pid": record.get("pid"),
            "host_identity": record.get("host_identity", record.get("host")),
            "output_file": record.get("output_file"),
            "output_path": record.get("output_path", record.get("output_file")),
        }
        log_fd = os.open(str(path), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            line = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
            os.write(log_fd, line)
            os.fsync(log_fd)
        finally:
            os.close(log_fd)
    finally:
        try:
            unlock(fd)
        finally:
            os.close(fd)


def _output_target(output_path: Path | str | None) -> Path:
    """Normalize a caller path while rejecting stream aliases."""
    raw_path = "" if output_path is None else str(output_path).strip()
    if raw_path.casefold() in {
        "",
        "-",
        "stdout",
        "stderr",
        "/dev/stdout",
        "/dev/stderr",
    }:
        raise LaneArbitrationError(
            "this resource requires output_path: redirect command output to a "
            "regular file, not stdout"
        )
    return Path(raw_path).expanduser().absolute()


def _validate_output_policy(max_bytes: int, truncate: bool) -> None:
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
        or not isinstance(truncate, bool)
    ):
        raise LaneArbitrationError(
            "output byte cap must be a positive integer and truncate must be bool"
        )


def _validate_existing_output(target: Path) -> None:
    # ``lexists`` distinguishes a missing target from a broken symlink without
    # turning an unexpected lstat failure into a successful admission.  The
    # final no-follow open and fstat remain the race-safe authority.
    if not os.path.lexists(str(target)):
        return
    existing = target.lstat()
    if stat.S_ISLNK(existing.st_mode):
        raise LaneArbitrationError(f"output_path {target} is a symlink")
    if not stat.S_ISREG(existing.st_mode):
        raise LaneArbitrationError(f"output_path {target} is not a regular file")
    if existing.st_nlink != 1:
        raise LaneArbitrationError(
            f"output_path {target} has {existing.st_nlink} hard links; refusing"
        )


def _output_open_flags(truncate: bool) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise LaneArbitrationError(
            "output_path cannot be opened safely: no-follow is unavailable"
        )
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | no_follow
    return flags if truncate else flags | os.O_APPEND


def _open_output_target(target: Path, truncate: bool) -> int:
    try:
        return os.open(str(target), _output_open_flags(truncate), 0o600)
    except OSError as exc:
        raise LaneArbitrationError(
            f"cannot open required output_path {target} without following links: {exc}"
        ) from exc


def _validate_open_output(
    fd: int, target: Path, *, max_bytes: int, truncate: bool
) -> int:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise LaneArbitrationError(f"output_path {target} is not a regular file")
    if info.st_nlink != 1:
        raise LaneArbitrationError(
            f"output_path {target} has {info.st_nlink} hard links; refusing"
        )
    if info.st_size > max_bytes and not truncate:
        raise LaneArbitrationError(
            f"output_path {target} exceeds the {max_bytes}-byte cap"
        )
    if truncate:
        os.ftruncate(fd, 0)
    return info.st_size


def _open_output_fd(
    output_path: Path | str | None,
    *,
    max_bytes: int = RESOURCE_OUTPUT_MAX_BYTES,
    truncate: bool = RESOURCE_OUTPUT_TRUNCATE,
) -> tuple[int, Path]:
    """Open a private regular output file with no-follow and size checks."""
    _validate_output_policy(max_bytes, truncate)
    target = _output_target(output_path)
    fd: int | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        _validate_existing_output(target)
        fd = _open_output_target(target, truncate)
        _validate_open_output(fd, target, max_bytes=max_bytes, truncate=truncate)
    except LaneArbitrationError:
        if fd is not None:
            os.close(fd)
        raise
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        raise LaneArbitrationError(
            f"cannot open required output_path {target} without following links: {exc}"
        ) from exc
    assert fd is not None
    return fd, target


def _coerce_output_view(data: bytes | bytearray | memoryview) -> memoryview:
    try:
        view = memoryview(data)
    except TypeError as exc:
        raise TypeError("bounded output accepts bytes-like data") from exc
    if view.itemsize != 1 or view.ndim != 1:
        return memoryview(view.tobytes())
    return view


def _validate_initial_output_size(max_bytes: int, initial_size: int) -> None:
    if (
        not isinstance(initial_size, int)
        or initial_size < 0
        or initial_size > max_bytes
    ):
        raise LaneArbitrationError("invalid bounded output writer configuration")


class BoundedOutputWriter:
    """A binary sink that enforces its byte boundary on every write.

    ``truncate=True`` writes the prefix that fits and discards the remainder;
    ``truncate=False`` refuses a write that would cross the boundary. The
    underlying descriptor is unbuffered so the cap applies while a command is
    still running, and close never pads a short file.
    """

    def __init__(
        self,
        stream: Any,
        *,
        max_bytes: int,
        truncate: bool,
        initial_size: int = 0,
    ) -> None:
        _validate_output_policy(max_bytes, truncate)
        _validate_initial_output_size(max_bytes, initial_size)
        self._stream = stream
        self.max_bytes = max_bytes
        self.truncate = truncate
        self.bytes_written = initial_size
        self.truncated = False

    @property
    def remaining(self) -> int:
        return self.max_bytes - self.bytes_written

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def write(self, data: bytes | bytearray | memoryview) -> int:
        """Write only data that fits, or refuse the whole overflowing write."""
        view = _coerce_output_view(data)
        size = len(view)
        if size == 0:
            return 0
        remaining = self.remaining
        if size > remaining and not self.truncate:
            raise LaneArbitrationError(
                f"output exceeds the configured {self.max_bytes}-byte cap"
            )
        allowed = min(size, remaining)
        written = self._write_output_chunk(view[:allowed]) if allowed else 0
        if allowed < size:
            self.truncated = True
        return written

    def _write_output_chunk(self, view: memoryview) -> int:
        written = self._stream.write(view.tobytes())
        if written is None:
            written = len(view)
        if written != len(view):
            raise LaneArbitrationError(
                "bounded output writer received a short underlying write"
            )
        self.bytes_written += written
        return written

    def writelines(self, lines: Iterator[bytes]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> BoundedOutputWriter:
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _validate_output_file(
    output_path: Path | str | None,
    *,
    max_bytes: int = RESOURCE_OUTPUT_MAX_BYTES,
    truncate: bool = RESOURCE_OUTPUT_TRUNCATE,
) -> Path:
    """Create/validate a bounded regular output file; stdout is never a sink."""
    fd, target = _open_output_fd(output_path, max_bytes=max_bytes, truncate=truncate)
    os.close(fd)
    return target


@contextmanager
def open_resource_output(
    output_path: Path | str,
    *,
    max_bytes: int = RESOURCE_OUTPUT_MAX_BYTES,
    truncate: bool = RESOURCE_OUTPUT_TRUNCATE,
) -> Iterator[Any]:
    """Open an output sink whose cap is enforced on each write."""
    fd, _ = _open_output_fd(output_path, max_bytes=max_bytes, truncate=truncate)
    stream: Any | None = None
    try:
        initial_size = os.fstat(fd).st_size
        stream = os.fdopen(fd, "ab", closefd=True, buffering=0)
        writer = BoundedOutputWriter(
            stream,
            max_bytes=max_bytes,
            truncate=truncate,
            initial_size=initial_size,
        )
    except (LaneArbitrationError, OSError, ValueError):
        if stream is not None:
            stream.close()
        else:
            os.close(fd)
        raise
    try:
        yield writer
    finally:
        writer.close()


def _validate_host_health(rule: ResourceRule, host_health: str | None) -> None:
    if rule.requires_healthy_host and not _is_healthy(host_health):
        raise LaneArbitrationError(
            f"{rule.name} denied: host health must be exactly 'healthy'; "
            "GPU admission is disabled while unhealthy"
        )


def _validate_host_capacity(
    rule: ResourceRule,
    snapshot: HostCapabilitySnapshot | None,
) -> None:
    if snapshot is None:
        return
    capacity = _host_resource_capacity(
        rule.name,
        snapshot,
        rule.capacity,
    )
    if capacity < 1:
        raise LaneArbitrationError(
            f"{rule.name} is not admitted on host {snapshot.identity!r}; "
            "this host is light-only "
            "until native capacity evidence is available"
        )


def _validate_rule_output(
    rule: ResourceRule, output_path: Path | str | None
) -> Path | None:
    if not rule.requires_output_file:
        return None
    _validate_output_policy(rule.output_max_bytes, rule.output_truncate)
    target = _output_target(output_path)
    if os.path.lexists(str(target)):
        _validate_existing_output(target)
    return target


def _prepare_rule_output(rule: ResourceRule, output_file: Path | None) -> None:
    """Create/truncate output only after the resource lease is published."""
    if output_file is not None:
        _validate_output_file(
            output_file,
            max_bytes=rule.output_max_bytes,
            truncate=rule.output_truncate,
        )


def _validate_host_admission(
    rule: ResourceRule,
    *,
    host_id: str | None,
    host_health: str | None,
    reboot_verified: bool,
    nvml_verified: bool,
    output_path: Path | str | None,
) -> tuple[HostCapabilitySnapshot | None, Path | None]:
    """Validate host identity/health and return normalized values for a record."""
    snapshot = (
        _host_snapshot(host_id, host_health, reboot_verified, nvml_verified)
        if rule.host_scoped
        else None
    )
    _validate_host_health(rule, host_health)
    _validate_host_capacity(rule, snapshot)
    return snapshot, _validate_rule_output(rule, output_path)


def _resource_status(
    name: str,
    rule: ResourceRule | None,
    path: Path | str | None,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    if rule is None:
        return _exclusive_lease_status(name, path)
    if name in _HOST_HEAVY_RESOURCES:
        holders = resource_lease_status(name, path)
        return holders[0] if holders else None
    if rule.capacity > 1:
        return resource_lease_status(name, path)
    return _exclusive_lease_status(name, path)


def _exclusive_lease_status(
    name: str, path: Path | str | None
) -> dict[str, Any] | None:
    scope = lane_scope(path)
    lease_file = _lease_dir(scope, name) / f"{_LANE_SAFE_RE.sub('-', name)}.lease"
    with _lease_mutex(scope, name):
        return _read_current_holder(lease_file)


def lease_status(
    name: str, path: Path | str | None = None
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """The live holder(s) of *name*, or ``None`` when an exclusive lease is free."""
    return _resource_status(name, _resource_rule(name), path)


def _uses_resource_gate(
    rule: ResourceRule | None,
    *,
    host_id: str | None,
    host_health: str | None,
    reboot_verified: bool,
    nvml_verified: bool,
    output_path: Path | str | None,
) -> bool:
    if rule is None:
        return False
    return any(
        (
            rule.host_scoped,
            rule.capacity > 1,
            host_id is not None,
            host_health is not None,
            output_path is not None,
            reboot_verified,
            nvml_verified,
        )
    )


def _legacy_lease_record(
    name: str,
    scope: LaneScope,
    operation: str,
    ttl_seconds: int,
    owner: dict[str, Any] | None,
) -> dict[str, Any]:
    local_host = _legacy_local_hostname()
    if local_host is None:
        raise LaneArbitrationError(
            "legacy lease admission requires a usable local hostname"
        )
    now = datetime.now(UTC)
    record: dict[str, Any] = {
        "name": name,
        "lane": scope.lane,
        "operation": operation,
        "pid": os.getpid(),
        "host": local_host,
        "acquired_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
    }
    if owner:
        record["owner"] = dict(owner)
    return record


def _acquire_legacy_lease(
    lease_file: Path, name: str, record: dict[str, Any], scope: LaneScope
) -> LeaseIdentity:
    with _lease_mutex(scope, name):
        holder = _read_current_holder(lease_file)
        if holder is not None:
            raise LeaseUnavailable(name, holder)
        return _write_exclusive_lease_file(
            lease_file, json.dumps(record, indent=2) + "\n"
        )


def _release_legacy_lease(
    lease_file: Path, name: str, identity: LeaseIdentity, scope: LaneScope
) -> None:
    with _lease_mutex(scope, name):
        _unlink_if_identity(lease_file, identity)


@contextmanager
def hold_lease(
    name: str,
    *,
    operation: str,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    path: Path | str | None = None,
    owner: dict[str, Any] | None = None,
    host_id: str | None = None,
    host_health: str | None = None,
    reboot_verified: bool = False,
    nvml_verified: bool = False,
    output_path: Path | str | None = None,
) -> Iterator[dict[str, Any]]:
    """Hold the exclusive lease *name*, or raise :class:`LeaseUnavailable`.

    Raising — rather than blocking — is the point: the caller is forced to make
    the deferral explicit. A dead holder's lease is reclaimed automatically, so a
    crashed lane cannot wedge the workspace.

    ``owner`` (D-CDX-14) is an optional, CALLER-DECLARED identity —
    e.g. ``{"fleet": "codex", "session": "..."}`` distinguishing the
    Claude/Codex/human/vLLM actor classes that arbitrate through this
    ledger — persisted into the lease record verbatim and returned by
    :func:`lease_status` so a holder can be attributed, not just located by
    ``lane``/``pid``/``host``. It is declared, not authenticated: this
    mechanism has no identity-verification step, so a caller can name any
    ``owner`` it likes. Nothing here (or in any caller) should treat an
    unset/mismatched ``owner`` as a security boundary — only the ``lane``
    lock itself (this lease + :func:`file_lock.lock_exclusive`) is enforced;
    ``owner`` is attribution for observability, not authorization.
    """
    reboot_verified, nvml_verified = _validate_verification_receipts(
        reboot_verified, nvml_verified
    )
    ttl_seconds = _validate_ttl(ttl_seconds)
    rule = _resource_rule(name)
    if _uses_resource_gate(
        rule,
        host_id=host_id,
        host_health=host_health,
        reboot_verified=reboot_verified,
        nvml_verified=nvml_verified,
        output_path=output_path,
    ):
        with hold_resource_lease(
            name,
            operation=operation,
            ttl_seconds=ttl_seconds,
            path=path,
            owner=owner,
            host_id=host_id,
            host_health=host_health,
            reboot_verified=reboot_verified,
            nvml_verified=nvml_verified,
            output_path=output_path,
        ) as record:
            yield record
        return

    scope = lane_scope(path)
    lease_file = _lease_dir(scope, name) / f"{_LANE_SAFE_RE.sub('-', name)}.lease"
    record = _legacy_lease_record(name, scope, operation, ttl_seconds, owner)
    lease_identity = _acquire_legacy_lease(lease_file, name, record, scope)
    try:
        yield record
    finally:
        _release_legacy_lease(lease_file, name, lease_identity, scope)


def _resource_lease_file(lease_dir: Path, name: str, slot: int, capacity: int) -> Path:
    safe_name = _LANE_SAFE_RE.sub("-", name)
    suffix = ".lease" if capacity == 1 else f".{slot}.lease"
    return lease_dir / f"{safe_name}{suffix}"


def _host_heavy_lease_file(lease_dir: Path) -> Path:
    """One extra coordination lease that serializes all interim heavy work."""
    return _resource_lease_file(lease_dir, _HOST_HEAVY_LEASE_NAME, 0, 1)


def _host_heavy_specific_files(lease_dir: Path) -> Iterator[Path]:
    """Yield the one-slot files whose presence also proves heavy occupancy."""
    for resource in sorted(_HOST_HEAVY_RESOURCES):
        yield _resource_lease_file(lease_dir, resource, 0, 1)


def _coordination_file(name: str, lease_dir: Path) -> Path | None:
    return _host_heavy_lease_file(lease_dir) if name in _HOST_HEAVY_RESOURCES else None


def _status_lease_files(name: str, rule: ResourceRule, lease_dir: Path) -> list[Path]:
    if name in _HOST_HEAVY_RESOURCES:
        return list(_host_heavy_specific_files(lease_dir))
    return [
        _resource_lease_file(lease_dir, name, slot, rule.capacity)
        for slot in range(rule.capacity)
    ]


def _status_holder(
    holder: dict[str, Any], name: str, coordination_holder: dict[str, Any] | None
) -> dict[str, Any]:
    if name not in _HOST_HEAVY_RESOURCES:
        return holder
    result = dict(holder)
    result["lease_state"] = "slot" if coordination_holder is not None else "orphan"
    return result


def _collect_status_holders(
    files: list[Path], name: str, coordination_holder: dict[str, Any] | None
) -> list[dict[str, Any]]:
    holders: list[dict[str, Any]] = []
    for lease_file in files:
        holder = _read_current_holder(lease_file)
        if holder is not None:
            holders.append(_status_holder(holder, name, coordination_holder))
    return holders


def _coordinator_status(holder: dict[str, Any]) -> dict[str, Any]:
    result = dict(holder)
    result["lease_state"] = "coordinator"
    return result


def resource_lease_status(
    name: str, path: Path | str | None = None
) -> list[dict[str, Any]]:
    """Return live holders for a capacity-limited host resource."""
    rule = _require_lease_rule(name)
    scope = lane_scope(path)
    lease_dir = _lease_dir(scope, name)
    with _lease_mutex(scope, name):
        coordination = _coordination_file(name, lease_dir)
        coordination_holder = (
            _read_current_holder(coordination) if coordination is not None else None
        )
        holders = _collect_status_holders(
            _status_lease_files(name, rule, lease_dir), name, coordination_holder
        )
        if coordination_holder is not None and not holders:
            # A crash between the two writes must still report the host-wide
            # coordination holder instead of falsely reporting the slot free.
            holders.append(_coordinator_status(coordination_holder))
    return holders


def _effective_resource_capacity(
    rule: ResourceRule,
    name: str,
    snapshot: HostCapabilitySnapshot | None,
) -> int:
    if snapshot is None:
        return rule.capacity
    return _host_resource_capacity(
        name,
        snapshot,
        rule.capacity,
    )


def _heavy_occupant(lease_dir: Path) -> dict[str, Any] | None:
    for candidate_file in _host_heavy_specific_files(lease_dir):
        holder = _read_current_holder(candidate_file)
        if holder is not None:
            return holder
    return None


def _host_occupant(
    name: str, lease_dir: Path, coordination_file: Path | None
) -> dict[str, Any] | None:
    if coordination_file is None:
        return None
    holder = _read_current_holder(coordination_file)
    return holder if holder is not None else _heavy_occupant(lease_dir)


def _free_resource_slot(
    name: str, rule: ResourceRule, lease_dir: Path, capacity: int
) -> tuple[Path | None, dict[str, Any] | None]:
    first_holder: dict[str, Any] | None = None
    for slot in range(capacity):
        candidate = _resource_lease_file(lease_dir, name, slot, rule.capacity)
        holder = _read_current_holder(candidate)
        if holder is not None:
            first_holder = first_holder or holder
            continue
        return candidate, first_holder
    return None, first_holder


def _resource_record_base(
    name: str,
    scope: LaneScope,
    *,
    slot: int,
    capacity: int,
    operation: str,
    ttl_seconds: int,
    host: str,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "name": name,
        "resource": name,
        "slot": slot,
        "capacity": capacity,
        "lane": scope.lane,
        "operation": operation,
        "pid": os.getpid(),
        "host": host,
        "host_identity": host,
        "acquired_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
    }


def _resource_record_optional(
    record: dict[str, Any],
    *,
    host_health: str | None,
    reboot_verified: bool,
    nvml_verified: bool,
    output_file: Path | None,
    rule: ResourceRule,
    owner: dict[str, Any] | None,
) -> None:
    if host_health is not None:
        record["host_health"] = host_health
    if reboot_verified:
        record["reboot_verified"] = True
    if nvml_verified:
        record["nvml_verified"] = True
    if output_file is not None:
        output_name = str(output_file)
        record["output_file"] = output_name
        record["output_path"] = output_name
        record["output_max_bytes"] = rule.output_max_bytes
        record["output_truncate"] = rule.output_truncate
    if owner:
        record["owner"] = dict(owner)


def _resource_record(
    name: str,
    scope: LaneScope,
    *,
    slot: int,
    capacity: int,
    operation: str,
    ttl_seconds: int,
    host: str | None,
    host_health: str | None,
    reboot_verified: bool,
    nvml_verified: bool,
    output_file: Path | None,
    rule: ResourceRule,
    owner: dict[str, Any] | None,
) -> dict[str, Any]:
    local_host = host if host is not None else resolve_host_identity()
    record = _resource_record_base(
        name,
        scope,
        slot=slot,
        capacity=capacity,
        operation=operation,
        ttl_seconds=ttl_seconds,
        host=local_host,
    )
    _resource_record_optional(
        record,
        host_health=host_health,
        reboot_verified=reboot_verified,
        nvml_verified=nvml_verified,
        output_file=output_file,
        rule=rule,
        owner=owner,
    )
    return record


LeaseIdentity = tuple[int, int]


def _lease_publication_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise LaneArbitrationError(
            "lease publication cannot be made safely: no-follow is unavailable"
        )
    return (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | no_follow
    )


def _unlink_if_identity(path: Path, identity: LeaseIdentity) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError as exc:
        _LOGGER.warning("lease file disappeared during identity-safe cleanup: %s", exc)
        return
    except OSError as exc:
        _LOGGER.error("could not inspect lease file %s during cleanup: %s", path, exc)
        return
    if (current.st_dev, current.st_ino) != identity:
        return
    try:
        path.unlink()
    except OSError as exc:
        _LOGGER.error("could not remove owned lease file %s: %s", path, exc)


def _close_failed_lease_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError as exc:
        _LOGGER.error("could not close failed lease publication descriptor: %s", exc)


def _write_lease_bytes(fd: int, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        count = os.write(fd, payload[written:])
        if count <= 0:
            raise OSError("lease publication made no write progress")
        written += count


def _write_exclusive_lease_file(path: Path, serialized: str) -> LeaseIdentity:
    """Publish a complete lease through an exclusive, no-follow descriptor."""
    fd: int | None = None
    identity: LeaseIdentity | None = None
    try:
        fd = os.open(str(path), _lease_publication_flags(), 0o600)
        info = os.fstat(fd)
        identity = (info.st_dev, info.st_ino)
        _write_lease_bytes(fd, serialized.encode("utf-8"))
        os.fsync(fd)
    except LaneArbitrationError:
        if fd is not None:
            _close_failed_lease_fd(fd)
        raise
    except OSError as exc:
        if fd is not None:
            _close_failed_lease_fd(fd)
        if identity is not None:
            _unlink_if_identity(path, identity)
        raise LaneArbitrationError(
            f"cannot publish lease file {path} without following links: {exc}"
        ) from exc
    assert fd is not None and identity is not None
    try:
        os.close(fd)
    except OSError as exc:
        _unlink_if_identity(path, identity)
        raise LaneArbitrationError(
            f"cannot close published lease file {path}: {exc}"
        ) from exc
    return identity


def _rollback_resource_files(published: dict[Path, LeaseIdentity]) -> None:
    for path, identity in reversed(tuple(published.items())):
        _unlink_if_identity(path, identity)


def _publish_resource_files(
    serialized: str, candidate: Path, coordination_file: Path | None
) -> dict[Path, LeaseIdentity]:
    published: dict[Path, LeaseIdentity] = {}
    try:
        for path in (coordination_file, candidate):
            if path is not None:
                published[path] = _write_exclusive_lease_file(path, serialized)
    except LaneArbitrationError as exc:
        _rollback_resource_files(published)
        raise LaneArbitrationError(
            f"resource lease transaction could not be committed: {exc}"
        ) from exc
    return published


def _release_resource_files(
    candidate: Path,
    coordination_file: Path | None,
    published: dict[Path, LeaseIdentity],
) -> None:
    for lease_file in (candidate, coordination_file):
        if lease_file is not None and lease_file in published:
            _unlink_if_identity(lease_file, published[lease_file])


def _lease_conflict(
    name: str, capacity: int, holder: dict[str, Any] | None, occupied: int
) -> LeaseUnavailable:
    status = dict(holder or {"name": name, "resource": name, "lane": "?"})
    status["capacity"] = capacity
    status["occupied"] = occupied
    return LeaseUnavailable(name, status)


def _admit_resource_lease(
    name: str,
    rule: ResourceRule,
    scope: LaneScope,
    *,
    capacity: int,
    operation: str,
    ttl_seconds: int,
    host_snapshot: HostCapabilitySnapshot | None,
    host_health: str | None,
    reboot_verified: bool,
    nvml_verified: bool,
    output_file: Path | None,
    owner: dict[str, Any] | None,
) -> tuple[dict[str, Any], Path, Path | None, dict[Path, LeaseIdentity]]:
    lease_dir = _lease_dir(scope, name)
    coordination_file = _coordination_file(name, lease_dir)
    with _lease_mutex(scope, name):
        occupied_holder = _host_occupant(name, lease_dir, coordination_file)
        if occupied_holder is not None:
            raise _lease_conflict(name, capacity, occupied_holder, 1)
        lease_candidate, first_holder = _free_resource_slot(
            name, rule, lease_dir, capacity
        )
        if lease_candidate is None:
            raise _lease_conflict(name, capacity, first_holder, capacity)
        slot = int(lease_candidate.stem.rsplit(".", 1)[-1]) if capacity > 1 else 0
        record = _resource_record(
            name,
            scope,
            slot=slot,
            capacity=capacity,
            operation=operation,
            ttl_seconds=ttl_seconds,
            host=host_snapshot.identity if host_snapshot is not None else None,
            host_health=host_health,
            reboot_verified=reboot_verified,
            nvml_verified=nvml_verified,
            output_file=output_file,
            rule=rule,
            owner=owner,
        )
        published = _publish_resource_files(
            json.dumps(record, indent=2) + "\n", lease_candidate, coordination_file
        )
        try:
            _prepare_rule_output(rule, output_file)
        except LaneArbitrationError:
            _rollback_resource_files(published)
            raise
    return record, lease_candidate, coordination_file, published


@contextmanager
def _run_resource_lease(
    name: str,
    scope: LaneScope,
    record: dict[str, Any],
    lease_file: Path,
    coordination_file: Path | None,
    published: dict[Path, LeaseIdentity],
) -> Iterator[dict[str, Any]]:
    acquired_logged = False
    try:
        try:
            _append_resource_log("acquired", record)
            acquired_logged = True
        except OSError as exc:
            with _lease_mutex(scope, name):
                _release_resource_files(lease_file, coordination_file, published)
            raise LaneArbitrationError(
                f"resource {name!r} acquired but its file-backed log could not be "
                f"written: {exc}"
            ) from exc
        yield record
    finally:
        with _lease_mutex(scope, name):
            _release_resource_files(lease_file, coordination_file, published)
        if acquired_logged:
            try:
                _append_resource_log("released", record)
            except OSError as exc:
                _LOGGER.warning(
                    "resource %s released but release event was not logged: %s",
                    name,
                    exc,
                )


@contextmanager
def hold_resource_lease(
    name: str,
    *,
    operation: str,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    path: Path | str | None = None,
    owner: dict[str, Any] | None = None,
    host_id: str | None = None,
    host_health: str | None = None,
    reboot_verified: bool = False,
    nvml_verified: bool = False,
    output_path: Path | str | None = None,
) -> Iterator[dict[str, Any]]:
    """Admit one bounded host resource slot, or defer without blocking.

    This is intentionally a small file-backed admission gate, not a scheduler:
    it selects the first free lease slot and leaves queueing, fairness, and
    native resource reservation to their existing authorities.
    """
    reboot_verified, nvml_verified = _validate_verification_receipts(
        reboot_verified, nvml_verified
    )
    ttl_seconds = _validate_ttl(ttl_seconds)
    rule = _require_lease_rule(name)
    host_snapshot, output_file = _validate_host_admission(
        rule,
        host_id=host_id,
        host_health=host_health,
        reboot_verified=reboot_verified,
        nvml_verified=nvml_verified,
        output_path=output_path,
    )
    capacity = _effective_resource_capacity(
        rule,
        name,
        host_snapshot,
    )
    if capacity < 1:
        raise LaneArbitrationError(
            f"{name} is not admitted on host "
            f"{host_snapshot.identity if host_snapshot else None!r}; "
            "this host is light-only "
            "until native capacity evidence is available"
        )

    scope = lane_scope(path)
    record, lease_file, coordination_file, published = _admit_resource_lease(
        name,
        rule,
        scope,
        capacity=capacity,
        operation=operation,
        ttl_seconds=ttl_seconds,
        host_snapshot=host_snapshot,
        host_health=host_health,
        reboot_verified=reboot_verified,
        nvml_verified=nvml_verified,
        output_file=output_file,
        owner=owner,
    )
    with _run_resource_lease(
        name, scope, record, lease_file, coordination_file, published
    ):
        yield record


@contextmanager
def guarded_tree_mutation(
    path: Path | str, *, operation: str, owner: str
) -> Iterator[LaneScope]:
    """The one choke point every global actor should route a tree mutation through.

    Combines the two halves that each fail alone: an exclusive lease held across
    the whole check-then-mutate (so the tree cannot go dirty between the check
    and the command), and the dirty/ownership refusal itself. It is deliberately
    **verb-agnostic** — ``checkout``, ``restore``, ``clean``, ``reset``, a branch
    switch and a stash all destroy uncommitted work, so guarding one verb just
    moves the hazard.

    **Residual gap, stated plainly:** a lease only binds actors that take it. An
    unwrapped external process still races, and no amount of leasing changes
    that. The gap closes only by making the guarded wrapper the *only* way the
    long operation is run — see the ``lane lease -- <command>`` form and
    ``reports/PROGRAM.md``. Do not record this as solved.
    """
    with hold_lease("canonical-mutation", operation=operation, path=path):
        require_resettable_tree(path, operation=operation, owner=owner)
        yield lane_scope(path)


# ---------------------------------------------------------------------------
# APPEND-ONLY — one fragment per writer, one generated view for readers
# ---------------------------------------------------------------------------
def _flow_line(record: dict[str, Any]) -> str:
    """One record as a single-line YAML flow mapping (the merge-safe unit)."""
    import yaml

    return (
        "- "
        + yaml.safe_dump(
            record, default_flow_style=True, sort_keys=False, width=10_000
        ).strip()
    )


@dataclass(frozen=True)
class FragmentStore:
    """An append-only record set: one fragment file per writing lane.

    A writer only ever touches ``<root>/<lane>.yaml``. Two lanes therefore write
    two different files, which git merges without a conflict and which no
    whole-file rewrite can clobber. Readers never read fragments — they read the
    single view :meth:`fold` produces.
    """

    root: Path
    key: str = "id"

    def fragment_for(self, lane: str) -> Path:
        return self.root / f"{_LANE_SAFE_RE.sub('-', lane)}.yaml"

    def read_fragment(self, lane: str) -> list[dict[str, Any]]:
        return self._load(self.fragment_for(lane))

    def append(self, record: dict[str, Any], *, lane: str) -> Path:
        """Append one immutable record to this lane's own fragment."""
        target = self.fragment_for(lane)
        target.parent.mkdir(parents=True, exist_ok=True)
        line = _flow_line(record) + "\n"
        fd = os.open(str(target), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        return target

    def rewrite_fragment(self, lane: str, records: list[dict[str, Any]]) -> Path:
        """Replace *this lane's* fragment. Never touches another lane's file."""
        target = self.fragment_for(lane)
        target.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(_flow_line(r) + "\n" for r in records)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, target)
        return target

    def lanes(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("*.yaml"))

    def fold(
        self, resolve: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Union every fragment into one deduplicated, deterministically ordered view.

        Records sharing a ``key`` are collapsed by *resolve* (default: the last
        one written wins), so a re-stated record supersedes rather than conflicts.
        """
        grouped: dict[str, list[dict[str, Any]]] = {}
        for lane in self.lanes():
            for record in self._load(self.fragment_for(lane)):
                grouped.setdefault(str(record.get(self.key, "")), []).append(record)
        pick = resolve or (lambda group: group[-1])
        return [pick(grouped[k]) for k in sorted(grouped)]

    @staticmethod
    def _load(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is None:
            return []
        if not isinstance(data, list) or any(not isinstance(r, dict) for r in data):
            raise LaneArbitrationError(f"fragment {path} is not a list of records")
        return data


def render_view(records: list[dict[str, Any]], *, header: list[str]) -> str:
    """Render a folded record set as the generated one-line-per-record view."""
    lines = list(header) + [_flow_line(r) for r in records]
    return "\n".join(lines) + "\n"


def write_view(path: Path, body: str) -> None:
    """Atomically replace a generated view file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# The classification itself — data, not code
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResourceRule:
    """One shared resource, the class it belongs to, and how that class is applied.

    ``scope`` says who contends for it: ``"repo"`` (every worktree of one
    repository) or ``"workspace"`` (every lane on the host — the shared ``.venv``
    and ``uv.lock`` are contended by ~26 worktrees spanning several repos, so a
    per-repo lease would not exclude the actor that actually collides with you).
    Host-scoped lease rules additionally declare a bounded ``capacity`` and may
    require a regular output file or healthy-host evidence before admission.
    """

    name: str
    arbitration: ArbitrationClass
    mechanism: str
    evidence: str
    scope: str = "repo"
    capacity: int = 1
    host_scoped: bool = False
    requires_output_file: bool = False
    requires_healthy_host: bool = False
    output_max_bytes: int = RESOURCE_OUTPUT_MAX_BYTES
    output_truncate: bool = RESOURCE_OUTPUT_TRUNCATE


_RULES_CACHE: list[ResourceRule] | None = None
RESOURCES_FILE = Path(__file__).with_name("lane_resources.yaml")


def _resource_positive_int(
    raw: dict[str, Any], key: str, default: int, label: str
) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool):
        raise LaneArbitrationError(
            f"resource {raw.get('name', '?')!r} has invalid {label}"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LaneArbitrationError(
            f"resource {raw.get('name', '?')!r} has invalid {label}"
        ) from exc
    if parsed < 1:
        raise LaneArbitrationError(
            f"resource {raw.get('name', '?')!r} {label} must be positive"
        )
    return parsed


def _resource_bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise LaneArbitrationError(
            f"resource {raw.get('name', '?')!r} {key} must be bool"
        )
    return value


def _parse_resource_rule(raw: object) -> ResourceRule:
    if not isinstance(raw, dict):
        raise LaneArbitrationError("lane_resources.yaml resource rows must be mappings")
    return ResourceRule(
        name=str(raw["name"]),
        arbitration=ArbitrationClass(str(raw["class"])),
        mechanism=str(raw["mechanism"]),
        evidence=str(raw["evidence"]),
        scope=str(raw.get("scope", "repo")),
        capacity=_resource_positive_int(raw, "capacity", 1, "capacity"),
        host_scoped=bool(raw.get("host_scoped", False)),
        requires_output_file=bool(raw.get("requires_output_file", False)),
        requires_healthy_host=bool(raw.get("requires_healthy_host", False)),
        output_max_bytes=_resource_positive_int(
            raw, "output_max_bytes", RESOURCE_OUTPUT_MAX_BYTES, "output cap"
        ),
        output_truncate=_resource_bool(
            raw, "output_truncate", RESOURCE_OUTPUT_TRUNCATE
        ),
    )


def _load_resource_rules() -> list[ResourceRule]:
    import yaml

    data = yaml.safe_load(RESOURCES_FILE.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not isinstance(data.get("resources"), list):
        raise LaneArbitrationError("lane_resources.yaml needs a resources list")
    return [_parse_resource_rule(raw) for raw in data["resources"]]


def resource_rules() -> list[ResourceRule]:
    """Every classified shared resource, loaded from ``lane_resources.yaml``."""
    global _RULES_CACHE
    if _RULES_CACHE is None:
        _RULES_CACHE = _load_resource_rules()
    return list(_RULES_CACHE)


def _resource_rule(name: str) -> ResourceRule | None:
    """Return a classification without weakening the legacy lease API."""
    return next((rule for rule in resource_rules() if rule.name == name), None)


def _require_lease_rule(name: str) -> ResourceRule:
    rule = _resource_rule(name)
    if rule is None:
        # Route through resource_class so the hard-error wording stays stable.
        resource_class(name)
        raise AssertionError("resource_class unexpectedly returned")
    if rule.arbitration is not ArbitrationClass.LEASE:
        raise LaneArbitrationError(
            f"{name} is {rule.arbitration.value}, not a LEASE-class resource"
        )
    return rule


def resource_class(name: str) -> ArbitrationClass:
    """The arbitration class registered for *name*."""
    for rule in resource_rules():
        if rule.name == name:
            return rule.arbitration
    raise LaneArbitrationError(
        f"unclassified shared resource {name!r} — classify it in "
        f"{RESOURCES_FILE.name} before contending for it"
    )


def lane_report(path: Path | str | None = None) -> dict[str, Any]:
    """Everything a lane (or a human) needs to know about its own isolation."""
    scope = lane_scope(path)
    parts = partitioned_paths(scope.tree)
    return {
        "lane": scope.lane,
        "tree": str(scope.tree),
        "canonical_checkout": str(scope.main_tree),
        "is_canonical": scope.is_canonical,
        "arbitration_dir": str(scope.arbitration_dir),
        "partitioned": {
            "cargo_target_dir": str(parts.cargo_target_dir),
            "pytest_basetemp": str(parts.pytest_basetemp),
            "scratch_dir": str(parts.scratch_dir),
            "precommit_home": str(parts.precommit_home),
            "stash_ref": parts.stash_ref,
        },
        "leases": {
            rule.name: lease_status(rule.name, scope.tree)
            for rule in resource_rules()
            if rule.arbitration is ArbitrationClass.LEASE
        },
        # Only the states worth an operator's attention — a clean lane sees an
        # empty list, a crashed one sees the exact path loudly (D-OB-12).
        "orphaned_precommit_patches": [
            p
            for p in orphaned_precommit_patches(scope.tree)
            if p["state"] in ("ORPHANED", "unknown")
        ],
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
