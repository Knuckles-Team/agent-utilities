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

import fcntl
import json
import os
import re
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

ARBITRATION_DIRNAME = "agent-lanes"
DEFAULT_LEASE_TTL_SECONDS = 1_800
_LANE_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
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


class LaneArbitrationError(RuntimeError):
    """Base for every refusal this module raises."""


class CanonicalCheckoutError(LaneArbitrationError):
    """A lane tried to mutate the canonical (main) checkout."""


class UnownedTreeError(LaneArbitrationError):
    """A global actor tried to reset a tree holding work it does not own."""


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
    stash_ref: str


def partitioned_paths(path: Path | str | None = None) -> PartitionedPaths:
    """Resolve this lane's private instance of every PARTITION-class resource.

    ``stash_ref`` matters most: ``refs/stash`` is a **single ref shared by every
    worktree**, so ``git stash`` in one lane is visible — and poppable — in all of
    them. A per-lane ref makes concurrent stashing safe, and is the only reason
    the blanket never-stash rule can ever be relaxed.
    """
    scope = lane_scope(path)
    lane_state = scope.arbitration_dir / "lanes" / scope.lane
    return PartitionedPaths(
        cargo_target_dir=scope.tree / "target-isolated",
        pytest_basetemp=lane_state / "pytest",
        scratch_dir=lane_state / "scratch",
        stash_ref=f"refs/lane/{scope.lane}/stash",
    )


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
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _holder_is_live(holder: dict[str, Any]) -> bool:
    """Whether a recorded holder still exists: unexpired, and its process alive."""
    expires = holder.get("expires_at")
    if expires:
        try:
            if datetime.fromisoformat(str(expires)) < datetime.now(UTC):
                return False
        except ValueError:
            return False
    if holder.get("host") != socket.gethostname():
        # Another host: expiry is the only evidence available, and it says live.
        return True
    pid = holder.get("pid")
    if not isinstance(pid, int):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lease_status(name: str, path: Path | str | None = None) -> dict[str, Any] | None:
    """The live holder of *name*, or ``None`` when the lease is free."""
    scope = lane_scope(path)
    lease_file = _lease_dir(scope, name) / f"{_LANE_SAFE_RE.sub('-', name)}.lease"
    with _lease_mutex(scope, name):
        if not lease_file.exists():
            return None
        holder = json.loads(lease_file.read_text(encoding="utf-8"))
        if _holder_is_live(holder):
            return holder
        lease_file.unlink()
        return None


@contextmanager
def hold_lease(
    name: str,
    *,
    operation: str,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    path: Path | str | None = None,
) -> Iterator[dict[str, Any]]:
    """Hold the exclusive lease *name*, or raise :class:`LeaseUnavailable`.

    Raising — rather than blocking — is the point: the caller is forced to make
    the deferral explicit. A dead holder's lease is reclaimed automatically, so a
    crashed lane cannot wedge the workspace.
    """
    scope = lane_scope(path)
    lease_file = _lease_dir(scope, name) / f"{_LANE_SAFE_RE.sub('-', name)}.lease"
    now = datetime.now(UTC)
    record = {
        "name": name,
        "lane": scope.lane,
        "operation": operation,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "acquired_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
    }
    with _lease_mutex(scope, name):
        if lease_file.exists():
            holder = json.loads(lease_file.read_text(encoding="utf-8"))
            if _holder_is_live(holder):
                raise LeaseUnavailable(name, holder)
        lease_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    try:
        yield record
    finally:
        with _lease_mutex(scope, name):
            if lease_file.exists():
                current = json.loads(lease_file.read_text(encoding="utf-8"))
                if (
                    current.get("pid") == record["pid"]
                    and current.get("acquired_at") == record["acquired_at"]
                ):
                    lease_file.unlink()


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
    """

    name: str
    arbitration: ArbitrationClass
    mechanism: str
    evidence: str
    scope: str = "repo"


_RULES_CACHE: list[ResourceRule] | None = None
RESOURCES_FILE = Path(__file__).with_name("lane_resources.yaml")


def resource_rules() -> list[ResourceRule]:
    """Every classified shared resource, loaded from ``lane_resources.yaml``."""
    global _RULES_CACHE
    if _RULES_CACHE is None:
        import yaml

        data = yaml.safe_load(RESOURCES_FILE.read_text(encoding="utf-8")) or {}
        rules = data.get("resources")
        if not isinstance(rules, list):
            raise LaneArbitrationError("lane_resources.yaml needs a resources list")
        _RULES_CACHE = [
            ResourceRule(
                name=str(r["name"]),
                arbitration=ArbitrationClass(str(r["class"])),
                mechanism=str(r["mechanism"]),
                evidence=str(r["evidence"]),
                scope=str(r.get("scope", "repo")),
            )
            for r in rules
        ]
    return list(_RULES_CACHE)


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
            "stash_ref": parts.stash_ref,
        },
        "leases": {
            rule.name: lease_status(rule.name, scope.tree)
            for rule in resource_rules()
            if rule.arbitration is ArbitrationClass.LEASE
        },
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
