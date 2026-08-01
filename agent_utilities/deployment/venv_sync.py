"""Sanctioned reconciler for the shared uv-workspace virtualenv.

CONCEPT:AU-OS.deployment.workspace-venv-reconciler
CONCEPT:AU-OS.safety.destructive-sync-refusal
CONCEPT:AU-OS.host.venv-drift-detector

Why this module exists
----------------------
The ecosystem is developed against ONE shared ``.venv`` at the uv-workspace
root, with ~75 workspace members installed editable into it and ~26 git
worktrees running their tests through it.  Two properties of that arrangement
are actively dangerous and both were paid for in production time:

1. **A bare ``uv sync`` destroys the environment.**  The workspace root project
   declares zero dependencies, so its target set is empty and ``uv sync``
   prunes everything that is not in it — a measured *557* uninstalls including
   every editable member.  The only correct invocation is
   ``uv sync --locked --all-packages --inexact``.  Neither ``--all-packages``
   nor ``--inexact`` has a ``UV_*`` environment equivalent, so the safe form
   cannot be made the shell default; it has to be made the *only* form any
   sanctioned path can construct.  :class:`SyncInvocation` therefore has no
   field that can drop a mandatory flag, and :func:`_assert_sanctioned` re-checks
   the argv immediately before ``subprocess`` sees it, so a future refactor that
   assembles an argv somewhere else still cannot get a destructive one past this
   module.
2. **Drift is silent.**  The venv sat ten days behind its own lock
   (``fastmcp`` 3.4.4 / ``mcp`` 1.28.1 against a lock wanting 4.0.0b1 / 2.0.0).
   An entire test module stopped collecting because one import failed, hiding
   thirteen real defects.  Nothing checked, so nobody knew.  :func:`detect_drift`
   is the check, and it is wired into ``agent-utilities doctor`` so it runs
   without anyone remembering to run it.

Design constraints this module honours
--------------------------------------
* **Stdlib only.**  This is the tool you reach for when the venv is broken, so
  it must not import anything the venv provides — including the rest of
  ``agent_utilities``.  Optional third-party helpers (``packaging``) are probed
  lazily and always have a stdlib fallback.
* **The workspace root is not a git repository.**  Discovery walks up looking
  for a ``pyproject.toml`` carrying ``[tool.uv.workspace]``; it never shells
  ``git rev-parse``, which is what made ``scripts/uv_workspace.py`` unusable
  from the root.
* **Another actor mutates the venv concurrently.**  Every mutation runs under an
  advisory exclusive lock and re-plans inside the lock, and every guardrail can
  answer ``defer`` rather than ``refuse`` so a busy environment is left alone
  instead of being fought over.
* **The lock has no version control.**  ``uv.lock`` is untracked and lives at a
  non-git root, so :class:`LockBackupStore` is the only rollback path there is;
  every mutation is wrapped in a checkpoint before it starts.
* **Abstraction first.**  Guardrails, activity probes and verify probes are
  registries of small protocols.  The default set encodes today's known hazards;
  adding a hazard means registering a probe, not editing a policy branch.

No new environment variables are introduced (configuration discipline): every
knob is a constructor/CLI argument, and the on/off state lives in one JSON
document under the XDG state directory.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


def redact_path_for_log(path: object) -> str:
    """Short, deterministic, non-reversible tag for a path in a log line.

    CONCEPT:AU-OS.observability.log-location-privacy — a raw filesystem path in
    a log line leaks host layout to anyone with log read access. Stdlib-only
    (this module's own "must not import anything the venv provides" constraint
    rules out the shared ``agent_utilities.security.log_redaction`` helper);
    the same input always hashes to the same tag within a process, so repeated
    log lines about the same path can still be correlated without the literal
    value ever appearing.
    """
    if not path:
        return "<empty>"
    digest = hashlib.sha256(str(path).encode("utf-8", errors="replace")).hexdigest()
    return f"<redacted:{digest[:12]}>"


__all__ = [
    "ActivityProbe",
    "ActivityRecord",
    "Backup",
    "DriftFinding",
    "DriftReport",
    "Guardrail",
    "LockBackupStore",
    "LockBusyError",
    "Member",
    "MemberInstallState",
    "PackageDelta",
    "PlanParseError",
    "ProbeResult",
    "PruneCandidate",
    "PruneOutcome",
    "PrunePlan",
    "SANCTIONED_SYNC_FLAGS",
    "SyncContext",
    "SyncInvocation",
    "SyncOutcome",
    "SyncPlan",
    "UnsafeInvocationError",
    "UpgradeOutcome",
    "Verdict",
    "VerifyProbe",
    "VenvSyncError",
    "Workspace",
    "WorkspaceNotFoundError",
    "classify_change",
    "detect_activity",
    "emit",
    "detect_drift",
    "evaluate_plan",
    "exclusive_lock",
    "main",
    "member_install_states",
    "plan_prune",
    "plan_sync",
    "prune",
    "redact_path_for_log",
    "register_activity_probe",
    "register_guardrail",
    "register_verify_probe",
    "rollback",
    "session_start_hint",
    "sync",
    "upgrade",
    "verify",
]


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────
class VenvSyncError(RuntimeError):
    """Base class for every failure this module raises."""


class WorkspaceNotFoundError(VenvSyncError):
    """No ancestor directory declares ``[tool.uv.workspace]``."""


class UnsafeInvocationError(VenvSyncError):
    """An argv that would destroy the environment was assembled.

    Raised by :func:`_assert_sanctioned`.  This is the last line of defence:
    it fires *before* ``subprocess`` runs, so the destructive command is never
    executed even if it was constructed by code outside this module.
    """


class PlanParseError(VenvSyncError):
    """``uv sync --dry-run`` output could not be parsed unambiguously.

    Deliberately fatal.  A plan we cannot read is a plan we cannot police, and
    the failure mode we are guarding against is precisely "silently uninstall
    more than expected", so the safe answer is to refuse rather than to guess.
    """


class LockBusyError(VenvSyncError):
    """Another reconciler holds the exclusive writer lock."""


class BackupNotFoundError(VenvSyncError):
    """The requested lock backup id does not exist."""


# ─────────────────────────────────────────────────────────────────────────────
# The sanctioned invocation — the destructive form is unrepresentable
# ─────────────────────────────────────────────────────────────────────────────
#: Flags that MUST be present on every ``uv sync`` this module runs.
#:
#: ``--locked``       never relock as a side effect of syncing; a manifest that
#:                    moved is an explicit ``upgrade``, not a silent swap.
#: ``--all-packages`` the workspace root declares no dependencies, so without
#:                    this the target set is empty and everything is pruned.
#: ``--inexact``      never uninstall packages outside the target set; other
#:                    lanes legitimately add packages to the shared venv.
SANCTIONED_SYNC_FLAGS: tuple[str, ...] = ("--locked", "--all-packages", "--inexact")


@dataclass(frozen=True)
class SyncInvocation:
    """The one sanctioned ``uv sync`` command line.

    There is intentionally no field that can remove a flag from
    :data:`SANCTIONED_SYNC_FLAGS`.  ``dry_run`` only ever *adds* ``--dry-run``.
    """

    dry_run: bool = False

    def argv(self, uv: str) -> list[str]:
        argv = [uv, "sync", *SANCTIONED_SYNC_FLAGS]
        if self.dry_run:
            argv.append("--dry-run")
        return argv


def _assert_sanctioned(argv: Sequence[str]) -> None:
    """Refuse to execute a ``uv sync`` missing any mandatory safety flag."""

    if "sync" not in argv:
        return
    missing = [flag for flag in SANCTIONED_SYNC_FLAGS if flag not in argv]
    if missing:
        raise UnsafeInvocationError(
            "refusing to run an unsanctioned `uv sync`: missing "
            f"{', '.join(missing)}. The shared workspace root declares no "
            "dependencies, so this invocation would prune every editable "
            "workspace member. The only sanctioned form is "
            f"`uv sync {' '.join(SANCTIONED_SYNC_FLAGS)}`."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Workspace discovery
# ─────────────────────────────────────────────────────────────────────────────
def canonical_name(name: str) -> str:
    """PEP 503 normalised distribution name."""

    return re.sub(r"[-_.]+", "-", name.strip()).lower()


@dataclass(frozen=True)
class Member:
    """One uv workspace member, resolved from the root manifest."""

    path: Path
    name: str

    @property
    def canonical(self) -> str:
        return canonical_name(self.name)


@dataclass(frozen=True)
class Workspace:
    """A uv workspace root plus the derived paths this module operates on."""

    root: Path
    uv: str = "uv"
    state_dir: Path = field(default_factory=lambda: Path())

    @property
    def pyproject(self) -> Path:
        return self.root / "pyproject.toml"

    @property
    def lock(self) -> Path:
        return self.root / "uv.lock"

    @property
    def venv(self) -> Path:
        return self.root / ".venv"

    @classmethod
    def discover(
        cls,
        start: Path | str | None = None,
        *,
        uv: str | None = None,
        state_dir: Path | None = None,
    ) -> Workspace:
        """Walk up from ``start`` to the nearest ``[tool.uv.workspace]`` root.

        Never shells out to git: ``/home/apps/workspace`` is not a repository,
        which is exactly why the pre-existing ``scripts/uv_workspace.py``
        launcher cannot run there.
        """

        origin = Path(start or Path.cwd()).resolve()
        candidates = (origin, *origin.parents) if origin.is_dir() else origin.parents
        for candidate in candidates:
            manifest = candidate / "pyproject.toml"
            if not manifest.is_file():
                continue
            try:
                document = tomllib.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError) as exc:  # noqa: BLE001 — deliberate DEBUG: workspace discovery walks EVERY ancestor directory, and an unreadable or non-uv pyproject.toml along that path is expected, not exceptional; it simply is not the workspace root. Warning here would fire on ordinary trees. The cause is preserved (interpolated) and the walk continues.
                logger.debug("skipping unreadable manifest %s: %s", manifest, exc)
                continue
            workspace = document.get("tool", {}).get("uv", {}).get("workspace")
            if isinstance(workspace, dict):
                resolved_uv = uv or shutil.which("uv") or "uv"
                root = candidate.resolve()
                return cls(
                    root=root,
                    uv=resolved_uv,
                    state_dir=state_dir or default_state_dir(root),
                )
        raise WorkspaceNotFoundError(
            f"no ancestor of {origin} declares [tool.uv.workspace]; "
            "pass the workspace root explicitly with --workspace"
        )

    def members(self) -> tuple[Member, ...]:
        """Resolve ``members`` minus ``exclude`` from the root manifest."""

        try:
            document = tomllib.loads(self.pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise VenvSyncError(
                f"workspace manifest {self.pyproject} is unreadable: {exc}"
            ) from exc
        config = document.get("tool", {}).get("uv", {}).get("workspace", {})
        included = _expand_member_patterns(self.root, config.get("members", []))
        excluded = _expand_member_patterns(self.root, config.get("exclude", []))
        resolved: list[Member] = []
        for path in sorted(included - excluded):
            name = _project_name(path)
            if name is not None:
                resolved.append(Member(path=path, name=name))
        return tuple(resolved)

    def site_packages(self) -> Path | None:
        """Locate the venv's ``site-packages`` without importing from it."""

        for pattern in ("lib/python*/site-packages", "Lib/site-packages"):
            for match in sorted(self.venv.glob(pattern)):
                if match.is_dir():
                    return match
        return None


def _expand_member_patterns(root: Path, patterns: Any) -> set[Path]:
    if not isinstance(patterns, list) or any(
        not isinstance(item, str) for item in patterns
    ):
        raise VenvSyncError(
            "tool.uv.workspace members/exclude must be lists of path patterns"
        )
    resolved: set[Path] = set()
    for pattern in patterns:
        for match in sorted(root.glob(pattern)):
            if match.is_dir() and (match / "pyproject.toml").is_file():
                resolved.add(match.resolve())
    return resolved


def _project_name(path: Path) -> str | None:
    manifest = path / "pyproject.toml"
    try:
        document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning(
            "workspace member %s has an unreadable manifest: %s",
            redact_path_for_log(path),
            exc,
        )
        return None
    name = document.get("project", {}).get("name")
    return name if isinstance(name, str) and name else None


def default_state_dir(root: Path) -> Path:
    """Per-workspace state under ``~/.local/state``, keyed by the root path.

    State lives outside the workspace on purpose: the lock backups and the
    pending-flip queue must survive ``rm -rf .venv`` and must never be something
    a lane can accidentally commit.

    The location is deliberately **not** configurable and reads no environment
    variable.  A git hook, a detached reconciler and an interactive shell run
    with three different environments; if the path moved with ``XDG_STATE_HOME``
    they would each see a different queue and a different backup set, and a
    flip enqueued by the hook would be invisible to the drain.  One fixed
    location is the property that makes the queue coherent.
    """

    base = Path.home() / ".local" / "state"
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    return base / "agent-utilities" / "venv-autosync" / f"{root.name}-{digest}"


# ─────────────────────────────────────────────────────────────────────────────
# Plan parsing
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PackageDelta:
    """One package uv would install, uninstall or replace."""

    name: str
    version: str = ""
    source: str | None = None
    url: str | None = None

    @property
    def canonical(self) -> str:
        return canonical_name(self.name)

    @property
    def is_local(self) -> bool:
        for candidate in (self.source, self.url):
            if candidate and candidate.startswith("file://"):
                return True
        return False


_HEADER_RE = re.compile(
    r"^Would (?P<verb>install|uninstall|update|downgrade) (?P<count>\d+) packages?$"
)
#: uv writes registry packages as ``name==version`` and local/direct-URL ones as
#: ``name @ file:///path`` — an editable member's *install* line uses the second
#: form while its *uninstall* line uses the first, so both must parse or a
#: rebuild looks like a pure removal.
_ENTRY_RE = re.compile(
    r"^\s*(?P<sign>[-+~])\s+(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:==(?P<version>[^\s(]+)|\s+@\s+(?P<url>\S+))"
    r"(?:\s+\((?:from\s+)?(?P<source>[^)]*)\))?\s*$"
)
_NO_CHANGES_RE = re.compile(r"^Would make no changes$")


@dataclass(frozen=True)
class SyncPlan:
    """What ``uv sync --dry-run`` says it would do to the environment."""

    installs: tuple[PackageDelta, ...] = ()
    uninstalls: tuple[PackageDelta, ...] = ()
    no_changes: bool = False
    raw: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.installs and not self.uninstalls

    @property
    def removals(self) -> tuple[PackageDelta, ...]:
        """Uninstalls that are NOT paired with a reinstall of the same package.

        uv renders a *replacement* as an uninstall line plus an install line —
        which is what every editable member looks like whenever its metadata is
        rebuilt.  Treating those as removals would make the guardrails refuse
        the correct, sanctioned sync (observed end-to-end: a dependency added to
        a member produced ``- demo==1.0.0`` / ``+ demo @ file://…``).  Only the
        *net* removals are the dangerous ones the guardrails care about.
        """

        reinstalled = {delta.canonical for delta in self.installs}
        return tuple(d for d in self.uninstalls if d.canonical not in reinstalled)

    @property
    def replacements(self) -> tuple[PackageDelta, ...]:
        """Packages uv would uninstall and immediately reinstall."""

        reinstalled = {delta.canonical for delta in self.installs}
        return tuple(d for d in self.uninstalls if d.canonical in reinstalled)

    @classmethod
    def parse(cls, text: str) -> SyncPlan:
        """Parse uv's dry-run report, failing closed on the dangerous direction.

        uv prints a ``Would <verb> N packages`` header followed by ``N``
        ``+``/``-``/``~`` entry lines.  The invariant that must hold for this
        module to be trustworthy is that we never *under-count uninstalls*: the
        whole guardrail rests on being able to say "this plan removes 557
        packages" with certainty.  So the uninstall header is checked against
        the entries parsed and a shortfall raises.  Install counts are not
        validated, because uv formats replacements differently across versions
        and a false alarm there would block legitimate syncs for no safety gain.
        """

        installs: list[PackageDelta] = []
        uninstalls: list[PackageDelta] = []
        declared_uninstalls = 0
        no_changes = False

        for line in text.splitlines():
            header = _HEADER_RE.match(line.strip())
            if header:
                if header.group("verb") == "uninstall":
                    declared_uninstalls += int(header.group("count"))
                continue
            if _NO_CHANGES_RE.match(line.strip()):
                no_changes = True
                continue
            entry = _ENTRY_RE.match(line)
            if not entry:
                continue
            delta = PackageDelta(
                name=entry.group("name"),
                version=entry.group("version") or "",
                source=entry.group("source"),
                url=entry.group("url"),
            )
            if entry.group("sign") == "-":
                uninstalls.append(delta)
            else:
                installs.append(delta)

        if len(uninstalls) < declared_uninstalls:
            raise PlanParseError(
                f"uv declared {declared_uninstalls} uninstall(s) but only "
                f"{len(uninstalls)} could be parsed; refusing to act on a plan "
                "whose removals cannot be enumerated exactly"
            )
        return cls(
            installs=tuple(installs),
            uninstalls=tuple(uninstalls),
            no_changes=no_changes,
            raw=text,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Activity detection — "is another lane mid-test?"
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ActivityRecord:
    """Evidence that something else is using the shared environment."""

    probe: str
    identifier: str
    detail: str


@runtime_checkable
class ActivityProbe(Protocol):
    """Answers "is somebody else using this venv right now?"."""

    name: str

    def busy(self, workspace: Workspace) -> Sequence[ActivityRecord]:
        """Return evidence of in-flight work (empty when idle)."""


#: Program names that mean "a build or test is running here".
#:
#: Matched against argv *tokens*, never against the joined command line: a
#: ``bash -c '…pytest…'`` wrapper carries the whole script in a single token, so
#: substring matching flagged every shell that merely mentioned a test command.
#: The real ``pytest`` child is still detected on its own, so nothing is lost.
_TEST_MARKERS = frozenset(
    {
        "pytest",
        "py.test",
        "unittest",
        "tox",
        "nox",
        "pre-commit",
        "maturin",
        "cargo",
        "rustc",
        "meson",
        "ninja",
    }
)
#: ``uv`` subcommands that mutate or depend on a stable environment.
_UV_SUBCOMMANDS = frozenset({"run", "sync", "lock", "pip", "build", "venv"})


def _build_or_test_command(cmdline: Sequence[str]) -> str | None:
    """Name the build/test tool this argv runs, or ``None``."""

    tokens = [Path(token).name for token in cmdline]
    if not tokens:
        return None
    if tokens[0] in ("uv", "uvx") and len(tokens) > 1 and tokens[1] in _UV_SUBCOMMANDS:
        return f"uv {tokens[1]}"
    for token in tokens:
        if token in _TEST_MARKERS:
            return token
    return None


def _looks_like_interpreter_or_test_command(cmdline: Sequence[str]) -> bool:
    """Whether ``cmdline`` is plausibly python/test work, not merely an
    inherited environment (D-VS-9).

    ``VIRTUAL_ENV`` is inherited by every descendant of an activated shell, so
    an unrelated child of that shell (``tail -40``, ``ls``, an editor) reads as
    "busy" purely because its parent activated the venv — a live run showed
    exactly this with ``tail -40``. Require the executable itself to look like
    a Python interpreter/tool or a recognised build/test command before
    treating an inherited ``VIRTUAL_ENV`` as evidence of in-flight work; the
    ``cmdline[0].startswith(venv_bin)`` and lease probes are unaffected and
    still catch the real cases directly.
    """
    if not cmdline:
        return False
    exe = Path(cmdline[0]).name
    if exe.startswith("python") or exe in {"pip", "pip3"}:
        return True
    return _build_or_test_command(cmdline) is not None


class ProcessActivityProbe:
    """Detect live processes bound to the shared venv by reading ``/proc``.

    Three independent signals, because no single one is sufficient:

    * ``cmdline[0]`` under ``<venv>/bin`` — catches ``.venv/bin/pytest`` and
      ``.venv/bin/python`` (``/proc/<pid>/exe`` is useless here: uv's venv
      python is a symlink to the interpreter in uv's own python store, so the
      resolved exe never mentions the venv);
    * ``VIRTUAL_ENV`` in the process environment — catches an activated shell
      and anything it spawned;
    * a build/test command whose working directory sits inside the workspace or
      any sibling worktree — catches ``uv run pytest`` and ``cargo`` builds that
      neither activate nor name the venv.
    """

    name = "process"

    def __init__(self, extra_roots: Sequence[Path] = ()) -> None:
        self._extra_roots = tuple(Path(p) for p in extra_roots)

    def busy(self, workspace: Workspace) -> Sequence[ActivityRecord]:
        proc = Path("/proc")
        if not proc.is_dir():
            return ()
        venv_bin = str(workspace.venv / "bin")
        roots = (workspace.root, *self._extra_roots)
        me = os.getpid()
        mine = _own_process_lineage()
        found: list[ActivityRecord] = []
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == me or pid in mine:
                continue
            cmdline = _read_proc_list(entry / "cmdline")
            if not cmdline:
                continue
            reason: str | None = None
            if cmdline[0].startswith(venv_bin):
                reason = "executing from the shared venv"
            elif _proc_environ_virtualenv(entry) == str(
                workspace.venv
            ) and _looks_like_interpreter_or_test_command(cmdline):
                reason = (
                    "VIRTUAL_ENV points at the shared venv (interpreter/test command)"
                )
            else:
                tool = _build_or_test_command(cmdline)
                if tool is not None:
                    cwd = _read_proc_link(entry / "cwd")
                    if cwd is not None and any(_is_within(cwd, root) for root in roots):
                        reason = f"{tool} running in {cwd}"
            if reason is None:
                continue
            found.append(
                ActivityRecord(
                    probe=self.name,
                    identifier=f"pid {pid}",
                    detail=f"{reason}: {' '.join(cmdline)[:160]}",
                )
            )
        return tuple(found)


class LeaseActivityProbe:
    """Honour explicit leases taken by cooperating lanes.

    The process probe cannot see work that is *about* to start, nor a lane that
    has paused between test commands but still needs a stable environment.  A
    lane can therefore declare a window explicitly::

        agent-utilities-venv lease acquire --owner lane-x --ttl 3600

    Leases expire, so a crashed lane cannot wedge the reconciler forever.
    """

    name = "lease"

    def busy(self, workspace: Workspace) -> Sequence[ActivityRecord]:
        directory = workspace.state_dir / "leases"
        if not directory.is_dir():
            return ()
        now = time.time()
        found: list[ActivityRecord] = []
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning(
                    "ignoring unreadable lease %s: %s", redact_path_for_log(path), exc
                )
                continue
            expires = float(payload.get("expires_at", 0.0))
            if expires <= now:
                _unlink_quietly(path)
                continue
            found.append(
                ActivityRecord(
                    probe=self.name,
                    identifier=str(payload.get("owner", path.stem)),
                    detail=(
                        f"lease held until "
                        f"{datetime.fromtimestamp(expires, UTC).isoformat()}"
                        f": {payload.get('reason', 'no reason given')}"
                    ),
                )
            )
        return tuple(found)


ACTIVITY_PROBES: list[ActivityProbe] = [ProcessActivityProbe(), LeaseActivityProbe()]


def register_activity_probe(
    probe: ActivityProbe, *, replace_existing: bool = True
) -> None:
    """Add (or replace) an activity probe."""

    if replace_existing:
        ACTIVITY_PROBES[:] = [p for p in ACTIVITY_PROBES if p.name != probe.name]
    ACTIVITY_PROBES.append(probe)


def detect_activity(workspace: Workspace) -> tuple[ActivityRecord, ...]:
    """Run every registered activity probe and collect the evidence."""

    records: list[ActivityRecord] = []
    for probe in ACTIVITY_PROBES:
        try:
            records.extend(probe.busy(workspace))
        except OSError as exc:
            # A probe that cannot read the system must not make the reconciler
            # claim the environment is idle; surface it as activity so the
            # decision fails safe (defer) instead of silently proceeding.
            records.append(
                ActivityRecord(
                    probe=probe.name,
                    identifier="probe-error",
                    detail=f"probe could not determine idleness: {exc}",
                )
            )
    return tuple(records)


def _own_process_lineage() -> frozenset[int]:
    """PIDs of this process' ancestors, so we never defer to ourselves."""

    lineage: set[int] = set()
    pid = os.getpid()
    for _ in range(32):
        stat = Path("/proc") / str(pid) / "stat"
        try:
            fields = stat.read_text(encoding="utf-8", errors="replace").rsplit(") ", 1)
        except OSError:
            break
        if len(fields) != 2:
            break
        parts = fields[1].split()
        if len(parts) < 2 or not parts[1].isdigit():
            break
        pid = int(parts[1])
        if pid <= 1 or pid in lineage:
            break
        lineage.add(pid)
    return frozenset(lineage)


def _read_proc_list(path: Path) -> list[str]:
    try:
        raw = path.read_bytes()
    except OSError:
        return []
    return [part for part in raw.decode("utf-8", "replace").split("\0") if part]


def _proc_environ_virtualenv(entry: Path) -> str | None:
    for item in _read_proc_list(entry / "environ"):
        if item.startswith("VIRTUAL_ENV="):
            return item.split("=", 1)[1]
    return None


def _read_proc_link(path: Path) -> Path | None:
    try:
        return Path(os.readlink(path))
    except OSError:
        return None


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Guardrails
# ─────────────────────────────────────────────────────────────────────────────
ALLOW = "allow"
REFUSE = "refuse"
DEFER = "defer"


@dataclass(frozen=True)
class Verdict:
    """A guardrail decision."""

    decision: str
    guardrail: str = "none"
    reason: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == ALLOW

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "guardrail": self.guardrail,
            "reason": self.reason,
            "data": self.data,
        }


@dataclass
class SyncContext:
    """Everything a guardrail needs to judge a proposed change."""

    workspace: Workspace
    reason: str = "manual"
    #: How many uninstalls the caller has explicitly sanctioned.  Zero for every
    #: automatic path; only an operator running ``prune`` raises it.
    allow_uninstalls: int = 0
    #: Set when the caller has explicitly accepted running while others work.
    ignore_activity: bool = False
    activity: tuple[ActivityRecord, ...] = ()
    lock_check_ok: bool | None = None
    lock_check_detail: str = ""


@runtime_checkable
class Guardrail(Protocol):
    """Vetoes or defers a proposed environment change."""

    name: str
    #: ``True`` when the guardrail only needs the context (runs before planning).
    pre_plan: bool

    def evaluate(self, plan: SyncPlan | None, ctx: SyncContext) -> Verdict | None:
        """Return a non-``allow`` verdict to stop the change, or ``None``."""


class ActivityGuardrail:
    """Defer — never refuse — while another lane is using the environment.

    Deferring is the whole point: swapping packages underneath a running test
    run produces failures that look like code defects.  The intent is queued and
    drained later rather than dropped.
    """

    name = "activity"
    pre_plan = True

    def evaluate(self, plan: SyncPlan | None, ctx: SyncContext) -> Verdict | None:
        if ctx.ignore_activity or not ctx.activity:
            return None
        return Verdict(
            decision=DEFER,
            guardrail=self.name,
            reason=(
                f"{len(ctx.activity)} in-flight activity record(s) on the shared "
                "environment; deferring rather than swapping packages underneath "
                "running work"
            ),
            data={"activity": [record.detail for record in ctx.activity]},
        )


class LockConsistencyGuardrail:
    """Refuse to sync when ``uv lock --check`` fails.

    A stale lock means somebody changed a manifest.  Resolving that is an
    explicit ``upgrade``/relock decision with a backup and a verification pass,
    never a side effect of "keep the venv current".
    """

    name = "lock_consistency"
    pre_plan = True

    def evaluate(self, plan: SyncPlan | None, ctx: SyncContext) -> Verdict | None:
        if ctx.lock_check_ok is not False:
            return None
        return Verdict(
            decision=REFUSE,
            guardrail=self.name,
            reason=(
                "uv.lock is out of date with the workspace manifests; a sync "
                "would install a resolution nobody reviewed. Run "
                "`agent-utilities-venv relock` (backed up + verified) instead."
            ),
            data={"detail": ctx.lock_check_detail},
        )


class MemberUninstallGuardrail:
    """Refuse any plan that would uninstall a workspace member.

    This is the guardrail that makes the measured catastrophe impossible: a
    bare ``uv sync`` plans 557 uninstalls including every editable member.
    """

    name = "member_uninstall"
    pre_plan = False

    def evaluate(self, plan: SyncPlan | None, ctx: SyncContext) -> Verdict | None:
        if plan is None or not plan.removals:
            return None
        members = {member.canonical for member in ctx.workspace.members()}
        offending = [d for d in plan.removals if d.canonical in members]
        if not offending:
            return None
        names = ", ".join(sorted(d.name for d in offending)[:8])
        return Verdict(
            decision=REFUSE,
            guardrail=self.name,
            reason=(
                f"plan would uninstall {len(offending)} editable workspace "
                f"member(s) ({names}"
                f"{', …' if len(offending) > 8 else ''}). This is the signature "
                "of an unsanctioned `uv sync`; the environment was not touched."
            ),
            data={"members": sorted(d.name for d in offending)},
        )


class LockedDistributionUninstallGuardrail:
    """Refuse to uninstall anything the lock still wants.

    ``--inexact`` means uv should only ever be adding or replacing.  An
    uninstall of a locked distribution means the plan was computed against a
    different target set than we think, so we stop.
    """

    name = "locked_uninstall"
    pre_plan = False

    def evaluate(self, plan: SyncPlan | None, ctx: SyncContext) -> Verdict | None:
        if plan is None or not plan.removals:
            return None
        locked = _locked_distribution_names(ctx.workspace)
        if not locked:
            return None
        offending = sorted({d.name for d in plan.removals if d.canonical in locked})
        if not offending:
            return None
        return Verdict(
            decision=REFUSE,
            guardrail=self.name,
            reason=(
                f"plan would uninstall {len(offending)} distribution(s) that "
                "uv.lock still requires; the environment was not touched."
            ),
            data={"distributions": offending[:32]},
        )


class UninstallBudgetGuardrail:
    """Refuse when a plan exceeds the caller's sanctioned uninstall budget."""

    name = "uninstall_budget"
    pre_plan = False

    def evaluate(self, plan: SyncPlan | None, ctx: SyncContext) -> Verdict | None:
        if plan is None:
            return None
        count = len(plan.removals)
        if count <= ctx.allow_uninstalls:
            return None
        return Verdict(
            decision=REFUSE,
            guardrail=self.name,
            reason=(
                f"plan would uninstall {count} package(s) but only "
                f"{ctx.allow_uninstalls} are sanctioned for this operation; "
                "re-run with an explicit --allow-uninstalls budget if this is "
                "intended."
            ),
            data={"count": count, "budget": ctx.allow_uninstalls},
        )


GUARDRAILS: list[Guardrail] = [
    ActivityGuardrail(),
    LockConsistencyGuardrail(),
    MemberUninstallGuardrail(),
    LockedDistributionUninstallGuardrail(),
    UninstallBudgetGuardrail(),
]


def register_guardrail(guardrail: Guardrail, *, replace_existing: bool = True) -> None:
    """Add (or replace) a guardrail."""

    if replace_existing:
        GUARDRAILS[:] = [g for g in GUARDRAILS if g.name != guardrail.name]
    GUARDRAILS.append(guardrail)


def evaluate_plan(plan: SyncPlan | None, ctx: SyncContext) -> Verdict:
    """Run the guardrails and return the first non-allow verdict.

    Refusals win over deferrals: a plan that would destroy the environment is
    wrong regardless of whether now is a good time to run it.
    """

    verdicts: list[Verdict] = []
    for guardrail in GUARDRAILS:
        if plan is None and not guardrail.pre_plan:
            continue
        verdict = guardrail.evaluate(plan, ctx)
        if verdict is not None and verdict.decision != ALLOW:
            verdicts.append(verdict)
    for verdict in verdicts:
        if verdict.decision == REFUSE:
            return verdict
    if verdicts:
        return verdicts[0]
    return Verdict(decision=ALLOW)


def _locked_distribution_names(workspace: Workspace) -> frozenset[str]:
    """Distribution names present in ``uv.lock``.

    Parsed with ``tomllib`` when the lock is well formed; a lock uv itself
    wrote always is.  A parse failure is reported and treated as "unknown",
    which simply makes this one guardrail inert — the member guardrail and the
    uninstall budget still apply.
    """

    try:
        document = tomllib.loads(workspace.lock.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning(
            "could not read %s for guardrail evaluation: %s", workspace.lock, exc
        )
        return frozenset()
    packages = document.get("package", [])
    if not isinstance(packages, list):
        return frozenset()
    return frozenset(
        canonical_name(pkg["name"])
        for pkg in packages
        if isinstance(pkg, dict) and isinstance(pkg.get("name"), str)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Exclusive writer lock
# ─────────────────────────────────────────────────────────────────────────────
@contextmanager
def exclusive_lock(workspace: Workspace, *, blocking: bool = False) -> Iterator[Path]:
    """Serialise every mutation of the shared venv/lock across processes.

    Advisory ``flock`` on a state-directory file.  Non-blocking by default so a
    merge hook that arrives while a reconcile is already running exits
    immediately instead of piling up — its intent stays queued and the running
    reconciler (or the next one) drains it.
    """

    workspace.state_dir.mkdir(parents=True, exist_ok=True)
    path = workspace.state_dir / "writer.lock"
    flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            raise LockBusyError(
                f"another reconciler holds {path}; not competing for the shared venv"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()} {datetime.now(UTC).isoformat()}\n")
        handle.flush()
        yield path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


# ─────────────────────────────────────────────────────────────────────────────
# Lock backups — the only rollback path an untracked lock has
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Backup:
    """One archived ``uv.lock`` plus the metadata needed to trust it."""

    id: str
    path: Path
    created_at: str
    digest: str
    reason: str
    verified: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": str(self.path),
            "created_at": self.created_at,
            "digest": self.digest,
            "reason": self.reason,
            "verified": self.verified,
            "meta": self.meta,
        }


class LockBackupStore:
    """Content-addressed archive of ``uv.lock`` revisions.

    ``uv.lock`` is untracked and lives at a non-git root, so there is no other
    way back.  Today's only successful rollback happened because somebody
    manually copied the file to a scratch directory first; this makes that
    automatic and unconditional.
    """

    def __init__(self, workspace: Workspace, *, retain: int = 50) -> None:
        self.workspace = workspace
        self.retain = retain
        self.directory = workspace.state_dir / "lock-backups"

    def create(self, reason: str, *, meta: dict[str, Any] | None = None) -> Backup:
        source = self.workspace.lock
        if not source.is_file():
            raise VenvSyncError(f"no lock to back up at {source}")
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_id = f"{stamp}-{digest[:12]}"
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{backup_id}.lock"
        if not target.exists():
            _atomic_write_bytes(target, payload)
        backup = Backup(
            id=backup_id,
            path=target,
            created_at=datetime.now(UTC).isoformat(),
            digest=digest,
            reason=reason,
            meta=meta or {},
        )
        _atomic_write_text(
            self.directory / f"{backup_id}.json",
            json.dumps(backup.as_dict(), indent=2, sort_keys=True),
        )
        self._prune()
        logger.info("lock backup %s created (%s)", backup_id, reason)
        return backup

    def list(self) -> list[Backup]:
        if not self.directory.is_dir():
            return []
        backups: list[Backup] = []
        for meta_path in sorted(self.directory.glob("*.json")):
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning(
                    "ignoring unreadable backup record %s: %s", meta_path, exc
                )
                continue
            backups.append(
                Backup(
                    id=str(payload.get("id", meta_path.stem)),
                    path=Path(payload.get("path", meta_path.with_suffix(".lock"))),
                    created_at=str(payload.get("created_at", "")),
                    digest=str(payload.get("digest", "")),
                    reason=str(payload.get("reason", "")),
                    verified=bool(payload.get("verified", False)),
                    meta=payload.get("meta") or {},
                )
            )
        return sorted(backups, key=lambda b: b.id)

    def get(self, backup_id: str) -> Backup:
        for backup in self.list():
            if backup.id == backup_id:
                return backup
        raise BackupNotFoundError(f"no lock backup with id {backup_id!r}")

    def latest(self) -> Backup | None:
        backups = self.list()
        return backups[-1] if backups else None

    def restore(self, backup_id: str | None = None) -> Backup:
        backup = self.get(backup_id) if backup_id else self.latest()
        if backup is None:
            raise BackupNotFoundError("no lock backups exist to restore from")
        payload = backup.path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != backup.digest:
            raise VenvSyncError(
                f"backup {backup.id} failed its own digest check; refusing to restore"
            )
        _atomic_write_bytes(self.workspace.lock, payload)
        logger.info("restored uv.lock from backup %s", backup.id)
        return backup

    def mark_verified(self, backup_id: str) -> Backup:
        backup = self.get(backup_id)
        verified = replace(backup, verified=True)
        _atomic_write_text(
            self.directory / f"{backup_id}.json",
            json.dumps(verified.as_dict(), indent=2, sort_keys=True),
        )
        return verified

    @contextmanager
    def checkpoint(
        self, reason: str, *, meta: dict[str, Any] | None = None
    ) -> Iterator[Backup]:
        """Back the lock up, then restore it if the block raises."""

        backup = self.create(reason, meta=meta)
        try:
            yield backup
        except BaseException as exc:
            self.restore(backup.id)
            raise VenvSyncError(
                f"restored uv.lock from checkpoint {backup.id} after a failed "
                f"{reason}: {exc}"
            ) from exc

    def _prune(self) -> None:
        backups = self.list()
        # Verified backups are the known-good rollback targets; retention only
        # ever discards unverified ones.
        removable = [b for b in backups if not b.verified]
        excess = len(backups) - self.retain
        for backup in removable[: max(0, excess)]:
            _unlink_quietly(backup.path)
            _unlink_quietly(self.directory / f"{backup.id}.json")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, payload: str) -> None:
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not remove %s: %s", redact_path_for_log(path), exc)


# ─────────────────────────────────────────────────────────────────────────────
# uv invocation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()


def run_uv(
    workspace: Workspace, args: Sequence[str], *, timeout: float = 3600.0
) -> CommandResult:
    """Run uv in the workspace root, refusing unsanctioned sync invocations."""

    argv = [workspace.uv, *args]
    _assert_sanctioned(argv)
    logger.debug("running %s", " ".join(argv))
    try:
        completed = subprocess.run(  # noqa: S603 — argv is constructed, never shell
            argv,
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise VenvSyncError(
            f"uv executable not found at {workspace.uv!r}: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise VenvSyncError(f"uv timed out after {timeout}s: {' '.join(argv)}") from exc
    return CommandResult(
        argv=tuple(argv),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def lock_check(workspace: Workspace) -> CommandResult:
    """``uv lock --check`` — is the lock current with the manifests?"""

    return run_uv(workspace, ["lock", "--check"], timeout=900.0)


def plan_sync(workspace: Workspace) -> SyncPlan:
    """Compute what the sanctioned sync would do, without touching anything."""

    result = run_uv(workspace, SyncInvocation(dry_run=True).argv(workspace.uv)[1:])
    if not result.ok:
        raise VenvSyncError(
            f"uv sync --dry-run failed (rc={result.returncode}): {result.output[-2000:]}"
        )
    return SyncPlan.parse(result.output)


# ─────────────────────────────────────────────────────────────────────────────
# Sync
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SyncOutcome:
    """The result of asking for the environment to be made current."""

    verdict: Verdict
    plan: SyncPlan | None
    applied: bool
    detail: str = ""
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.verdict.decision in (ALLOW,) and (
            self.applied or (self.plan is not None and self.plan.is_empty)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.as_dict(),
            "applied": self.applied,
            "detail": self.detail,
            "duration_s": round(self.duration_s, 3),
            "plan": {
                "installs": [d.name for d in (self.plan.installs if self.plan else ())],
                "uninstalls": [
                    d.name for d in (self.plan.uninstalls if self.plan else ())
                ],
                "empty": self.plan.is_empty if self.plan else None,
            },
        }


def sync(
    workspace: Workspace,
    *,
    reason: str = "manual",
    apply: bool = True,
    ignore_activity: bool = False,
    allow_uninstalls: int = 0,
    hold_lock: bool = True,
) -> SyncOutcome:
    """Bring the shared venv to the locked state, or explain why we did not.

    The whole safety story lives here:

    1. take the exclusive writer lock (``defer`` if somebody else holds it);
    2. gather context — activity probes and ``uv lock --check``;
    3. run the pre-plan guardrails (``defer`` on busy, ``refuse`` on stale lock);
    4. compute the plan with ``--dry-run`` **using the sanctioned flags**;
    5. run the plan guardrails (``refuse`` on any member/locked uninstall);
    6. only then apply — with the identical, re-asserted argv.
    """

    started = time.monotonic()
    try:
        if hold_lock:
            with exclusive_lock(workspace):
                return _sync_locked(
                    workspace,
                    reason=reason,
                    apply=apply,
                    ignore_activity=ignore_activity,
                    allow_uninstalls=allow_uninstalls,
                    started=started,
                )
        return _sync_locked(
            workspace,
            reason=reason,
            apply=apply,
            ignore_activity=ignore_activity,
            allow_uninstalls=allow_uninstalls,
            started=started,
        )
    except LockBusyError as exc:
        return SyncOutcome(
            verdict=Verdict(
                decision=DEFER,
                guardrail="writer_lock",
                reason=str(exc),
            ),
            plan=None,
            applied=False,
            detail=str(exc),
            duration_s=time.monotonic() - started,
        )


def _sync_locked(
    workspace: Workspace,
    *,
    reason: str,
    apply: bool,
    ignore_activity: bool,
    allow_uninstalls: int,
    started: float,
) -> SyncOutcome:
    check = lock_check(workspace)
    ctx = SyncContext(
        workspace=workspace,
        reason=reason,
        allow_uninstalls=allow_uninstalls,
        ignore_activity=ignore_activity,
        activity=detect_activity(workspace),
        lock_check_ok=check.ok,
        lock_check_detail=check.output[-2000:],
    )

    verdict = evaluate_plan(None, ctx)
    if not verdict.allowed:
        return SyncOutcome(
            verdict=verdict,
            plan=None,
            applied=False,
            detail=verdict.reason,
            duration_s=time.monotonic() - started,
        )

    plan = plan_sync(workspace)
    verdict = evaluate_plan(plan, ctx)
    if not verdict.allowed:
        return SyncOutcome(
            verdict=verdict,
            plan=plan,
            applied=False,
            detail=verdict.reason,
            duration_s=time.monotonic() - started,
        )

    if plan.is_empty:
        return SyncOutcome(
            verdict=verdict,
            plan=plan,
            applied=False,
            detail="environment already matches the lock; nothing to do",
            duration_s=time.monotonic() - started,
        )

    if not apply:
        return SyncOutcome(
            verdict=verdict,
            plan=plan,
            applied=False,
            detail=(
                f"plan approved but not applied (--dry-run): "
                f"{len(plan.installs)} install(s), {len(plan.removals)} removal(s)"
            ),
            duration_s=time.monotonic() - started,
        )

    result = run_uv(workspace, SyncInvocation().argv(workspace.uv)[1:])
    if not result.ok:
        raise VenvSyncError(
            f"sanctioned uv sync failed (rc={result.returncode}): {result.output[-2000:]}"
        )
    return SyncOutcome(
        verdict=verdict,
        plan=plan,
        applied=True,
        detail=(
            f"applied {len(plan.installs)} install(s), "
            f"{len(plan.removals)} removal(s), {len(plan.replacements)} replacement(s)"
        ),
        duration_s=time.monotonic() - started,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prune (D-VS-8)
#
# ``--inexact`` — permanently forced into SANCTIONED_SYNC_FLAGS, unconditionally
# — means uv's own dry-run plan NEVER reports a genuinely extraneous package as
# a removal. Verified live 2026-07-31 against a real throwaway uv workspace: an
# installed distribution absent from the lock survived `sync(workspace,
# allow_uninstalls=1)` completely untouched — `plan_sync()` returned
# `installs=[], uninstalls=[]` even though the package was undeniably present
# and undeniably not in `uv.lock`. `UninstallBudgetGuardrail` was therefore
# unreachable through the only code path that fed it a plan: not merely
# "unexercised" as originally filed, but structurally dead code. Fixed here by
# giving the SAME guardrail machinery a plan source that can actually contain
# removals, computed without ever touching `uv sync` at all — so it can never
# produce the destructive bare-sync argv `_assert_sanctioned` exists to catch.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PruneCandidate:
    """An installed distribution that is neither locked nor a workspace member."""

    name: str
    version: str


@dataclass(frozen=True)
class PrunePlan:
    """What `prune()` would remove — computed by set difference, not `uv sync`."""

    candidates: tuple[PruneCandidate, ...]

    @property
    def is_empty(self) -> bool:
        return not self.candidates

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates": [f"{c.name}=={c.version}" for c in self.candidates],
        }


def plan_prune(workspace: Workspace) -> PrunePlan:
    """Find installed distributions `--inexact` deliberately leaves behind.

    Read-only: compares `.dist-info` on disk against `uv.lock`'s package set
    and the workspace's own member names. Never invokes `uv sync`, so it sees
    exactly what a hand-audit of the venv would see, independent of any sync
    flag.
    """

    site = workspace.site_packages()
    if site is None or not site.is_dir():
        return PrunePlan(candidates=())
    installed = _installed_distributions(site)
    locked = _locked_distribution_names(workspace)
    member_names = {m.canonical for m in workspace.members()}
    candidates = tuple(
        PruneCandidate(name=record.name, version=record.version)
        for canon, record in sorted(installed.items())
        if canon not in locked and canon not in member_names
    )
    return PrunePlan(candidates=candidates)


@dataclass(frozen=True)
class PruneOutcome:
    """The result of asking for extraneous packages to be removed."""

    plan: PrunePlan
    applied: bool
    refused: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "applied": self.applied,
            "refused": self.refused,
            "detail": self.detail,
        }


def prune(
    workspace: Workspace,
    *,
    allow_uninstalls: int,
    apply: bool = True,
    ignore_activity: bool = False,
) -> PruneOutcome:
    """Remove up to `allow_uninstalls` packages that are installed but not
    locked and not a workspace member — the class `sync()` can never reach.

    Deliberately budgeted and explicit: `allow_uninstalls` has no default
    that removes anything (`prune(workspace, allow_uninstalls=0)` always
    refuses), matching `--allow-uninstalls`'s existing "any uninstall
    refuses by default" contract elsewhere in this module. Removes one
    package per `uv pip uninstall` call — never `uv sync` — so this path can
    never produce the destructive bare-sync argv `_assert_sanctioned` exists
    to catch, and a locked or member distribution can never appear as a
    candidate in the first place (excluded by construction in `plan_prune`).
    """

    if allow_uninstalls <= 0:
        raise VenvSyncError(
            "prune() refuses without an explicit allow_uninstalls > 0 budget "
            "(0 is the same as sync()'s default: any uninstall refuses)"
        )
    if not ignore_activity:
        activity = detect_activity(workspace)
        if activity:
            names = "; ".join(f"{r.probe}:{r.identifier}" for r in activity[:5])
            return PruneOutcome(
                plan=PrunePlan(candidates=()),
                applied=False,
                detail=f"deferred: environment busy ({names})",
            )

    plan = plan_prune(workspace)
    if plan.is_empty:
        return PruneOutcome(plan=plan, applied=False, detail="nothing to prune")
    if len(plan.candidates) > allow_uninstalls:
        names = ", ".join(c.name for c in plan.candidates[:8])
        return PruneOutcome(
            plan=plan,
            applied=False,
            refused=True,
            detail=(
                f"plan would remove {len(plan.candidates)} package(s) "
                f"({names}) but only {allow_uninstalls} are sanctioned for "
                "this operation; re-run with a larger --allow-uninstalls "
                "budget if this is intended"
            ),
        )
    if not apply:
        return PruneOutcome(
            plan=plan,
            applied=False,
            detail=f"plan approved but not applied (--dry-run): {len(plan.candidates)} removal(s)",
        )

    python = workspace.venv / "bin" / "python"
    for candidate in plan.candidates:
        result = run_uv(
            workspace, ["pip", "uninstall", "--python", str(python), candidate.name]
        )
        if not result.ok:
            raise VenvSyncError(
                f"uv pip uninstall {candidate.name} failed "
                f"(rc={result.returncode}): {result.output[-2000:]}"
            )
    return PruneOutcome(
        plan=plan, applied=True, detail=f"removed {len(plan.candidates)} package(s)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Verify probes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ProbeResult:
    """One post-change health assertion."""

    name: str
    ok: bool | None
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@runtime_checkable
class VerifyProbe(Protocol):
    """Asserts the environment is healthy after a change."""

    name: str

    def check(self, workspace: Workspace) -> ProbeResult:
        """Return ``ok=True``/``False``, or ``ok=None`` when inapplicable."""


class LockCheckProbe:
    """The lock still matches the manifests."""

    name = "lock_check"

    def check(self, workspace: Workspace) -> ProbeResult:
        result = lock_check(workspace)
        return ProbeResult(
            name=self.name,
            ok=result.ok,
            detail="uv.lock is current" if result.ok else result.output[-600:],
        )


class CleanPlanProbe:
    """The environment matches the lock exactly."""

    name = "clean_plan"

    def check(self, workspace: Workspace) -> ProbeResult:
        try:
            plan = plan_sync(workspace)
        except VenvSyncError as exc:
            return ProbeResult(name=self.name, ok=False, detail=str(exc))
        if plan.is_empty:
            return ProbeResult(
                name=self.name, ok=True, detail="environment matches the lock"
            )
        return ProbeResult(
            name=self.name,
            ok=False,
            detail=(
                f"{len(plan.installs)} install(s) and {len(plan.uninstalls)} "
                "uninstall(s) still outstanding"
            ),
        )


#: Modules imported to prove the environment is usable, not merely installed.
#: ``agent_utilities.mcp.child_resilience`` is here for a specific reason: it is
#: the module whose silent ``ImportError`` under a stale ``fastmcp`` stopped an
#: entire test module from collecting and hid thirteen defects for ten days.
DEFAULT_IMPORT_PROBE_MODULES: tuple[str, ...] = (
    "agent_utilities",
    "agent_utilities.mcp.child_resilience",
)


class ImportProbe:
    """Import canary modules inside the shared venv's interpreter.

    A canary that is simply *not installed* in this environment is not a
    failure — it is inapplicable, and reported as ``ok=None``.  A canary that is
    installed but raises on import IS a failure, and that distinction is the
    entire value of the probe: the ten-day outage was ``agent_utilities.mcp
    .child_resilience`` present-but-unimportable against a stale ``fastmcp``,
    which reads as "installed and broken", not "absent".
    """

    name = "imports"

    def __init__(self, modules: Sequence[str] = DEFAULT_IMPORT_PROBE_MODULES) -> None:
        self.modules = tuple(modules)

    def check(self, workspace: Workspace) -> ProbeResult:
        python = workspace.venv / "bin" / "python"
        if not python.exists():
            python = workspace.venv / "Scripts" / "python.exe"
        if not python.exists():
            return ProbeResult(
                name=self.name, ok=None, detail=f"no interpreter under {workspace.venv}"
            )
        script = "\n".join(
            [
                "import importlib, importlib.util, json, sys",
                "bad, absent = {}, []",
                f"for name in {list(self.modules)!r}:",
                "    top = name.split('.')[0]",
                "    try:",
                "        found = importlib.util.find_spec(top) is not None",
                "    except BaseException:",
                "        found = True",
                "    if not found:",
                "        absent.append(name)",
                "        continue",
                "    try:",
                "        importlib.import_module(name)",
                "    except BaseException as exc:",
                "        bad[name] = f'{type(exc).__name__}: {exc}'",
                "json.dump({'bad': bad, 'absent': absent}, sys.stdout)",
            ]
        )
        try:
            completed = subprocess.run(  # noqa: S603 — argv is constructed, never shell
                [str(python), "-c", script],
                cwd=str(workspace.root),
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ProbeResult(
                name=self.name, ok=False, detail=f"import probe could not run: {exc}"
            )
        try:
            payload = json.loads(completed.stdout or "{}")
            failures = dict(payload.get("bad", {}))
            absent = list(payload.get("absent", []))
        except (ValueError, TypeError) as exc:
            return ProbeResult(
                name=self.name,
                ok=False,
                detail=(
                    f"import probe produced unparseable output ({exc}): "
                    f"{(completed.stdout or completed.stderr)[-400:]}"
                ),
            )
        if failures:
            return ProbeResult(
                name=self.name,
                ok=False,
                detail="; ".join(f"{k} -> {v}" for k, v in sorted(failures.items())),
            )
        checked = len(self.modules) - len(absent)
        if not checked:
            return ProbeResult(
                name=self.name,
                ok=None,
                detail=f"no canary module is installed here (absent: {absent})",
            )
        return ProbeResult(
            name=self.name,
            ok=True,
            detail=(
                f"{checked} canary module(s) import cleanly"
                + (f"; not installed here: {absent}" if absent else "")
            ),
        )


class SdkFloorProbe:
    """Reuse ``check_mcp_sdk_floor`` rather than re-deriving MCP SDK floors.

    A sibling lane already built the "derive the required floor from installed
    metadata, compare against what is installed" check.  Duplicating that logic
    here would give us two answers that can disagree, so this probe simply calls
    it inside the shared venv when it is available and reports ``ok=None`` when
    the running tree does not have it.
    """

    name = "mcp_sdk_floor"

    def check(self, workspace: Workspace) -> ProbeResult:
        python = workspace.venv / "bin" / "python"
        if not python.exists():
            return ProbeResult(name=self.name, ok=None, detail="no venv interpreter")
        script = (
            "import json, sys\n"
            "try:\n"
            "    from agent_utilities.mcp.protocol_compat import check_mcp_sdk_floor\n"
            "except Exception as exc:\n"
            "    json.dump({'ok': None, 'detail': f'unavailable: {type(exc).__name__}: {exc}'}, sys.stdout)\n"
            "else:\n"
            "    json.dump(check_mcp_sdk_floor(), sys.stdout)\n"
        )
        try:
            completed = subprocess.run(  # noqa: S603 — argv is constructed, never shell
                [str(python), "-c", script],
                cwd=str(workspace.root),
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            payload = json.loads(completed.stdout or "{}")
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            return ProbeResult(
                name=self.name, ok=None, detail=f"floor probe could not run: {exc}"
            )
        return ProbeResult(
            name=self.name,
            ok=payload.get("ok"),
            detail=str(payload.get("detail", "")),
        )


VERIFY_PROBES: list[VerifyProbe] = [
    LockCheckProbe(),
    CleanPlanProbe(),
    ImportProbe(),
    SdkFloorProbe(),
]


def register_verify_probe(probe: VerifyProbe, *, replace_existing: bool = True) -> None:
    """Add (or replace) a verification probe."""

    if replace_existing:
        VERIFY_PROBES[:] = [p for p in VERIFY_PROBES if p.name != probe.name]
    VERIFY_PROBES.append(probe)


def verify(workspace: Workspace) -> tuple[ProbeResult, ...]:
    """Run every verify probe.  ``ok is False`` anywhere means "roll back"."""

    return tuple(probe.check(workspace) for probe in VERIFY_PROBES)


# ─────────────────────────────────────────────────────────────────────────────
# Editable members: source change vs metadata change
# ─────────────────────────────────────────────────────────────────────────────
SOURCE_ONLY = "source-only"
METADATA = "metadata"
NATIVE = "native"
LOCK = "lock"

_METADATA_FILENAMES = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "MANIFEST.in",
        "requirements.txt",
        "requirements-dev.txt",
    }
)
_NATIVE_FILENAMES = frozenset({"Cargo.toml", "Cargo.lock", "build.rs"})
_NATIVE_SUFFIXES = frozenset({".rs", ".c", ".h", ".pyx", ".pxd"})


def classify_change(paths: Sequence[str]) -> str:
    """Classify a merge's changed paths into the action it actually requires.

    This is the distinction the whole "flip on merge" feature turns on:

    * **source-only** — a ``.py`` edit inside an editable member is *already*
      live the instant the merge writes the file, for the member and for every
      one of its downstream dependents, because an editable install resolves
      imports through the source tree.  Reinstalling would be pure cost.
    * **metadata** — ``pyproject.toml``/``setup.cfg`` changes are baked into the
      installed ``.dist-info`` at install time.  A dependency change also
      invalidates the resolution shared by every dependent, so it needs a
      relock followed by a sync; an entry-point or version change needs the
      member reinstalled.
    * **native** — Rust/C sources need the extension rebuilt; an editable
      install does not track those.
    * **lock** — ``uv.lock`` itself moved, so the environment must be re-synced.

    The classification is the cheap *trigger-time* signal.  It is deliberately
    biased toward doing work: anything not recognised as source-only escalates.
    :func:`member_install_states` is the authoritative check that runs anyway.
    """

    verdicts = {SOURCE_ONLY}
    for raw in paths:
        name = Path(raw).name
        suffix = Path(raw).suffix
        if name == "uv.lock":
            verdicts.add(LOCK)
        elif name in _METADATA_FILENAMES:
            verdicts.add(METADATA)
        elif name in _NATIVE_FILENAMES or suffix in _NATIVE_SUFFIXES:
            verdicts.add(NATIVE)
    for level in (METADATA, NATIVE, LOCK):
        if level in verdicts:
            return level
    return SOURCE_ONLY


@dataclass(frozen=True)
class MemberInstallState:
    """How one editable workspace member is actually installed right now."""

    member: Member
    installed: bool
    editable: bool
    source_version: str | None
    installed_version: str | None
    source_entry_points: tuple[str, ...]
    installed_entry_points: tuple[str, ...]
    differences: tuple[str, ...]

    @property
    def stale(self) -> bool:
        return bool(self.differences)


def member_install_states(workspace: Workspace) -> tuple[MemberInstallState, ...]:
    """Compare every member's *source* metadata against its installed record.

    Scope note, deliberately narrow to avoid crying wolf: dependency drift is
    already detected exactly and cheaply by ``uv lock --check`` (a member whose
    ``dependencies`` moved makes the lock stale by definition).  Re-deriving a
    requirement-by-requirement diff here would only add a second, fuzzier answer
    to the same question.  What ``uv lock --check`` genuinely cannot see is
    metadata that does **not** affect resolution — a version bump, an added or
    renamed console script, a member that got installed non-editable — and those
    are exactly the fields this compares.
    """

    site = workspace.site_packages()
    installed = _installed_distributions(site) if site else {}
    states: list[MemberInstallState] = []
    for member in workspace.members():
        source_version, source_eps, dynamic_version = _source_metadata(member.path)
        record = installed.get(member.canonical)
        differences: list[str] = []
        if record is None:
            states.append(
                MemberInstallState(
                    member=member,
                    installed=False,
                    editable=False,
                    source_version=source_version,
                    installed_version=None,
                    source_entry_points=source_eps,
                    installed_entry_points=(),
                    differences=("not installed into the shared venv",),
                )
            )
            continue
        if not record.editable:
            differences.append("installed non-editable: source edits will NOT be live")
        if (
            not dynamic_version
            and source_version is not None
            and record.version != source_version
        ):
            differences.append(
                f"version {record.version} installed but {source_version} declared"
            )
        if set(source_eps) != set(record.entry_points):
            added = sorted(set(source_eps) - set(record.entry_points))
            removed = sorted(set(record.entry_points) - set(source_eps))
            differences.append(
                "console scripts differ"
                + (f"; missing {added}" if added else "")
                + (f"; stale {removed}" if removed else "")
            )
        states.append(
            MemberInstallState(
                member=member,
                installed=True,
                editable=record.editable,
                source_version=source_version,
                installed_version=record.version,
                source_entry_points=source_eps,
                installed_entry_points=record.entry_points,
                differences=tuple(differences),
            )
        )
    return tuple(states)


@dataclass(frozen=True)
class _InstalledRecord:
    name: str
    version: str
    editable: bool
    entry_points: tuple[str, ...]


def _installed_distributions(site: Path) -> dict[str, _InstalledRecord]:
    """Read ``.dist-info`` directly — we may not be running inside this venv."""

    records: dict[str, _InstalledRecord] = {}
    for dist_info in sorted(site.glob("*.dist-info")):
        metadata = dist_info / "METADATA"
        try:
            headers = _parse_rfc822(
                metadata.read_text(encoding="utf-8", errors="replace")
            )
        except OSError as exc:
            logger.warning("unreadable dist metadata %s: %s", metadata, exc)
            continue
        name = headers.get("name")
        if not name:
            continue
        records[canonical_name(name)] = _InstalledRecord(
            name=name,
            version=headers.get("version", ""),
            editable=_is_editable(dist_info),
            entry_points=_read_entry_points(dist_info / "entry_points.txt"),
        )
    return records


def _parse_rfc822(text: str) -> dict[str, str]:
    """Minimal single-valued header parse (stdlib ``email`` is overkill here)."""

    headers: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            break
        if ":" not in line or line[:1].isspace():
            continue
        key, _, value = line.partition(":")
        headers.setdefault(key.strip().lower(), value.strip())
    return headers


def _is_editable(dist_info: Path) -> bool:
    direct_url = dist_info / "direct_url.json"
    try:
        payload = json.loads(direct_url.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        logger.warning("unreadable direct_url for %s: %s", dist_info.name, exc)
        return False
    return bool(payload.get("dir_info", {}).get("editable"))


def _read_entry_points(path: Path) -> tuple[str, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except OSError as exc:
        logger.warning("unreadable entry points %s: %s", redact_path_for_log(path), exc)
        return ()
    scripts: list[str] = []
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        if section == "console_scripts" and "=" in stripped:
            scripts.append(stripped.split("=", 1)[0].strip())
    return tuple(sorted(scripts))


def _source_metadata(path: Path) -> tuple[str | None, tuple[str, ...], bool]:
    manifest = path / "pyproject.toml"
    try:
        document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("unreadable member manifest %s: %s", manifest, exc)
        return None, (), True
    project = document.get("project", {})
    dynamic = project.get("dynamic", [])
    version = project.get("version")
    scripts = project.get("scripts", {})
    return (
        version if isinstance(version, str) else None,
        tuple(sorted(scripts)) if isinstance(scripts, dict) else (),
        isinstance(dynamic, list) and "version" in dynamic,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Drift detection
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DriftFinding:
    code: str
    severity: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "detail": self.detail,
            "data": self.data,
        }


@dataclass(frozen=True)
class DriftReport:
    findings: tuple[DriftFinding, ...]

    @property
    def status(self) -> str:
        ranks = {"ok": 0, "warn": 1, "fail": 2}
        worst = max((ranks[f.severity] for f in self.findings), default=0)
        return {0: "ok", 1: "warn", 2: "fail"}[worst]

    @property
    def summary(self) -> str:
        problems = [f for f in self.findings if f.severity != "ok"]
        if not problems:
            return "shared venv is current with uv.lock"
        return "; ".join(f.detail for f in problems)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "findings": [f.as_dict() for f in self.findings],
        }


def detect_drift(workspace: Workspace, *, include_floor: bool = True) -> DriftReport:
    """Answer "is the shared venv still what the lock says it should be?".

    Runs read-only.  This is the check whose absence let the environment rot for
    ten days, so it is wired into ``agent-utilities doctor`` and into the merge
    reconciler rather than left as a command somebody has to remember.
    """

    findings: list[DriftFinding] = []

    if not workspace.venv.is_dir():
        findings.append(
            DriftFinding(
                code="venv_missing",
                severity="fail",
                detail=f"no virtualenv at {workspace.venv}",
            )
        )
        return DriftReport(findings=tuple(findings))

    check = lock_check(workspace)
    findings.append(
        DriftFinding(
            code="lock_current",
            severity="ok" if check.ok else "fail",
            detail=(
                "uv.lock is current with the workspace manifests"
                if check.ok
                else "uv.lock is STALE — a manifest moved without a relock"
            ),
            data={"detail": "" if check.ok else check.output[-1200:]},
        )
    )

    if check.ok:
        try:
            plan = plan_sync(workspace)
        except VenvSyncError as exc:
            findings.append(
                DriftFinding(
                    code="plan_unavailable",
                    severity="fail",
                    detail=f"could not compute a sync plan: {exc}",
                )
            )
            plan = None
        if plan is not None:
            findings.append(
                DriftFinding(
                    code="env_current",
                    severity="ok" if not plan.installs else "fail",
                    detail=(
                        "installed packages match uv.lock"
                        if not plan.installs
                        else (
                            f"{len(plan.installs)} package(s) are BEHIND the lock — "
                            "the environment is not running what the lock resolves"
                        )
                    ),
                    data={
                        "installs": [
                            f"{d.name}=={d.version}" for d in plan.installs[:40]
                        ]
                    },
                )
            )
            if plan.removals:
                findings.append(
                    DriftFinding(
                        code="env_extraneous",
                        severity="warn",
                        detail=(
                            f"{len(plan.removals)} installed package(s) are not in "
                            "the lock; --inexact leaves them alone deliberately"
                        ),
                        data={"packages": [d.name for d in plan.removals[:40]]},
                    )
                )

    states = member_install_states(workspace)
    stale = [s for s in states if s.stale]
    findings.append(
        DriftFinding(
            code="member_metadata",
            severity="ok" if not stale else "warn",
            detail=(
                f"all {len(states)} workspace member(s) installed editable and current"
                if not stale
                else (
                    f"{len(stale)} member(s) have metadata that differs from their "
                    "source (version / console scripts / editability)"
                )
            ),
            data={"stale": {s.member.name: list(s.differences) for s in stale[:20]}},
        )
    )

    if include_floor:
        floor = SdkFloorProbe().check(workspace)
        findings.append(
            DriftFinding(
                code="mcp_sdk_floor",
                severity={True: "ok", False: "fail", None: "ok"}[floor.ok],
                detail=floor.detail or "mcp sdk floor probe returned no detail",
            )
        )

    pending = _pending_intent_count(workspace)
    if pending:
        findings.append(
            DriftFinding(
                code="pending_flips",
                severity="warn",
                detail=(
                    f"{pending} merge flip(s) are queued and not yet applied "
                    "(deferred because the environment was busy)"
                ),
                data={"count": pending},
            )
        )

    return DriftReport(findings=tuple(findings))


def _pending_intent_count(workspace: Workspace) -> int:
    directory = workspace.state_dir / "intents"
    if not directory.is_dir():
        return 0
    return len(list(directory.glob("*.json")))


def session_start_hint(workspace: Workspace) -> str | None:
    """A near-zero-cost drift check, safe to run on every agent session start.

    CONCEPT:AU-OS.deployment.workspace-venv-reconciler (D-VS-6). ``detect_drift``
    is the full picture, but two of its three checks (``lock_check``,
    ``plan_sync``) spawn a real ``uv`` subprocess each — ~10s combined, an
    operator-visible latency this lane declined to impose on every session
    start by default. ``member_install_states`` is pure Python (reads
    dist-info + ``pyproject.toml`` off disk, no subprocess), so it is the one
    piece of drift detection cheap enough to run unconditionally. It also
    covers the single highest-signal case: an editable member whose *source*
    has moved (version, console scripts, editability) without a resync —
    exactly what went undetected for ten days before this lane existed.

    Returns ``None`` when there is nothing to say (clean, or the check itself
    could not run) so a hooked-up caller stays silent on the common case;
    never raises, so it can never turn a session-start hook into a failure.
    """
    try:
        if not workspace.venv.is_dir():
            return None
        stale = [s for s in member_install_states(workspace) if s.stale]
    except (OSError, VenvSyncError):  # noqa: BLE001 — best-effort session hint,
        # never the reason a session fails to start; caller sees silence.
        logger.debug("session_start_hint: drift check failed", exc_info=True)
        return None
    if not stale:
        return None
    names = ", ".join(sorted(s.member.name for s in stale)[:5])
    more = f" (+{len(stale) - 5} more)" if len(stale) > 5 else ""
    return (
        f"agent-utilities-venv: {len(stale)} editable member(s) look stale in "
        f"the shared venv ({names}{more}) — run `agent-utilities-venv status` "
        "for the full report, `agent-utilities-venv sync` to reconcile."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Upgrade / relock / rollback
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class UpgradeOutcome:
    """Result of moving the lock forward and proving the result works."""

    verdict: Verdict
    backup: Backup | None
    plan: SyncPlan | None
    probes: tuple[ProbeResult, ...]
    applied: bool
    rolled_back: bool
    detail: str

    @property
    def ok(self) -> bool:
        return self.applied and not self.rolled_back

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.as_dict(),
            "backup": self.backup.as_dict() if self.backup else None,
            "applied": self.applied,
            "rolled_back": self.rolled_back,
            "detail": self.detail,
            "probes": [p.as_dict() for p in self.probes],
        }


def upgrade(
    workspace: Workspace,
    packages: Sequence[str] = (),
    *,
    all_packages: bool = False,
    reason: str = "manual upgrade",
    ignore_activity: bool = False,
    allow_uninstalls: int = 0,
    hold_lock: bool = True,
) -> UpgradeOutcome:
    """Move dependencies forward, sync, verify, and roll back on failure.

    One command does the whole loop the operator asked for — relock, verify,
    roll back — because the dangerous part of an upgrade is not the relock, it
    is discovering three days later that it half worked and having no way back.

    ``hold_lock=False`` is for callers that already hold the exclusive writer
    lock (the merge reconciler does).  ``flock`` is per open-file-description,
    so re-acquiring it from the same process would falsely report "busy".
    """

    if not packages and not all_packages:
        raise VenvSyncError(
            "upgrade needs at least one --package, or --all to move everything"
        )

    try:
        if hold_lock:
            with exclusive_lock(workspace):
                return _upgrade_locked(
                    workspace,
                    packages,
                    all_packages=all_packages,
                    reason=reason,
                    ignore_activity=ignore_activity,
                    allow_uninstalls=allow_uninstalls,
                )
        return _upgrade_locked(
            workspace,
            packages,
            all_packages=all_packages,
            reason=reason,
            ignore_activity=ignore_activity,
            allow_uninstalls=allow_uninstalls,
        )
    except LockBusyError as exc:
        return UpgradeOutcome(
            verdict=Verdict(decision=DEFER, guardrail="writer_lock", reason=str(exc)),
            backup=None,
            plan=None,
            probes=(),
            applied=False,
            rolled_back=False,
            detail=str(exc),
        )


def _upgrade_locked(
    workspace: Workspace,
    packages: Sequence[str],
    *,
    all_packages: bool,
    reason: str,
    ignore_activity: bool,
    allow_uninstalls: int,
) -> UpgradeOutcome:
    ctx = SyncContext(
        workspace=workspace,
        reason=reason,
        allow_uninstalls=allow_uninstalls,
        ignore_activity=ignore_activity,
        activity=detect_activity(workspace),
    )
    busy = ActivityGuardrail().evaluate(None, ctx)
    if busy is not None:
        return UpgradeOutcome(
            verdict=busy,
            backup=None,
            plan=None,
            probes=(),
            applied=False,
            rolled_back=False,
            detail=busy.reason,
        )

    store = LockBackupStore(workspace)
    backup = store.create(
        reason, meta={"packages": list(packages), "all": all_packages}
    )

    args = ["lock"]
    if all_packages:
        args.append("--upgrade")
    for name in packages:
        args.extend(["--upgrade-package", name])
    result = run_uv(workspace, args, timeout=3600.0)
    if not result.ok:
        store.restore(backup.id)
        return UpgradeOutcome(
            verdict=Verdict(
                decision=REFUSE,
                guardrail="relock",
                reason=f"uv lock failed: {result.output[-1200:]}",
            ),
            backup=backup,
            plan=None,
            probes=(),
            applied=False,
            rolled_back=True,
            detail="lock restored from backup after a failed relock",
        )

    outcome = sync(
        workspace,
        reason=f"{reason} (post-relock)",
        apply=True,
        ignore_activity=True,  # already gated above, inside the writer lock
        allow_uninstalls=allow_uninstalls,
        hold_lock=False,
    )
    if not outcome.verdict.allowed:
        store.restore(backup.id)
        return UpgradeOutcome(
            verdict=outcome.verdict,
            backup=backup,
            plan=outcome.plan,
            probes=(),
            applied=False,
            rolled_back=True,
            detail=(
                "guardrail vetoed the post-upgrade plan; uv.lock restored from "
                f"backup {backup.id}"
            ),
        )

    probes = verify(workspace)
    failed = [p for p in probes if p.ok is False]
    if failed:
        store.restore(backup.id)
        recovery = sync(
            workspace,
            reason="rollback re-sync",
            apply=True,
            ignore_activity=True,
            allow_uninstalls=allow_uninstalls,
            hold_lock=False,
        )
        return UpgradeOutcome(
            verdict=Verdict(
                decision=REFUSE,
                guardrail="verify",
                reason="; ".join(f"{p.name}: {p.detail}" for p in failed),
            ),
            backup=backup,
            plan=outcome.plan,
            probes=probes,
            applied=False,
            rolled_back=True,
            detail=(
                f"verification failed; uv.lock restored from {backup.id} and the "
                f"environment re-synced ({recovery.detail})"
            ),
        )

    store.mark_verified(backup.id)
    return UpgradeOutcome(
        verdict=Verdict(decision=ALLOW),
        backup=backup,
        plan=outcome.plan,
        probes=probes,
        applied=True,
        rolled_back=False,
        detail=outcome.detail,
    )


def rollback(
    workspace: Workspace,
    backup_id: str | None = None,
    *,
    ignore_activity: bool = False,
) -> UpgradeOutcome:
    """Restore an archived ``uv.lock`` and re-sync the environment onto it."""

    try:
        with exclusive_lock(workspace):
            store = LockBackupStore(workspace)
            ctx = SyncContext(
                workspace=workspace,
                reason="rollback",
                ignore_activity=ignore_activity,
                activity=detect_activity(workspace),
            )
            busy = ActivityGuardrail().evaluate(None, ctx)
            if busy is not None:
                return UpgradeOutcome(
                    verdict=busy,
                    backup=None,
                    plan=None,
                    probes=(),
                    applied=False,
                    rolled_back=False,
                    detail=busy.reason,
                )
            # Archive the pre-rollback state too, so a rollback is itself undoable.
            store.create("pre-rollback snapshot")
            restored = store.restore(backup_id)
            outcome = sync(
                workspace,
                reason=f"rollback to {restored.id}",
                apply=True,
                ignore_activity=True,
                hold_lock=False,
            )
            probes = verify(workspace)
            return UpgradeOutcome(
                verdict=outcome.verdict,
                backup=restored,
                plan=outcome.plan,
                probes=probes,
                applied=outcome.applied,
                rolled_back=True,
                detail=f"restored {restored.id}: {outcome.detail}",
            )
    except LockBusyError as exc:
        return UpgradeOutcome(
            verdict=Verdict(decision=DEFER, guardrail="writer_lock", reason=str(exc)),
            backup=None,
            plan=None,
            probes=(),
            applied=False,
            rolled_back=False,
            detail=str(exc),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Leases
# ─────────────────────────────────────────────────────────────────────────────
def acquire_lease(workspace: Workspace, owner: str, *, ttl: float, reason: str) -> Path:
    """Declare a window in which the environment must not be swapped."""

    directory = workspace.state_dir / "leases"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{re.sub(r'[^A-Za-z0-9_.-]', '-', owner)}.json"
    _atomic_write_text(
        path,
        json.dumps(
            {
                "owner": owner,
                "reason": reason,
                "acquired_at": datetime.now(UTC).isoformat(),
                "expires_at": time.time() + ttl,
                "pid": os.getpid(),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    return path


def release_lease(workspace: Workspace, owner: str) -> bool:
    """Drop a lease early."""

    path = (
        workspace.state_dir
        / "leases"
        / f"{re.sub(r'[^A-Za-z0-9_.-]', '-', owner)}.json"
    )
    if not path.exists():
        return False
    _unlink_quietly(path)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def emit(payload: dict[str, Any], *, as_json: bool) -> None:
    """Render a result payload as JSON or as an indented human report."""
    if as_json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True, default=str)
        sys.stdout.write("\n")
        return
    _print_human(payload)


def _print_human(payload: dict[str, Any], indent: int = 0) -> None:
    pad = "  " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            sys.stdout.write(f"{pad}{key}:\n")
            _print_human(value, indent + 1)
        elif isinstance(value, list):
            sys.stdout.write(f"{pad}{key}:\n")
            for item in value:
                if isinstance(item, dict):
                    _print_human(item, indent + 1)
                    sys.stdout.write("\n")
                else:
                    sys.stdout.write(f"{pad}  - {item}\n")
        else:
            sys.stdout.write(f"{pad}{key}: {value}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-utilities-venv",
        description=(
            "The sanctioned entry point for the shared uv-workspace virtualenv. "
            "Every sync it runs uses `uv sync "
            f"{' '.join(SANCTIONED_SYNC_FLAGS)}`; the destructive bare form "
            "cannot be produced by any code path here."
        ),
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="workspace root (default: discovered by walking up for [tool.uv.workspace])",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--verbose", action="store_true", help="log uv invocations to stderr"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="drift report: lock, environment, members, floors")

    sub.add_parser("plan", help="show the sanctioned sync plan (read-only)")

    sync_cmd = sub.add_parser("sync", help="make the environment match uv.lock")
    sync_cmd.add_argument("--dry-run", action="store_true")
    sync_cmd.add_argument(
        "--ignore-activity",
        action="store_true",
        help="proceed even though other lanes are using the environment",
    )
    sync_cmd.add_argument(
        "--allow-uninstalls",
        type=int,
        default=0,
        metavar="N",
        help=(
            "sanction up to N uninstalls of a LOCKED distribution or workspace "
            "member (default 0 — any such uninstall refuses). Never reaches a "
            "merely-extraneous package: this sync always runs --inexact, so "
            "uv's own plan never proposes removing one; use `prune` for that"
        ),
    )
    sync_cmd.add_argument("--reason", default="manual")

    prune_cmd = sub.add_parser(
        "prune",
        help=(
            "remove installed packages that are neither locked nor a "
            "workspace member (D-VS-8) — the class --inexact deliberately "
            "leaves behind; never runs `uv sync`"
        ),
    )
    prune_cmd.add_argument(
        "--allow-uninstalls",
        type=int,
        default=0,
        metavar="N",
        required=True,
        help="required: sanction up to N removals (0 always refuses, on purpose)",
    )
    prune_cmd.add_argument("--dry-run", action="store_true")
    prune_cmd.add_argument(
        "--ignore-activity",
        action="store_true",
        help="proceed even though other lanes are using the environment",
    )

    up = sub.add_parser(
        "upgrade", help="move dependencies forward, verify, auto-roll-back"
    )
    up.add_argument("--package", action="append", default=[], dest="packages")
    up.add_argument("--all", action="store_true", dest="all_packages")
    up.add_argument("--ignore-activity", action="store_true")
    up.add_argument("--reason", default="manual upgrade")

    relock = sub.add_parser(
        "relock", help="re-resolve uv.lock in place (backed up + verified)"
    )
    relock.add_argument("--ignore-activity", action="store_true")

    rb = sub.add_parser("rollback", help="restore an archived uv.lock and re-sync")
    rb.add_argument("--to", dest="backup_id", default=None)
    rb.add_argument("--ignore-activity", action="store_true")

    sub.add_parser("backups", help="list archived uv.lock revisions")
    sub.add_parser("verify", help="run the post-change health probes")
    sub.add_parser("members", help="per-member editable install state")
    sub.add_parser("activity", help="what the activity probes currently see")
    sub.add_parser(
        "session-hint",
        help=(
            "near-zero-cost drift check (D-VS-6), safe for a SessionStart hook: "
            "silent when clean, never raises, no `uv` subprocess"
        ),
    )

    lease = sub.add_parser("lease", help="declare/release a do-not-swap window")
    lease.add_argument("action", choices=["acquire", "release", "list"])
    lease.add_argument("--owner", default=f"pid-{os.getpid()}")
    lease.add_argument("--ttl", type=float, default=3600.0)
    lease.add_argument("--reason", default="in-flight work")

    # Trigger/hook verbs live in venv_autosync and are imported lazily so a
    # broken environment can still run `status`/`sync`.
    autosync = sub.add_parser("autosync", help="merge-triggered auto-flip control")
    autosync.add_argument(
        "action",
        choices=["on", "off", "status", "install", "uninstall", "drain", "trigger"],
    )
    autosync.add_argument(
        "--repo",
        action="append",
        default=[],
        help="repository to install/uninstall hooks in (default: every workspace member)",
    )
    autosync.add_argument("--event", default="manual")
    autosync.add_argument("--inline", action="store_true", help="reconcile in-process")
    autosync.add_argument("--ignore-activity", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        workspace = Workspace.discover(args.workspace)
    except WorkspaceNotFoundError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    try:
        return _dispatch(args, workspace)
    except VenvSyncError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


def _dispatch(args: argparse.Namespace, workspace: Workspace) -> int:
    as_json = args.json
    if args.command == "status":
        report = detect_drift(workspace)
        emit(report.as_dict(), as_json=as_json)
        return {"ok": 0, "warn": 0, "fail": 3}[report.status]

    if args.command == "plan":
        plan = plan_sync(workspace)
        ctx = SyncContext(workspace=workspace, activity=detect_activity(workspace))
        verdict = evaluate_plan(plan, ctx)
        emit(
            {
                "installs": [f"{d.name}=={d.version}" for d in plan.installs],
                "uninstalls": [f"{d.name}=={d.version}" for d in plan.uninstalls],
                "verdict": verdict.as_dict(),
            },
            as_json=as_json,
        )
        return 0 if verdict.allowed else 3

    if args.command == "sync":
        sync_outcome = sync(
            workspace,
            reason=args.reason,
            apply=not args.dry_run,
            ignore_activity=args.ignore_activity,
            allow_uninstalls=args.allow_uninstalls,
        )
        emit(sync_outcome.as_dict(), as_json=as_json)
        return 0 if sync_outcome.verdict.allowed else 3

    if args.command == "prune":
        prune_outcome = prune(
            workspace,
            allow_uninstalls=args.allow_uninstalls,
            apply=not args.dry_run,
            ignore_activity=args.ignore_activity,
        )
        emit(prune_outcome.as_dict(), as_json=as_json)
        return 0 if not prune_outcome.refused else 3

    if args.command in ("upgrade", "relock"):
        upgrade_outcome = upgrade(
            workspace,
            packages=getattr(args, "packages", []),
            all_packages=getattr(args, "all_packages", False)
            or args.command == "relock",
            reason=getattr(args, "reason", args.command),
            ignore_activity=args.ignore_activity,
        )
        emit(upgrade_outcome.as_dict(), as_json=as_json)
        return 0 if upgrade_outcome.ok else 3

    if args.command == "rollback":
        rollback_outcome = rollback(
            workspace, args.backup_id, ignore_activity=args.ignore_activity
        )
        emit(rollback_outcome.as_dict(), as_json=as_json)
        return 0 if rollback_outcome.verdict.allowed else 3

    if args.command == "backups":
        emit(
            {"backups": [b.as_dict() for b in LockBackupStore(workspace).list()]},
            as_json=as_json,
        )
        return 0

    if args.command == "verify":
        probes = verify(workspace)
        emit({"probes": [p.as_dict() for p in probes]}, as_json=as_json)
        return 0 if all(p.ok is not False for p in probes) else 3

    if args.command == "members":
        states = member_install_states(workspace)
        emit(
            {
                "members": [
                    {
                        "name": s.member.name,
                        "installed": s.installed,
                        "editable": s.editable,
                        "source_version": s.source_version,
                        "installed_version": s.installed_version,
                        "differences": list(s.differences),
                    }
                    for s in states
                ],
                "stale": sum(1 for s in states if s.stale),
            },
            as_json=as_json,
        )
        return 0 if not any(s.stale for s in states) else 3

    if args.command == "session-hint":
        # D-VS-6: deliberately never propagates a raised error and always
        # exits 0 — a SessionStart hook command that can fail a session is a
        # worse outcome than staying silent about drift for one session.
        hint = session_start_hint(workspace)
        if hint is not None:
            if as_json:
                emit({"hint": hint}, as_json=True)
            else:
                print(hint)
        return 0

    if args.command == "activity":
        records = detect_activity(workspace)
        emit(
            {
                "busy": bool(records),
                "records": [
                    {"probe": r.probe, "id": r.identifier, "detail": r.detail}
                    for r in records
                ],
            },
            as_json=as_json,
        )
        return 0

    if args.command == "lease":
        if args.action == "acquire":
            path = acquire_lease(
                workspace, args.owner, ttl=args.ttl, reason=args.reason
            )
            emit({"lease": str(path), "owner": args.owner}, as_json=as_json)
            return 0
        if args.action == "release":
            emit({"released": release_lease(workspace, args.owner)}, as_json=as_json)
            return 0
        lease_records = LeaseActivityProbe().busy(workspace)
        emit(
            {
                "leases": [
                    {"owner": r.identifier, "detail": r.detail} for r in lease_records
                ]
            },
            as_json=as_json,
        )
        return 0

    if args.command == "autosync":
        from agent_utilities.deployment import venv_autosync

        return venv_autosync.dispatch(args, workspace, as_json=as_json)

    raise VenvSyncError(f"unhandled command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover - console entry
    raise SystemExit(main())
