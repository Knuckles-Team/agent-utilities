"""Merge-triggered auto-flip for the shared uv-workspace virtualenv.

CONCEPT:AU-OS.deployment.merge-triggered-venv-flip

The operator's requirement: *"when we merge to main locally, we want a
development process that automatically flips that live (even for things with
many downstream relationships)"*.

Why git hooks, and why they only *enqueue*
------------------------------------------
Four mechanisms were available and only one observes the actual event:

* a **``post-merge`` git hook** fires exactly once, synchronously, at the moment
  a merge lands — zero polling, zero latency, and it knows the repository, the
  branch and (via ``ORIG_HEAD``) precisely which files moved.  Nothing else
  knows *what changed*, which is what decides whether any work is needed at all;
* a **filesystem watcher** would have to poll ~75 repositories, cannot
  distinguish a merge from an editor save, and adds a daemon to supervise;
* a **``make`` target** is not automatic — the failure mode being fixed here is
  precisely that nobody remembered to run the command;
* **CI** cannot help: the flip is local, and the workspace root is not even a
  git repository.

Hooks live in the repository's *common* git directory, so installing once in a
member checkout also covers every one of its linked worktrees — which is what
makes this tractable across ~26 concurrent worktrees.

The hook itself does almost nothing.  It writes an **intent record** and returns,
because the one thing this must never do is swap packages underneath a lane that
is mid-test.  A detached reconciler drains the queue under the same guardrails as
every other mutation, and a deferred intent stays queued (and is reported as
pending drift) rather than being dropped.

What actually needs to happen on a merge
----------------------------------------
Editable installs make most merges free:

* **source-only change** — already live for the member *and every downstream
  dependent*, because they all import through the source tree.  The reconciler
  records it and does nothing.
* **metadata change** (``pyproject.toml`` dependencies/scripts/version) — the
  installed ``.dist-info`` and, for a dependency change, the whole workspace
  resolution are now stale.  This is the case with the wide blast radius — and
  therefore the case the requirement actually named — so ``on_metadata_change``
  defaults to *relock*: back up, re-resolve, sync, verify, auto-roll-back.  That
  default is only defensible because of the guardrail stack it runs inside; see
  :class:`AutosyncConfig` for the full reasoning and for when to prefer the
  conservative *propose* mode instead.
* **native change** (Rust/C) — the compiled extension must be rebuilt; a sync
  does that.
* **``uv.lock`` moved** — sync.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agent_utilities.deployment.venv_sync import (
    ALLOW,
    DEFER,
    LOCK,
    METADATA,
    NATIVE,
    SOURCE_ONLY,
    LockBusyError,
    VenvSyncError,
    Workspace,
    classify_change,
    exclusive_lock,
    member_install_states,
    sync,
    upgrade,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AutosyncConfig",
    "GitHookTrigger",
    "Intent",
    "TriggerBackend",
    "dispatch",
    "drain",
    "load_config",
    "save_config",
    "trigger",
]

HOOK_EVENTS: tuple[str, ...] = ("post-merge", "post-checkout", "post-rewrite")
_BLOCK_START = "# >>> agent-utilities venv autosync (managed) >>>"
_BLOCK_END = "# <<< agent-utilities venv autosync (managed) <<<"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — one JSON document, no new environment variables
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AutosyncConfig:
    """The on/off switch and the policy knobs, all in one place."""

    #: Master switch.  ``autosync off`` sets this and every hook becomes inert
    #: without having to uninstall anything — turning it back on is one command.
    enabled: bool = False
    #: Branches whose advance means "this is now what runs".
    flip_branches: tuple[str, ...] = ("main", "master")
    #: What to do when a merge changes packaging metadata.
    #:
    #: ``relock``   — **default.** Run the full backed-up, verified,
    #:                auto-rolled-back ``upgrade --all``;
    #: ``propose``  — record it loudly and leave the relock to an operator;
    #: ``sync-only``— sync against the existing lock and report the staleness.
    #:
    #: Why ``relock`` is the default, given that it re-resolves a lock shared by
    #: every worktree: the requirement this feature exists to satisfy is
    #: "merging to main flips that live **even for things with many downstream
    #: relationships**", and a dependency change with many dependents is
    #: precisely that case.  ``propose`` stops exactly there, so it would honour
    #: the letter of "keep the venv current" while declining the one case that
    #: was actually asked for.
    #:
    #: What makes that safe rather than reckless is the guardrail stack this
    #: policy runs inside, and the default is only defensible *because* of it:
    #:
    #:   * :class:`~agent_utilities.deployment.venv_sync.ActivityGuardrail`
    #:     defers while any lane is mid-test, so a relock never lands underneath
    #:     running work;
    #:   * :class:`~agent_utilities.deployment.venv_sync.LockBackupStore`
    #:     archives ``uv.lock`` before the mutation starts;
    #:   * the verify probes run after it, and any failure **auto-rolls-back**
    #:     the lock and re-syncs the environment;
    #:   * a refusal outranks a deferral, so a plan that would net-remove a
    #:     workspace member is rejected regardless of timing.
    #:
    #: Remove any one of those and ``propose`` would be the right default.
    #:
    #: Prefer ``propose`` when the environment has many concurrent editors who
    #: must not have their resolution move under them, or while a large merge
    #: campaign is in flight — the relock would be correct but its timing would
    #: not be. It is a supported mode, not a deprecated one.
    on_metadata_change: str = "relock"
    #: Repositories whose hooks this tool manages.
    installed_repos: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["flip_branches"] = list(self.flip_branches)
        payload["installed_repos"] = list(self.installed_repos)
        return payload


def _config_path(workspace: Workspace) -> Path:
    return workspace.state_dir / "autosync.json"


def load_config(workspace: Workspace) -> AutosyncConfig:
    path = _config_path(workspace)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AutosyncConfig()
    except (OSError, ValueError) as exc:
        raise VenvSyncError(f"autosync config {path} is unreadable: {exc}") from exc
    known = {f for f in AutosyncConfig.__dataclass_fields__}
    unknown = sorted(set(payload) - known)
    if unknown:
        logger.warning(
            "ignoring unknown autosync config key(s): %s", ", ".join(unknown)
        )
    return AutosyncConfig(
        enabled=bool(payload.get("enabled", False)),
        flip_branches=tuple(payload.get("flip_branches", AutosyncConfig.flip_branches)),
        on_metadata_change=str(
            payload.get("on_metadata_change", AutosyncConfig.on_metadata_change)
        ),
        installed_repos=tuple(payload.get("installed_repos", ())),
    )


def save_config(workspace: Workspace, config: AutosyncConfig) -> Path:
    path = _config_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Intents
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Intent:
    """One recorded "make this merge live" request."""

    id: str
    created_at: str
    repo: str
    branch: str
    event: str
    change_class: str
    head: str = ""
    previous: str = ""
    changed_paths: tuple[str, ...] = ()
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed_paths"] = list(self.changed_paths[:200])
        return payload


def _intent_dir(workspace: Workspace) -> Path:
    return workspace.state_dir / "intents"


def enqueue(workspace: Workspace, intent: Intent) -> Path:
    directory = _intent_dir(workspace)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{intent.id}.json"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(intent.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp, path)
    return path


def pending(workspace: Workspace) -> list[Intent]:
    directory = _intent_dir(workspace)
    if not directory.is_dir():
        return []
    intents: list[Intent] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("ignoring unreadable intent %s: %s", path, exc)
            continue
        intents.append(
            Intent(
                id=str(payload.get("id", path.stem)),
                created_at=str(payload.get("created_at", "")),
                repo=str(payload.get("repo", "")),
                branch=str(payload.get("branch", "")),
                event=str(payload.get("event", "")),
                change_class=str(payload.get("change_class", METADATA)),
                head=str(payload.get("head", "")),
                previous=str(payload.get("previous", "")),
                changed_paths=tuple(payload.get("changed_paths", ())),
                note=str(payload.get("note", "")),
            )
        )
    return intents


def _clear(workspace: Workspace, intents: Sequence[Intent]) -> None:
    directory = _intent_dir(workspace)
    for intent in intents:
        path = directory / f"{intent.id}.json"
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("could not drain intent %s: %s", path, exc)


def _record_run(workspace: Workspace, payload: dict[str, Any]) -> Path:
    directory = workspace.state_dir / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{stamp}.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    for stale in sorted(directory.glob("*.json"))[:-200]:
        try:
            stale.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("could not prune run record %s: %s", stale, exc)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Trigger backends
# ─────────────────────────────────────────────────────────────────────────────
@runtime_checkable
class TriggerBackend(Protocol):
    """Installs/removes whatever observes "a merge landed"."""

    name: str

    def install(self, workspace: Workspace, repo: Path) -> dict[str, Any]: ...

    def uninstall(self, workspace: Workspace, repo: Path) -> dict[str, Any]: ...

    def status(self, workspace: Workspace, repo: Path) -> dict[str, Any]: ...


def _git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(  # noqa: S603 — argv is constructed, never shell
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VenvSyncError(f"git {' '.join(args)} failed in {repo}: {exc}") from exc
    if completed.returncode != 0:
        raise VenvSyncError(
            f"git {' '.join(args)} failed in {repo} (rc={completed.returncode}): "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    return completed.stdout.strip()


def hooks_dir(repo: Path) -> Path:
    """The hooks directory git will actually consult for ``repo``.

    Resolves ``core.hooksPath`` when set, otherwise ``<common-git-dir>/hooks``.
    The *common* directory is the point: it is shared by every linked worktree,
    so one install covers the canonical checkout and all of its worktrees.
    """

    try:
        configured = _git(repo, "config", "--get", "core.hooksPath")
    except VenvSyncError:
        configured = ""
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else (repo / path)
    common = _git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common) / "hooks"


def _trigger_script_path(workspace: Workspace) -> Path:
    return workspace.state_dir / "bin" / "venv-autosync-trigger"


def write_trigger_script(workspace: Workspace) -> Path:
    """Emit the one script every hook calls.

    Indirection on purpose: hooks are written once and never need rewriting when
    the interpreter or the checkout moves — only this script does.  It is plain
    POSIX shell plus a stdlib-only Python module invocation, so it keeps working
    when the shared venv is broken (which is exactly when it matters).
    """

    path = _trigger_script_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    log = workspace.state_dir / "trigger.log"
    checkout = Path(__file__).resolve().parents[2]
    launcher = checkout / "scripts" / "venvctl"
    # Prefer the launcher: it falls back to loading the reconciler standalone
    # when `agent_utilities` is not importable, which is precisely the state a
    # broken shared venv leaves the machine in.  `-m` cannot do that, because
    # it executes the package __init__ chain first.
    if launcher.is_file():
        entry = f'"$PY" {shlex.quote(str(launcher))}'
    else:
        entry = (
            f"PYTHONPATH={shlex.quote(str(checkout))}"
            '"${PYTHONPATH:+:$PYTHONPATH}" '
            '"$PY" -m agent_utilities.deployment.venv_sync'
        )
    interpreters = [
        str(workspace.venv / "bin" / "python"),
        sys.executable,
        "python3",
    ]
    body = f"""#!/bin/sh
# Generated by agent_utilities.deployment.venv_autosync — do not edit.
# CONCEPT:AU-OS.deployment.merge-triggered-venv-flip
set -u
LOG={shlex.quote(str(log))}
for PY in {" ".join(shlex.quote(i) for i in interpreters)}; do
    if [ -x "$PY" ] || command -v "$PY" >/dev/null 2>&1; then
        {entry} autosync "$@" >>"$LOG" 2>&1
        exit 0
    fi
done
echo "agent-utilities venv autosync: no usable interpreter; see $LOG" >&2
exit 0
"""
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _hook_block(workspace: Workspace, event: str) -> str:
    script = _trigger_script_path(workspace)
    log = workspace.state_dir / "trigger.log"
    return "\n".join(
        [
            _BLOCK_START,
            "# Enqueues a 'make this merge live' intent for the shared uv workspace",
            "# venv. Never blocks the merge and never fails it: a trigger problem is",
            "# written to the log below and announced, not swallowed.",
            f"if [ -x {shlex.quote(str(script))} ]; then",
            f"    {shlex.quote(str(script))} trigger --event {shlex.quote(event)} "
            '--repo "$(git rev-parse --show-toplevel)" || \\',
            f'        echo "agent-utilities venv autosync trigger failed; see '
            f'{log}" >&2',
            "fi",
            _BLOCK_END,
            "",
        ]
    )


class GitHookTrigger:
    """Install/remove the managed block in ``post-merge``/-``checkout``/-``rewrite``.

    Composes with hooks that already exist (pre-commit owns some of them in this
    repository), by appending a delimited block and removing only that block on
    uninstall.  Never clobbers a file it did not create.
    """

    name = "git-hook"

    def install(self, workspace: Workspace, repo: Path) -> dict[str, Any]:
        write_trigger_script(workspace)
        directory = hooks_dir(repo)
        directory.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for event in HOOK_EVENTS:
            path = directory / event
            block = _hook_block(workspace, event)
            existing = path.read_text(encoding="utf-8") if path.is_file() else ""
            if _BLOCK_START in existing:
                body = _strip_block(existing) + block
            elif existing.strip():
                body = existing.rstrip("\n") + "\n\n" + block
            else:
                body = "#!/bin/sh\n# agent-utilities venv autosync\n\n" + block
            path.write_text(body, encoding="utf-8")
            path.chmod(0o755)
            written.append(str(path))
        return {"repo": str(repo), "hooks": written, "hooks_dir": str(directory)}

    def uninstall(self, workspace: Workspace, repo: Path) -> dict[str, Any]:
        directory = hooks_dir(repo)
        removed: list[str] = []
        for event in HOOK_EVENTS:
            path = directory / event
            if not path.is_file():
                continue
            existing = path.read_text(encoding="utf-8")
            if _BLOCK_START not in existing:
                continue
            remainder = _strip_block(existing)
            if remainder.strip() in (
                "",
                "#!/bin/sh",
                "#!/bin/sh\n# agent-utilities venv autosync",
            ):
                path.unlink()
            else:
                path.write_text(remainder, encoding="utf-8")
            removed.append(str(path))
        return {"repo": str(repo), "removed": removed}

    def status(self, workspace: Workspace, repo: Path) -> dict[str, Any]:
        directory = hooks_dir(repo)
        return {
            "repo": str(repo),
            "hooks_dir": str(directory),
            "installed": {
                event: (directory / event).is_file()
                and _BLOCK_START in (directory / event).read_text(encoding="utf-8")
                for event in HOOK_EVENTS
            },
        }


def _strip_block(text: str) -> str:
    out: list[str] = []
    skipping = False
    for line in text.splitlines(keepends=True):
        if line.strip() == _BLOCK_START:
            skipping = True
            continue
        if line.strip() == _BLOCK_END:
            skipping = False
            continue
        if not skipping:
            out.append(line)
    return "".join(out)


TRIGGER_BACKENDS: dict[str, TriggerBackend] = {"git-hook": GitHookTrigger()}


def register_trigger_backend(backend: TriggerBackend) -> None:
    """Register an alternative observer (a watcher, an IDE integration, …)."""

    TRIGGER_BACKENDS[backend.name] = backend


# ─────────────────────────────────────────────────────────────────────────────
# Trigger handling
# ─────────────────────────────────────────────────────────────────────────────
def _changed_paths(repo: Path) -> tuple[tuple[str, ...], str, str, bool]:
    """Return (paths, head, previous, known).

    ``known=False`` means we could not determine what moved, which escalates the
    change class — this fails toward doing work, never toward skipping it.
    """

    try:
        head = _git(repo, "rev-parse", "HEAD")
    except VenvSyncError as exc:
        logger.warning("could not read HEAD in %s: %s", repo, exc)
        return (), "", "", False
    for previous_ref in ("ORIG_HEAD", "HEAD@{1}"):
        try:
            previous = _git(repo, "rev-parse", previous_ref)
            names = _git(repo, "diff", "--name-only", f"{previous}..{head}")
        except VenvSyncError as exc:  # noqa: BLE001 — deliberate DEBUG: this is a CASCADE probe over candidate refs (ORIG_HEAD, then HEAD@{1}); a fresh clone or a repo with no reflog legitimately has neither, so absence is the normal case, not a failure. The loop continues to the next candidate and the final `return` reports the genuine outcome. The cause is preserved (interpolated).
            logger.debug("%s unavailable in %s: %s", previous_ref, repo, exc)
            continue
        return tuple(n for n in names.splitlines() if n), head, previous, True
    return (), head, "", False


def trigger(
    workspace: Workspace,
    repo: Path,
    *,
    event: str,
    inline: bool = False,
) -> dict[str, Any]:
    """Handle one hook firing: decide, record, and (maybe) kick the reconciler."""

    config = load_config(workspace)
    if not config.enabled:
        return {
            "action": "skipped",
            "why": "autosync is off (`autosync on` enables it)",
        }

    repo = repo.resolve()
    if not _is_live_checkout(workspace, repo):
        return {
            "action": "skipped",
            "why": (
                f"{repo} is not the checkout installed into {workspace.venv}; "
                "merging in a linked worktree does not change what is live"
            ),
        }

    try:
        branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    except VenvSyncError as exc:
        return {"action": "skipped", "why": f"could not read the branch: {exc}"}
    if branch not in config.flip_branches:
        return {
            "action": "skipped",
            "why": f"branch {branch!r} is not one of {list(config.flip_branches)}",
        }

    paths, head, previous, known = _changed_paths(repo)
    change_class = classify_change(paths) if known else METADATA
    note = "" if known else "changed paths unknown; escalated to metadata"

    if change_class == SOURCE_ONLY:
        stale = [s for s in member_install_states(workspace) if s.stale]
        if not stale:
            record = {
                "action": "already-live",
                "why": (
                    "source-only change in an editable member: it and every "
                    "downstream dependent already import the merged source"
                ),
                "repo": str(repo),
                "branch": branch,
                "changed": len(paths),
            }
            _record_run(workspace, record)
            return record
        note = f"{len(stale)} member(s) already had stale install metadata"
        change_class = METADATA

    intent = Intent(
        id=f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{random.randrange(1 << 24):06x}",
        created_at=datetime.now(UTC).isoformat(),
        repo=str(repo),
        branch=branch,
        event=event,
        change_class=change_class,
        head=head,
        previous=previous,
        changed_paths=paths,
        note=note,
    )
    enqueue(workspace, intent)

    if inline:
        return {
            "action": "queued+drained",
            "intent": intent.id,
            "drain": drain(workspace),
        }
    _spawn_reconciler(workspace)
    return {"action": "queued", "intent": intent.id, "change_class": change_class}


def _is_live_checkout(workspace: Workspace, repo: Path) -> bool:
    """Is ``repo`` one of the member directories actually installed editable?

    A linked worktree lives outside the workspace tree, so a merge there changes
    nothing about what the shared venv runs.  Only an advance in the member
    checkout that the editable install points at is a real flip.
    """

    return any(member.path == repo for member in workspace.members())


def _spawn_reconciler(workspace: Workspace) -> int | None:
    """Start the drain detached, so the merge returns immediately."""

    script = _trigger_script_path(workspace)
    argv = (
        [str(script), "drain"]
        if os.access(script, os.X_OK)
        else [
            sys.executable,
            "-m",
            "agent_utilities.deployment.venv_sync",
            "autosync",
            "drain",
        ]
    )
    log = workspace.state_dir / "reconciler.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(  # noqa: S603 — argv is constructed, never shell
                argv,
                cwd=str(workspace.root),
                stdout=handle,
                stderr=handle,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        return process.pid
    except OSError as exc:
        logger.error(
            "could not spawn the venv reconciler (%s); intent stays queued", exc
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Reconciler
# ─────────────────────────────────────────────────────────────────────────────
_CLASS_RANK = {SOURCE_ONLY: 0, LOCK: 1, NATIVE: 2, METADATA: 3}


def drain(workspace: Workspace, *, ignore_activity: bool = False) -> dict[str, Any]:
    """Apply every queued flip, under the same guardrails as any other mutation.

    Deferred intents stay queued — that is what makes "never fire while another
    lane is mid-test" safe rather than lossy: the flip is late, never lost, and
    :func:`agent_utilities.deployment.venv_sync.detect_drift` reports the backlog
    so a stuck queue is visible instead of silent.
    """

    intents = pending(workspace)
    if not intents:
        return {"action": "idle", "pending": 0}

    config = load_config(workspace)
    worst = max(intents, key=lambda i: _CLASS_RANK.get(i.change_class, 3))
    payload: dict[str, Any] = {
        "pending": len(intents),
        "change_class": worst.change_class,
        "policy": config.on_metadata_change,
        "repos": sorted({i.repo for i in intents}),
    }

    try:
        with exclusive_lock(workspace):
            if worst.change_class == METADATA and config.on_metadata_change == "relock":
                upgrade_outcome = upgrade(
                    workspace,
                    all_packages=True,
                    reason="merge flip: metadata change",
                    ignore_activity=ignore_activity,
                    hold_lock=False,
                )
                payload["upgrade"] = upgrade_outcome.as_dict()
                applied = upgrade_outcome.ok
                verdict = upgrade_outcome.verdict
            else:
                sync_outcome = sync(
                    workspace,
                    reason=f"merge flip: {worst.change_class}",
                    apply=True,
                    ignore_activity=ignore_activity,
                    hold_lock=False,
                )
                payload["sync"] = sync_outcome.as_dict()
                applied = sync_outcome.verdict.allowed
                verdict = sync_outcome.verdict
                if (
                    worst.change_class == METADATA
                    and config.on_metadata_change == "propose"
                    and not sync_outcome.verdict.allowed
                ):
                    payload["proposal"] = (
                        "a merge changed packaging metadata, so uv.lock no longer "
                        "matches the manifests. Review and apply it with "
                        "`agent-utilities-venv relock` (backed up, verified, "
                        "auto-rolled-back), or set on_metadata_change=relock to "
                        "have this happen automatically."
                    )
    except LockBusyError as exc:
        payload.update(
            {"action": "deferred", "why": str(exc), "drained": 0, "kept": len(intents)}
        )
        _record_run(workspace, payload)
        return payload

    if applied or verdict.decision == ALLOW:
        _clear(workspace, intents)
        payload.update({"action": "applied", "drained": len(intents), "kept": 0})
    elif verdict.decision == DEFER:
        payload.update({"action": "deferred", "drained": 0, "kept": len(intents)})
    else:
        payload.update({"action": "refused", "drained": 0, "kept": len(intents)})
    _record_run(workspace, payload)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# CLI dispatch (called from venv_sync.main)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_repos(workspace: Workspace, requested: Sequence[str]) -> list[Path]:
    if requested:
        return [Path(item).resolve() for item in requested]
    return [member.path for member in workspace.members()]


def dispatch(args: argparse.Namespace, workspace: Workspace, *, as_json: bool) -> int:
    from agent_utilities.deployment.venv_sync import emit

    action = args.action
    config = load_config(workspace)

    if action in ("on", "off"):
        config.enabled = action == "on"
        save_config(workspace, config)
        emit(
            {
                "enabled": config.enabled,
                "config": str(_config_path(workspace)),
                "note": (
                    "hooks stay installed either way; this switch alone turns the "
                    "auto-flip on and off"
                ),
            },
            as_json=as_json,
        )
        return 0

    if action == "status":
        backend = TRIGGER_BACKENDS["git-hook"]
        repos = _resolve_repos(workspace, config.installed_repos)
        statuses = []
        for repo in repos:
            try:
                statuses.append(backend.status(workspace, repo))
            except VenvSyncError as exc:
                statuses.append({"repo": str(repo), "error": str(exc)})
        emit(
            {
                "enabled": config.enabled,
                "on_metadata_change": config.on_metadata_change,
                "flip_branches": list(config.flip_branches),
                "pending_intents": len(pending(workspace)),
                "trigger_script": str(_trigger_script_path(workspace)),
                "repos": statuses,
            },
            as_json=as_json,
        )
        return 0

    if action in ("install", "uninstall"):
        backend = TRIGGER_BACKENDS["git-hook"]
        repos = _resolve_repos(workspace, args.repo)
        results = []
        installed = set(config.installed_repos)
        for repo in repos:
            try:
                if action == "install":
                    results.append(backend.install(workspace, repo))
                    installed.add(str(repo))
                else:
                    results.append(backend.uninstall(workspace, repo))
                    installed.discard(str(repo))
            except VenvSyncError as exc:
                results.append({"repo": str(repo), "error": str(exc)})
        config.installed_repos = tuple(sorted(installed))
        save_config(workspace, config)
        emit(
            {
                "action": action,
                "count": len(results),
                "enabled": config.enabled,
                "results": results,
            },
            as_json=as_json,
        )
        return 0

    if action == "drain":
        emit(drain(workspace, ignore_activity=args.ignore_activity), as_json=as_json)
        return 0

    if action == "trigger":
        repo = Path(args.repo[0]) if args.repo else Path.cwd()
        emit(
            trigger(workspace, repo, event=args.event, inline=args.inline),
            as_json=as_json,
        )
        return 0

    raise VenvSyncError(f"unhandled autosync action {action!r}")
