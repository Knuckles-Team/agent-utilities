"""Merge-trigger regression tests.

CONCEPT:AU-OS.deployment.merge-triggered-venv-flip

Covers the four things that decide whether this feature is safe to leave on:
the off switch really is off, a merge in a *linked worktree* does not flip the
live environment, a source-only merge does no work at all, and a refused or
deferred flip keeps its intent queued rather than dropping it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent_utilities.deployment import venv_autosync, venv_sync
from agent_utilities.deployment.venv_autosync import (
    AutosyncConfig,
    GitHookTrigger,
    Intent,
    drain,
    load_config,
    pending,
    save_config,
    trigger,
)
from agent_utilities.deployment.venv_sync import METADATA, SOURCE_ONLY, Workspace

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is required for the merge trigger"
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(repo),
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
    )
    return completed.stdout.strip()


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    _write(
        root / "pyproject.toml",
        '[project]\nname = "root"\nversion = "0.1.0"\ndependencies = []\n'
        '\n[tool.uv.workspace]\nmembers = ["pkgs/*"]\n',
    )
    member = root / "pkgs" / "alpha"
    _write(member / "pyproject.toml", '[project]\nname = "alpha"\nversion = "1.0.0"\n')
    _write(member / "alpha.py", "VALUE = 1\n")
    _write(root / "uv.lock", 'version = 1\n\n[[package]]\nname = "alpha"\nversion = "1.0.0"\n')

    site = root / ".venv" / "lib" / "python3.13" / "site-packages"
    dist = site / "alpha-1.0.0.dist-info"
    _write(dist / "METADATA", "Metadata-Version: 2.4\nName: alpha\nVersion: 1.0.0\n")
    _write(dist / "direct_url.json", json.dumps({"dir_info": {"editable": True}}))

    _git(member, "init", "-q", "-b", "main")
    _git(member, "add", "-A")
    _git(member, "commit", "-q", "-m", "initial")

    return Workspace.discover(root, uv="uv", state_dir=tmp_path / "state")


def _enable(workspace: Workspace, **overrides: object) -> AutosyncConfig:
    config = AutosyncConfig(enabled=True, **overrides)  # type: ignore[arg-type]
    save_config(workspace, config)
    return config


# ─────────────────────────────────────────────────────────────────────────────
# The switch
# ─────────────────────────────────────────────────────────────────────────────
def test_default_is_off_and_the_trigger_is_inert(workspace: Workspace) -> None:
    assert load_config(workspace).enabled is False
    result = trigger(workspace, workspace.root / "pkgs" / "alpha", event="post-merge")
    assert result["action"] == "skipped"
    assert "off" in result["why"]
    assert pending(workspace) == []


def test_on_and_off_roundtrip_without_touching_hooks(workspace: Workspace) -> None:
    save_config(workspace, AutosyncConfig(enabled=True))
    assert load_config(workspace).enabled is True
    save_config(workspace, AutosyncConfig(enabled=False))
    assert load_config(workspace).enabled is False


def test_unknown_config_keys_are_reported_not_silently_dropped(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    path = workspace.state_dir / "autosync.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"enabled": True, "typo_key": 1}), encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert load_config(workspace).enabled is True
    assert "typo_key" in caplog.text


# ─────────────────────────────────────────────────────────────────────────────
# What counts as a flip
# ─────────────────────────────────────────────────────────────────────────────
def test_source_only_merge_is_already_live_and_queues_nothing(
    workspace: Workspace,
) -> None:
    _enable(workspace)
    member = workspace.root / "pkgs" / "alpha"
    _write(member / "alpha.py", "VALUE = 2\n")
    _git(member, "add", "-A")
    _git(member, "commit", "-q", "-m", "source change")
    _git(member, "update-ref", "ORIG_HEAD", "HEAD~1")

    result = trigger(workspace, member, event="post-merge")
    assert result["action"] == "already-live"
    assert "downstream" in result["why"]
    assert pending(workspace) == []


def test_metadata_merge_queues_an_intent(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(workspace)
    monkeypatch.setattr(venv_autosync, "_spawn_reconciler", lambda ws: None)
    member = workspace.root / "pkgs" / "alpha"
    _write(member / "pyproject.toml", '[project]\nname = "alpha"\nversion = "1.0.0"\ndependencies = ["anyio"]\n')
    _git(member, "add", "-A")
    _git(member, "commit", "-q", "-m", "dependency change")
    _git(member, "update-ref", "ORIG_HEAD", "HEAD~1")

    result = trigger(workspace, member, event="post-merge")
    assert result["action"] == "queued"
    assert result["change_class"] == METADATA
    queued = pending(workspace)
    assert len(queued) == 1
    assert "pyproject.toml" in queued[0].changed_paths


def test_a_merge_on_a_non_flip_branch_is_skipped(workspace: Workspace) -> None:
    _enable(workspace)
    member = workspace.root / "pkgs" / "alpha"
    _git(member, "checkout", "-q", "-b", "feat/x")
    result = trigger(workspace, member, event="post-merge")
    assert result["action"] == "skipped"
    assert "feat/x" in result["why"]


def test_a_merge_in_a_linked_worktree_does_not_flip_the_live_environment(
    workspace: Workspace, tmp_path: Path
) -> None:
    """Only the checkout the editable install points at is 'live'."""

    _enable(workspace)
    member = workspace.root / "pkgs" / "alpha"
    outside = tmp_path / "worktrees" / "alpha-feature"
    _git(member, "worktree", "add", "-q", "-b", "feat/y", str(outside))

    # The worktree check runs before the branch check on purpose: even a
    # worktree sitting on a flip branch is not what the editable install points
    # at, so merging there changes nothing about what is live.
    result = trigger(workspace, outside, event="post-merge")
    assert result["action"] == "skipped"
    assert "not the checkout installed into" in result["why"]
    assert pending(workspace) == []


def test_unknown_changed_paths_escalate_rather_than_skip(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failing to read the diff must never be read as 'nothing changed'."""

    _enable(workspace)
    monkeypatch.setattr(venv_autosync, "_spawn_reconciler", lambda ws: None)
    monkeypatch.setattr(
        venv_autosync, "_changed_paths", lambda repo: ((), "abc", "", False)
    )
    result = trigger(workspace, workspace.root / "pkgs" / "alpha", event="post-merge")
    assert result["action"] == "queued"
    assert result["change_class"] == METADATA
    assert "escalated" in pending(workspace)[0].note


# ─────────────────────────────────────────────────────────────────────────────
# Draining
# ─────────────────────────────────────────────────────────────────────────────
def _queue(workspace: Workspace, change_class: str = METADATA) -> Intent:
    intent = Intent(
        id="intent-1",
        created_at="2026-07-31T00:00:00Z",
        repo=str(workspace.root / "pkgs" / "alpha"),
        branch="main",
        event="post-merge",
        change_class=change_class,
    )
    venv_autosync.enqueue(workspace, intent)
    return intent


def test_drain_keeps_the_intent_when_a_guardrail_refuses(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(workspace, on_metadata_change="propose")
    _queue(workspace)
    monkeypatch.setattr(
        venv_autosync,
        "sync",
        lambda ws, **kw: venv_sync.SyncOutcome(
            verdict=venv_sync.Verdict(
                decision=venv_sync.REFUSE,
                guardrail="lock_consistency",
                reason="uv.lock is out of date",
            ),
            plan=None,
            applied=False,
        ),
    )
    result = drain(workspace)
    assert result["action"] == "refused"
    assert result["kept"] == 1
    assert len(pending(workspace)) == 1
    assert "relock" in result["proposal"]


def test_drain_keeps_the_intent_when_deferred(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(workspace)
    _queue(workspace, change_class=SOURCE_ONLY)
    monkeypatch.setattr(
        venv_autosync,
        "sync",
        lambda ws, **kw: venv_sync.SyncOutcome(
            verdict=venv_sync.Verdict(
                decision=venv_sync.DEFER, guardrail="activity", reason="busy"
            ),
            plan=None,
            applied=False,
        ),
    )
    result = drain(workspace)
    assert result["action"] == "deferred"
    assert len(pending(workspace)) == 1


def test_drain_clears_the_queue_once_applied(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(workspace)
    _queue(workspace, change_class=SOURCE_ONLY)
    monkeypatch.setattr(
        venv_autosync,
        "sync",
        lambda ws, **kw: venv_sync.SyncOutcome(
            verdict=venv_sync.Verdict(decision=venv_sync.ALLOW),
            plan=venv_sync.SyncPlan(),
            applied=True,
            detail="applied",
        ),
    )
    result = drain(workspace)
    assert result["action"] == "applied"
    assert pending(workspace) == []


def test_the_default_metadata_policy_is_relock(workspace: Workspace) -> None:
    """A dependency change with many dependents is the case the requirement named.

    'propose' would honour "keep the venv current" while declining exactly the
    case that was asked for, so the default is 'relock' -- safe only because of
    the guardrail stack the other tests in this file pin down.
    """

    assert AutosyncConfig().on_metadata_change == "relock"
    assert load_config(workspace).on_metadata_change == "relock"
    save_config(workspace, AutosyncConfig(enabled=True))
    assert load_config(workspace).on_metadata_change == "relock"


def test_propose_mode_stays_supported_and_does_not_relock(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The conservative mode is a supported choice, not a removed one."""

    _enable(workspace, on_metadata_change="propose")
    _queue(workspace)
    monkeypatch.setattr(
        venv_autosync,
        "upgrade",
        lambda ws, **kw: pytest.fail("propose must never relock"),
    )
    monkeypatch.setattr(
        venv_autosync,
        "sync",
        lambda ws, **kw: venv_sync.SyncOutcome(
            verdict=venv_sync.Verdict(decision=venv_sync.ALLOW),
            plan=venv_sync.SyncPlan(),
            applied=True,
            detail="synced against the existing lock",
        ),
    )
    assert drain(workspace)["policy"] == "propose"


def test_drain_relocks_only_when_the_policy_says_so(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enable(workspace, on_metadata_change="relock")
    _queue(workspace)
    calls: list[str] = []

    def _upgrade(ws: Workspace, **kw: object) -> venv_sync.UpgradeOutcome:
        calls.append("upgrade")
        return venv_sync.UpgradeOutcome(
            verdict=venv_sync.Verdict(decision=venv_sync.ALLOW),
            backup=None,
            plan=None,
            probes=(),
            applied=True,
            rolled_back=False,
            detail="relocked",
        )

    monkeypatch.setattr(venv_autosync, "upgrade", _upgrade)
    monkeypatch.setattr(
        venv_autosync,
        "sync",
        lambda ws, **kw: pytest.fail("sync must not run under the relock policy"),
    )
    result = drain(workspace)
    assert calls == ["upgrade"]
    assert result["action"] == "applied"


def test_drain_on_an_empty_queue_is_idle(workspace: Workspace) -> None:
    assert drain(workspace)["action"] == "idle"


# ─────────────────────────────────────────────────────────────────────────────
# Hook installation
# ─────────────────────────────────────────────────────────────────────────────
def test_hooks_install_into_the_common_git_dir_and_are_idempotent(
    workspace: Workspace,
) -> None:
    member = workspace.root / "pkgs" / "alpha"
    backend = GitHookTrigger()
    first = backend.install(workspace, member)
    assert len(first["hooks"]) == len(venv_autosync.HOOK_EVENTS)
    hook = Path(first["hooks"][0])
    assert hook.is_file()
    body = hook.read_text(encoding="utf-8")
    backend.install(workspace, member)
    assert hook.read_text(encoding="utf-8") == body
    assert body.count(venv_autosync._BLOCK_START) == 1
    assert backend.status(workspace, member)["installed"]["post-merge"] is True


def test_hook_installation_composes_with_a_pre_existing_hook(
    workspace: Workspace,
) -> None:
    member = workspace.root / "pkgs" / "alpha"
    hooks = venv_autosync.hooks_dir(member)
    hooks.mkdir(parents=True, exist_ok=True)
    existing = hooks / "post-merge"
    existing.write_text("#!/bin/sh\necho pre-existing\n", encoding="utf-8")

    backend = GitHookTrigger()
    backend.install(workspace, member)
    assert "echo pre-existing" in existing.read_text(encoding="utf-8")

    backend.uninstall(workspace, member)
    remaining = existing.read_text(encoding="utf-8")
    assert "echo pre-existing" in remaining
    assert venv_autosync._BLOCK_START not in remaining


def test_uninstall_removes_a_hook_this_tool_created_outright(
    workspace: Workspace,
) -> None:
    member = workspace.root / "pkgs" / "alpha"
    backend = GitHookTrigger()
    backend.install(workspace, member)
    hooks = venv_autosync.hooks_dir(member)
    backend.uninstall(workspace, member)
    assert not (hooks / "post-merge").exists()


def test_the_trigger_script_is_executable_and_self_contained(
    workspace: Workspace,
) -> None:
    script = venv_autosync.write_trigger_script(workspace)
    assert script.stat().st_mode & 0o111
    body = script.read_text(encoding="utf-8")
    assert "autosync" in body
    assert ("scripts/venvctl" in body) or ("venv_sync" in body)
    assert str(workspace.venv / "bin" / "python") in body
