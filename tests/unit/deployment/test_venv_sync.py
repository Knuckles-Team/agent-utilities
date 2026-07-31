"""Guardrail regression tests for the shared-venv reconciler.

CONCEPT:AU-OS.safety.destructive-sync-refusal

The two behaviours that matter are proved against *real* uv output, not a
hand-written approximation: ``DESTRUCTIVE_PLAN`` below is the verbatim shape of
``uv sync --locked --dry-run`` in the live workspace, which reported 557
uninstalls including every editable member.  If a future refactor lets that plan
through, these tests fail.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent_utilities.deployment import venv_autosync, venv_sync
from agent_utilities.deployment.venv_sync import (
    ALLOW,
    DEFER,
    METADATA,
    NATIVE,
    REFUSE,
    SOURCE_ONLY,
    ActivityRecord,
    LockBackupStore,
    PlanParseError,
    SyncContext,
    SyncInvocation,
    SyncPlan,
    UnsafeInvocationError,
    Workspace,
    WorkspaceNotFoundError,
    classify_change,
    evaluate_plan,
    exclusive_lock,
    member_install_states,
)

# Verbatim head/tail of the observed destructive plan (2026-07-31).
DESTRUCTIVE_PLAN = """Would use project environment at: .venv
Resolved 726 packages in 64ms
Would uninstall 5 packages
 - aenum==3.1.15
 - agent-terminal-ui==2.0.0 (from file:///ws/agent-packages/agent-terminal-ui)
 - agent-utilities==2.1.1 (from file:///ws/agent-packages/agent-utilities)
 - aiofile==3.11.1
 - anyio==4.14.2
"""

CLEAN_PLAN = """Would use project environment at: .venv
Resolved 726 packages in 66ms
Checked 384 packages in 15ms
Would make no changes
"""

BEHIND_PLAN = """Would use project environment at: .venv
Resolved 726 packages in 66ms
Would install 2 packages
 + fastmcp==4.0.0b1
 + mcp==2.0.0
"""


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """A miniature uv workspace: two members, a lock, and an installed venv."""

    root = tmp_path / "ws"
    _write(
        root / "pyproject.toml",
        '[project]\nname = "root-project"\nversion = "0.1.0"\ndependencies = []\n'
        '\n[tool.uv.workspace]\nmembers = ["pkgs/*"]\nexclude = ["pkgs/excluded"]\n',
    )
    _write(
        root / "pkgs" / "alpha" / "pyproject.toml",
        '[project]\nname = "alpha"\nversion = "1.2.3"\n\n[project.scripts]\nalpha = "alpha:main"\n',
    )
    _write(
        root / "pkgs" / "beta" / "pyproject.toml",
        '[project]\nname = "beta"\nversion = "0.1.0"\n',
    )
    _write(
        root / "pkgs" / "excluded" / "pyproject.toml",
        '[project]\nname = "excluded"\nversion = "0.0.1"\n',
    )
    _write(
        root / "uv.lock",
        'version = 1\n\n[[package]]\nname = "alpha"\nversion = "1.2.3"\n'
        '\n[[package]]\nname = "beta"\nversion = "0.1.0"\n'
        '\n[[package]]\nname = "anyio"\nversion = "4.14.2"\n',
    )

    site = root / ".venv" / "lib" / "python3.13" / "site-packages"
    for name, version, editable, scripts in (
        ("alpha", "1.2.3", True, ["alpha"]),
        ("beta", "0.1.0", True, []),
    ):
        dist = site / f"{name}-{version}.dist-info"
        _write(dist / "METADATA", f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n")
        _write(
            dist / "direct_url.json",
            json.dumps({"url": "file:///x", "dir_info": {"editable": editable}}),
        )
        if scripts:
            _write(
                dist / "entry_points.txt",
                "[console_scripts]\n" + "\n".join(f"{s} = {name}:main" for s in scripts) + "\n",
            )
    (root / ".venv" / "bin").mkdir(parents=True, exist_ok=True)

    return Workspace.discover(root, uv="uv", state_dir=tmp_path / "state")


# ─────────────────────────────────────────────────────────────────────────────
# The destructive form must be unrepresentable
# ─────────────────────────────────────────────────────────────────────────────
def test_sanctioned_invocation_always_carries_every_safety_flag() -> None:
    argv = SyncInvocation().argv("uv")
    assert argv[:2] == ["uv", "sync"]
    for flag in venv_sync.SANCTIONED_SYNC_FLAGS:
        assert flag in argv
    assert "--dry-run" in SyncInvocation(dry_run=True).argv("uv")


@pytest.mark.parametrize(
    "argv",
    [
        ["uv", "sync"],
        ["uv", "sync", "--locked"],
        ["uv", "sync", "--locked", "--all-packages"],
        ["uv", "sync", "--all-packages", "--inexact"],
        ["uv", "sync", "--locked", "--inexact"],
    ],
)
def test_assert_sanctioned_refuses_every_incomplete_sync(argv: list[str]) -> None:
    with pytest.raises(UnsafeInvocationError) as excinfo:
        venv_sync._assert_sanctioned(argv)
    assert "unsanctioned" in str(excinfo.value)


def test_assert_sanctioned_ignores_non_sync_commands() -> None:
    venv_sync._assert_sanctioned(["uv", "lock", "--check"])


def test_run_uv_never_executes_an_unsanctioned_sync(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal happens before subprocess is reached, not after."""

    def _explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("subprocess must not be reached for an unsafe argv")

    monkeypatch.setattr(subprocess, "run", _explode)
    with pytest.raises(UnsafeInvocationError):
        venv_sync.run_uv(workspace, ["sync"])


# ─────────────────────────────────────────────────────────────────────────────
# Plan parsing
# ─────────────────────────────────────────────────────────────────────────────
def test_plan_parses_uninstalls_with_local_sources() -> None:
    plan = SyncPlan.parse(DESTRUCTIVE_PLAN)
    assert len(plan.uninstalls) == 5
    assert not plan.installs
    names = {d.name for d in plan.uninstalls}
    assert {"agent-utilities", "agent-terminal-ui"} <= names
    local = [d for d in plan.uninstalls if d.is_local]
    assert {d.name for d in local} == {"agent-utilities", "agent-terminal-ui"}


def test_plan_parses_clean_and_behind_states() -> None:
    clean = SyncPlan.parse(CLEAN_PLAN)
    assert clean.is_empty and clean.no_changes
    behind = SyncPlan.parse(BEHIND_PLAN)
    assert [d.name for d in behind.installs] == ["fastmcp", "mcp"]
    assert not behind.uninstalls


def test_plan_parse_fails_closed_when_uninstalls_cannot_be_enumerated() -> None:
    """Under-counting removals is the one direction that must never be silent."""

    truncated = "Would uninstall 400 packages\n - anyio==4.14.2\n"
    with pytest.raises(PlanParseError):
        SyncPlan.parse(truncated)


# ─────────────────────────────────────────────────────────────────────────────
# Guardrails
# ─────────────────────────────────────────────────────────────────────────────
def test_member_uninstall_guardrail_refuses_the_real_destructive_plan(
    workspace: Workspace,
) -> None:
    plan = SyncPlan.parse(
        DESTRUCTIVE_PLAN.replace("agent-terminal-ui", "alpha").replace(
            "agent-utilities", "beta"
        )
    )
    verdict = evaluate_plan(plan, SyncContext(workspace=workspace, ignore_activity=True))
    assert verdict.decision == REFUSE
    assert verdict.guardrail == "member_uninstall"
    assert set(verdict.data["members"]) == {"alpha", "beta"}


def test_locked_distribution_guardrail_refuses_removing_a_locked_package(
    workspace: Workspace,
) -> None:
    venv_sync.GUARDRAILS[:] = [
        g for g in venv_sync.GUARDRAILS if g.name != "member_uninstall"
    ]
    try:
        plan = SyncPlan.parse("Would uninstall 1 packages\n - anyio==4.14.2\n")
        verdict = evaluate_plan(
            plan, SyncContext(workspace=workspace, ignore_activity=True)
        )
        assert verdict.decision == REFUSE
        assert verdict.guardrail == "locked_uninstall"
    finally:
        venv_sync.register_guardrail(venv_sync.MemberUninstallGuardrail())


def test_uninstall_budget_allows_a_sanctioned_prune(workspace: Workspace) -> None:
    plan = SyncPlan.parse("Would uninstall 1 packages\n - leftover==1.0\n")
    ctx = SyncContext(workspace=workspace, ignore_activity=True)
    assert evaluate_plan(plan, ctx).decision == REFUSE
    ctx.allow_uninstalls = 1
    assert evaluate_plan(plan, ctx).decision == ALLOW


def test_activity_defers_but_a_destructive_plan_still_refuses(
    workspace: Workspace,
) -> None:
    """Refusal outranks deferral: a wrong plan is wrong at any time."""

    busy = SyncContext(
        workspace=workspace,
        activity=(ActivityRecord(probe="test", identifier="pid 1", detail="pytest"),),
    )
    assert evaluate_plan(SyncPlan.parse(CLEAN_PLAN), busy).decision == DEFER

    destructive = SyncPlan.parse(DESTRUCTIVE_PLAN.replace("agent-utilities", "alpha"))
    assert evaluate_plan(destructive, busy).decision == REFUSE


def test_stale_lock_refuses_rather_than_silently_relocking(
    workspace: Workspace,
) -> None:
    ctx = SyncContext(
        workspace=workspace,
        ignore_activity=True,
        lock_check_ok=False,
        lock_check_detail="the lockfile is not up-to-date",
    )
    verdict = evaluate_plan(None, ctx)
    assert verdict.decision == REFUSE
    assert verdict.guardrail == "lock_consistency"


def test_sync_defers_when_the_writer_lock_is_held(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _busy(*args: object, **kwargs: object) -> object:
        raise venv_sync.LockBusyError("another reconciler holds the lock")

    monkeypatch.setattr(venv_sync, "exclusive_lock", _busy)
    outcome = venv_sync.sync(workspace, reason="test")
    assert outcome.verdict.decision == DEFER
    assert outcome.verdict.guardrail == "writer_lock"
    assert outcome.applied is False


def test_exclusive_lock_is_actually_exclusive(workspace: Workspace) -> None:
    with exclusive_lock(workspace):
        with pytest.raises(venv_sync.LockBusyError):
            with exclusive_lock(workspace):
                pytest.fail("the writer lock was granted twice")


# ─────────────────────────────────────────────────────────────────────────────
# Backups and rollback
# ─────────────────────────────────────────────────────────────────────────────
def test_backup_restore_roundtrip(workspace: Workspace) -> None:
    store = LockBackupStore(workspace)
    original = workspace.lock.read_text(encoding="utf-8")
    backup = store.create("before an upgrade")
    workspace.lock.write_text("version = 1\n# mutated\n", encoding="utf-8")
    restored = store.restore(backup.id)
    assert restored.id == backup.id
    assert workspace.lock.read_text(encoding="utf-8") == original


def test_checkpoint_restores_the_lock_when_the_block_raises(
    workspace: Workspace,
) -> None:
    store = LockBackupStore(workspace)
    original = workspace.lock.read_text(encoding="utf-8")
    with pytest.raises(venv_sync.VenvSyncError) as excinfo:
        with store.checkpoint("relock"):
            workspace.lock.write_text("broken\n", encoding="utf-8")
            raise RuntimeError("resolution failed")
    assert workspace.lock.read_text(encoding="utf-8") == original
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_verified_backups_survive_retention(workspace: Workspace) -> None:
    store = LockBackupStore(workspace, retain=1)
    first = store.create("one")
    store.mark_verified(first.id)
    workspace.lock.write_text("version = 1\n# two\n", encoding="utf-8")
    store.create("two")
    workspace.lock.write_text("version = 1\n# three\n", encoding="utf-8")
    store.create("three")
    assert first.id in {b.id for b in store.list()}


# ─────────────────────────────────────────────────────────────────────────────
# Editable source change vs metadata change
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["agent_utilities/mcp/kg_server.py"], SOURCE_ONLY),
        (["README.md", "docs/x.md"], SOURCE_ONLY),
        (["pyproject.toml"], METADATA),
        (["a/setup.cfg", "b/main.py"], METADATA),
        (["src/lib.rs"], NATIVE),
        (["uv.lock"], venv_sync.LOCK),
        (["pyproject.toml", "src/lib.rs"], METADATA),
        ([], SOURCE_ONLY),
    ],
)
def test_classify_change(paths: list[str], expected: str) -> None:
    assert classify_change(paths) == expected


def test_member_install_states_are_clean_for_a_matching_install(
    workspace: Workspace,
) -> None:
    states = {s.member.name: s for s in member_install_states(workspace)}
    assert set(states) == {"alpha", "beta"}
    assert not any(s.stale for s in states.values())


def test_member_install_states_flag_version_and_script_and_editability_skew(
    workspace: Workspace,
) -> None:
    site = workspace.site_packages()
    assert site is not None
    dist = site / "alpha-1.2.3.dist-info"
    # A version bump merged into source but never reinstalled.
    _write(
        workspace.root / "pkgs" / "alpha" / "pyproject.toml",
        '[project]\nname = "alpha"\nversion = "2.0.0"\n\n[project.scripts]\n'
        'alpha = "alpha:main"\nalpha-extra = "alpha:extra"\n',
    )
    # And beta installed non-editable, so its source edits are NOT live.
    _write(
        site / "beta-0.1.0.dist-info" / "direct_url.json",
        json.dumps({"url": "file:///x", "dir_info": {"editable": False}}),
    )
    states = {s.member.name: s for s in member_install_states(workspace)}
    alpha = states["alpha"]
    assert alpha.stale
    assert any("2.0.0 declared" in d for d in alpha.differences)
    assert any("console scripts differ" in d for d in alpha.differences)
    assert any("non-editable" in d for d in states["beta"].differences)
    assert dist.is_dir()


def test_member_install_states_report_an_uninstalled_member(
    workspace: Workspace,
) -> None:
    _write(
        workspace.root / "pkgs" / "gamma" / "pyproject.toml",
        '[project]\nname = "gamma"\nversion = "0.0.1"\n',
    )
    states = {s.member.name: s for s in member_install_states(workspace)}
    assert states["gamma"].installed is False
    assert states["gamma"].stale


def test_dynamic_versions_are_not_reported_as_skew(workspace: Workspace) -> None:
    _write(
        workspace.root / "pkgs" / "alpha" / "pyproject.toml",
        '[project]\nname = "alpha"\ndynamic = ["version"]\n\n[project.scripts]\n'
        'alpha = "alpha:main"\n',
    )
    states = {s.member.name: s for s in member_install_states(workspace)}
    assert not states["alpha"].stale


# ─────────────────────────────────────────────────────────────────────────────
# Workspace discovery
# ─────────────────────────────────────────────────────────────────────────────
def test_discovery_walks_up_without_git(workspace: Workspace, tmp_path: Path) -> None:
    nested = workspace.root / "pkgs" / "alpha"
    assert Workspace.discover(nested, state_dir=tmp_path / "s").root == workspace.root


def test_discovery_raises_when_no_workspace_is_above(tmp_path: Path) -> None:
    lonely = tmp_path / "nowhere" / "deep"
    lonely.mkdir(parents=True)
    with pytest.raises(WorkspaceNotFoundError):
        Workspace.discover(lonely)


def test_members_honour_exclude(workspace: Workspace) -> None:
    assert {m.name for m in workspace.members()} == {"alpha", "beta"}


# ─────────────────────────────────────────────────────────────────────────────
# Drift detection
# ─────────────────────────────────────────────────────────────────────────────
def test_detect_drift_reports_a_stale_lock_as_fail(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        venv_sync,
        "lock_check",
        lambda ws: venv_sync.CommandResult(
            argv=("uv", "lock", "--check"),
            returncode=1,
            stdout="",
            stderr="the lockfile is not up-to-date",
        ),
    )
    report = venv_sync.detect_drift(workspace, include_floor=False)
    assert report.status == "fail"
    assert "STALE" in report.summary


def test_detect_drift_reports_packages_behind_the_lock(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        venv_sync,
        "lock_check",
        lambda ws: venv_sync.CommandResult(("uv",), 0, "", ""),
    )
    monkeypatch.setattr(venv_sync, "plan_sync", lambda ws: SyncPlan.parse(BEHIND_PLAN))
    report = venv_sync.detect_drift(workspace, include_floor=False)
    assert report.status == "fail"
    codes = {f.code: f.severity for f in report.findings}
    assert codes["env_current"] == "fail"


def test_detect_drift_is_ok_when_everything_matches(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        venv_sync, "lock_check", lambda ws: venv_sync.CommandResult(("uv",), 0, "", "")
    )
    monkeypatch.setattr(venv_sync, "plan_sync", lambda ws: SyncPlan.parse(CLEAN_PLAN))
    report = venv_sync.detect_drift(workspace, include_floor=False)
    assert report.status == "ok"


def test_detect_drift_surfaces_a_stuck_flip_queue(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        venv_sync, "lock_check", lambda ws: venv_sync.CommandResult(("uv",), 0, "", "")
    )
    monkeypatch.setattr(venv_sync, "plan_sync", lambda ws: SyncPlan.parse(CLEAN_PLAN))
    venv_autosync.enqueue(
        workspace,
        venv_autosync.Intent(
            id="deferred-1",
            created_at="2026-07-31T00:00:00Z",
            repo=str(workspace.root / "pkgs" / "alpha"),
            branch="main",
            event="post-merge",
            change_class=METADATA,
        ),
    )
    report = venv_sync.detect_drift(workspace, include_floor=False)
    assert report.status == "warn"
    assert any(f.code == "pending_flips" for f in report.findings)
