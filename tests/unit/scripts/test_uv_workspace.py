"""Regression coverage for uv execution from XDG-hosted git worktrees."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "uv_workspace.py"
SPEC = importlib.util.spec_from_file_location("uv_workspace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
uv_workspace = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(uv_workspace)


@pytest.fixture
def workspace_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    canonical = workspace / "agent-packages" / "agent-utilities"
    sibling = workspace / "agent-packages" / "epistemic-graph"
    worktree = tmp_path / "xdg-state" / "repository-worktrees" / "agent-utilities"
    for project, name in (
        (canonical, "agent-utilities"),
        (sibling, "epistemic-graph"),
        (worktree, "agent-utilities"),
    ):
        project.mkdir(parents=True)
        (project / "pyproject.toml").write_text(
            f'[project]\nname = "{name}"\nversion = "1.0.0"\n',
            encoding="utf-8",
        )
    (workspace / "pyproject.toml").write_text(
        "\n".join(
            (
                '[project]\nname = "ecosystem"\nversion = "1.0.0"',
                "[tool.uv.workspace]",
                'members = ["agent-packages/*"]',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return workspace, canonical, worktree


def test_shadow_workspace_substitutes_only_current_worktree(
    tmp_path: Path,
    workspace_layout: tuple[Path, Path, Path],
) -> None:
    workspace, canonical, worktree = workspace_layout

    shadow = uv_workspace.shadow_workspace(
        worktree,
        canonical,
        workspace,
        state_root=tmp_path / "state",
    )

    assert not (shadow / "pyproject.toml").is_symlink()
    assert (shadow / "pyproject.toml").read_bytes() == (
        workspace / "pyproject.toml"
    ).read_bytes()
    assert not (shadow / "uv.lock").is_symlink()
    assert (shadow / "uv.lock").read_bytes() == (workspace / "uv.lock").read_bytes()
    assert (shadow / "agent-packages" / "agent-utilities").resolve() == worktree
    assert (
        shadow / "agent-packages" / "epistemic-graph"
    ).resolve() == workspace / "agent-packages" / "epistemic-graph"


def test_shadow_workspace_never_replaces_non_symlink(
    tmp_path: Path,
    workspace_layout: tuple[Path, Path, Path],
) -> None:
    workspace, canonical, worktree = workspace_layout
    state = tmp_path / "state"
    shadow = uv_workspace.shadow_workspace(
        worktree,
        canonical,
        workspace,
        state_root=state,
    )
    (shadow / uv_workspace._SHADOW_MARKER).unlink()

    with pytest.raises(RuntimeError, match="refusing to replace unmanaged shadow"):
        uv_workspace.shadow_workspace(
            worktree,
            canonical,
            workspace,
            state_root=state,
        )


def test_uv_plan_selects_worktree_package_and_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setenv("PYTHONPATH", "/would/mask/sibling/scripts")
    worktree = tmp_path / "worktree"
    shadow = tmp_path / "shadow"

    plan = uv_workspace.uv_plan(
        ["run", "--all-extras", "python", "-c", "pass"],
        worktree=worktree,
        shadow=shadow,
    )
    command, environment = list(plan.execute), plan.environment

    assert command[:8] == [
        "/usr/bin/uv",
        "--project",
        str(shadow),
        "run",
        "--no-sync",
        "--locked",
        "--package",
        "agent-utilities",
    ]
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(worktree / ".venv")
    assert environment["EPISTEMIC_GRAPH_NATIVE_ARTIFACT_CACHE"] == str(
        Path.home() / ".cache" / "epistemic-graph" / "native-artifacts" / "v1"
    )
    assert "PYTHONPATH" not in environment


def test_uv_plans_share_native_cache_across_hermetic_worktrees(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")
    user_home = tmp_path / "user-home"
    monkeypatch.setenv("HOME", str(user_home))

    environments: list[dict[str, str]] = []
    for identity in ("first", "second"):
        worktree = tmp_path / identity / "worktree"
        shadow = tmp_path / identity / "xdg-state" / "shadow"
        monkeypatch.setenv("XDG_CACHE_HOME", str(shadow / "cache"))
        monkeypatch.setenv(
            "EPISTEMIC_GRAPH_NATIVE_ARTIFACT_CACHE",
            str(worktree / "unsafe-inherited-cache"),
        )

        environments.append(
            uv_workspace.uv_plan(
                ["run", "python", "-c", "pass"],
                worktree=worktree,
                shadow=shadow,
            ).environment
        )

    expected = user_home / ".cache" / "epistemic-graph" / "native-artifacts" / "v1"
    assert {
        environment["EPISTEMIC_GRAPH_NATIVE_ARTIFACT_CACHE"]
        for environment in environments
    } == {str(expected)}
    assert all(
        not Path(environment["EPISTEMIC_GRAPH_NATIVE_ARTIFACT_CACHE"]).is_relative_to(
            tmp_path / identity
        )
        for identity, environment in zip(("first", "second"), environments, strict=True)
    )


def test_doctor_evidence_proves_worktree_and_lock_provenance(
    tmp_path: Path,
    workspace_layout: tuple[Path, Path, Path],
) -> None:
    workspace, canonical, worktree = workspace_layout
    shadow = uv_workspace.shadow_workspace(
        worktree,
        canonical,
        workspace,
        state_root=tmp_path / "state",
    )

    payload = uv_workspace.doctor_payload(
        worktree,
        canonical,
        workspace,
        shadow,
    )

    assert json.loads(json.dumps(payload))["status"] == "ok"
    assert payload["external_worktree"] is True
    assert payload["member_resolves_to_worktree"] is True
    assert payload["manifest_is_generated_copy"] is True
    assert payload["manifest_matches_canonical"] is True
    assert payload["lock_is_generated_copy"] is True
    assert payload["lock_matches_canonical"] is True


def test_shadow_lock_cannot_mutate_canonical_lock(
    tmp_path: Path,
    workspace_layout: tuple[Path, Path, Path],
) -> None:
    workspace, canonical, worktree = workspace_layout
    canonical_lock = workspace / "uv.lock"
    original = canonical_lock.read_bytes()
    shadow = uv_workspace.shadow_workspace(
        worktree,
        canonical,
        workspace,
        state_root=tmp_path / "state",
    )

    (shadow / "uv.lock").write_text("changed by synthetic uv\n", encoding="utf-8")

    assert canonical_lock.read_bytes() == original


def test_uv_execution_cannot_mutate_canonical_manifest_or_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_layout: tuple[Path, Path, Path],
) -> None:
    workspace, canonical, worktree = workspace_layout
    shadow = uv_workspace.shadow_workspace(
        worktree,
        canonical,
        workspace,
        state_root=tmp_path / "state",
    )
    canonical_manifest = workspace / "pyproject.toml"
    canonical_lock = workspace / "uv.lock"
    original_manifest = canonical_manifest.read_bytes()
    original_lock = canonical_lock.read_bytes()

    def mutate_generated_inputs(*_args: object, **_kwargs: object) -> SimpleNamespace:
        (shadow / "pyproject.toml").write_text("synthetic mutation\n", encoding="utf-8")
        (shadow / "uv.lock").write_text("synthetic mutation\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(uv_workspace.subprocess, "run", mutate_generated_inputs)

    with pytest.raises(RuntimeError, match="changed a lock-governed workspace input"):
        uv_workspace.run_uv(
            ["/usr/bin/uv", "lock", "--locked"],
            worktree=worktree,
            environment={},
            workspace=workspace,
            shadow=shadow,
        )

    assert canonical_manifest.read_bytes() == original_manifest
    assert canonical_lock.read_bytes() == original_lock


def test_lock_invocation_is_always_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")

    command = uv_workspace.uv_plan(
        ["lock"],
        worktree=tmp_path / "worktree",
        shadow=tmp_path / "shadow",
    ).execute

    assert list(command[-2:]) == ["lock", "--locked"]


def test_sync_invocation_is_always_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")

    command = uv_workspace.uv_plan(
        ["sync"],
        worktree=tmp_path / "worktree",
        shadow=tmp_path / "shadow",
    ).execute

    assert list(command[-4:]) == ["sync", "--locked", "--package", "agent-utilities"]


# ---------------------------------------------------------------------------
# Environment partitioning (D-VI-1). One worktree serves many concurrent
# invocations asking for DIFFERENT dependency selections; they used to share one
# `.venv` and rewrite it underneath each other.
# ---------------------------------------------------------------------------
def test_distinct_selections_never_share_an_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The collision itself: two selections, one worktree, two environments."""
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")
    worktree = tmp_path / "worktree"

    directories = {
        uv_workspace.uv_plan(
            arguments,
            worktree=worktree,
            shadow=tmp_path / "shadow",
        ).environment_path
        for arguments in (
            ["run", "--all-extras", "pytest"],
            ["run", "pytest"],
            ["run", "--extra", "graph", "pytest"],
            ["run", "--extra", "graph", "--extra", "owl", "pytest"],
        )
    }

    assert len(directories) == 4, "each dependency selection must own its environment"
    assert all(path.parent == worktree for path in directories)


def test_canonical_selection_keeps_the_conventional_venv_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`.venv/bin` is on the PATH built by bootstrap.sh and both CI workflows."""
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")
    worktree = tmp_path / "worktree"

    plan = uv_workspace.uv_plan(
        ["run", "--all-extras", "pytest"],
        worktree=worktree,
        shadow=tmp_path / "shadow",
    )

    assert plan.environment_path == worktree / ".venv"


def test_selection_identity_ignores_order_and_repetition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Equivalent requests must not fork the environment and double the disk."""
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")
    worktree = tmp_path / "worktree"

    directories = {
        uv_workspace.uv_plan(
            arguments,
            worktree=worktree,
            shadow=tmp_path / "shadow",
        ).environment_path
        for arguments in (
            ["run", "--extra", "graph", "--extra", "owl", "pytest"],
            ["run", "--extra", "owl", "--extra", "graph", "pytest"],
            ["run", "--extra=owl", "--extra", "graph", "--extra", "owl", "pytest"],
            ["run", "--locked", "--extra", "graph", "--extra", "owl", "pytest"],
        )
    }

    assert len(directories) == 1


def test_run_synchronises_before_exec_and_never_syncs_during_the_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """uv frees the environment lock BEFORE exec, so the child must not resync."""
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")

    plan = uv_workspace.uv_plan(
        ["run", "--all-extras", "pytest", "tests", "-q"],
        worktree=tmp_path / "worktree",
        shadow=tmp_path / "shadow",
    )

    assert len(plan.prepare) == 1
    assert "sync" in plan.prepare[0]
    assert "--locked" in plan.prepare[0]
    assert "--all-extras" in plan.prepare[0]
    assert "--no-sync" in plan.execute, (
        "the child must run against a frozen environment"
    )


def test_unrecognized_flag_degrades_instead_of_guessing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An identity we cannot derive falls back to uv's own behaviour, loudly."""
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")

    plan = uv_workspace.uv_plan(
        ["run", "--some-future-uv-flag", "pytest"],
        worktree=tmp_path / "worktree",
        shadow=tmp_path / "shadow",
    )

    assert plan.selection_recognized is False
    assert plan.prepare == ()
    assert "--no-sync" not in plan.execute


def test_failed_preparation_never_execs_the_child(tmp_path: Path) -> None:
    """A half-built environment must not be handed to a test run."""
    workspace = tmp_path / "workspace"
    shadow = tmp_path / "shadow"
    worktree = tmp_path / "worktree"
    for directory in (workspace, shadow, worktree):
        directory.mkdir(parents=True)
    for directory in (workspace, shadow):
        (directory / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (directory / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    returncode = uv_workspace.run_uv(
        [sys.executable, "-c", "raise SystemExit('the child must never run')"],
        worktree=worktree,
        environment=dict(os.environ),
        workspace=workspace,
        shadow=shadow,
        prepare=[[sys.executable, "-c", "raise SystemExit(3)"]],
    )

    assert returncode == 3


def test_partitioned_environment_is_a_classified_resource() -> None:
    """An unclassified shared resource is exactly how this defect survived."""
    from agent_utilities.governance import lanes

    assert (
        lanes.resource_class("uv-project-environment")
        is lanes.ArbitrationClass.PARTITION
    )


def test_concurrent_shadow_refresh_never_collides(
    tmp_path: Path,
    workspace_layout: tuple[Path, Path, Path],
) -> None:
    """The shadow is keyed by worktree, so concurrent invocations refresh it at once.

    A fixed staging filename made them race: one invocation's rename consumed the
    file another had just written and the second died with FileNotFoundError
    before uv ever started — which is how the acceptance run for the environment
    partition first failed.
    """
    workspace, canonical, worktree = workspace_layout
    state_root = tmp_path / "state"
    errors: list[BaseException] = []
    start = threading.Barrier(8)

    def refresh() -> None:
        start.wait()
        for _ in range(25):
            try:
                uv_workspace.shadow_workspace(
                    worktree,
                    canonical,
                    workspace,
                    state_root=state_root,
                )
            except BaseException as exc:  # noqa: BLE001 - recorded, then asserted
                errors.append(exc)
                return

    threads = [threading.Thread(target=refresh) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"concurrent shadow refresh collided: {errors[:3]}"
    shadow = uv_workspace.shadow_workspace(
        worktree, canonical, workspace, state_root=state_root
    )
    assert (shadow / "uv.lock").read_bytes() == (workspace / "uv.lock").read_bytes()
    assert not list(shadow.glob(".*.tmp")), "staging files must never be left behind"


# ---------------------------------------------------------------------------
# D-ORC-33. The venv is already partitioned per worktree+selection (above), but
# every worktree's `uv sync` still contends for the SAME shared `~/.cache/uv`
# and the same /home spindle. At the peak of a 13-20 lane wave, load average
# hit ~26 on 24 cores and swap was 100% exhausted; one lane's `uv sync` stalled
# at 0% CPU for 17+ minutes. An exclusive lease would defeat the partitioning
# above (20 lanes can each legitimately need a sync at once); a small capped
# pool bounds the disk/cache contention without serialising to one lane.
# ---------------------------------------------------------------------------
def test_dependency_sync_slot_caps_concurrent_holders_at_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Never more than `_DEPENDENCY_SYNC_POOL_CAPACITY` slots held at once, and
    every waiter eventually gets in (no deadlock, no lost wakeups)."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    capacity = uv_workspace._DEPENDENCY_SYNC_POOL_CAPACITY
    contenders = capacity * 3
    lock = threading.Lock()
    current = 0
    peak = 0
    entries = 0
    start = threading.Barrier(contenders)

    def contend() -> None:
        nonlocal current, peak, entries
        start.wait()
        with uv_workspace._dependency_sync_slot():
            with lock:
                current += 1
                peak = max(peak, current)
                entries += 1
            threading.Event().wait(0.02)
            with lock:
                current -= 1

    threads = [threading.Thread(target=contend) for _ in range(contenders)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), (
        "a waiter never got a slot -- the pool deadlocked"
    )
    assert entries == contenders, "every contender must eventually acquire a slot"
    assert peak <= capacity, f"observed {peak} concurrent holders, cap is {capacity}"
    assert peak == capacity, (
        "the test is vacuous unless contention actually saturated the pool"
    )


def test_uv_plan_marks_bare_sync_and_lock_as_heavy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`sync`/`lock` have no separate `prepare` step -- `execute` IS the sync,
    so it must be pool-gated directly or D-ORC-33's contention is unguarded."""
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")

    for arguments in (["sync"], ["lock"]):
        plan = uv_workspace.uv_plan(
            arguments, worktree=tmp_path / "worktree", shadow=tmp_path / "shadow"
        )
        assert plan.prepare == ()
        assert plan.execute_is_heavy_sync is True, arguments


def test_uv_plan_marks_recognized_run_not_heavy_at_execute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A recognised `run` pool-gates its `prepare` sync step (see run_uv()), but
    its `execute` is `run --no-sync` -- not itself heavy, so gating it too would
    cap ordinary test/build execution, not just disk/cache contention."""
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")

    plan = uv_workspace.uv_plan(
        ["run", "--all-extras", "pytest"],
        worktree=tmp_path / "worktree",
        shadow=tmp_path / "shadow",
    )
    assert len(plan.prepare) == 1
    assert plan.execute_is_heavy_sync is False


def test_uv_plan_marks_unrecognized_run_as_heavy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unrecognised `run` has no `prepare` step and falls through to a plain
    `uv run` with no `--no-sync` -- uv performs its own implicit sync as a side
    effect, so `execute` must be pool-gated here too."""
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")

    plan = uv_workspace.uv_plan(
        ["run", "--some-future-uv-flag", "pytest"],
        worktree=tmp_path / "worktree",
        shadow=tmp_path / "shadow",
    )
    assert plan.prepare == ()
    assert plan.execute_is_heavy_sync is True


def test_run_uv_pool_gates_the_prepare_sync_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pins the D-ORC-33 wiring itself: `run_uv()` must acquire a pool slot for
    every `prepare` step. Proven against the restored bug: reverting the
    `with _dependency_sync_slot():` wrap around the `prepare` loop in
    `run_uv()` drops the recorded call count to 0 and this assertion fails."""
    calls: list[str] = []
    from contextlib import contextmanager

    @contextmanager
    def spy():
        calls.append("acquired")
        yield

    monkeypatch.setattr(uv_workspace, "_dependency_sync_slot", spy)
    workspace = tmp_path / "workspace"
    shadow = tmp_path / "shadow"
    worktree = tmp_path / "worktree"
    for directory in (workspace, shadow, worktree):
        directory.mkdir(parents=True)
    for directory in (workspace, shadow):
        (directory / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (directory / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    returncode = uv_workspace.run_uv(
        [sys.executable, "-c", "pass"],
        worktree=worktree,
        environment=dict(os.environ),
        workspace=workspace,
        shadow=shadow,
        prepare=[[sys.executable, "-c", "pass"]],
    )

    assert returncode == 0
    assert calls == ["acquired"], "the prepare (sync) step must be pool-gated"


def test_run_uv_pool_gates_a_heavy_execute_with_no_prepare_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pins the OTHER half: a bare `sync`/`lock`-shaped invocation has no
    `prepare` step, so `execute_is_heavy_sync=True` must gate `execute`
    directly. Proven against the restored bug the same way as above."""
    calls: list[str] = []
    from contextlib import contextmanager

    @contextmanager
    def spy():
        calls.append("acquired")
        yield

    monkeypatch.setattr(uv_workspace, "_dependency_sync_slot", spy)
    workspace = tmp_path / "workspace"
    shadow = tmp_path / "shadow"
    worktree = tmp_path / "worktree"
    for directory in (workspace, shadow, worktree):
        directory.mkdir(parents=True)
    for directory in (workspace, shadow):
        (directory / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (directory / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    returncode = uv_workspace.run_uv(
        [sys.executable, "-c", "pass"],
        worktree=worktree,
        environment=dict(os.environ),
        workspace=workspace,
        shadow=shadow,
        prepare=(),
        execute_is_heavy_sync=True,
    )

    assert returncode == 0
    assert calls == ["acquired"], "a heavy execute with no prepare must be pool-gated"


def test_run_uv_does_not_pool_gate_an_ordinary_test_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The actual test/build run (`run --no-sync ...`) must stay OUTSIDE the
    pool -- gating it too would cap general lane parallelism, not just the
    disk/cache contention D-ORC-33 measured."""
    calls: list[str] = []
    from contextlib import contextmanager

    @contextmanager
    def spy():
        calls.append("acquired")
        yield

    monkeypatch.setattr(uv_workspace, "_dependency_sync_slot", spy)
    workspace = tmp_path / "workspace"
    shadow = tmp_path / "shadow"
    worktree = tmp_path / "worktree"
    for directory in (workspace, shadow, worktree):
        directory.mkdir(parents=True)
    for directory in (workspace, shadow):
        (directory / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (directory / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    returncode = uv_workspace.run_uv(
        [sys.executable, "-c", "pass"],
        worktree=worktree,
        environment=dict(os.environ),
        workspace=workspace,
        shadow=shadow,
        prepare=(),
        execute_is_heavy_sync=False,
    )

    assert returncode == 0
    assert calls == [], "a non-heavy execute must not wait on the sync pool"


# ---------------------------------------------------------------------------
# D-W2T-3: a second `uv_workspace.py run` invocation against the SAME
# environment, launched while a first invocation's long-running child was
# still executing, used to trigger a second concurrent `uv sync` that
# corrupted the environment out from under the running child (measured:
# `.venv/bin/pytest` briefly missing mid-run). `_acquire_environment_activity`
# is the readers-writer flock that closes this specific race.
# ---------------------------------------------------------------------------
def test_environment_activity_lets_only_one_sync_win_while_a_reader_is_active(
    tmp_path: Path,
) -> None:
    """A live reader (an exec'd child) must make a concurrent writer lose."""
    environment_path = tmp_path / ".venv"
    environment_path.mkdir()

    handle_a, won_a = uv_workspace._acquire_environment_activity(
        environment_path, want_sync=True
    )
    assert won_a is True, "the first, uncontended caller must win the sync"
    uv_workspace._downgrade_environment_activity(handle_a)  # A is now a reader

    results: dict[str, bool] = {}

    def contend() -> None:
        handle_b, won_b = uv_workspace._acquire_environment_activity(
            environment_path, want_sync=True
        )
        results["b_won"] = won_b
        uv_workspace._release_environment_activity(handle_b)

    thread = threading.Thread(target=contend)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert results["b_won"] is False, (
        "B must lose the sync race while A is still reading -- winning here is "
        "exactly the D-W2T-3 corruption window (a second `uv sync` against an "
        "environment a sibling's child is actively running in)"
    )

    uv_workspace._release_environment_activity(handle_a)

    # Once A releases, the environment is free again and a fresh sync can win.
    handle_c, won_c = uv_workspace._acquire_environment_activity(
        environment_path, want_sync=True
    )
    assert won_c is True
    uv_workspace._release_environment_activity(handle_c)


def test_environment_activity_readers_never_block_each_other(tmp_path: Path) -> None:
    """Two invocations that both skip syncing (or already synced) may read
    the SAME environment concurrently -- this is not an exclusive lock for
    ordinary concurrent test runs, only for a sync racing a reader."""
    environment_path = tmp_path / ".venv"
    environment_path.mkdir()

    handle_a, _ = uv_workspace._acquire_environment_activity(
        environment_path, want_sync=False
    )
    results: dict[str, bool] = {}

    def contend() -> None:
        handle_b, won_b = uv_workspace._acquire_environment_activity(
            environment_path, want_sync=False
        )
        results["b_acquired"] = True
        uv_workspace._release_environment_activity(handle_b)

    thread = threading.Thread(target=contend)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "a second reader must not block behind the first"
    assert results.get("b_acquired") is True
    uv_workspace._release_environment_activity(handle_a)


def test_run_uv_never_overlaps_two_syncs_against_the_same_environment(
    tmp_path: Path,
) -> None:
    """End-to-end proof at the `run_uv()` level, using real subprocesses.

    Invocation A's `prepare` step is a real (slow) command that stamps
    SYNC-START/SYNC-END around a sleep; invocation B starts mid-way through
    A's prepare/exec and shares the SAME `environment_path`. Proven against
    the restored bug: with `_acquire_environment_activity` replaced by a
    no-op that always reports a win (the pre-fix shape -- no coordination at
    all), the same test observes an OVERLAPPING SYNC-START before the first
    SYNC-END, reproducing the corruption window this fix closes.
    """
    workspace = tmp_path / "workspace"
    shadow = tmp_path / "shadow"
    worktree = tmp_path / "worktree"
    for directory in (workspace, shadow, worktree):
        directory.mkdir(parents=True)
    for directory in (workspace, shadow):
        (directory / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (directory / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    environment_path = worktree / ".venv"
    environment_path.mkdir()
    sync_log = tmp_path / "sync_log.txt"

    prepare_cmd = [
        sys.executable,
        "-c",
        "import time, pathlib, sys; p = pathlib.Path(sys.argv[1]); "
        "f = p.open('a'); f.write('SYNC-START\\n'); f.close(); "
        "time.sleep(0.4); "
        "f = p.open('a'); f.write('SYNC-END\\n'); f.close()",
        str(sync_log),
    ]
    long_child_cmd = [sys.executable, "-c", "import time; time.sleep(1.0)"]

    results: dict[str, int] = {}

    def invocation_a() -> None:
        results["a_rc"] = uv_workspace.run_uv(
            long_child_cmd,
            worktree=worktree,
            environment=dict(os.environ),
            workspace=workspace,
            shadow=shadow,
            prepare=[prepare_cmd],
            environment_path=environment_path,
        )

    def invocation_b() -> None:
        threading.Event().wait(0.15)  # start while A is still mid-sync/reading
        results["b_rc"] = uv_workspace.run_uv(
            [sys.executable, "-c", "pass"],
            worktree=worktree,
            environment=dict(os.environ),
            workspace=workspace,
            shadow=shadow,
            prepare=[prepare_cmd],
            environment_path=environment_path,
        )

    thread_a = threading.Thread(target=invocation_a)
    thread_b = threading.Thread(target=invocation_b)
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=15)
    thread_b.join(timeout=15)
    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert results.get("a_rc") == 0
    assert results.get("b_rc") == 0

    lines = sync_log.read_text().splitlines()
    depth = 0
    overlapping = False
    for line in lines:
        if line == "SYNC-START":
            if depth > 0:
                overlapping = True
            depth += 1
        elif line == "SYNC-END":
            depth -= 1
    assert not overlapping, (
        "two `uv sync`-shaped prepare steps overlapped against the SAME "
        "environment -- this is the D-W2T-3 corruption window"
    )


# ---------------------------------------------------------------------------
# Interpreter guard (D-SP-4). `uv run pytest` does not fail when the environment
# lacks pytest — it falls through to the system one and runs the project's tests
# under /usr/bin/python against system site-packages.
# ---------------------------------------------------------------------------
def _console_script(directory: Path, name: str, interpreter: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / name
    script.write_text(f"#!{interpreter}\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def test_python_tool_outside_the_environment_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The exact shape that reported fastmcp 3.3.1 from /usr/bin/python."""
    system_bin = tmp_path / "usr" / "bin"
    system_pytest = _console_script(system_bin, "pytest", "/usr/bin/python")
    monkeypatch.setenv("PATH", str(system_bin))
    environment = tmp_path / "worktree" / ".venv-base"
    (environment / "bin").mkdir(parents=True)

    foreign = uv_workspace.foreign_python_console_script("pytest", environment)

    assert foreign == system_pytest


def test_tool_installed_in_the_environment_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    system_bin = tmp_path / "usr" / "bin"
    _console_script(system_bin, "pytest", "/usr/bin/python")
    monkeypatch.setenv("PATH", str(system_bin))
    environment = tmp_path / "worktree" / ".venv"
    _console_script(environment / "bin", "pytest", str(environment / "bin" / "python"))

    assert uv_workspace.foreign_python_console_script("pytest", environment) is None


def test_non_python_command_outside_the_environment_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """`bash` and `find` legitimately live outside the environment."""
    system_bin = tmp_path / "usr" / "bin"
    binary = system_bin / "bash"
    system_bin.mkdir(parents=True)
    binary.write_bytes(b"\x7fELF not a script")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(system_bin))
    environment = tmp_path / "worktree" / ".venv-base"
    (environment / "bin").mkdir(parents=True)

    assert uv_workspace.foreign_python_console_script("bash", environment) is None


def test_run_refuses_the_foreign_interpreter_before_executing_anything(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_layout: tuple[Path, Path, Path],
) -> None:
    """The guard must fire between preparation and exec, so nothing runs wrong."""
    workspace, _canonical, worktree = workspace_layout
    shadow = tmp_path / "shadow"
    shadow.mkdir(parents=True)
    for directory in (workspace, shadow):
        (directory / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (shadow / "pyproject.toml").write_text("[project]\n", encoding="utf-8")

    system_bin = tmp_path / "usr" / "bin"
    _console_script(system_bin, "pytest", "/usr/bin/python")
    monkeypatch.setenv("PATH", str(system_bin))
    environment = worktree / ".venv-base"
    (environment / "bin").mkdir(parents=True)
    executed: list[list[str]] = []

    def record(command: list[str], **_kwargs: object) -> SimpleNamespace:
        executed.append(list(command))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(uv_workspace.subprocess, "run", record)

    with pytest.raises(RuntimeError, match="refusing to run 'pytest'"):
        uv_workspace.run_uv(
            ["uv", "run", "pytest"],
            worktree=worktree,
            environment={},
            workspace=workspace,
            shadow=shadow,
            prepare=[["uv", "sync"]],
            environment_path=environment,
            command_name="pytest",
        )

    assert executed == [["uv", "sync"]], "preparation only; the child never ran"


def test_plan_identifies_the_command_it_must_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")

    for arguments, expected in (
        (["run", "pytest", "-q"], "pytest"),
        (["run", "--all-extras", "pytest", "tests"], "pytest"),
        (["run", "--extra", "graph", "ruff", "check"], "ruff"),
        (["run", "python", "-m", "pytest"], "python"),
        (["run", "--some-future-uv-flag", "pytest"], None),
    ):
        plan = uv_workspace.uv_plan(
            arguments,
            worktree=tmp_path / "worktree",
            shadow=tmp_path / "shadow",
        )
        assert plan.command_name == expected, arguments
