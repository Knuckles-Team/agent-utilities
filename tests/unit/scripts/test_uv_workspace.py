"""Regression coverage for uv execution from XDG-hosted git worktrees."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
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


def test_uv_invocation_selects_worktree_package_and_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")
    monkeypatch.setenv("PYTHONPATH", "/would/mask/sibling/scripts")
    worktree = tmp_path / "worktree"
    shadow = tmp_path / "shadow"

    command, environment = uv_workspace.uv_invocation(
        ["run", "--all-extras", "python", "-c", "pass"],
        worktree=worktree,
        shadow=shadow,
    )

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


def test_uv_invocations_share_native_cache_across_hermetic_worktrees(
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

        _command, environment = uv_workspace.uv_invocation(
            ["run", "python", "-c", "pass"],
            worktree=worktree,
            shadow=shadow,
        )
        environments.append(environment)

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

    command, _environment = uv_workspace.uv_invocation(
        ["lock"],
        worktree=tmp_path / "worktree",
        shadow=tmp_path / "shadow",
    )

    assert command[-2:] == ["lock", "--locked"]


def test_sync_invocation_is_always_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(uv_workspace.shutil, "which", lambda _name: "/usr/bin/uv")

    command, _environment = uv_workspace.uv_invocation(
        ["sync"],
        worktree=tmp_path / "worktree",
        shadow=tmp_path / "shadow",
    )

    assert command[-4:] == ["sync", "--locked", "--package", "agent-utilities"]


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
    assert "--no-sync" in plan.execute, "the child must run against a frozen environment"


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
        lanes.resource_class("uv-project-environment") is lanes.ArbitrationClass.PARTITION
    )
