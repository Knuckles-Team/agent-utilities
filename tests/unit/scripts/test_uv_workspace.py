"""Regression coverage for uv execution from XDG-hosted git worktrees."""

from __future__ import annotations

import importlib.util
import json
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

    assert command[:7] == [
        "/usr/bin/uv",
        "--project",
        str(shadow),
        "run",
        "--locked",
        "--package",
        "agent-utilities",
    ]
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(worktree / ".venv")
    assert "PYTHONPATH" not in environment


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
