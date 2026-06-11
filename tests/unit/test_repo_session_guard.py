"""Regression guard for the conftest repo-mutation session guard.

History: the session guard used to run ``git checkout -- <strays>`` and
``os.remove`` against the LIVE repository working tree at sessionfinish,
wiping any uncommitted edits a developer or agent made *while* the suite ran
(mid-run edits are absent from the session-start snapshot, so they were
indistinguishable from test pollution). These tests pin the safe contract:
report-only by default, destructive cleanup only on explicit opt-in.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _guard_module():
    """Find the loaded tests/conftest.py module regardless of its import name.

    Matched by ``__file__`` (not ``hasattr``): the conftest installs MagicMock
    stand-ins for absent heavy deps, and mocks claim every attribute exists.
    """
    conftest_path = str(Path(__file__).resolve().parents[1] / "conftest.py")
    for mod in list(sys.modules.values()):
        if getattr(mod, "__file__", None) == conftest_path:
            return mod
    pytest.skip("session-guard conftest not loaded")


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "guard-repo"
    repo.mkdir()
    run = lambda *args: subprocess.run(  # noqa: E731
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    )
    run("init", "-q")
    run("config", "user.email", "guard@test.local")
    run("config", "user.name", "Guard Test")
    (repo / "tracked.txt").write_text("original\n")
    run("add", "tracked.txt")
    run("commit", "-q", "-m", "seed")
    return repo


class TestSessionGuardIsNonDestructiveByDefault:
    def test_default_mode_is_warn(self, monkeypatch):
        guard = _guard_module()
        monkeypatch.delenv("AGENT_UTILITIES_TEST_REPO_GUARD", raising=False)
        assert guard._au_guard_mode() == "warn"

    def test_unknown_mode_falls_back_to_warn(self, monkeypatch):
        guard = _guard_module()
        monkeypatch.setenv("AGENT_UTILITIES_TEST_REPO_GUARD", "nuke-it-all")
        assert guard._au_guard_mode() == "warn"

    def test_warn_mode_leaves_concurrent_edits_intact(self, tmp_path, capsys):
        guard = _guard_module()
        repo = _make_repo(tmp_path)
        (repo / "tracked.txt").write_text("agent edit mid-run\n")
        (repo / "SENTINEL_UNTRACKED.txt").write_text("agent work\n")

        guard._au_enforce_session_guard(
            ["tracked.txt"],
            ["SENTINEL_UNTRACKED.txt"],
            repo_root=str(repo),
            mode="warn",
        )

        assert (repo / "tracked.txt").read_text() == "agent edit mid-run\n"
        assert (repo / "SENTINEL_UNTRACKED.txt").exists()
        err = capsys.readouterr().err
        assert "tracked.txt" in err and "SENTINEL_UNTRACKED.txt" in err

    def test_off_mode_is_silent_and_intact(self, tmp_path, capsys):
        guard = _guard_module()
        repo = _make_repo(tmp_path)
        (repo / "tracked.txt").write_text("edit\n")

        guard._au_enforce_session_guard(
            ["tracked.txt"], [], repo_root=str(repo), mode="off"
        )

        assert (repo / "tracked.txt").read_text() == "edit\n"
        assert capsys.readouterr().err == ""


class TestSessionGuardRevertOptIn:
    def test_revert_mode_restores_tree(self, tmp_path):
        guard = _guard_module()
        repo = _make_repo(tmp_path)
        (repo / "tracked.txt").write_text("test pollution\n")
        (repo / "stray.txt").write_text("test pollution\n")

        guard._au_enforce_session_guard(
            ["tracked.txt"], ["stray.txt"], repo_root=str(repo), mode="revert"
        )

        assert (repo / "tracked.txt").read_text() == "original\n"
        assert not (repo / "stray.txt").exists()
