"""D-OB-12: the ``--all-files`` unstaged-work safeguard actually protects and
actually detects a drop — not just documentation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def _module():
    source = Path(__file__).parents[3] / "scripts" / "safe_precommit_all_files.py"
    spec = importlib.util.spec_from_file_location("safe_precommit_all_files", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "docs").mkdir()
    (root / "docs" / "concept_reservations.yaml").write_text(
        "# ledger\n", encoding="utf-8"
    )
    (root / "tracked.txt").write_text("original\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    return root


def test_no_unstaged_changes_skips_backup(tmp_path: Path, capsys) -> None:
    module = _module()
    root = _repo(tmp_path)
    module._run_precommit = lambda root, argv: 0  # noqa: SLF001 - stub
    status = module.main([], cwd=root)
    assert status == 0
    out = capsys.readouterr().out
    assert "nothing to protect" in out
    assert not (root / ".git" / "precommit-all-files-backups").exists()


def test_unstaged_changes_are_backed_up_and_verified_intact(
    tmp_path: Path, capsys
) -> None:
    module = _module()
    root = _repo(tmp_path)
    (root / "tracked.txt").write_text("original\nedited\n", encoding="utf-8")

    module._run_precommit = lambda root, argv: 0  # noqa: SLF001 - real hooks not exercised
    status = module.main([], cwd=root)
    assert status == 0

    backups = list((root / ".git" / "precommit-all-files-backups").glob("*.patch"))
    assert len(backups) == 1
    out = capsys.readouterr().out
    assert "backed up unstaged changes" in out
    assert "survived the run intact" in out
    # Nothing was actually touched — the file still carries the edit.
    assert (root / "tracked.txt").read_text(encoding="utf-8") == "original\nedited\n"


def test_dirty_concept_reservations_triggers_the_named_warning(
    tmp_path: Path, capsys
) -> None:
    module = _module()
    root = _repo(tmp_path)
    (root / "docs" / "concept_reservations.yaml").write_text(
        "# ledger\n- {id: X}\n", encoding="utf-8"
    )

    module._run_precommit = lambda root, argv: 0  # noqa: SLF001
    module.main([], cwd=root)
    out = capsys.readouterr().out
    assert "docs/concept_reservations.yaml" in out
    assert "D-OB-12" in out


def test_a_dropped_unstaged_change_is_detected_after_the_run(
    tmp_path: Path, capsys
) -> None:
    """Simulate exactly the D-OB-12 failure mode: a "hook" silently reverts
    the unstaged edit back to the committed content during the run."""
    module = _module()
    root = _repo(tmp_path)
    (root / "tracked.txt").write_text("original\nedited\n", encoding="utf-8")

    def _revert_during_run(root: Path, argv: list[str]) -> int:
        (root / "tracked.txt").write_text("original\n", encoding="utf-8")
        return 0

    module._run_precommit = _revert_during_run  # noqa: SLF001
    status = module.main([], cwd=root)
    assert status == 0  # pre-commit's own exit code is still surfaced faithfully
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "git apply --3way" in out


def test_precommit_exit_status_is_surfaced(tmp_path: Path) -> None:
    module = _module()
    root = _repo(tmp_path)
    module._run_precommit = lambda root, argv: 1  # noqa: SLF001
    assert module.main([], cwd=root) == 1


@pytest.mark.parametrize("has_diff", [True, False])
def test_diff_still_applies_matches_working_tree_state(
    tmp_path: Path, has_diff: bool
) -> None:
    module = _module()
    root = _repo(tmp_path)
    (root / "tracked.txt").write_text("original\nedited\n", encoding="utf-8")
    diff = module._unstaged_diff(root)  # noqa: SLF001
    backup = root / "backup.patch"
    backup.write_text(diff, encoding="utf-8")
    if not has_diff:
        (root / "tracked.txt").write_text("original\n", encoding="utf-8")
    assert module._diff_still_applies(root, backup) is has_diff  # noqa: SLF001


# ---------------------------------------------------------------------------
# D-CDX-49: the wrapper must reach hooks in BOTH the canonical checkout shape
# (``.git`` is a directory) and a linked worktree shape (``.git`` is a
# gitdir-pointer FILE) — the prior ``root / ".git" / "..."`` backup path
# raised ``NotADirectoryError`` before any hook ran in every linked worktree.
# ---------------------------------------------------------------------------


def _linked_worktree(root: Path, tmp_path: Path, branch: str = "lane") -> Path:
    wt = tmp_path / f"worktree-{branch}"
    _git(root, "worktree", "add", str(wt), "-b", branch)
    return wt


def test_git_dir_is_a_directory_in_the_canonical_checkout(tmp_path: Path) -> None:
    module = _module()
    root = _repo(tmp_path)
    git_dir = module._git_dir(root)  # noqa: SLF001
    assert git_dir == (root / ".git").resolve()
    assert git_dir.is_dir()


def test_git_dir_resolves_the_per_worktree_admin_dir_in_a_linked_worktree(
    tmp_path: Path,
) -> None:
    module = _module()
    root = _repo(tmp_path)
    wt = _linked_worktree(root, tmp_path)
    # The defining shape of a linked worktree: ``.git`` under it is a FILE,
    # never a directory.
    assert (wt / ".git").is_file()
    git_dir = module._git_dir(wt)  # noqa: SLF001
    assert git_dir.is_dir()
    assert git_dir != (wt / ".git")
    assert str(git_dir).startswith(str((root / ".git" / "worktrees").resolve()))


def test_wrapper_backs_up_and_reaches_hooks_in_a_linked_worktree_with_unstaged_changes(
    tmp_path: Path, capsys
) -> None:
    """The literal D-CDX-49 reproduction: an unstaged change in a linked
    worktree used to raise ``NotADirectoryError`` before ``pre-commit`` (the
    hooks) ever ran. Prove the wrapper now reaches them."""
    module = _module()
    root = _repo(tmp_path)
    wt = _linked_worktree(root, tmp_path)
    (wt / "tracked.txt").write_text("original\nedited-in-worktree\n", encoding="utf-8")

    reached_hooks = {"called": False}

    def _record_and_succeed(root: Path, argv: list[str]) -> int:
        reached_hooks["called"] = True
        return 0

    module._run_precommit = _record_and_succeed  # noqa: SLF001
    status = module.main([], cwd=wt)

    assert status == 0
    assert reached_hooks["called"], "wrapper must reach hooks, not raise first"
    backups = list(
        (module._git_dir(wt) / "precommit-all-files-backups").glob("*.patch")  # noqa: SLF001
    )
    assert len(backups) == 1
    out = capsys.readouterr().out
    assert "backed up unstaged changes" in out
    assert "survived the run intact" in out
    assert (wt / "tracked.txt").read_text(encoding="utf-8") == (
        "original\nedited-in-worktree\n"
    )


def test_linked_worktree_backup_is_private_to_that_worktree(tmp_path: Path) -> None:
    """Two lanes in two linked worktrees of the same repo must not share (or
    collide on) a backup directory — each worktree gets its own admin dir."""
    module = _module()
    root = _repo(tmp_path)
    wt_a = _linked_worktree(root, tmp_path, branch="lane-a")
    wt_b = _linked_worktree(root, tmp_path, branch="lane-b")
    assert module._git_dir(wt_a) != module._git_dir(wt_b)  # noqa: SLF001

    (wt_a / "tracked.txt").write_text("original\nfrom-a\n", encoding="utf-8")
    (wt_b / "tracked.txt").write_text("original\nfrom-b\n", encoding="utf-8")
    module._run_precommit = lambda root, argv: 0  # noqa: SLF001
    assert module.main([], cwd=wt_a) == 0
    assert module.main([], cwd=wt_b) == 0

    backups_a = list((module._git_dir(wt_a) / "precommit-all-files-backups").glob("*"))  # noqa: SLF001
    backups_b = list((module._git_dir(wt_b) / "precommit-all-files-backups").glob("*"))  # noqa: SLF001
    assert len(backups_a) == 1
    assert len(backups_b) == 1
