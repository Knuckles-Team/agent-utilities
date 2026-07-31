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
