"""Capability E proof: merged-tree helper (incident 2).

Incident: `git diff main..HEAD` was used to ask "does this branch delete
X?" A two-dot diff against a moving `main` shows every commit main gained
since divergence as a "deletion" from the branch's point of view. This
constructs the exact shape: `main` gains a file AFTER a feature branch
diverged; the feature branch never touches that file. The naive two-dot
diff reports it as deleted; `files_deleted_by_merge` (via `git merge-tree
--write-tree`) must correctly report no deletion.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_utilities.measurement.merged_tree import (
    files_deleted_by_merge,
    merged_tree,
    naive_two_dot_diff_deletions,
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "test@example.com")
    _git(r, "config", "user.name", "test")

    (r / "A.txt").write_text("a\n")
    _git(r, "add", "A.txt")
    _git(r, "commit", "-q", "-m", "init")

    # feature branch diverges HERE, never sees B.txt
    _git(r, "branch", "feature")

    # main moves forward with a NEW file — this is "other lanes' work
    # landing on main after the branch point".
    (r / "B.txt").write_text("b\n")
    _git(r, "add", "B.txt")
    _git(r, "commit", "-q", "-m", "main gains B.txt")

    # feature branch makes its own, unrelated change (never touches B.txt)
    _git(r, "checkout", "-q", "feature")
    (r / "C.txt").write_text("c\n")
    _git(r, "add", "C.txt")
    _git(r, "commit", "-q", "-m", "feature adds C.txt")
    _git(r, "checkout", "-q", "main")

    return r


def test_naive_two_dot_diff_falsely_reports_a_deletion(repo: Path):
    """Reproduce the false-alarm instrument first, to prove the incident is real."""
    naive = naive_two_dot_diff_deletions(repo, "main", "feature")
    assert "B.txt" in naive, (
        "the incident's own instrument (two-dot diff) must show B.txt as "
        "'deleted' by feature — that's the false alarm being guarded against"
    )


def test_merged_tree_does_not_delete_the_file_the_naive_diff_flagged(repo: Path):
    """The correct instrument: merging feature into main keeps B.txt."""
    real_deletions = files_deleted_by_merge(repo, "main", "feature")
    assert "B.txt" not in real_deletions, (
        "a real merge of feature into main does NOT delete B.txt — the "
        "naive two-dot diff was wrong, and files_deleted_by_merge must not "
        "repeat its mistake"
    )
    assert real_deletions == set()


def test_merged_tree_contains_files_from_both_sides(repo: Path):
    result = merged_tree(repo, "main", "feature")
    assert not result.had_conflicts
    ls = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", result.tree_oid],
        capture_output=True,
        text=True,
        check=True,
    )
    files = set(ls.stdout.split())
    assert files == {"A.txt", "B.txt", "C.txt"}


def test_files_deleted_by_merge_catches_a_real_deletion(repo: Path):
    """Sanity check the other direction: a branch that DOES delete a file
    must still be caught, so this isn't just always returning empty."""
    _git(repo, "checkout", "-q", "-b", "deleter", "feature")
    (repo / "A.txt").unlink()
    _git(repo, "add", "A.txt")
    _git(repo, "commit", "-q", "-m", "actually delete A.txt")

    real_deletions = files_deleted_by_merge(repo, "main", "deleter")
    assert real_deletions == {"A.txt"}
