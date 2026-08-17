"""Regression tests for BUG-067's guard: pre-commit's own stash mechanism parks
uncommitted work in ``$PRE_COMMIT_HOME/patch<timestamp>-<pid>`` (never plain
``git stash``), and a killed run leaves it there looking like disposable cache.
``scripts/check_pre_commit_patch_safety.py`` classifies every parked patch
against a target repo and FAILS (exit 1) the moment one would still apply
cleanly and is not yet present in the tree -- exactly the "would destroy a
worker's uncommitted diff outright" near-miss BUG-067 recorded.

Every test below drives the gate against a REAL, isolated git repository and
REAL patch files on disk (never a mock of ``git apply``/the filesystem) so a
regression in the actual classification logic is caught, not just a regression
in a test double standing in for it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _gate() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "check_pre_commit_patch_safety.py"
    )
    spec = importlib.util.spec_from_file_location("pre_commit_patch_safety_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register under its own name before exec: the module's frozen dataclass
    # (PatchVerdict) resolves its string-annotated field types via
    # sys.modules[cls.__module__] at class-definition time (`from __future__
    # import annotations`), which raises AttributeError on None if the module
    # is exec'd without first being registered.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
            "HOME": str(repo),
            "PATH": "/usr/bin:/bin",
        },
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init", "-q")
    (repo / "tracked.txt").write_text("line one\nline two\n", encoding="utf-8")
    _run_git(repo, "add", "tracked.txt")
    _run_git(repo, "commit", "-q", "-m", "initial")
    return repo


def _uncommitted_diff_then_revert(repo: Path, contents: str) -> str:
    """Simulate exactly what pre-commit's own stash mechanism captures: an
    unstaged edit's diff, taken BEFORE the working tree is blanked back to
    HEAD -- the same ``git diff`` shape ``staged_files_only.py`` writes to
    the patch file."""
    (repo / "tracked.txt").write_text(contents, encoding="utf-8")
    diff = _run_git(repo, "diff", "--no-color", "tracked.txt").stdout
    _run_git(repo, "checkout", "--", "tracked.txt")
    return diff


# ---------------------------------------------------------------------------
# find_patches: only the pre-commit patch-name shape is picked up
# ---------------------------------------------------------------------------


def test_find_patches_matches_only_the_pre_commit_patch_name_pattern(
    tmp_path: Path,
) -> None:
    gate = _gate()
    patch_dir = tmp_path / "cache"
    patch_dir.mkdir()
    (patch_dir / "patch1786999999-999999").write_text("diff\n", encoding="utf-8")
    (patch_dir / "patch12-34").write_text("diff\n", encoding="utf-8")
    (patch_dir / "not-a-patch.txt").write_text("noise\n", encoding="utf-8")
    (patch_dir / "patch-missing-digits").write_text("noise\n", encoding="utf-8")
    (patch_dir / "README.md").write_text("noise\n", encoding="utf-8")

    found = {p.name for p in gate.find_patches(patch_dir)}

    assert found == {"patch1786999999-999999", "patch12-34"}


def test_find_patches_on_missing_directory_returns_empty(tmp_path: Path) -> None:
    gate = _gate()
    assert gate.find_patches(tmp_path / "does-not-exist") == []


# ---------------------------------------------------------------------------
# Known-good: a clean/empty patch dir passes
# ---------------------------------------------------------------------------


def test_main_passes_on_empty_patch_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _gate()
    repo = _init_repo(tmp_path)
    empty_dir = tmp_path / "empty-cache"
    empty_dir.mkdir()

    exit_code = gate.main(
        ["--repository-root", str(repo), "--patch-dir", str(empty_dir)]
    )

    assert exit_code == 0
    assert "clean" in capsys.readouterr().out


def test_main_passes_when_patch_dir_does_not_exist_at_all(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _gate()
    repo = _init_repo(tmp_path)

    exit_code = gate.main(
        ["--repository-root", str(repo), "--patch-dir", str(tmp_path / "never-created")]
    )

    assert exit_code == 0


# ---------------------------------------------------------------------------
# Known-bad: a genuinely live, unapplied patch is detected and FAILS the gate
# ---------------------------------------------------------------------------


def test_classify_detects_a_live_unapplied_patch(tmp_path: Path) -> None:
    """The exact BUG-067 near-miss shape: a killed pre-commit run's own stash
    diff, sitting in the cache, that would apply cleanly and is not yet in the
    tree -- this is uncommitted work that a cache-reclaim sweep would destroy."""
    gate = _gate()
    repo = _init_repo(tmp_path)
    diff = _uncommitted_diff_then_revert(repo, "line one\nline two\nlive edit\n")
    assert diff, "setup produced no diff -- test fixture is broken"

    patch = tmp_path / "cache" / "patch1786999999-111111"
    patch.parent.mkdir()
    patch.write_text(diff, encoding="utf-8")

    verdict = gate.classify(repo, patch)

    assert verdict.classification == "LIVE-UNAPPLIED"
    assert "DESTROYED" in verdict.detail


def test_main_fails_on_a_live_unapplied_patch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _gate()
    repo = _init_repo(tmp_path)
    diff = _uncommitted_diff_then_revert(repo, "line one\nline two\nlive edit\n")
    patch_dir = tmp_path / "cache"
    patch_dir.mkdir()
    (patch_dir / "patch1786999999-111111").write_text(diff, encoding="utf-8")

    exit_code = gate.main(
        ["--repository-root", str(repo), "--patch-dir", str(patch_dir)]
    )

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "LIVE-UNAPPLIED" in out


# ---------------------------------------------------------------------------
# already-in-tree: content that has since been committed is not a false alarm
# ---------------------------------------------------------------------------


def test_classify_marks_already_committed_content_as_safe(tmp_path: Path) -> None:
    gate = _gate()
    repo = _init_repo(tmp_path)
    diff = _uncommitted_diff_then_revert(repo, "line one\nline two\nlive edit\n")

    # The work was actually recovered/committed since the patch was written.
    (repo / "tracked.txt").write_text(
        "line one\nline two\nlive edit\n", encoding="utf-8"
    )
    _run_git(repo, "commit", "-aq", "-m", "recovered")

    patch = tmp_path / "cache" / "patch1786999999-222222"
    patch.parent.mkdir()
    patch.write_text(diff, encoding="utf-8")

    verdict = gate.classify(repo, patch)

    assert verdict.classification == "already-in-tree"


def test_main_passes_when_every_patch_is_already_in_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = _gate()
    repo = _init_repo(tmp_path)
    diff = _uncommitted_diff_then_revert(repo, "line one\nline two\nlive edit\n")
    (repo / "tracked.txt").write_text(
        "line one\nline two\nlive edit\n", encoding="utf-8"
    )
    _run_git(repo, "commit", "-aq", "-m", "recovered")

    patch_dir = tmp_path / "cache"
    patch_dir.mkdir()
    (patch_dir / "patch1786999999-222222").write_text(diff, encoding="utf-8")

    exit_code = gate.main(
        ["--repository-root", str(repo), "--patch-dir", str(patch_dir)]
    )

    assert exit_code == 0
    assert "FAILED" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# stale-or-foreign: a patch belonging to neither this tree's past nor present
# ---------------------------------------------------------------------------


def test_classify_marks_unrelated_patch_as_stale_or_foreign(tmp_path: Path) -> None:
    gate = _gate()
    repo = _init_repo(tmp_path)
    # Targets tracked.txt (which DOES exist in this repo) but with context
    # lines that match neither its current nor any prior content -- neither
    # `git apply --check` nor `--check --reverse` can find a hunk to anchor
    # on, which is exactly what a patch from a DIFFERENT repository sharing
    # this same per-user cache looks like.
    foreign_diff = (
        "diff --git a/tracked.txt b/tracked.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/tracked.txt\n"
        "+++ b/tracked.txt\n"
        "@@ -1,2 +1,2 @@\n"
        " this context line never existed in this repo's tracked.txt\n"
        "-neither did this deleted line\n"
        "+content from a completely different repository entirely\n"
    )
    patch = tmp_path / "cache" / "patch1786999999-333333"
    patch.parent.mkdir()
    patch.write_text(foreign_diff, encoding="utf-8")

    verdict = gate.classify(repo, patch)

    assert verdict.classification == "stale-or-foreign"


# ---------------------------------------------------------------------------
# default patch-dir resolution honours the documented precedence
# ---------------------------------------------------------------------------


def test_default_patch_dir_prefers_pre_commit_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()
    monkeypatch.setenv("PRE_COMMIT_HOME", str(tmp_path / "pch"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert gate._default_patch_dir() == tmp_path / "pch"


def test_default_patch_dir_falls_back_to_xdg_cache_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()
    monkeypatch.delenv("PRE_COMMIT_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert gate._default_patch_dir() == tmp_path / "xdg" / "pre-commit"


def test_default_patch_dir_falls_back_to_home_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()
    monkeypatch.delenv("PRE_COMMIT_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert gate._default_patch_dir() == tmp_path / ".cache" / "pre-commit"
