"""Proof that ``scripts/_git_scan.py`` is safe under BUG-043/BUG-174's
confirmed ambient-env condition, and that the pre-extraction shape was NOT.

D-LGI-1: a real ``git commit`` exports ``GIT_DIR``/``GIT_INDEX_FILE`` into
every hook subprocess it runs. ``tests/conftest.py`` strips those at
collection time, AND installs a runtime guard (``_guarded_popen_init``) that
refuses to spawn ``git`` at all while they are set -- so these tests
deliberately restore the true, unpatched ``Popen.__init__`` (same pattern as
``tests/unit/test_conftest_git_env_isolation.py``) and re-set the vars
pointed at their OWN throwaway ``tmp_path`` fixture repo, to reproduce the
ambient condition safely without any risk to this repository's real index.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module():
    source = ROOT / "scripts" / "_git_scan.py"
    spec = importlib.util.spec_from_file_location("_git_scan", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _conftest_module():
    conftest_path = str(Path(__file__).resolve().parents[2] / "conftest.py")
    for module in list(sys.modules.values()):
        if getattr(module, "__file__", None) == conftest_path:
            return module
    pytest.skip("root conftest not loaded")


def _allow_ambient_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deliberately bypass ``tests/conftest.py``'s ``LeakedGitPointerEnvError``
    guard for this test only -- this file's whole purpose is to prove
    ``_git_scan``'s behavior under a real ambient ``GIT_DIR``/``GIT_INDEX_FILE``,
    which the guard would otherwise (correctly, for every OTHER test) refuse
    to let run. ``monkeypatch`` reverts the restore at teardown.
    """
    conftest = _conftest_module()
    monkeypatch.setattr(subprocess.Popen, "__init__", conftest._TRUE_POPEN_INIT)


def _init_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git", "-C", str(root),
            "-c", "user.email=gate@example.invalid",
            "-c", "user.name=gate",
            "commit", "--allow-empty", "-q", "-m", "init",
        ],
        check=True,
    )


def _commit_all(root: Path) -> None:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        [
            "git", "-C", str(root),
            "-c", "user.email=gate@example.invalid",
            "-c", "user.name=gate",
            "commit", "-q", "-m", "add files",
        ],
        check=True,
    )


def _fixture_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _init_repo(root)
    (root / "other.py").write_text("# root-level\n", encoding="utf-8")
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "module.py").write_text("# pkg module\n", encoding="utf-8")
    (pkg / "notes.md").write_text("# not python\n", encoding="utf-8")
    _commit_all(root)
    return root


def test_ambient_env_reproduces_the_confirmed_git_quirk_directly(
    tmp_path, monkeypatch
):
    """Ground truth: plain ``-C`` output is subdir-relative; under ambient
    GIT_DIR/GIT_INDEX_FILE the SAME command reverts to root-relative -- the
    exact defect this module exists to route around. If this stops
    reproducing, the rest of this file is not proving what it claims to.
    """
    root = _fixture_repo(tmp_path)
    plain = subprocess.run(
        ["git", "-C", str(root / "pkg"), "ls-files", "--", "*.py"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    assert plain == ["module.py"]

    _allow_ambient_git_env(monkeypatch)
    monkeypatch.setenv("GIT_DIR", str(root / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(root / ".git" / "index"))
    ambient = subprocess.run(
        ["git", "-C", str(root / "pkg"), "ls-files", "--", "*.py"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    # Under ambient GIT_DIR/GIT_INDEX_FILE, `-C`'s own directory scoping is
    # defeated too: it lists the WHOLE repo (root-relative), not just `pkg/`.
    assert sorted(ambient) == ["other.py", "pkg/module.py"]
    assert ambient != plain


def test_tracked_or_walked_matches_under_plain_invocation(tmp_path):
    scan = _module()
    root = _fixture_repo(tmp_path)
    result = scan.tracked_or_walked(root / "pkg", "*.py", root=root)
    assert result == [root / "pkg" / "module.py"]


def test_tracked_or_walked_matches_under_ambient_git_env(tmp_path, monkeypatch):
    """The actual regression proof: identical call, ambient env simulating a
    real hook subprocess, SAME correct result -- BUG-043's broken shape
    would return ``[]`` here instead.
    """
    scan = _module()
    root = _fixture_repo(tmp_path)
    _allow_ambient_git_env(monkeypatch)
    monkeypatch.setenv("GIT_DIR", str(root / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(root / ".git" / "index"))
    result = scan.tracked_or_walked(root / "pkg", "*.py", root=root)
    assert result == [root / "pkg" / "module.py"]


def test_tracked_or_walked_no_pattern_scans_everything_under_target(
    tmp_path, monkeypatch
):
    scan = _module()
    root = _fixture_repo(tmp_path)
    _allow_ambient_git_env(monkeypatch)
    monkeypatch.setenv("GIT_DIR", str(root / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(root / ".git" / "index"))
    result = scan.tracked_or_walked(root / "pkg", root=root)
    assert set(result) == {root / "pkg" / "module.py", root / "pkg" / "notes.md"}


def test_tracked_or_walked_at_root_itself_is_unaffected(tmp_path, monkeypatch):
    """When target IS root, the ambient quirk cannot manifest (the prefix is
    empty either way) -- both conditions must already agree, and did before
    this fix too. Regression guard against breaking the unaffected case.
    """
    scan = _module()
    root = _fixture_repo(tmp_path)
    plain = scan.tracked_or_walked(root, "*.py", root=root)
    _allow_ambient_git_env(monkeypatch)
    monkeypatch.setenv("GIT_DIR", str(root / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(root / ".git" / "index"))
    ambient = scan.tracked_or_walked(root, "*.py", root=root)
    expected = sorted([root / "other.py", root / "pkg" / "module.py"])
    assert sorted(plain) == sorted(ambient) == expected


def test_tracked_or_walked_falls_back_outside_a_git_repo(tmp_path):
    scan = _module()
    stray = tmp_path / "not_a_repo"
    stray.mkdir()
    (stray / "a.py").write_text("x = 1\n", encoding="utf-8")
    (stray / "b.txt").write_text("not python\n", encoding="utf-8")
    result = scan.tracked_or_walked(stray, "*.py", root=stray)
    assert result == [stray / "a.py"]


def test_tracked_or_walked_target_outside_root_preserves_dash_c_behavior(tmp_path):
    """A synthetic fixture rooted at its own tmp_path, unrelated to the
    caller's real repo root -- the pre-extraction ``ValueError`` carve-out.
    """
    scan = _module()
    unrelated_root = tmp_path / "unrelated"
    unrelated_root.mkdir()
    fixture = _fixture_repo(tmp_path)
    result = scan.tracked_or_walked(fixture / "pkg", "*.py", root=unrelated_root)
    assert result == [fixture / "pkg" / "module.py"]


def test_repo_root_of_finds_the_nearest_git_ancestor(tmp_path):
    scan = _module()
    root = _fixture_repo(tmp_path)
    nested = root / "pkg"
    assert scan.repo_root_of(nested) == root
    assert scan.repo_root_of(root) == root


def test_repo_root_of_none_with_no_git_ancestor(tmp_path):
    scan = _module()
    lonely = tmp_path / "lonely" / "deep"
    lonely.mkdir(parents=True)
    assert scan.repo_root_of(lonely) is None


def test_repo_root_of_is_cross_repo_safe_unlike_git_rev_parse(tmp_path, monkeypatch):
    """The cross-repo contamination this function exists to avoid: under
    another repo's ambient GIT_DIR, ``git -C <foreign> rev-parse
    --show-toplevel`` can silently answer for the WRONG repository (or in
    the worst confirmed case, ``ls-files`` returns the ambient repo's own
    file list instead of erroring). Pure filesystem detection is immune
    because it never invokes git at all.
    """
    scan = _module()
    ambient_repo = _fixture_repo(tmp_path)
    foreign_repo = tmp_path / "foreign"
    foreign_repo.mkdir()
    _allow_ambient_git_env(monkeypatch)
    _init_repo(foreign_repo)
    monkeypatch.setenv("GIT_DIR", str(ambient_repo / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(ambient_repo / ".git" / "index"))
    assert scan.repo_root_of(foreign_repo) == foreign_repo
