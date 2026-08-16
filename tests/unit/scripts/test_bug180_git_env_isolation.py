"""BUG-180: a real `git commit`/`git push` exports GIT_DIR/GIT_INDEX_FILE (and
siblings) into every `language: system` pre-commit hook it runs. `git -C
<other-repo-root> ls-files` does NOT override them -- `-C` only changes the
working directory; the repository these env vars name still wins over
path-based discovery (the same mechanism D-LGI-1 already fixed for the pytest
session via `tests/conftest.py`, but that fix does not reach a gate script
invoked directly as its own hook process).

Confirmed live 2026-08-15: running `scripts/check_liveness.py` (which imports
`scripts/check_wiring.py`'s `_tracked_or_walked`) under a real `git commit`
reported a false liveness regression -- reconciled `orphan_modules: 4 -> 196`,
`dead_definitions: 527 -> 579` -- with zero source changes. Reproduced
deterministically by hand-setting GIT_DIR/GIT_INDEX_FILE to the worktree's
real gitdir/index: identical 196/579. Root cause: `_tracked_or_walked`'s
`git -C str(root) ls-files` call passed no explicit `env=`, so the inherited
vars redirected it to the wrong repository, it returned an empty/wrong
tracked list, and the function silently fell back to an untracked `rglob`
walk that sweeps in `.venv`, build output, and other gitignored content.

Fixed by `scripts/_git_subprocess_env.py::strip_inherited_git_repository_env()`,
called at import time by both `scripts/check_wiring.py` and
`scripts/check_current_only_contract.py` (the two sites proven to break a
real gate). This test proves the known-bad input directly against
`check_wiring._tracked_or_walked`: a synthetic git repo with a KNOWN
git-tracked file set, called while GIT_DIR/GIT_INDEX_FILE point somewhere
else entirely (simulating the inherited-env shape a real `git commit`
produces) -- without the fix this returns the wrong (walked, not tracked)
file set; with it, it returns the correct git-tracked set regardless of what
the process inherited.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts" / "check_wiring.py"
SPEC = importlib.util.spec_from_file_location("check_wiring_bug180", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECK_WIRING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECK_WIRING
SPEC.loader.exec_module(CHECK_WIRING)

ENV_MODULE = Path(__file__).parents[3] / "scripts" / "_git_subprocess_env.py"
ENV_SPEC = importlib.util.spec_from_file_location("_git_subprocess_env_bug180", ENV_MODULE)
assert ENV_SPEC is not None and ENV_SPEC.loader is not None
GIT_ENV = importlib.util.module_from_spec(ENV_SPEC)
sys.modules[ENV_SPEC.name] = GIT_ENV
ENV_SPEC.loader.exec_module(GIT_ENV)


def _make_synthetic_repo(tmp_path: Path) -> Path:
    """A real git repo with exactly one tracked file and one gitignored file
    of the same extension, so `ls-files` and `rglob` disagree observably."""
    repo = tmp_path / "synthetic-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "bug180@test.local"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "bug180"], cwd=repo, check=True)
    (repo / "tracked.py").write_text("# tracked\n")
    (repo / "untracked_build_output.py").write_text("# should be gitignored\n")
    (repo / ".gitignore").write_text("untracked_build_output.py\n")
    subprocess.run(["git", "add", "tracked.py", ".gitignore"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)
    return repo


def _poisoned_env_pointing_elsewhere(tmp_path: Path) -> Path:
    """A second, unrelated real git repo whose GIT_DIR/GIT_INDEX_FILE, if
    inherited, would redirect a `git -C <synthetic repo> ...` call here
    instead -- exactly the shape a real `git commit` produces for its hooks."""
    decoy = tmp_path / "decoy-repo"
    decoy.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=decoy, check=True)
    return decoy


def test_known_bad_input_poisoned_git_dir_no_longer_corrupts_tracked_or_walked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_synthetic_repo(tmp_path)
    decoy = _poisoned_env_pointing_elsewhere(tmp_path)

    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(decoy / ".git" / "index"))

    # The fix: strip before calling, exactly as check_wiring.py now does at
    # import time.
    GIT_ENV.strip_inherited_git_repository_env()

    found = CHECK_WIRING._tracked_or_walked(repo, "*.py")
    names = {p.name for p in found}

    assert names == {"tracked.py"}, (
        "poisoned GIT_DIR/GIT_INDEX_FILE must not leak into a scoped "
        f"`git -C {repo} ls-files` call once stripped -- got {names!r}"
    )


def test_known_bad_input_reproduces_corruption_without_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: proves the test above is actually exercising the
    defect, not a no-op. With the poisoned env left in place and NO strip
    applied, `_tracked_or_walked` genuinely attempts to spawn `git -C <repo>
    ls-files` with a leaked GIT_DIR pointer still in its environment.

    Outside pytest (a real `git commit` hook subprocess, no conftest loaded)
    that spawn would succeed and silently redirect to the decoy repository --
    exactly what corrupted `check_liveness.py`'s counts live (BUG-180). Inside
    THIS suite, `tests/conftest.py`'s own D-LGI-1/GOC-71 runtime backstop
    (`_guarded_popen_init`) intercepts any git spawn carrying a leaked
    GIT_DIR/GIT_WORK_TREE and raises loudly instead of letting it corrupt
    silently -- a stronger, session-wide protection than this one call site's
    fix. Either failure mode is evidence of the same exposure; pytest's own
    backstop just converts "silently wrong" into "loudly wrong" for anything
    reached through it, which a standalone `python3 scripts/check_liveness.py`
    git-hook invocation is not.
    """
    repo = _make_synthetic_repo(tmp_path)
    decoy = _poisoned_env_pointing_elsewhere(tmp_path)

    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(decoy / ".git" / "index"))

    conftest = sys.modules.get("tests.conftest") or sys.modules.get("conftest")
    leaked_pointer_error = (
        getattr(conftest, "LeakedGitPointerEnvError", None) if conftest else None
    )

    if leaked_pointer_error is not None:
        with pytest.raises(leaked_pointer_error):
            CHECK_WIRING._tracked_or_walked(repo, "*.py")
    else:  # pragma: no cover - only if run outside this repo's conftest
        found = CHECK_WIRING._tracked_or_walked(repo, "*.py")
        names = {p.name for p in found}
        assert names != {"tracked.py"}, (
            "expected the poisoned env to corrupt this call when unstripped"
        )
