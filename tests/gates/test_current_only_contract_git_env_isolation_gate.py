"""NE-044 sub-gate 3 acceptance (current-only contract): does
``scripts/check_current_only_contract.py`` run "from the exact tree with NO
ambient Git-state dependency" (``aad9ab52``, BUG-180)?

BUG-180: a real ``git commit``/``git push`` exports ``GIT_DIR``/
``GIT_INDEX_FILE`` (and siblings) into every ``language: system`` pre-commit
hook it runs. ``_tracked_or_walked``'s ``git -C <scan_root> ls-files`` call
does not override them -- ``-C`` only changes the working directory; the
repository these env vars name still wins over path-based discovery -- so an
unstripped hook process can silently resolve against the WRONG repository
and (per the module's own fallback) drop to an untracked ``rglob`` walk that
sweeps in ``.venv``, build output, and other gitignored content. Fixed via
``scripts/_git_subprocess_env.py::strip_inherited_git_repository_env()``,
called once at this module's import time.

Two proofs, at two different altitudes:

1. **Unit-level, known-bad input** (mirrors
   ``tests/unit/scripts/test_bug180_git_env_isolation.py``'s proof for the
   sibling ``check_wiring.py._tracked_or_walked``, applied to THIS module's
   own copy of the same function): a poisoned ``GIT_DIR``/``GIT_INDEX_FILE``
   pointing at an unrelated decoy repository must not corrupt a scoped
   ``_tracked_or_walked`` call once the fix's strip has run.

2. **Real subprocess, real tree** (the crux this NE-044 track was assigned
   to prove): spawn ``scripts/check_current_only_contract.py`` as a genuine
   child process -- exactly the shape a real ``language: system`` pre-commit
   hook runs it -- once with a clean environment and once with
   ``GIT_DIR``/``GIT_INDEX_FILE`` pointed at a different repository, and
   assert the gate's verdict against the SAME real repository tree is
   byte-for-byte identical either way.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _REPO_ROOT / "scripts" / "check_current_only_contract.py"

_SPEC = importlib.util.spec_from_file_location(
    "check_current_only_contract_ne044", _MODULE
)
assert _SPEC is not None and _SPEC.loader is not None
gate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gate
_SPEC.loader.exec_module(gate)


# ---------------------------------------------------------------------------
# 1. Unit-level known-bad input against _tracked_or_walked itself.
# ---------------------------------------------------------------------------


def _make_synthetic_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "synthetic-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "ne044@test.local"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "ne044"], cwd=repo, check=True)
    (repo / "tracked.py").write_text("# tracked\n")
    (repo / "untracked_build_output.py").write_text("# should be gitignored\n")
    (repo / ".gitignore").write_text("untracked_build_output.py\n")
    subprocess.run(["git", "add", "tracked.py", ".gitignore"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)
    return repo


def _decoy_repo(tmp_path: Path) -> Path:
    decoy = tmp_path / "decoy-repo"
    decoy.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=decoy, check=True)
    return decoy


def test_known_bad_input_poisoned_git_dir_does_not_corrupt_tracked_or_walked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the module's own import-time strip already applied (it ran when
    this file imported the module above), a poisoned ``GIT_DIR``/
    ``GIT_INDEX_FILE`` set AFTER import must not corrupt a fresh
    ``_tracked_or_walked`` call -- re-applying the strip (exactly what a
    fresh process does at its own import time) makes the poison harmless
    regardless of when it was introduced relative to import."""
    repo = _make_synthetic_repo(tmp_path)
    decoy = _decoy_repo(tmp_path)

    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(decoy / ".git" / "index"))

    gate.strip_inherited_git_repository_env()

    found = gate._tracked_or_walked(repo)
    names = {p.name for p in found}

    assert names == {"tracked.py", ".gitignore"}


# ---------------------------------------------------------------------------
# 2. Real subprocess, real tree -- the actual pre-commit hook shape.
# ---------------------------------------------------------------------------


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for name in (
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_WORK_TREE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
    ):
        env.pop(name, None)
    return env


def _poisoned_env(decoy_git_dir: Path) -> dict[str, str]:
    env = _clean_env()
    env["GIT_DIR"] = str(decoy_git_dir)
    env["GIT_INDEX_FILE"] = str(decoy_git_dir / "index")
    return env


def _run_gate(env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_MODULE), "--new-only"],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_gate_verdict_against_the_real_tree_is_identical_plain_vs_ambient_git_env(
    tmp_path: Path,
) -> None:
    """The crux this track was assigned to prove: run the REAL gate against
    the REAL repository tree twice -- once plain, once with ``GIT_DIR``/
    ``GIT_INDEX_FILE`` pointed at an unrelated decoy repository -- and assert
    the verdict (stdout + exit code) is byte-for-byte identical either way.
    A gate whose answer changes here is a confirmed defect."""
    decoy = tmp_path / "decoy-repo"
    decoy.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=decoy, check=True)

    plain = _run_gate(_clean_env())
    ambient = _run_gate(_poisoned_env(decoy / ".git"))

    assert plain.stdout == ambient.stdout
    assert plain.stderr == ambient.stderr
    assert plain.returncode == ambient.returncode
    # Not a trivially-empty comparison: the gate must have actually scanned
    # something real either way (``--new-only`` reports findings on stderr,
    # a clean summary line on stdout).
    assert "Current-only contract violations:" in plain.stderr
