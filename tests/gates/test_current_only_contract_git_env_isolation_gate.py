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
    # WD10-P-AUPUSH: this test calls the REAL check_current_only_contract.py
    # TWICE, and the module's own check_report() is O(files x lines x
    # needles) -- at this repo's current scale (~4,657 tracked text files,
    # RETIRED_IDENTIFIERS + RAW_ROUTE_FRAGMENTS at 189 needles) that is
    # ~350M substring/regex checks per invocation. Measured directly
    # (`/usr/bin/time -v .venv/bin/python3 scripts/check_current_only_
    # contract.py --new-only`, exit 0): 293.03s user / 5:09.73 wall / 94%
    # CPU -- CPU-bound the entire time (py-spy: parked in
    # `_needle_matches` <- `check_report`), reproduced at BOTH high host
    # load (~50-90) and low host load (~26), so this is NOT a scheduling-
    # contention artifact this program's GOC-70 doctrine covers -- it is a
    # genuine algorithmic cost that has grown past the timeout's original
    # budget as the repo/needle-list grew. 120s was already impossible at
    # measurement time (one call alone needs ~2.5x that). 900s is a
    # stopgap with real headroom (~3x the measured worst case), not a fix:
    # the real fix is algorithmic (e.g. one combined alternation regex
    # instead of up to 189 separate `.search()` calls per line, or a
    # needle-prefix index) and is out of this lane's scope -- flagged to
    # the program ledger, not attempted here, to avoid rushing a change to
    # the word-boundary-vs-plain-substring matching semantics this needle
    # system depends on (see `_needle_matches`'s own docstring).
    return subprocess.run(
        [sys.executable, str(_MODULE), "--new-only"],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
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
    # something real either way. WD10-P-AUPUSH: this used to assert
    # "Current-only contract violations:" appears in `plain.stderr` -- true
    # while this repo carried 2 genuine new violations, but `--new-only`
    # prints NOTHING on either stream once `report.new` is empty (see
    # main()'s own `if report.new:` guard), which is now this repo's
    # correct, fixed state (0 new violations) -- so that assertion no
    # longer proves liveness, it just happens to prove "were there
    # violations right now", an unrelated fact this test should not
    # depend on. Prove liveness instead with a THIRD invocation, without
    # `--new-only`, whose summary line prints unconditionally regardless
    # of the violation count.
    summary = subprocess.run(
        [sys.executable, str(_MODULE)],
        cwd=_REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert "Current-only contract:" in summary.stdout
