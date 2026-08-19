"""NE-056 -- the fast-tier gates must run under the repo's REAL interpreter,
even though :func:`agent_utilities.governance.merge_queue.materialized` verifies
the merged tree in a throwaway detached worktree that has no ``.venv`` of its
own (it is untracked scratch space, not a checkout of anything that would
carry one).

Two things are proven here, matching the two failure modes NE-056 named:

1. :func:`test_branch_lands_when_only_the_canonical_checkout_has_a_venv` --
   the queue actually LANDS a candidate end to end when the interpreter lives
   only in the canonical checkout, never in the merged-tree scratch dir the
   gate runs the candidate from. This is the positive case: a correctly
   configured repo must not be refused just because its `.venv` is not a git
   object.
2. :func:`test_a_genuinely_unexecutable_interpreter_is_refused_not_crashed`
   and :func:`test_a_genuinely_unexecutable_interpreter_does_not_crash_the_batch`
   -- the negative case the fix must NOT relax: when the resolved interpreter
   path truly cannot be executed (not merely absent from the merged tree --
   genuinely broken), the gate is refused with a named, actionable Check, and
   the refusal is a Check like any other -- it does not raise an exception
   that takes the rest of the batch down with it. "A gate that cannot run is
   refused, never assumed clean" -- and never crashes the queue either.
"""

from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

import pytest

from agent_utilities.governance import merge_queue as mq


def _run(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        args, cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _write(root: Path, rel: str, body: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def _commit(root: Path, message: str) -> str:
    _run(["git", "add", "-A"], root)
    _run(["git", "commit", "-qm", message], root)
    return _run(["git", "rev-parse", "HEAD"], root)


@pytest.fixture
def canonical(tmp_path: Path) -> Path:
    """A canonical checkout on ``main`` with a REAL ``.venv/bin/python`` --
    the fixture ``test_merge_queue.py`` deliberately omits (its repos have no
    venv at all, so ``_interpreter`` there always falls back to
    ``sys.executable``, which happens to be correct under pytest but proves
    nothing about resolving a DIFFERENT interpreter that lives in a different
    tree). Here the venv is a real, separate path from the process's own
    ``sys.executable`` string, so a check that reports this exact path proves
    it did not just inherit the ambient interpreter.
    """
    root = tmp_path / "canonical"
    root.mkdir(parents=True)
    _run(["git", "init", "-b", "main"], root)
    _run(["git", "config", "user.email", "queue@test"], root)
    _run(["git", "config", "user.name", "Queue Test"], root)
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/core.py", "VALUE = 1\n")
    _commit(root, "base")
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    interpreter = venv_bin / "python"
    interpreter.symlink_to(sys.executable)
    return root


def _lane(canonical: Path, name: str) -> Path:
    path = canonical.parent / name
    _run(["git", "worktree", "add", "-q", str(path), "-b", name], canonical)
    return path


def _branch(canonical: Path, name: str, files: dict[str, str]) -> Path:
    tree = _lane(canonical, name)
    for rel, body in files.items():
        _write(tree, rel, body)
    _commit(tree, f"{name}: work")
    return tree


# ---------------------------------------------------------------------------
# The positive case: a correctly configured repo is never refused for this.
# ---------------------------------------------------------------------------
def test_branch_lands_when_only_the_canonical_checkout_has_a_venv(
    canonical: Path,
) -> None:
    """Drives the real gate-execution path (``run_queue`` -> ``integrate_batch``
    -> ``materialized`` -> ``run_fast_gate``): the candidate is verified in a
    detached worktree that git created for this run alone and that has never
    had a ``.venv`` written into it. The gate must still run, under the
    canonical checkout's interpreter, and land.
    """
    lane = _branch(canonical, "lane-a", {"pkg/extra.py": "EXTRA = 1\n"})
    mq.enqueue(path=lane)

    result = mq.run_queue(path=canonical, prune=False)

    assert result["landed"] == 1, result
    outcome = result["outcomes"][0]
    assert outcome["landed"] is True
    assert (canonical / "pkg" / "extra.py").is_file()

    # The verdict names the interpreter it is a claim about, and that
    # interpreter is the CANONICAL checkout's own -- not the ambient
    # sys.executable that happens to be running this test process, and not
    # any interpreter path inside the (nonexistent) materialized-tree venv.
    reported = outcome["gate"]["interpreter"]
    assert reported == str(canonical / ".venv" / "bin" / "python")
    assert reported != sys.executable


def test_direct_gate_run_against_a_venv_less_materialized_tree(
    canonical: Path,
) -> None:
    """The same proof one level down, driving :func:`run_fast_gate` directly
    against a hand-built ``materialized()`` tree -- isolates the claim to the
    gate-execution path itself, independent of the queueing/landing machinery
    exercised by the test above.
    """
    scope = mq.lane_scope(canonical)
    head = _run(["git", "rev-parse", "HEAD"], canonical)
    with mq.materialized(canonical, head, scope=scope) as tree:
        assert not (tree / ".venv").exists(), (
            "fixture invariant: no venv in the merged tree"
        )
        gate = mq.run_fast_gate(tree, changed=[], duplicates=[], scope=scope)
    assert gate.ok, gate.as_dict()
    assert gate.interpreter == str(canonical / ".venv" / "bin" / "python")


# ---------------------------------------------------------------------------
# The negative case: a genuinely broken interpreter is refused, not assumed
# clean -- and does not crash the run either.
# ---------------------------------------------------------------------------
@pytest.fixture
def canonical_with_broken_venv(tmp_path: Path) -> Path:
    """Like ``canonical``, but ``.venv/bin/python`` resolves to a real file
    that cannot be executed (no execute bit, no shebang) -- a stale venv, a
    half-written symlink, a permissions error: the class of failure that
    previously escaped ``subprocess.run`` as an uncaught ``OSError``.
    """
    root = tmp_path / "canonical"
    root.mkdir(parents=True)
    _run(["git", "init", "-b", "main"], root)
    _run(["git", "config", "user.email", "queue@test"], root)
    _run(["git", "config", "user.name", "Queue Test"], root)
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/core.py", "VALUE = 1\n")
    _commit(root, "base")
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    broken = venv_bin / "python"
    broken.write_text("not a real interpreter\n", encoding="utf-8")
    broken.chmod(stat.S_IRUSR | stat.S_IWUSR)  # readable, NOT executable
    return root


def test_a_genuinely_unexecutable_interpreter_is_refused_not_crashed(
    canonical_with_broken_venv: Path,
) -> None:
    canonical = canonical_with_broken_venv
    scope = mq.lane_scope(canonical)
    head = _run(["git", "rev-parse", "HEAD"], canonical)
    # A changed module so import-smoke actually attempts to invoke the
    # (broken) interpreter rather than short-circuiting on "no changed
    # modules".
    with mq.materialized(canonical, head, scope=scope) as tree:
        gate = mq.run_fast_gate(
            tree, changed=["pkg/core.py"], duplicates=[], scope=scope
        )

    assert gate.ok is False
    failing = {c.name: c for c in gate.failures()}
    assert "import-smoke" in failing
    detail = failing["import-smoke"].detail
    assert "could not execute" in detail
    assert "refused" in detail
    assert "never assumed clean" in detail


def test_a_genuinely_unexecutable_interpreter_does_not_crash_the_batch(
    canonical_with_broken_venv: Path,
) -> None:
    """The full ``run_queue`` path: the candidate is REFUSED (reported as an
    ordinary, non-landed outcome) rather than the whole call raising and
    losing every other candidate in the batch -- the exact crash NE-056
    described ("every merge tonight was done by hand").
    """
    canonical = canonical_with_broken_venv
    lane = _branch(canonical, "lane-a", {"pkg/core.py": "VALUE = 2\n"})
    mq.enqueue(path=lane)

    result = mq.run_queue(path=canonical, prune=False)  # must not raise

    assert result["landed"] == 0
    outcome = result["outcomes"][0]
    assert outcome["landed"] is False
    assert (
        "import-smoke" in outcome["reason"] or "could not execute" in outcome["reason"]
    )
