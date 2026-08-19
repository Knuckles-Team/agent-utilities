"""NE-058 -- ``prune: true`` must remove *only* the worktree it just landed.

The incident: draining the queue with ``prune=True`` orphaned every OTHER
linked worktree of the shared repo in the same call -- root-caused (by
reading ``repository_manager/worktree.py`` directly) to
``WorktreeManager.remove()`` following its own targeted ``git worktree
remove`` with a BARE, repo-wide ``git worktree prune``. ``prune_landed()`` no
longer calls that accelerator at all; :func:`agent_utilities.governance
.merge_queue._prune_landed_inline` is the only implementation now, and it
never issues an unscoped ``git worktree`` command.

Three things are proven here:

1. :func:`test_prune_removes_only_the_landed_worktree_a_sibling_survives` --
   the positive case and the literal DoD requirement: land one candidate with
   ``prune=True`` while an unrelated sibling worktree sits alongside it, and
   prove the sibling's git registration is untouched (it can still run git
   commands -- the exact operation NE-058's incident broke).
2. :func:`test_prune_refuses_outright_when_a_sibling_worktree_is_dirty` --
   the bonus guard: a prune must not even start while some OTHER lane is
   mid-edit.
3. :func:`test_prune_postcondition_catches_collateral_worktree_loss` --
   proves the post-condition guard itself fires (not just that it is
   unnecessary in the happy path) by simulating the exact failure shape the
   old accelerator produced: a sibling's registration vanishing alongside the
   one actually being pruned.
"""

from __future__ import annotations

import subprocess
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
    root = tmp_path / "canonical"
    root.mkdir(parents=True)
    _run(["git", "init", "-b", "main"], root)
    _run(["git", "config", "user.email", "queue@test"], root)
    _run(["git", "config", "user.name", "Queue Test"], root)
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/core.py", "VALUE = 1\n")
    _commit(root, "base")
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


def test_prune_removes_only_the_landed_worktree_a_sibling_survives(
    canonical: Path,
) -> None:
    """The literal DoD proof: a sibling worktree's registration survives a
    `prune: true` drain of an unrelated candidate."""
    candidate_lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    # A completely unrelated sibling worktree, never enqueued, sitting on its
    # own branch -- exactly the shape of the 8 live tracks NE-058 orphaned.
    sibling_lane = _lane(canonical, "sibling-lane")

    mq.enqueue(path=candidate_lane)
    result = mq.run_queue(path=canonical, prune=True)

    assert result["landed"] == 1
    outcome = result["outcomes"][0]
    assert outcome["landed"] is True
    assert outcome["prune"]["pruned"] is True

    # The landed candidate's own worktree really is gone.
    assert not candidate_lane.exists()
    assert _run(["git", "branch", "--list", "lane-a"], canonical) == ""

    # The UNRELATED sibling worktree is fully intact and still a working git
    # operation -- not `fatal: not a git repository: .../.git/worktrees/...`.
    assert sibling_lane.is_dir()
    status = subprocess.run(  # noqa: S603
        ["git", "status", "--porcelain"],
        cwd=str(sibling_lane),
        capture_output=True,
        text=True,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    assert "fatal" not in (status.stderr or "")
    listing = _run(["git", "worktree", "list", "--porcelain"], canonical)
    assert str(sibling_lane) in listing
    assert _run(["git", "branch", "--list", "sibling-lane"], canonical) != ""


def test_prune_refuses_outright_when_a_sibling_worktree_is_dirty(
    canonical: Path,
) -> None:
    """Bonus guard: a prune refuses to run at all -- not just skip the
    target -- while ANY sibling lane worktree holds uncommitted work."""
    candidate_lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    sibling_lane = _lane(canonical, "sibling-lane")
    _write(sibling_lane, "pkg/scratch.py", "WIP = 1\n")  # uncommitted, on purpose

    mq.enqueue(path=candidate_lane)
    landed = mq.run_queue(path=canonical, prune=False)
    assert landed["landed"] == 1

    candidate = mq.Candidate(
        branch="lane-a", lane="lane-a", worktree=str(candidate_lane)
    )
    result = mq.prune_landed(
        candidate, repo_name="canonical", base="main", repo=canonical
    )

    assert result["pruned"] is False
    assert "uncommitted work" in result["reason"]
    assert "sibling-lane" in result["reason"] or str(sibling_lane) in result["reason"]
    # Refused BEFORE touching anything: the landed candidate's own worktree
    # and branch are still exactly where they were.
    assert candidate_lane.is_dir()
    assert _run(["git", "branch", "--list", "lane-a"], canonical) != ""


def test_prune_refuses_when_worktree_registry_cannot_be_read(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed registry read is not an empty registry: without the
    before-snapshot, pruning cannot prove that sibling registrations survive."""
    candidate_lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    tip = _run(["git", "rev-parse", "lane-a"], canonical)
    _run(["git", "merge", "--ff-only", "lane-a"], canonical)

    monkeypatch.setattr(mq, "_worktree_registrations", lambda repo: None)
    result = mq.prune_landed(
        mq.Candidate(branch="lane-a", lane="lane-a", worktree=str(candidate_lane)),
        repo_name="canonical",
        base="main",
        repo=canonical,
    )

    assert result["pruned"] is False
    assert "worktree registrations" in result["reason"]
    assert candidate_lane.is_dir()
    assert _run(["git", "rev-parse", "lane-a"], canonical) == tip


def test_prune_postcondition_catches_collateral_worktree_loss(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves the post-condition guard itself fires, by simulating exactly
    the failure shape NE-058 produced: some OTHER worktree's registration
    disappears in between the before/after snapshot (as `git worktree prune`
    would cause), alongside the one actually being pruned."""
    candidate_lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    sibling_lane = _lane(canonical, "sibling-lane")
    mq.enqueue(path=candidate_lane)
    landed = mq.run_queue(path=canonical, prune=False)
    assert landed["landed"] == 1

    real_registrations = mq._worktree_registrations
    calls = {"n": 0}

    def _fake_registrations(repo: Path) -> dict[str, str]:
        calls["n"] += 1
        snapshot = real_registrations(repo)
        if calls["n"] >= 2:
            # Simulate collateral damage: the sibling's registration is gone
            # from the AFTER snapshot too, exactly as an unscoped
            # `git worktree prune` could cause.
            snapshot = {
                path: sha
                for path, sha in snapshot.items()
                if Path(path).resolve() != sibling_lane.resolve()
            }
        return snapshot

    monkeypatch.setattr(mq, "_worktree_registrations", _fake_registrations)

    candidate = mq.Candidate(
        branch="lane-a", lane="lane-a", worktree=str(candidate_lane)
    )
    with pytest.raises(mq.MergeQueueError, match="REFUSED post-hoc"):
        mq._prune_landed_inline(candidate, repo=canonical, base="main")
