"""Proofs that the merge queue catches what the bulk reconciliation gate caught.

Continuous merge is only an improvement if it reproduces, mechanically, the three
things the human adversarial review used to catch. Each test here reproduces one
of those incidents in a real git repository and asserts the queue now refuses it —
not that a function exists.

* an add/add duplicate where **git silently dropped one of two new node classes**
  → :func:`test_duplicate_symbols_across_two_candidates_are_reported`
* ``D-OB-17``: two branches that **git auto-merges cleanly into a tree that
  ``ImportError``s** → :func:`test_clean_merge_that_does_not_import_is_rejected`
* the canonical tree used as a merge arena → :func:`test_landing_is_fast_forward_only`
  and :func:`test_conflicting_candidate_never_touches_the_canonical_tree`
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agent_utilities.governance import lanes
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
    """A canonical checkout on ``main`` holding an importable package."""
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
    """A lane worktree that has already committed *files* on its own branch."""
    tree = _lane(canonical, name)
    for rel, body in files.items():
        _write(tree, rel, body)
    _commit(tree, f"{name}: work")
    return tree


# ---------------------------------------------------------------------------
# The queue itself — APPEND-ONLY, no lane can drop another's candidate
# ---------------------------------------------------------------------------
def test_two_lanes_enqueue_concurrently_without_clobbering(canonical: Path) -> None:
    a = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    b = _branch(canonical, "lane-b", {"pkg/b.py": "B = 1\n"})
    mq.enqueue(path=a)
    mq.enqueue(path=b)
    branches = {c.branch for c in mq.queued(canonical)}
    assert branches == {"lane-a", "lane-b"}
    # Two DIFFERENT fragment files — that is what makes the enqueue lock-free.
    store = mq.queue_store(canonical)
    assert sorted(store.lanes()) == ["lane-a", "lane-b"]


def test_terminal_state_supersedes_rather_than_edits(canonical: Path) -> None:
    a = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    mq.enqueue(path=a)
    mq.withdraw("lane-a", reason="changed my mind", path=a)
    assert mq.queued(canonical) == []
    everything = {c.branch: c for c in mq._all_candidates(canonical)}
    assert everything["lane-a"].state == mq.WITHDRAWN


def test_fold_resolves_cross_lane_state_by_recorded_time_not_lane_name_sort(
    canonical: Path,
) -> None:
    """D-F6-1: a candidate enqueued from a lane whose name sorts AFTER
    "canonical" alphabetically ('z' > 'c'), then landed via a state
    transition recorded from the canonical lane (the real shape: enqueue
    from a candidate's own worktree, land from the canonical checkout),
    must resolve to its terminal state -- not get stuck reporting "queued"
    forever because FragmentStore.fold()'s default `group[-1]` picked
    whichever lane's NAME sorted alphabetically last ('zzz-late-lane'),
    which happened to hold the OLDER queued record, over the canonical
    lane's newer terminal one."""
    lane = _branch(canonical, "zzz-late-lane", {"pkg/z.py": "Z = 1\n"})
    mq.enqueue(path=lane)
    candidate = mq.queued(canonical)[0]
    # The terminal write is recorded from `canonical`, a DIFFERENT lane than
    # the one that enqueued it -- exactly what `land()`'s callers do.
    mq._record_state(candidate, mq.LANDED, "", canonical)
    store = mq.queue_store(canonical)
    assert {"zzz-late-lane", "canonical"} <= set(store.lanes())  # premise: 2 fragments
    resolved = {c.branch: c for c in mq._all_candidates(canonical)}
    assert resolved["zzz-late-lane"].state == mq.LANDED
    assert mq.queued(canonical) == []  # not stuck reporting queued forever


def _folded_state_under_old_group_last_resolve(canonical: Path, branch: str) -> str:
    """What FragmentStore.fold()'s OLD default (group[-1], no resolve=) would
    have reported for *branch* against the fragments as they exist RIGHT NOW.

    Used only to prove a D-CVG-9 test is not vacuous (D-ORC-17): reads the
    SAME on-disk fragments the fixed code just resolved, through the
    UNPATCHED default resolver, so a test that would already pass under the
    restored bug is caught rather than silently accepted."""
    store = mq.queue_store(canonical)
    for record in store.fold():  # no resolve= -> the pre-D-F6-1 default
        if record.get("id") == branch:
            return str(record.get("state", ""))
    raise AssertionError(f"{branch!r} not found in any fragment")


def test_withdraw_then_reenqueue_from_an_earlier_sorting_lane_is_not_silently_lost(
    canonical: Path,
) -> None:
    """D-CVG-9 (production incident, lane-converge-0801): a candidate is
    withdrawn from `canonical` (fragment "canonical.yaml"), then
    RE-ENQUEUED from a lane whose fragment sorts ALPHABETICALLY BEFORE
    "canonical" (e.g. "au-pc-lint-0801.yaml", 'a' < 'c'). Under the D-F6-1
    bug, `enqueue()`'s own return value happily says {"enqueued": True,
    "state": "queued"} (it just echoes back the record it wrote), but the
    GLOBAL folded view -- what `queued()`/`run_queue()` actually act on --
    stayed permanently stuck on the withdrawal, because "au-pc-lint-0801"
    sorts BEFORE "canonical" and so its records are never last in
    `fold()`'s lane-sorted grouping. The failure is SILENT: the CLI reports
    success, the candidate never actually queues.

    D-ORC-17: proves non-vacuousness by re-resolving the SAME on-disk
    fragments through the OLD group[-1] resolver and asserting THAT reports
    the wrong (stuck) state -- so this test would have caught the bug had
    it existed when this test was written, not just today."""
    lane = _branch(canonical, "au-pc-lint-0801", {"pkg/x.py": "X = 1\n"})
    mq.enqueue(path=lane)
    withdrawn = mq.queued(canonical)[0]
    mq._record_state(withdrawn, mq.WITHDRAWN, "changed mind", canonical)
    mq.enqueue(path=lane)  # re-enqueue, chronologically AFTER the withdrawal

    old_state = _folded_state_under_old_group_last_resolve(canonical, "au-pc-lint-0801")
    assert old_state != mq.QUEUED, (
        "D-ORC-17: this fixture does not reproduce D-CVG-9 under the OLD "
        f"resolver (got {old_state!r}) -- the test below would be vacuous"
    )

    resolved = {c.branch: c for c in mq._all_candidates(canonical)}
    assert resolved["au-pc-lint-0801"].state == mq.QUEUED
    assert "au-pc-lint-0801" in {c.branch for c in mq.queued(canonical)}


def test_withdraw_then_reenqueue_reverse_lane_ordering_is_not_silently_lost(
    canonical: Path,
) -> None:
    """D-CVG-9, the mirror-image ordering: withdrawn from a lane sorting
    AFTER "canonical" ('z' > 'c'), re-enqueued via a state transition
    written INTO canonical's own fragment -- proving the bug (and the fix)
    is about recency, not about which side of the alphabet a lane name
    falls on."""
    lane = _branch(canonical, "zzz-lane", {"pkg/z.py": "Z = 1\n"})
    mq.enqueue(path=lane)
    withdrawn = mq.queued(canonical)[0]
    mq._record_state(withdrawn, mq.WITHDRAWN, "changed mind", lane)
    # Re-enqueue via a state transition recorded from `canonical` (a
    # DIFFERENT, alphabetically-EARLIER-sorting fragment than "zzz-lane").
    revived = [c for c in mq._all_candidates(canonical) if c.branch == "zzz-lane"][0]
    mq._record_state(revived, mq.QUEUED, "", canonical)

    old_state = _folded_state_under_old_group_last_resolve(canonical, "zzz-lane")
    assert old_state != mq.QUEUED, (
        "D-ORC-17: this fixture does not reproduce D-CVG-9 (reverse "
        f"ordering) under the OLD resolver (got {old_state!r}) -- the test "
        "below would be vacuous"
    )

    resolved = {c.branch: c for c in mq._all_candidates(canonical)}
    assert resolved["zzz-lane"].state == mq.QUEUED
    assert "zzz-lane" in {c.branch for c in mq.queued(canonical)}


def test_a_genuinely_withdrawn_candidate_is_never_resurrected(
    canonical: Path,
) -> None:
    """The other half of D-CVG-9's report: the SAME class of bug also
    revived stale entries whose fragments happened to sort after
    "canonical". A candidate withdrawn and never touched again must stay
    withdrawn -- not resurface as "queued" merely because of where its
    lane's fragment file falls alphabetically. D-ORC-17: this fixture is
    proven non-vacuous below (the restored bug DOES flip this one to
    "queued", the opposite direction from the other two tests here)."""
    lane = _branch(canonical, "dead-lane", {"pkg/d.py": "D = 1\n"})
    mq.enqueue(path=lane)
    candidate = mq.queued(canonical)[0]
    mq._record_state(candidate, mq.WITHDRAWN, "truly abandoned", canonical)

    old_state = _folded_state_under_old_group_last_resolve(canonical, "dead-lane")
    assert old_state != mq.WITHDRAWN, (
        "D-ORC-17: this fixture does not reproduce the false-resurrection "
        f"half of D-CVG-9 under the OLD resolver (got {old_state!r}) -- "
        "the test below would be vacuous"
    )

    resolved = {c.branch: c for c in mq._all_candidates(canonical)}
    assert resolved["dead-lane"].state == mq.WITHDRAWN
    assert "dead-lane" not in {c.branch for c in mq.queued(canonical)}


def test_enqueue_refuses_the_base_itself(canonical: Path) -> None:
    with pytest.raises(mq.MergeQueueError, match="named branch"):
        mq.enqueue("main", path=canonical)


# ---------------------------------------------------------------------------
# The candidate is tested AS MERGED, never as it sat on its branch
# ---------------------------------------------------------------------------
def test_clean_merge_that_does_not_import_is_rejected(canonical: Path) -> None:
    """D-OB-17: git merges both branches without one conflict marker; the result
    does not import. Each branch imports fine on its own, so *nothing* short of
    building and testing the merged tree can see this."""
    _branch(
        canonical,
        "lane-provider",
        {"pkg/core.py": "VALUE = 1\n\n\ndef helper() -> int:\n    return 2\n"},
    )
    consumer = _branch(
        canonical,
        "lane-consumer",
        {"pkg/consumer.py": "from pkg.core import missing_helper\n"},
    )
    # Both branches are individually fine against their own base...
    assert mq.trial_merge(canonical, "main", "lane-provider").ok
    assert mq.trial_merge(canonical, "main", "lane-consumer").ok
    mq.enqueue(path=consumer)
    scope = lanes.lane_scope(canonical)
    head, accepted, conflicted = mq._build_chain(
        canonical, "main", mq.queued(canonical), scope=scope
    )
    assert conflicted == [] and len(accepted) == 1  # git is perfectly happy
    changed = mq.changed_paths(canonical, "main", "lane-consumer")
    with mq.materialized(canonical, head, scope=scope) as tree:
        gate = mq.run_fast_gate(tree, changed=changed, duplicates=[], scope=scope)
    assert gate.ok is False
    smoke = next(c for c in gate.checks if c.name == "import-smoke")
    assert smoke.ok is False
    assert "missing_helper" in smoke.detail


def test_materialized_tree_is_removed_and_never_the_canonical_one(
    canonical: Path,
) -> None:
    scope = lanes.lane_scope(canonical)
    head = _run(["git", "rev-parse", "main"], canonical)
    with mq.materialized(canonical, head, scope=scope) as tree:
        assert tree.resolve() != canonical.resolve()
        assert (tree / "pkg" / "core.py").is_file()
        captured = tree
    assert not captured.exists()


# ---------------------------------------------------------------------------
# The cross-branch duplicate-symbol scan
# ---------------------------------------------------------------------------
def test_duplicate_symbols_across_two_candidates_are_reported(
    canonical: Path,
) -> None:
    """Two lanes that never saw each other each add a ``CandidateClaim``. They are
    in DIFFERENT files, so git merges both silently and every test passes — the
    duplicate is only visible when the two candidates are compared to each other,
    which is the one thing a per-branch check structurally cannot do."""
    _branch(
        canonical,
        "lane-extract",
        {"pkg/extract.py": "class CandidateClaim:\n    pass\n"},
    )
    _branch(
        canonical,
        "lane-resolve",
        {"pkg/resolve.py": "class CandidateClaim:\n    pass\n"},
    )
    assert mq.trial_merge(canonical, "main", "lane-extract").ok
    dups = mq.duplicate_definitions(canonical, "main", ["lane-extract", "lane-resolve"])
    assert [d["symbol"] for d in dups] == ["CandidateClaim"]
    assert {s["branch"] for s in dups[0]["added_by"]} == {
        "lane-extract",
        "lane-resolve",
    }


def test_a_name_already_on_main_is_shared_ancestry_not_a_collision(
    canonical: Path,
) -> None:
    _write(canonical, "pkg/shared.py", "class Existing:\n    pass\n")
    _commit(canonical, "add Existing to main")
    _branch(
        canonical,
        "lane-x",
        {"pkg/shared.py": "class Existing:\n    pass\n\n\nX = 1\n"},
    )
    _branch(
        canonical,
        "lane-y",
        {"pkg/shared.py": "class Existing:\n    pass\n\n\nY = 1\n"},
    )
    assert mq.duplicate_definitions(canonical, "main", ["lane-x", "lane-y"]) == []


def test_duplicate_scan_fails_the_gate_with_both_sites_named(
    canonical: Path,
) -> None:
    """A rejection has to hand back what a human or agent needs to COMBINE the two
    — five collisions were previously combined by hand at the gate, which is only
    possible if both call sites are named."""
    scope = lanes.lane_scope(canonical)
    dups = [
        {
            "symbol": "Fragment",
            "added_by": [
                {"branch": "lane-a", "at": "pkg/a.py:10"},
                {"branch": "lane-b", "at": "pkg/b.py:20"},
            ],
        }
    ]
    gate = mq.run_fast_gate(canonical, changed=[], duplicates=dups, scope=scope)
    assert gate.ok is False
    check = next(c for c in gate.checks if c.name == "cross-branch-duplicate-symbols")
    assert "pkg/a.py:10" in check.detail and "pkg/b.py:20" in check.detail


# ---------------------------------------------------------------------------
# Landing — the canonical tree only ever fast-forwards, and is never an arena
# ---------------------------------------------------------------------------
def test_landing_is_fast_forward_only(canonical: Path) -> None:
    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    mq.enqueue(path=lane)
    before = _run(["git", "rev-parse", "main"], canonical)
    result = mq.run_queue(path=canonical, prune=False)
    after = _run(["git", "rev-parse", "main"], canonical)
    assert result["landed"] == 1
    assert after != before
    # main's new tip contains the old tip: a fast-forward, by construction.
    subprocess.run(  # noqa: S603
        ["git", "merge-base", "--is-ancestor", before, after],
        cwd=str(canonical),
        check=True,
    )
    assert (canonical / "pkg" / "a.py").is_file()  # the canonical tree came along


def test_conflicting_candidate_never_touches_the_canonical_tree(
    canonical: Path,
) -> None:
    """The incident the reconciliation-merge lease exists for: a conflict resolved
    on the shared base left 26 commits on a detached HEAD with no ref. Here a
    conflict is detected entirely in the object database, so there is no
    half-applied state for anything to strand."""
    # Cut the branch FIRST, then move main underneath it — the ordinary shape of
    # a lane that has been open a while.
    lane = _branch(canonical, "lane-a", {"pkg/core.py": "VALUE = 'lane'\n"})
    _write(canonical, "pkg/core.py", "VALUE = 'main'\n")
    _commit(canonical, "main moves")
    mq.enqueue(path=lane)
    before = _run(["git", "rev-parse", "main"], canonical)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 0 and result["rejected"] == 1
    assert "pkg/core.py" in result["outcomes"][0]["conflicts"]
    assert _run(["git", "rev-parse", "main"], canonical) == before
    assert _run(["git", "status", "--porcelain"], canonical) == ""
    assert _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], canonical) == "main"


def test_landing_defers_when_the_canonical_tree_holds_someone_elses_work(
    canonical: Path,
) -> None:
    """The worked example from AGENTS.md: merging into a dirty canonical tree is
    the hazard, so the correct outcome is to defer, not to force."""
    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    mq.enqueue(path=lane)
    (canonical / "pkg" / "core.py").write_text("UNCOMMITTED = 1\n", encoding="utf-8")
    with pytest.raises(lanes.UnownedTreeError):
        mq.run_queue(path=lane, prune=False)
    assert mq.queued(canonical)[0].branch == "lane-a"  # still queued, not lost


def test_a_second_runner_defers_on_the_existing_lease(canonical: Path) -> None:
    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    mq.enqueue(path=lane)
    with lanes.hold_lease(mq.MERGE_LEASE, operation="other runner", path=canonical):
        with pytest.raises(lanes.LeaseUnavailable):
            mq.run_queue(path=canonical, prune=False)


# ---------------------------------------------------------------------------
# Batching + bisection — throughput without mis-attributing a failure
# ---------------------------------------------------------------------------
def test_batch_lands_many_candidates_through_one_gate(canonical: Path) -> None:
    for name in ("lane-a", "lane-b", "lane-c"):
        lane = _branch(canonical, name, {f"pkg/{name}.py": "X = 1\n"})
        mq.enqueue(path=lane)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 3
    assert {o["batch_size"] for o in result["outcomes"]} == {3}
    for name in ("lane-a", "lane-b", "lane-c"):
        assert (canonical / "pkg" / f"{name}.py").is_file()


def test_bisection_blames_only_the_candidate_that_actually_failed(
    canonical: Path,
) -> None:
    """One bad candidate in a batch of four must not reject three innocent ones —
    otherwise batching trades throughput for exactly the false attribution the
    queue exists to remove."""
    for name in ("lane-a", "lane-b", "lane-c"):
        lane = _branch(canonical, name, {f"pkg/{name}.py": "X = 1\n"})
        mq.enqueue(path=lane)
    bad = _branch(canonical, "lane-bad", {"pkg/bad.py": "import pkg.nonexistent\n"})
    mq.enqueue(path=bad)
    result = mq.run_queue(path=canonical, prune=False)
    landed = {o["branch"] for o in result["outcomes"] if o["landed"]}
    rejected = {o["branch"] for o in result["outcomes"] if not o["landed"]}
    assert landed == {"lane-a", "lane-b", "lane-c"}
    assert rejected == {"lane-bad"}


def test_a_conflicting_candidate_does_not_poison_its_batch(canonical: Path) -> None:
    good = _branch(canonical, "lane-good", {"pkg/good.py": "G = 1\n"})
    bad = _branch(canonical, "lane-clash", {"pkg/core.py": "VALUE = 'lane'\n"})
    _write(canonical, "pkg/core.py", "VALUE = 'main'\n")
    _commit(canonical, "main moves")
    mq.enqueue(path=bad)
    mq.enqueue(path=good)
    result = mq.run_queue(path=canonical, prune=False)
    outcomes = {o["branch"]: o["landed"] for o in result["outcomes"]}
    assert outcomes == {"lane-clash": False, "lane-good": True}


# ---------------------------------------------------------------------------
# Targeted-test selection stays targeted, or says so
# ---------------------------------------------------------------------------
def test_targeted_selection_maps_changed_source_to_its_tests(
    canonical: Path,
) -> None:
    _write(canonical, "tests/unit/test_core.py", "def test_x():\n    assert True\n")
    _write(
        canonical, "tests/unit/test_core_wiring.py", "def test_y():\n    assert True\n"
    )
    _write(canonical, "tests/unit/test_other.py", "def test_z():\n    assert True\n")
    _commit(canonical, "tests")
    selected = mq.select_tests(canonical, ["pkg/core.py"])
    assert selected == ["tests/unit/test_core.py", "tests/unit/test_core_wiring.py"]


def test_an_untargeted_selection_defers_instead_of_blowing_the_budget(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate that silently becomes a full run is how a gate stops being run at
    all (D-OP-4). It must say it deferred, and stay green."""
    monkeypatch.setattr(mq, "MAX_TARGETED_TEST_FILES", 1)
    _write(canonical, "tests/unit/test_core.py", "def test_x():\n    assert True\n")
    _write(
        canonical, "tests/unit/test_core_extra.py", "def test_y():\n    assert True\n"
    )
    _commit(canonical, "tests")
    scope = lanes.lane_scope(canonical)
    gate = mq.run_fast_gate(
        canonical, changed=["pkg/core.py"], duplicates=[], scope=scope
    )
    check = next(c for c in gate.checks if c.name == "targeted-tests")
    assert check.ok is True and check.deferred is True
    assert "post-merge suite" in check.detail


# ---------------------------------------------------------------------------
# Merge is not deploy
# ---------------------------------------------------------------------------
def test_promotion_state_reports_an_undecoupled_fleet_as_undecoupled(
    canonical: Path,
) -> None:
    state = mq.promotion_state(canonical)
    assert state["decoupled"] is False
    assert "armed deploy" in state["reason"]


def test_promotion_state_counts_merges_not_yet_promoted(canonical: Path) -> None:
    _run(["git", "update-ref", mq.PROMOTION_REF, "main"], canonical)
    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    mq.enqueue(path=lane)
    mq.run_queue(path=canonical, prune=False)
    state = mq.promotion_state(canonical)
    assert state["decoupled"] is True
    assert state["unpromoted_commits"] > 0
    # The deployed ref did NOT move: merging is not deploying.
    assert state["deployed"] != state["main"]


# ---------------------------------------------------------------------------
# Fail-closed pruning (D-ORC-21: a guarded prune that works even when
# repository-manager is not importable, instead of always keeping the branch)
# ---------------------------------------------------------------------------
def _block_repository_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every ``import repository_manager...`` raise, as in a minimal env."""
    import builtins

    real_import = builtins.__import__

    def _no_rm(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("repository_manager"):
            raise ImportError("repository_manager is not installed")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_rm)


def test_prune_without_repository_manager_refuses_an_unknown_branch(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inline fallback still fails closed for a branch it cannot find —
    it never invents a way to prune something it cannot verify."""
    _block_repository_manager(monkeypatch)
    candidate = mq.Candidate(branch="lane-a", lane="lane-a", worktree=str(canonical))
    result = mq.prune_landed(
        candidate, repo_name="canonical", base="main", repo=canonical
    )
    assert result["pruned"] is False
    assert result["accelerator"] == "inline (repository-manager not importable)"
    assert "does not exist" in result["reason"]


def test_prune_without_repository_manager_deletes_a_genuinely_landed_branch(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-ORC-21: the queue used to keep EVERY landed branch when
    repository-manager was not importable, regardless of whether the branch was
    genuinely safe to delete. The inline guarded prune must actually prune a
    branch that has landed (is now an ancestor of main) with a clean worktree —
    anchoring it first, and never using `git branch -D`."""
    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    mq.enqueue(path=lane)
    landed = mq.run_queue(path=canonical, prune=False)
    assert landed["landed"] == 1

    _block_repository_manager(monkeypatch)
    candidate = mq.Candidate(branch="lane-a", lane="lane-a", worktree=str(lane))
    result = mq.prune_landed(
        candidate, repo_name="canonical", base="main", repo=canonical
    )

    assert result["pruned"] is True
    assert result["accelerator"] == "inline (repository-manager not importable)"
    assert not lane.exists()  # worktree removed
    assert _run(["git", "branch", "--list", "lane-a"], canonical) == ""  # ref deleted
    # anchored before deletion — the commit is still reachable, never orphaned
    anchor = _run(
        ["git", "rev-parse", "--verify", "--quiet", "refs/lane-backup/lane-a"],
        canonical,
    )
    assert anchor == landed["outcomes"][0]["to"] or anchor  # anchor resolved


def test_prune_without_repository_manager_refuses_unmerged_commits(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A branch that was never actually landed must never be deleted, even with
    `-d` (which would itself refuse) — this pins the refusal at the merge-base
    check, before git is even asked, and proves the branch survives."""
    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})  # never merged to main

    _block_repository_manager(monkeypatch)
    candidate = mq.Candidate(branch="lane-a", lane="lane-a", worktree=str(lane))
    result = mq.prune_landed(
        candidate, repo_name="canonical", base="main", repo=canonical
    )

    assert result["pruned"] is False
    assert "not (or no longer) reachable" in result["reason"]
    assert _run(["git", "branch", "--list", "lane-a"], canonical) != ""
    assert lane.is_dir()


def test_prune_without_repository_manager_skips_a_worktree_still_dirty(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`merged` (an ancestor of base) is not the same as `unoccupied` — a lane
    that landed part of its work and kept editing must not be pruned out from
    under it (the exact shape of D-FE-9, reproduced here for the inline path)."""
    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    mq.enqueue(path=lane)
    landed = mq.run_queue(path=canonical, prune=False)
    assert landed["landed"] == 1
    _write(lane, "pkg/a.py", "A = 2\n")  # uncommitted work, never committed

    _block_repository_manager(monkeypatch)
    candidate = mq.Candidate(branch="lane-a", lane="lane-a", worktree=str(lane))
    result = mq.prune_landed(
        candidate, repo_name="canonical", base="main", repo=canonical
    )

    assert result["pruned"] is False
    assert "uncommitted work" in result["reason"]
    assert lane.is_dir()
    assert _run(["git", "branch", "--list", "lane-a"], canonical) != ""


# ---------------------------------------------------------------------------
# Queue visibility (D-ORC-20: queued != landed, and nothing drains the queue
# automatically — the queue must say so rather than let a lane misread
# `state: queued` as "handed off")
# ---------------------------------------------------------------------------
def test_enqueue_says_explicitly_that_queued_is_not_landed(canonical: Path) -> None:
    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    result = mq.enqueue(path=lane)
    assert "D-ORC-20" in result["note"]
    assert "merge-queue run" in result["note"]
    assert result["queue_depth"] == 1


def test_status_warns_when_the_oldest_candidate_is_stale(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    mq.enqueue(path=lane)
    monkeypatch.setattr(mq, "STALE_QUEUE_THRESHOLD_SECONDS", -1)
    report = mq.queue_report(canonical)
    assert report["stale_queue_warning"] is not None
    assert "lane-a" in report["stale_queue_warning"]
    # D-MQR-7: the warning no longer cites D-ORC-20 -- that reference claimed
    # "nothing drives the queue automatically," which became false once
    # merge-queue-runner.timer started draining it (see queue_report's own
    # docstring). The warning now points at the runner instead.
    assert "merge-queue-runner.timer" in report["stale_queue_warning"]


def test_status_does_not_warn_for_a_freshly_queued_candidate(canonical: Path) -> None:
    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    mq.enqueue(path=lane)
    report = mq.queue_report(canonical)
    assert report["stale_queue_warning"] is None


def test_queue_report_publishes_the_latency_budget(canonical: Path) -> None:
    """The budget is a contract a lane relies on when deciding whether to wait.
    An unpublished budget is one nobody can hold the queue to."""
    report = mq.queue_report(canonical)
    assert report["budget_seconds"] == mq.FAST_GATE_BUDGET_SECONDS
    assert report["lease"] == mq.MERGE_LEASE
    assert report["batch_size"] == mq.DEFAULT_BATCH_SIZE


# ---------------------------------------------------------------------------
# Wiring — the CLI entrypoint actually reaches the queue
# ---------------------------------------------------------------------------
def test_cli_entrypoint_drives_the_real_queue(
    canonical: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wire-First: the capability is reached from a live entrypoint, driving the
    real module — not a mock standing in for the seam."""
    import json

    from agent_utilities.cli import main

    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    assert main(["merge-queue", "enqueue", "--path", str(lane)]) == 0
    capsys.readouterr()

    assert main(["merge-queue", "status", "--path", str(canonical)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [c["id"] for c in report["queued"]] == ["lane-a"]

    assert main(["merge-queue", "run", "--path", str(canonical), "--no-prune"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["landed"] == 1
    assert (canonical / "pkg" / "a.py").is_file()


def test_cli_exits_75_so_a_shell_stops_instead_of_proceeding(
    canonical: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal is worthless if `&&` still proceeds after it — the same contract
    `lane lease` publishes."""
    from agent_utilities.cli import main

    lane = _branch(canonical, "lane-a", {"pkg/a.py": "A = 1\n"})
    main(["merge-queue", "enqueue", "--path", str(lane)])
    capsys.readouterr()
    with lanes.hold_lease(mq.MERGE_LEASE, operation="other runner", path=canonical):
        assert main(["merge-queue", "run", "--path", str(canonical)]) == 75


# ---------------------------------------------------------------------------
# Zero conflicts is not a safety property
# ---------------------------------------------------------------------------
CONTRACT = """\
import pathlib, sys
root = pathlib.Path(sys.argv[sys.argv.index("--repository-root") + 1])
src = (root / "pkg" / "identity.py").read_text()
sys.exit(1 if "resolve_placement" in src else 0)
"""


def _with_contract(canonical: Path) -> None:
    """Give the fixture repo a fix on `main` plus the invariant that guards it."""
    _write(canonical, "pkg/identity.py", "def mint():\n    return Session()\n")
    _write(canonical, "scripts/security/check_identity_contract.py", CONTRACT)
    _commit(canonical, "main: the fix, and the contract that guards it")


def test_a_branch_that_reverts_a_fix_merges_cleanly_and_is_caught(
    canonical: Path,
) -> None:
    """The headline case. A lane forks AFTER a fix, reverts it, and touches nothing
    else main touched. Git reports **no conflict** — it is answering "did two people
    edit the same lines", a question about text on the branch, which is simply not a
    question about whether the merged tree still upholds an invariant. Only running
    the invariant against the merged tree answers that."""
    _with_contract(canonical)
    lane = _branch(
        canonical,
        "lane-revert",
        {
            "pkg/identity.py": "def mint():\n    placement = resolve_placement()\n    return Session()\n"
        },
    )
    _write(canonical, "pkg/elsewhere.py", "UNRELATED = 1\n")
    _commit(canonical, "main: unrelated work elsewhere")

    trial = mq.trial_merge(canonical, "main", "lane-revert")
    assert trial.ok, "the revert must merge CLEANLY — that is the whole point"

    mq.enqueue(path=lane)
    before = _run(["git", "rev-parse", "main"], canonical)
    result = mq.run_queue(path=canonical, prune=False)

    assert result["landed"] == 0 and result["rejected"] == 1
    assert _run(["git", "rev-parse", "main"], canonical) == before
    gate = result["outcomes"][0]["gate"]
    failed = [c["name"] for c in gate["checks"] if not c["ok"]]
    assert failed == ["contract-checks"]


def test_the_same_branch_passes_before_the_fix_exists(canonical: Path) -> None:
    """Control: the mechanism is the contract, not the string. With no fix on main
    to revert, the identical branch lands."""
    _write(canonical, "scripts/security/check_identity_contract.py", CONTRACT)
    _write(canonical, "pkg/identity.py", "def mint():\n    return Session()\n")
    _commit(canonical, "main: contract only")
    lane = _branch(canonical, "lane-ok", {"pkg/other.py": "X = 1\n"})
    mq.enqueue(path=lane)
    assert mq.run_queue(path=canonical, prune=False)["landed"] == 1


def test_contracts_are_discovered_from_the_merged_tree_not_this_module(
    canonical: Path,
) -> None:
    """A candidate that ADDS an invariant has it enforced against itself in the same
    gate run — which is what makes this generalize without editing merge_queue.py."""
    lane = _branch(
        canonical,
        "lane-adds-contract",
        {
            "scripts/security/check_identity_contract.py": CONTRACT,
            "pkg/identity.py": "def mint():\n    placement = resolve_placement()\n",
        },
    )
    mq.enqueue(path=lane)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 0
    assert "check_identity_contract.py" in result["outcomes"][0]["reason"]


def test_a_repo_with_no_contracts_is_a_genuine_empty_not_a_refusal(
    canonical: Path,
) -> None:
    """Fail-closed applies to a DEGRADED read, not to an honest absence. A repo that
    has never had a contract must not have every merge refused — that is a gate
    nobody can use, which is the same outcome as no gate at all."""
    check = mq.run_contract_checks(
        canonical,
        interpreter=mq._interpreter(canonical),
        env=dict(os.environ),
        baseline=set(),
    )
    assert check.ok is True
    assert "genuine empty, not a degraded read" in check.detail


def test_deleting_the_contract_is_not_a_way_to_satisfy_it(canonical: Path) -> None:
    """The degraded read that DOES matter: the base has an invariant and the merged
    tree does not. Without comparing against the base, a candidate could land by
    leaving nothing behind to fail."""
    _with_contract(canonical)
    lane = _lane(canonical, "lane-deletes")
    (lane / "scripts" / "security" / "check_identity_contract.py").unlink()
    _commit(lane, "lane-deletes: remove the contract")
    mq.enqueue(path=lane)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 0
    assert "DROPS contract check(s)" in result["outcomes"][0]["reason"]
    assert "check_identity_contract.py" in result["outcomes"][0]["reason"]


def test_contract_baseline_reads_the_base_ref(canonical: Path) -> None:
    _with_contract(canonical)
    assert mq.contract_baseline(canonical, "main") == {"check_identity_contract.py"}


def test_contract_step_stays_inside_its_share_of_the_budget(canonical: Path) -> None:
    """A security check that blows the budget gets the whole gate bypassed."""
    _with_contract(canonical)
    check = mq.run_contract_checks(
        canonical,
        interpreter=mq._interpreter(canonical),
        env=dict(os.environ),
        baseline=mq.contract_baseline(canonical, "main"),
    )
    assert check.ok is True
    assert check.seconds < mq.CONTRACT_CHECK_BUDGET_SECONDS
    assert check.seconds < mq.FAST_GATE_BUDGET_SECONDS / 10


# ---------------------------------------------------------------------------
# Differential (regression) gating for targeted tests
# CONCEPT:AU-OS.governance.test-regression-baseline
#
# `main` itself can be red. `contract_baseline` already established that a gate
# has to compare the merged tree against the base rather than judge it alone;
# these four tests prove `targeted-tests` now does the identical thing at the
# individual-test-id level — the one step that previously had none.
# ---------------------------------------------------------------------------
def test_parse_failing_test_ids_reads_failed_and_error_lines() -> None:
    """Pins the parsing contract against pytest's own stable short-summary shape,
    independent of any real subprocess."""
    stdout = (
        "..F.E\n"
        "=================== short test summary info ===================\n"
        "FAILED tests/unit/test_x.py::test_a - AssertionError: boom\n"
        "FAILED tests/unit/test_x.py::test_b[1] - assert 2 == 1\n"
        "ERROR tests/unit/test_y.py::test_c\n"
        "2 failed, 1 error in 0.10s\n"
    )
    assert mq._parse_failing_test_ids(stdout) == {
        "tests/unit/test_x.py::test_a",
        "tests/unit/test_x.py::test_b[1]",
        "tests/unit/test_y.py::test_c",
    }


def test_regression_new_failure_is_rejected(canonical: Path) -> None:
    """(a) A genuine NEW failure introduced by the branch is REJECTED — the whole
    point of the differential gate, proven against a `main` that is otherwise
    fully green for this selection (so there is no pre-existing red to hide
    behind)."""
    _write(canonical, "pkg/greeter.py", "def greet():\n    return 'hi'\n")
    _write(
        canonical,
        "tests/unit/test_greeter.py",
        "from pkg.greeter import greet\n\n\ndef test_greet():\n    assert greet() == 'hi'\n",
    )
    _commit(canonical, "main: greeter + its test, both green")
    lane = _branch(
        canonical,
        "lane-break-greeter",
        {"pkg/greeter.py": "def greet():\n    return 'bye'\n"},
    )
    mq.enqueue(path=lane)
    before = _run(["git", "rev-parse", "main"], canonical)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 0 and result["rejected"] == 1
    assert _run(["git", "rev-parse", "main"], canonical) == before  # never touched
    gate = result["outcomes"][0]["gate"]
    check = next(c for c in gate["checks"] if c["name"] == "targeted-tests")
    assert check["ok"] is False
    assert "NEW failure" in check["detail"]
    assert "tests/unit/test_greeter.py::test_greet" in check["detail"]


def test_regression_pre_existing_failure_is_allowed_and_reported(
    canonical: Path,
) -> None:
    """(b) A failure present IDENTICALLY on base and merged is ALLOWED to land —
    but the gate must say so explicitly, with a count, so pre-existing red stays
    loudly visible rather than reading as a silent pass (property 1: this is
    reporting, never masking)."""
    _write(canonical, "pkg/flaky.py", "BROKEN = True\n")
    _write(
        canonical,
        "tests/unit/test_flaky.py",
        "from pkg.flaky import BROKEN\n\n\n"
        "def test_flaky():\n    assert not BROKEN, 'pre-existing bug'\n",
    )
    _commit(canonical, "main: a pre-existing, already-red test")
    lane = _branch(
        canonical,
        "lane-touches-flaky",
        # Lands in the targeted selection (touches the same module) WITHOUT
        # fixing the bug — an unrelated, harmless change.
        {"pkg/flaky.py": "# unrelated note\nBROKEN = True\n"},
    )
    mq.enqueue(path=lane)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 1 and result["rejected"] == 0
    gate = result["outcomes"][0]["gate"]
    check = next(c for c in gate["checks"] if c["name"] == "targeted-tests")
    assert check["ok"] is True
    assert "1 pre-existing failure" in check["detail"]
    assert "tests/unit/test_flaky.py::test_flaky" in check["detail"]
    assert "allowed" in check["detail"]


def test_regression_unproducible_baseline_refuses_not_allows(canonical: Path) -> None:
    """(c) When the base's copy of the exact targeted selection cannot even be
    collected, the candidate is REFUSED — never treated as though the base had
    "no pre-existing failures" (property 2: fail-closed on a degraded read, the
    single easiest thing to get wrong here)."""
    _write(canonical, "pkg/odd.py", "VALUE = 1\n")
    _write(
        canonical, "tests/unit/test_odd.py", "def test_odd(:\n    pass\n"
    )  # a genuine syntax error, already on main
    _commit(canonical, "main: a targeted test file that does not even collect")
    lane = _lane(canonical, "lane-touches-odd")
    _write(
        lane,
        "tests/unit/test_odd.py",
        "def test_odd():\n    assert False, 'now valid syntax, still a real bug'\n",
    )
    _commit(lane, "lane-touches-odd: fix the syntax, not the bug")
    mq.enqueue(path=lane)
    before = _run(["git", "rev-parse", "main"], canonical)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 0 and result["rejected"] == 1
    assert _run(["git", "rev-parse", "main"], canonical) == before  # never touched
    gate = result["outcomes"][0]["gate"]
    check = next(c for c in gate["checks"] if c["name"] == "targeted-tests")
    assert check["ok"] is False
    assert "REFUSED" in check["detail"]
    assert "could not be produced" in check["detail"]
    # NOT the allow-everything fallback this design explicitly forbids: a refusal
    # never reports a regression diff (new/pre-existing/fixed counts), because
    # computing one requires exactly the baseline that could not be produced.
    assert "NEW failure" not in check["detail"]
    assert "pre-existing failure(s) also" not in check["detail"]


def test_regression_fix_is_allowed_and_counted_as_improvement(canonical: Path) -> None:
    """(d) A test that fails on base but PASSES on merged is ALLOWED, and the fix
    is counted explicitly in the gate's report as an improvement — not just a
    silent green."""
    _write(canonical, "pkg/fixme.py", "def compute():\n    return 1\n")
    _write(
        canonical,
        "tests/unit/test_fixme.py",
        "from pkg.fixme import compute\n\n\ndef test_fixme():\n    assert compute() == 2\n",
    )
    _commit(canonical, "main: a real, pre-existing bug")
    lane = _branch(
        canonical,
        "lane-fixes-fixme",
        {"pkg/fixme.py": "def compute():\n    return 2\n"},
    )
    mq.enqueue(path=lane)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 1 and result["rejected"] == 0
    gate = result["outcomes"][0]["gate"]
    check = next(c for c in gate["checks"] if c["name"] == "targeted-tests")
    assert check["ok"] is True
    assert "FIXES" in check["detail"]
    assert "tests/unit/test_fixme.py::test_fixme" in check["detail"]


def test_baseline_is_cached_and_reused_across_calls(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline for a static base + selection is computed once and served
    from cache on a second call — this is what keeps differential gating from
    doubling every candidate's cost rather than paying once per batch."""
    _write(canonical, "pkg/flaky.py", "BROKEN = True\n")
    _write(
        canonical,
        "tests/unit/test_flaky.py",
        "from pkg.flaky import BROKEN\n\n\ndef test_flaky():\n    assert not BROKEN\n",
    )
    _commit(canonical, "main: pre-existing red")
    scope = lanes.lane_scope(canonical)
    interpreter = mq._interpreter(canonical)
    env = dict(os.environ)
    tests = ["tests/unit/test_flaky.py"]

    first = mq.compute_test_baseline(
        canonical, "main", tests, scope=scope, interpreter=interpreter, env=env
    )
    assert first.readable is True
    assert "tests/unit/test_flaky.py::test_flaky" in first.failing

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("materialized() must not run again on a cache hit")

    monkeypatch.setattr(mq, "materialized", _boom)
    second = mq.compute_test_baseline(
        canonical, "main", tests, scope=scope, interpreter=interpreter, env=env
    )
    assert second.readable is True
    assert second.failing == first.failing


def test_unreadable_baseline_is_not_cached(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable baseline may be a transient fluke rather than a durable fact
    about `base_sha` — caching it would turn one bad run into a standing refusal
    for every later candidate. Only a READABLE result is ever cached."""
    _write(canonical, "pkg/x.py", "X = 1\n")
    _write(canonical, "tests/unit/test_x.py", "def test_x():\n    assert True\n")
    _commit(canonical, "main")
    scope = lanes.lane_scope(canonical)
    interpreter = mq._interpreter(canonical)
    env = dict(os.environ)
    tests = ["tests/unit/test_x.py"]

    monkeypatch.setattr(mq, "_timed_run", lambda *a, **k: (None, 999.0))
    first = mq.compute_test_baseline(
        canonical, "main", tests, scope=scope, interpreter=interpreter, env=env
    )
    assert first.readable is False
    base_sha = mq._require_git(["rev-parse", "main"], canonical)
    cache_path = mq._baseline_cache_path(scope, base_sha, interpreter=interpreter)
    assert mq._load_file_baseline_cache(cache_path) == {}


def test_baseline_cache_is_per_file_and_serves_a_subset_selection_free(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-MW-10: a selection that is a SUBSET of one already baselined at the same
    base_sha must be answered entirely from cache — no subprocess, no
    materialized() worktree — because this is exactly the shape
    integrate_batch's bisection produces on every retry (a sub-batch's selection
    is by construction a subset of its parent batch's)."""
    _write(canonical, "pkg/a.py", "A = 1\n")
    _write(canonical, "pkg/b.py", "B = 1\n")
    _write(canonical, "tests/unit/test_a.py", "def test_a():\n    assert True\n")
    _write(
        canonical,
        "tests/unit/test_b.py",
        "def test_b():\n    assert False\n",
    )
    _commit(canonical, "main: two independent test files, one pre-existing red")
    scope = lanes.lane_scope(canonical)
    interpreter = mq._interpreter(canonical)
    env = dict(os.environ)
    superset = ["tests/unit/test_a.py", "tests/unit/test_b.py"]

    parent = mq.compute_test_baseline(
        canonical, "main", superset, scope=scope, interpreter=interpreter, env=env
    )
    assert parent.readable is True
    assert parent.failing == {"tests/unit/test_b.py::test_b"}

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "materialized() must not run again for a subset already covered "
            "by the parent batch's baseline"
        )

    monkeypatch.setattr(mq, "materialized", _boom)
    monkeypatch.setattr(
        mq,
        "_timed_run",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("no subprocess needed for an all-cached subset")
        ),
    )
    for subset in (["tests/unit/test_a.py"], ["tests/unit/test_b.py"], superset):
        sub = mq.compute_test_baseline(
            canonical, "main", subset, scope=scope, interpreter=interpreter, env=env
        )
        assert sub.readable is True
        assert sub.failing == {
            fid for fid in parent.failing if fid.split("::")[0] in subset
        }


def test_baseline_only_runs_the_not_yet_cached_delta(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A selection that PARTIALLY overlaps an already-baselined one only pays for
    the genuinely new file — the already-known file is not re-run."""
    _write(canonical, "tests/unit/test_a.py", "def test_a():\n    assert True\n")
    _write(canonical, "tests/unit/test_c.py", "def test_c():\n    assert True\n")
    _commit(canonical, "main")
    scope = lanes.lane_scope(canonical)
    interpreter = mq._interpreter(canonical)
    env = dict(os.environ)

    mq.compute_test_baseline(
        canonical,
        "main",
        ["tests/unit/test_a.py"],
        scope=scope,
        interpreter=interpreter,
        env=env,
    )

    real_timed_run = mq._timed_run
    seen: list[list[str]] = []

    def _tracking(argv: list[str], *a: object, **k: object) -> object:
        seen.append(list(argv))
        return real_timed_run(argv, *a, **k)

    monkeypatch.setattr(mq, "_timed_run", _tracking)
    result = mq.compute_test_baseline(
        canonical,
        "main",
        ["tests/unit/test_a.py", "tests/unit/test_c.py"],
        scope=scope,
        interpreter=interpreter,
        env=env,
    )
    assert result.readable is True
    assert result.failing == frozenset()
    assert len(seen) == 1
    (argv,) = seen
    assert "tests/unit/test_a.py" not in argv
    assert "tests/unit/test_c.py" in argv


# ---------------------------------------------------------------------------
# D-MW-9: differential contract-check gating proves the hole is closed — a
# contract script that ALREADY carries debt on main (exactly
# check_current_only_contract.py's real shape: ~490 pre-existing violations)
# must not deadlock the queue, while a candidate that adds a genuinely NEW
# violation to that same already-red script must still be rejected.
# ---------------------------------------------------------------------------

ITEMIZED_CONTRACT = """\
import pathlib, sys
root = pathlib.Path(sys.argv[sys.argv.index("--repository-root") + 1])
src = (root / "pkg" / "core.py").read_text()
violations = sorted(line.strip() for line in src.splitlines() if line.strip().startswith("BAD_"))
if violations:
    print("itemized gate failed:")
    for v in violations:
        print(f"- {v}")
    sys.exit(1)
sys.exit(0)
"""


def _with_itemized_contract(canonical: Path) -> None:
    """Give the fixture repo a contract with ONE pre-existing violation already
    on main — the real shape of check_current_only_contract.py right now."""
    _write(canonical, "pkg/core.py", "VALUE = 1\nBAD_existing = 1\n")
    _write(canonical, "scripts/security/check_itemized_contract.py", ITEMIZED_CONTRACT)
    _commit(canonical, "main: a contract with one pre-existing, unfixed violation")


def test_new_violation_on_an_already_red_contract_script_is_rejected(
    canonical: Path,
) -> None:
    """The itemized-diff proof: main is ALREADY red for this script (BAD_existing),
    and a candidate that adds a SECOND, distinct violation to the same file must
    still be rejected — proving new-vs-pre-existing is judged per violation LINE,
    not merely by whether the script was already failing."""
    _with_itemized_contract(canonical)
    lane = _branch(
        canonical,
        "lane-adds-new-violation",
        {"pkg/core.py": "VALUE = 1\nBAD_existing = 1\nBAD_new = 2\n"},
    )
    mq.enqueue(path=lane)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 0 and result["rejected"] == 1
    gate = result["outcomes"][0]["gate"]
    check = next(c for c in gate["checks"] if c["name"] == "contract-checks")
    assert check["ok"] is False
    assert "NEW violation" in check["detail"]
    assert "BAD_new = 2" in check["detail"]
    assert "pre-existing" in check["detail"]
    assert (
        "BAD_existing"
        not in check["detail"].split("NEW violation")[1].split("pre-existing")[0]
    )


def test_pre_existing_contract_debt_does_not_block_an_unrelated_candidate(
    canonical: Path,
) -> None:
    """The escape-the-deadlock proof: main already carries contract debt (exactly
    the ~490-violation check_current_only_contract.py shape), and a candidate that
    never touches the offending file must land — the debt is reported, not
    silenced, but it is not blocking."""
    _with_itemized_contract(canonical)
    lane = _branch(canonical, "lane-unrelated", {"pkg/elsewhere.py": "UNRELATED = 1\n"})
    mq.enqueue(path=lane)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 1 and result["rejected"] == 0
    gate = result["outcomes"][0]["gate"]
    check = next(c for c in gate["checks"] if c["name"] == "contract-checks")
    assert check["ok"] is True
    assert "BAD_existing" in check["detail"]
    assert "pre-existing" in check["detail"]


# ---------------------------------------------------------------------------
# D-MQ-FP-1: a contract script's own INCIDENTAL report text (a count, a
# range boundary — anything derived from the size/shape of what it scanned
# rather than from a violation's identity) must never be diffed as if it
# were a violation, and the differential baseline run must be anchored to
# the SAME base_sha the queue itself is using, not whatever a script
# defaults to on its own. Reproduces the false-reject a release.yml-only
# candidate hit in production: `contract-checks: ...: NEW violation(s) not
# present on the base ref:` with the actual violation list empty — the
# ONLY thing that differed between the base-tree and merged-tree runs was
# incidental report metadata, not a real finding.
# ---------------------------------------------------------------------------


def test_output_lines_excludes_bare_numeric_json_fields() -> None:
    """Unit-level pin on the exact shape :func:`mq._is_volatile_report_metadata`
    strips: a standalone ``"key": <number>`` JSON member, with or without a
    trailing comma, is incidental report metadata, not a violation."""
    proc = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=('{\n  "ok": false,\n  "addedLines": 33,\n  "seconds": 0.42\n}\n'),
        stderr="",
    )
    lines = mq._output_lines(proc)
    assert '"addedLines": 33,' not in lines
    assert '"seconds": 0.42' not in lines
    # Genuine content survives: the boolean field, and anything that is not a
    # bare "key": number member (a quoted string value, a per-violation
    # free-text line) must never be dropped from the comparison.
    assert '"ok": false,' in lines


def test_output_lines_keeps_violation_shaped_lines_that_end_in_digits() -> None:
    """The narrowness guarantee: a real per-violation line that merely
    contains digits (a file:line locator, a string value with a numeric
    suffix) must NOT be mistaken for volatile metadata — only an entire line
    that is nothing but ``"key": <number>`` is excluded."""
    proc = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout='- pkg/core.py:42: BAD_thing_v2\n"file": "pkg/core42.py",\n',
        stderr="",
    )
    lines = mq._output_lines(proc)
    assert "- pkg/core.py:42: BAD_thing_v2" in lines
    assert '"file": "pkg/core42.py",' in lines


def test_contract_check_argv_forwards_base_only_when_declared(tmp_path: Path) -> None:
    """Unit-level pin on :func:`mq._contract_check_argv`'s detection rule —
    identical in shape to the pre-existing ``--repository-root`` rule it is
    modeled on."""
    declares = tmp_path / "declares.py"
    declares.write_text("import argparse\np.add_argument('--base')\n")
    silent = tmp_path / "silent.py"
    silent.write_text("import sys\nsys.exit(0)\n")

    argv = mq._contract_check_argv(
        declares,
        Path("declares.py"),
        interpreter="python3",
        tree=tmp_path,
        base_sha="deadbeef",
    )
    assert argv[-2:] == ["--base", "deadbeef"]

    argv_silent = mq._contract_check_argv(
        silent,
        Path("silent.py"),
        interpreter="python3",
        tree=tmp_path,
        base_sha="deadbeef",
    )
    assert "--base" not in argv_silent

    # No base_sha available (e.g. output_baseline is None) — never forwarded,
    # even for a script that declares support.
    argv_no_sha = mq._contract_check_argv(
        declares,
        Path("declares.py"),
        interpreter="python3",
        tree=tmp_path,
    )
    assert "--base" not in argv_no_sha


# Like ITEMIZED_CONTRACT, but also prints a volatile line ahead of the
# violation list — a bare count derived from the tree it was invoked
# against (here, total commit count, exactly the shape `addedLines` in
# check_secret_history.py's JSON report has: a number that legitimately
# differs between "run on the base tree" and "run on the merged tree"
# even when the violations themselves are identical, because the merged
# tree simply has one more commit).
VOLATILE_ITEMIZED_CONTRACT = """\
import pathlib, subprocess, sys
root = pathlib.Path(sys.argv[sys.argv.index("--repository-root") + 1])
count = subprocess.run(
    ["git", "rev-list", "--count", "HEAD"], cwd=str(root),
    capture_output=True, text=True, check=True,
).stdout.strip()
print(f'"commitCount": {count},')
src = (root / "pkg" / "core.py").read_text()
violations = sorted(line.strip() for line in src.splitlines() if line.strip().startswith("BAD_"))
if violations:
    print("itemized gate failed:")
    for v in violations:
        print(f"- {v}")
    sys.exit(1)
sys.exit(0)
"""


def _with_volatile_itemized_contract(canonical: Path) -> None:
    """Same pre-existing-debt shape as :func:`_with_itemized_contract`, but the
    contract script also emits a volatile per-run count."""
    _write(canonical, "pkg/core.py", "VALUE = 1\nBAD_existing = 1\n")
    _write(
        canonical,
        "scripts/security/check_volatile_contract.py",
        VOLATILE_ITEMIZED_CONTRACT,
    )
    _commit(canonical, "main: a contract with pre-existing debt AND volatile output")


def test_volatile_report_metadata_does_not_block_an_unrelated_candidate(
    canonical: Path,
) -> None:
    """The false-reject reproduction: main already carries contract debt, and a
    candidate that never touches the offending file adds exactly one commit —
    which, pre-fix, changed the script's ``commitCount`` line between the
    base-tree run and the merged-tree run and was misread as a NEW violation
    even though the violation SET (``BAD_existing`` only) never changed. This
    must land, with the debt reported but not blocking — the same escape this
    module already proves for itemized text (see
    ``test_pre_existing_contract_debt_does_not_block_an_unrelated_candidate``),
    now proven for a script whose report also carries an incidental number."""
    _with_volatile_itemized_contract(canonical)
    lane = _branch(canonical, "lane-unrelated", {"pkg/elsewhere.py": "UNRELATED = 1\n"})
    mq.enqueue(path=lane)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 1 and result["rejected"] == 0, result
    gate = result["outcomes"][0]["gate"]
    check = next(c for c in gate["checks"] if c["name"] == "contract-checks")
    assert check["ok"] is True
    assert "BAD_existing" in check["detail"]
    assert "pre-existing" in check["detail"]
    # The volatile count itself must never surface as a blocking "NEW
    # violation" line — proving it was excluded from the diff, not merely
    # that this particular run happened not to trip on it.
    assert "NEW violation" not in check["detail"]


def test_a_genuinely_new_violation_still_blocks_alongside_volatile_output(
    canonical: Path,
) -> None:
    """The critical other half: stripping volatile report metadata from the
    comparison must never mask a REAL new violation printed by the same
    script. A candidate that adds a second, distinct violation to the
    already-red file must still be rejected, with the new violation
    (not the pre-existing one) named in the reason."""
    _with_volatile_itemized_contract(canonical)
    lane = _branch(
        canonical,
        "lane-adds-new-violation",
        {"pkg/core.py": "VALUE = 1\nBAD_existing = 1\nBAD_new = 2\n"},
    )
    mq.enqueue(path=lane)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 0 and result["rejected"] == 1
    gate = result["outcomes"][0]["gate"]
    check = next(c for c in gate["checks"] if c["name"] == "contract-checks")
    assert check["ok"] is False
    assert "NEW violation" in check["detail"]
    assert "BAD_new = 2" in check["detail"]


# A script that logs the literal ``--base`` value it was invoked with, to a
# path named by an env var the test controls — proves the fast tier forwards
# ``--base <base_sha>`` (the SAME commit anchoring the differential baseline)
# to a script that declares support for it, exactly as it already does for
# ``--repository-root``.
BASE_LOGGING_CONTRACT = """\
import os, pathlib, sys
if "--base" in sys.argv:
    value = sys.argv[sys.argv.index("--base") + 1]
    pathlib.Path(os.environ["MQ_TEST_BASE_LOG"]).write_text(value)
sys.exit(0)
"""


def test_base_sha_is_forwarded_to_a_script_that_declares_it(
    canonical: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D-MQ-FP-1 root cause, closed: a script that scans "the range not yet
    merged" needs to be told what that range is anchored to. Left undeclared,
    such a script silently falls back to its OWN idea of the base (measured
    in production as ``origin/main`` sitting 100+ commits behind local
    ``main``), scanning an entirely different — and volatile — range on the
    base-tree run vs. the merged-tree run. The fast tier must instead forward
    the SAME base_sha :func:`compute_contract_baseline` already anchors its
    own baseline run to."""
    log = tmp_path / "base_seen.txt"
    monkeypatch.setenv("MQ_TEST_BASE_LOG", str(log))
    _write(canonical, "scripts/security/check_base_logging.py", BASE_LOGGING_CONTRACT)
    _commit(canonical, "main: a contract that logs the --base it receives")
    expected_base_sha = _run(["git", "rev-parse", "main"], canonical)

    lane = _branch(canonical, "lane-x", {"pkg/other.py": "X = 1\n"})
    mq.enqueue(path=lane)
    result = mq.run_queue(path=canonical, prune=False)

    assert result["landed"] == 1, result
    assert log.is_file(), "the script never saw --base at all"
    assert log.read_text() == expected_base_sha


# ---------------------------------------------------------------------------
# Regenerate-on-land — a conflict confined to GENERATED_FILES is resolved by
# regenerating them from the merged truth, never by rejecting the candidate
# or picking a side (CONCEPT:AU-OS.governance.merge-queue-regenerate-on-land)
# ---------------------------------------------------------------------------


def _with_fake_generators(canonical: Path) -> None:
    """Stub ``scripts/{build_concepts_yaml,gen_docs,gen_agents_md}.py`` that
    write FIXED, deterministic content — enough to prove the regenerate-on-land
    mechanism runs the real commands and uses their output, without needing the
    real doc-generation logic in a synthetic test repo."""
    _write(
        canonical,
        "scripts/build_concepts_yaml.py",
        "from pathlib import Path\n"
        "Path('docs').mkdir(exist_ok=True)\n"
        "Path('docs/concepts.yaml').write_text('regenerated: true\\n')\n",
    )
    _write(
        canonical,
        "scripts/gen_docs.py",
        "from pathlib import Path\n"
        "Path('README.md').write_text('REGENERATED README\\n')\n",
    )
    _write(
        canonical,
        "scripts/gen_agents_md.py",
        "from pathlib import Path\n"
        "Path('docs').mkdir(exist_ok=True)\n"
        "Path('AGENTS.md').write_text('REGENERATED AGENTS\\n')\n"
        "Path('docs/project_structure.md').write_text('REGENERATED STRUCTURE\\n')\n",
    )
    _write(canonical, "docs/concepts.yaml", "concepts: stale\n")
    _write(canonical, "README.md", "stale readme\n")
    _write(canonical, "AGENTS.md", "stale agents\n")
    _write(canonical, "docs/project_structure.md", "stale structure\n")
    _commit(canonical, "add fake generators + generated files")


def test_conflict_confined_to_generated_files_is_regenerated_not_rejected(
    canonical: Path,
) -> None:
    """Two lanes each hand-edit README.md (simulating each having run its own
    stale copy of the generator locally) alongside an UNRELATED source change.
    Merging the two branches conflicts on README.md alone -- the real-world
    shape the module docstring describes ("every land regenerates ... so
    every other branch then conflicts"). The queue must resolve this itself
    by regenerating GENERATED_FILES from the merged tree, landing BOTH
    candidates, rather than rejecting either."""
    _with_fake_generators(canonical)
    a = _branch(
        canonical,
        "lane-a",
        {"pkg/a.py": "A = 1\n", "README.md": "lane-a's stale regeneration\n"},
    )
    b = _branch(
        canonical,
        "lane-b",
        {"pkg/b.py": "B = 1\n", "README.md": "lane-b's stale regeneration\n"},
    )
    mq.enqueue(path=a)
    mq.enqueue(path=b)
    # Confirm the premise BEFORE resolution: git itself cannot auto-merge the
    # two branches' README.md edits -- that is the conflict this feature
    # must catch and resolve, not something already harmless.
    scope = lanes.lane_scope(canonical)
    head0 = mq._require_git(["rev-parse", "main"], canonical)
    raw_trial = mq.trial_merge(canonical, head0, "lane-a")
    assert raw_trial.ok
    chained = mq._commit_trial(canonical, raw_trial.tree, [head0], "premise: lane-a")
    raw_trial_2 = mq.trial_merge(canonical, chained, "lane-b")
    assert not raw_trial_2.ok and "README.md" in raw_trial_2.conflicts

    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 2 and result["rejected"] == 0, result
    head = mq._require_git(["rev-parse", "main"], canonical)
    with mq.materialized(canonical, head, scope=scope) as tree:
        # Regenerated from the (fake) generator, not either lane's stale copy.
        assert (tree / "README.md").read_text() == "REGENERATED README\n"
        assert (tree / "docs" / "concepts.yaml").read_text() == "regenerated: true\n"
        assert (tree / "AGENTS.md").read_text() == "REGENERATED AGENTS\n"
        assert (
            tree / "docs" / "project_structure.md"
        ).read_text() == "REGENERATED STRUCTURE\n"
        # Both lanes' REAL (non-generated) source changes are present.
        assert (tree / "pkg" / "a.py").read_text() == "A = 1\n"
        assert (tree / "pkg" / "b.py").read_text() == "B = 1\n"


def test_conflict_outside_generated_files_is_still_rejected(canonical: Path) -> None:
    """The narrowness guarantee: if even one conflicted path falls OUTSIDE
    GENERATED_FILES, regeneration must NOT kick in -- this stays a real,
    human-resolvable conflict."""
    _with_fake_generators(canonical)
    c = _branch(canonical, "lane-c", {"pkg/core.py": "VALUE = 1\nC = 1\n"})
    d = _branch(canonical, "lane-d", {"pkg/core.py": "VALUE = 1\nD = 1\n"})
    mq.enqueue(path=c)
    mq.enqueue(path=d)
    result = mq.run_queue(path=canonical, prune=False)
    assert result["landed"] == 1 and result["rejected"] == 1, result
    rejected = next(o for o in result["outcomes"] if not o["landed"])
    assert "pkg/core.py" in rejected["reason"]


# ---------------------------------------------------------------------------
# D-RMD-1 / D-RMQ-2 — land() must write the DECLARED base ref, and must verify it
# ---------------------------------------------------------------------------
def test_land_writes_the_declared_base_ref_when_canonical_is_not_on_it(
    canonical: Path,
) -> None:
    """The bug: ``merge --ff-only`` writes HEAD, not the declared base.

    It landed correctly only while the canonical checkout happened to sit on
    ``base`` — true for agent-utilities, false the moment the queue drives any
    other repo, a release branch, or a checkout left mid-bisect.
    """
    lane = _branch(canonical, "lane-x", {"pkg/x.py": "X = 1\n"})
    commit = _run(["git", "rev-parse", "lane-x"], canonical)
    # Put the canonical checkout on something OTHER than main — the exact
    # configuration the coincidence hid.
    _run(["git", "checkout", "-q", "-b", "parked"], canonical)
    assert _run(["git", "symbolic-ref", "--short", "HEAD"], canonical) == "parked"
    scope = lanes.lane_scope(canonical)

    mq.land(canonical, commit, base="main", scope=scope)

    # main must have advanced even though HEAD was elsewhere...
    assert _run(["git", "rev-parse", "refs/heads/main"], canonical) == commit
    # ...and the parked branch must NOT have been touched.
    assert _run(["git", "rev-parse", "parked"], canonical) != commit
    assert lane.exists()


def test_the_post_condition_catches_a_wrong_write_target_by_itself(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore D-RMD-1 verbatim, leaving ONLY the post-condition in place.

    This is the anti-vacuity proof: the assertion must do real work on its own,
    so it also catches the NEXT variant of the same mistake. If this test ever
    passes with the post-condition removed, the assertion is decoration.
    """
    lane = _branch(canonical, "lane-y", {"pkg/y.py": "Y = 1\n"})
    commit = _run(["git", "rev-parse", "lane-y"], canonical)
    _run(["git", "checkout", "-q", "-b", "parked"], canonical)
    scope = lanes.lane_scope(canonical)

    real_run_git = mq._run_git

    def _buggy(args: list[str], cwd: Path):  # type: ignore[no-untyped-def]
        # The original defect: CAS write to the declared ref becomes a merge
        # into whatever HEAD happens to be.
        if args[:1] == ["update-ref"]:
            return real_run_git(["merge", "--ff-only", commit], scope.main_tree)
        return real_run_git(args, cwd)

    monkeypatch.setattr(mq, "_run_git", _buggy)

    with pytest.raises(mq.MergeQueueError, match="POST-CONDITION FAILED"):
        mq.land(canonical, commit, base="main", scope=scope)

    # And the damage the post-condition prevents: main never moved, so nothing
    # may be reported landed or pruned as landed.
    assert _run(["git", "rev-parse", "refs/heads/main"], canonical) != commit
    assert lane.exists()


def test_land_refuses_when_the_base_is_checked_out_in_another_worktree(
    canonical: Path,
) -> None:
    """``update-ref`` would move the branch out from under another tree's HEAD.

    Git forbids this for ``checkout``/``branch -f`` but NOT for ``update-ref``,
    so the refusal has to be ours.
    """
    lane = _branch(canonical, "lane-z", {"pkg/z.py": "Z = 1\n"})
    commit = _run(["git", "rev-parse", "lane-z"], canonical)
    # Park the canonical checkout FIRST — git refuses to add a worktree for a
    # branch that is already checked out somewhere.
    _run(["git", "checkout", "-q", "-b", "parked"], canonical)
    other = canonical.parent / "holds-main"
    _run(["git", "worktree", "add", "-q", str(other), "main"], canonical)
    scope = lanes.lane_scope(canonical)

    with pytest.raises(mq.MergeQueueError, match="checked out in another worktree"):
        mq.land(canonical, commit, base="main", scope=scope)
    assert lane.exists()
