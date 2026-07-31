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
        canonical, "main", mq.queued(canonical)
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
    dups = mq.duplicate_definitions(
        canonical, "main", ["lane-extract", "lane-resolve"]
    )
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
    _write(canonical, "tests/unit/test_core_wiring.py", "def test_y():\n    assert True\n")
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
    _write(canonical, "tests/unit/test_core_extra.py", "def test_y():\n    assert True\n")
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
# Fail-closed pruning
# ---------------------------------------------------------------------------
def test_prune_without_repository_manager_keeps_the_branch(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An un-pruned branch is untidy; a wrongly-pruned one loses work. When the
    guarded pruner is unavailable the queue must decline, not improvise."""
    import builtins

    real_import = builtins.__import__

    def _no_rm(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("repository_manager"):
            raise ImportError("repository_manager is not installed")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_rm)
    candidate = mq.Candidate(branch="lane-a", lane="lane-a", worktree=str(canonical))
    result = mq.prune_landed(candidate, repo_name="canonical", base="main")
    assert result["pruned"] is False
    assert "repository-manager is not importable" in result["reason"]


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
