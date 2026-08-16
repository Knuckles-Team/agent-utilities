"""Proofs that the four lane-arbitration mechanisms actually refuse, defer, and survive.

These are wiring tests: each one reproduces a collision that really happened and
asserts the mechanism now makes it impossible, rather than asserting that a
function exists.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from agent_utilities import cli as au_cli
from agent_utilities.governance import concept_allocator as ca
from agent_utilities.governance import lanes


def _load_check_lane_guard():
    """Import ``scripts/check_lane_guard.py`` as a module (it is a script, not
    a package member, so every repo's pre-commit hook `exec`s it directly)."""
    import importlib.util

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "check_lane_guard", repo_root / "scripts" / "check_lane_guard.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        args, cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _init_repo(root: Path) -> Path:
    """A canonical checkout on `main` carrying an empty concept ledger."""
    root.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-b", "main"], root)
    _run(["git", "config", "user.email", "lane@test"], root)
    _run(["git", "config", "user.name", "Lane Test"], root)
    (root / "agent_utilities").mkdir(exist_ok=True)
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "concepts.yaml").write_text(
        yaml.safe_dump({"concepts": []}), encoding="utf-8"
    )
    (root / "docs" / "concept_reservations.yaml").write_text("", encoding="utf-8")
    _run(["git", "add", "-A"], root)
    _run(["git", "commit", "-qm", "base"], root)
    return root


@pytest.fixture(autouse=True)
def _isolate_workspace_arbitration_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route "workspace"-scoped leases into an isolated ``tmp_path`` instead of
    the REAL host-wide ``~/.local/state/agent-utilities/arbitration`` directory
    that live lane orchestration on this host actually uses for
    ``dependency-lock``/``epistemic-graph-daemon``/``workspace-scripts``
    (``lanes.workspace_arbitration_dir()`` — see ``lane_resources.yaml``:
    those three names are deliberately host-wide, not per-repo, because the
    shared venv/uv.lock and the shared local epistemic-graph daemon really are
    contended across every worktree of every repo on the host).

    Passing ``path=canonical`` to ``hold_lease``/``lease_status`` does NOT
    isolate these tests from that real directory -- ``_lease_dir`` ignores
    ``path`` for workspace-scoped resources by design (that is the behaviour
    under test). Without this fixture these tests take a REAL lease that a
    genuinely concurrent lane/agent on this host could be holding (a spurious
    ``LeaseUnavailable`` unrelated to the code under test) or that this test
    process could itself briefly hold, deferring a real concurrent full-suite
    run or ``uv sync`` elsewhere on the host. It also makes every one of these
    tests race every OTHER worker under ``pytest-xdist -n auto`` on the same
    real lease file, since they all pass through the same un-isolated function.
    """
    workspace_dir = tmp_path / "workspace-arbitration"

    def _fake_workspace_arbitration_dir() -> Path:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir

    monkeypatch.setattr(
        lanes, "workspace_arbitration_dir", _fake_workspace_arbitration_dir
    )


@pytest.fixture
def canonical(tmp_path: Path) -> Path:
    return _init_repo(tmp_path / "canonical")


def _add_worktree(canonical: Path, name: str) -> Path:
    path = canonical.parent / name
    _run(["git", "worktree", "add", "-q", str(path), "-b", name], canonical)
    return path


# ---------------------------------------------------------------------------
# READ-ONLY — the canonical checkout guard genuinely refuses
# ---------------------------------------------------------------------------
def test_scope_distinguishes_canonical_from_linked_worktree(canonical: Path) -> None:
    lane = _add_worktree(canonical, "lane-a")
    assert lanes.lane_scope(canonical).is_canonical is True
    scope = lanes.lane_scope(lane)
    assert scope.is_canonical is False
    assert scope.lane == "lane-a"
    # Both resolve to the SAME arbitration directory — that is what makes
    # cross-worktree arbitration possible at all.
    assert scope.arbitration_dir == lanes.lane_scope(canonical).arbitration_dir


def test_guard_refuses_to_edit_the_canonical_checkout(canonical: Path) -> None:
    with pytest.raises(lanes.CanonicalCheckoutError, match="worktree add"):
        lanes.require_mutable_tree(canonical, operation="edit")


def test_guard_allows_a_linked_worktree(canonical: Path) -> None:
    lane = _add_worktree(canonical, "lane-b")
    assert lanes.require_mutable_tree(lane, operation="edit").lane == "lane-b"


# ---------------------------------------------------------------------------
# check_lane_guard.py — the pre-commit hook every repo's `.pre-commit-config.yaml`
# `exec`s directly. Regression: a PUSH stages nothing, and the hook's
# `always_run: true` lane-guard check fired for every repo's pre-push gate too
# -- an empty `staged` list fell through the old `if staged and ...:` carve-out
# straight into REFUSED, blocking pushes fleet-wide from every canonical
# checkout even though nothing was being committed.
# ---------------------------------------------------------------------------
def test_lane_guard_allows_a_canonical_push_with_nothing_staged(
    canonical: Path,
) -> None:
    clg = _load_check_lane_guard()
    scope = lanes.lane_scope(canonical)
    assert scope.is_canonical is True
    assert clg._check_canonical(scope, staged=[]) is None


def test_lane_guard_still_refuses_a_real_canonical_commit(canonical: Path) -> None:
    clg = _load_check_lane_guard()
    scope = lanes.lane_scope(canonical)
    assert clg._check_canonical(scope, staged=["some/real/file.py"]) is not None


def test_lane_guard_allows_a_pure_bumpversion_commit(canonical: Path) -> None:
    (canonical / ".bumpversion.cfg").write_text(
        "[bumpversion:file:pyproject.toml]\n", encoding="utf-8"
    )
    clg = _load_check_lane_guard()
    scope = lanes.lane_scope(canonical)
    assert clg._check_canonical(scope, staged=["pyproject.toml"]) is None


def test_guard_allows_the_canonical_merge_back(canonical: Path) -> None:
    """The one sanctioned canonical mutation, detected from git's own state."""
    head = _run(["git", "rev-parse", "HEAD"], canonical)
    (canonical / ".git" / "MERGE_HEAD").write_text(head + "\n", encoding="utf-8")
    assert lanes.require_mutable_tree(canonical, operation="merge").is_canonical


def test_global_actor_cannot_reset_a_dirty_tree_it_does_not_own(
    canonical: Path,
) -> None:
    """The exact collision: a background sync resetting a lane mid-pre-commit."""
    lane = _add_worktree(canonical, "lane-c")
    (lane / "work_in_progress.py").write_text(
        "# 20 minutes of work\n", encoding="utf-8"
    )
    assert lanes.tree_has_uncommitted_work(lane) is True
    with pytest.raises(lanes.UnownedTreeError, match="uncommitted work"):
        lanes.require_resettable_tree(lane, operation="reset", owner="background-sync")


def test_a_lane_may_reset_its_own_tree(canonical: Path) -> None:
    lane = _add_worktree(canonical, "lane-d")
    (lane / "scratch.py").write_text("x = 1\n", encoding="utf-8")
    lanes.require_resettable_tree(lane, operation="reset", owner="lane-d")


def test_a_clean_foreign_tree_stays_resettable(canonical: Path) -> None:
    lane = _add_worktree(canonical, "lane-e")
    lanes.require_resettable_tree(lane, operation="reset", owner="background-sync")


# ---------------------------------------------------------------------------
# LEASE — the second holder genuinely defers
# ---------------------------------------------------------------------------
def test_lease_defers_a_second_holder_and_frees_on_exit(canonical: Path) -> None:
    with lanes.hold_lease("dependency-lock", operation="uv sync", path=canonical):
        holder = lanes.lease_status("dependency-lock", canonical)
        assert holder is not None and holder["operation"] == "uv sync"
        with pytest.raises(lanes.LeaseUnavailable, match="defer, do not proceed"):
            with lanes.hold_lease(
                "dependency-lock", operation="second relock", path=canonical
            ):
                pytest.fail("a second lane acquired a held lease")
    assert lanes.lease_status("dependency-lock", canonical) is None


def test_lease_is_shared_across_worktrees_not_per_worktree(canonical: Path) -> None:
    """A lease taken from one worktree must block a lane in another."""
    lane = _add_worktree(canonical, "lane-f")
    with lanes.hold_lease("precommit-all-files", operation="gate", path=canonical):
        with pytest.raises(lanes.LeaseUnavailable):
            with lanes.hold_lease("precommit-all-files", operation="gate", path=lane):
                pytest.fail("worktrees took independent leases")


def test_hold_lease_persists_and_exposes_the_declared_owner(canonical: Path) -> None:
    """D-CDX-14: an optional caller-declared owner identity (fleet/session,
    distinguishing Claude/Codex/human/vLLM actors) is persisted into the
    lease record and readable via lease_status - not discarded."""
    owner = {"fleet": "codex", "session": "codex-session-42"}
    with lanes.hold_lease(
        "dependency-lock", operation="uv sync", path=canonical, owner=owner
    ):
        holder = lanes.lease_status("dependency-lock", canonical)
        assert holder is not None
        assert holder["owner"] == owner
    assert lanes.lease_status("dependency-lock", canonical) is None


def test_hold_lease_without_an_owner_omits_the_key(canonical: Path) -> None:
    """Backward compatible: no owner passed means no owner key at all - not a
    default/placeholder value that could be mistaken for a real one."""
    with lanes.hold_lease("dependency-lock", operation="uv sync", path=canonical):
        holder = lanes.lease_status("dependency-lock", canonical)
        assert holder is not None
        assert "owner" not in holder


def test_workspace_scoped_lease_leaves_the_repo_scope(canonical: Path) -> None:
    """The shared venv is contended across repos, so its lease cannot live in one."""
    assert lanes.resource_scope("dependency-lock") == "workspace"
    assert lanes.resource_scope("precommit-all-files") == "repo"
    repo_leases = lanes.lane_scope(canonical).arbitration_dir / "leases"
    with lanes.hold_lease("dependency-lock", operation="uv sync", path=canonical):
        assert not (repo_leases / "dependency-lock.lease").exists()
        assert (
            lanes.workspace_arbitration_dir() / "leases" / "dependency-lock.lease"
        ).exists()
    with lanes.hold_lease("precommit-all-files", operation="gate", path=canonical):
        assert (repo_leases / "precommit-all-files.lease").exists()


def test_engine_daemon_lease_is_workspace_scoped_like_the_shared_venv(
    canonical: Path,
) -> None:
    """The shared local epistemic-graph daemon is one host-wide singleton —
    engine_resolver's share-running-local/autostart-shared-supervised
    precedence hands it to every entrypoint on the host, across every repo's
    worktrees, exactly like the shared venv (`dependency-lock`)."""
    assert lanes.resource_scope("epistemic-graph-daemon") == "workspace"
    repo_leases = lanes.lane_scope(canonical).arbitration_dir / "leases"
    with lanes.hold_lease(
        "epistemic-graph-daemon", operation="pytest full-suite", path=canonical
    ):
        assert not (repo_leases / "epistemic-graph-daemon.lease").exists()
        assert (
            lanes.workspace_arbitration_dir()
            / "leases"
            / "epistemic-graph-daemon.lease"
        ).exists()
    assert lanes.lease_status("epistemic-graph-daemon", canonical) is None


def test_engine_daemon_lease_defers_a_second_lanes_full_suite(canonical: Path) -> None:
    """The exact incident: a second lane's full-suite run must defer rather
    than pile onto the same shared epistemic-graph daemon."""
    lane = _add_worktree(canonical, "lane-eg")
    with lanes.hold_lease(
        "epistemic-graph-daemon", operation="lane-a full-suite", path=canonical
    ):
        with pytest.raises(lanes.LeaseUnavailable, match="defer, do not proceed"):
            with lanes.hold_lease(
                "epistemic-graph-daemon", operation="lane-b full-suite", path=lane
            ):
                pytest.fail("a second lane's full-suite run acquired a held lease")
    assert lanes.lease_status("epistemic-graph-daemon", canonical) is None


def test_workspace_scripts_lease_is_workspace_scoped(canonical: Path) -> None:
    """D-ORC-28: /home/apps/workspace/scripts/ is not a git worktree of any repo
    and cannot be PARTITIONed like cargo/pytest/scratch — it is one directory
    every lane on the host invokes, so its edit lease must be workspace-scoped
    like the shared venv and the engine daemon, not repo-scoped."""
    assert lanes.resource_scope("workspace-scripts") == "workspace"
    assert lanes.resource_class("workspace-scripts") is lanes.ArbitrationClass.LEASE
    repo_leases = lanes.lane_scope(canonical).arbitration_dir / "leases"
    with lanes.hold_lease(
        "workspace-scripts",
        operation="editing scripts/deferred_registry.py",
        path=canonical,
    ):
        assert not (repo_leases / "workspace-scripts.lease").exists()
        assert (
            lanes.workspace_arbitration_dir() / "leases" / "workspace-scripts.lease"
        ).exists()
    assert lanes.lease_status("workspace-scripts", canonical) is None


def test_workspace_scripts_lease_defers_a_racing_editor(canonical: Path) -> None:
    """The exact D-ORC-28 incident: two lanes editing scripts/deferred_registry.py
    at once left a broken intermediate state (consumption code referencing an
    argparse attribute that did not exist yet) visible to every lane's CLI
    invocation. The lease must make a second, concurrent editor defer instead
    of racing the first one's in-progress edit."""
    lane = _add_worktree(canonical, "lane-scripts")
    with lanes.hold_lease(
        "workspace-scripts",
        operation="lane-a editing deferred_registry.py",
        path=canonical,
    ):
        with pytest.raises(lanes.LeaseUnavailable, match="defer, do not proceed"):
            with lanes.hold_lease(
                "workspace-scripts",
                operation="lane-b editing deferred_registry.py",
                path=lane,
            ):
                pytest.fail("a second lane's edit acquired a held lease")
    assert lanes.lease_status("workspace-scripts", canonical) is None


def test_guarded_tree_mutation_refuses_and_releases(canonical: Path) -> None:
    """One choke point: lease held across the whole check-then-mutate."""
    lane = _add_worktree(canonical, "lane-guarded")
    (lane / "wip.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(lanes.UnownedTreeError):
        with lanes.guarded_tree_mutation(
            lane, operation="checkout", owner="background-sync"
        ):
            pytest.fail("mutation proceeded against a dirty foreign tree")
    # The refusal must not leak the lease, or one bad call wedges every actor.
    assert lanes.lease_status("canonical-mutation", lane) is None


def test_lease_reclaims_a_dead_holder(canonical: Path) -> None:
    """A crashed lane must not wedge the workspace forever."""
    lease_dir = lanes.lane_scope(canonical).arbitration_dir / "leases"
    lease_dir.mkdir(parents=True, exist_ok=True)
    import json
    import socket

    (lease_dir / "dependency-lock.lease").write_text(
        json.dumps(
            {
                "name": "dependency-lock",
                "lane": "ghost",
                "operation": "crashed",
                "pid": 2**22,  # above any live pid on Linux
                "host": socket.gethostname(),
                "expires_at": "2999-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    assert lanes.lease_status("dependency-lock", canonical) is None
    with lanes.hold_lease("dependency-lock", operation="recovered", path=canonical):
        pass


def test_expired_lease_is_reclaimed(canonical: Path) -> None:
    with pytest.raises(lanes.LeaseUnavailable):
        with lanes.hold_lease("dependency-lock", operation="a", path=canonical):
            with lanes.hold_lease("dependency-lock", operation="b", path=canonical):
                pass
    with lanes.hold_lease(
        "dependency-lock", operation="a", ttl_seconds=-1, path=canonical
    ):
        # Already expired, so a second lane may proceed rather than wedge.
        with lanes.hold_lease("dependency-lock", operation="b", path=canonical):
            pass


# ---------------------------------------------------------------------------
# PARTITION — nothing is shared
# ---------------------------------------------------------------------------
def test_partitioned_paths_differ_per_lane_and_never_use_refs_stash(
    canonical: Path,
) -> None:
    one = lanes.partitioned_paths(_add_worktree(canonical, "lane-g"))
    two = lanes.partitioned_paths(_add_worktree(canonical, "lane-h"))
    assert one.cargo_target_dir != two.cargo_target_dir
    assert one.pytest_basetemp != two.pytest_basetemp
    assert one.scratch_dir != two.scratch_dir
    assert one.precommit_home != two.precommit_home
    assert one.stash_ref != two.stash_ref
    assert one.stash_ref == "refs/lane/lane-g/stash" != "refs/stash"
    # It must not be the shared pre-commit store every lane used to collide on.
    shared_store = Path.home() / ".cache" / "pre-commit"
    assert one.precommit_home != shared_store
    assert two.precommit_home != shared_store


def test_precommit_home_is_classified_partition() -> None:
    assert lanes.resource_class("precommit-home") is lanes.ArbitrationClass.PARTITION


def test_partitioned_paths_creates_the_temp_root_on_disk(canonical: Path) -> None:
    """Reproduces the real failure: a lane whose ``~/.al/<token>`` had never
    been created before exported ``PYTEST_ADDOPTS=--basetemp=<token>/pytest``
    that pointed at a nonexistent parent. Pytest's own basetemp creation is
    ``mkdir(exist_ok=True)`` with no ``parents=True``, so the very first
    ``pytest`` run under a fresh lane's ``env`` output failed with a raw
    ``FileNotFoundError`` before any test collected — observed for real via
    ``guardrail-prod-profile`` in a freshly created worktree. ``env`` must
    leave both the pytest-basetemp parent and the scratch (``TMPDIR``)
    directory itself already present, exactly as ``workspace_arbitration_dir``
    already guarantees for the shared-venv lease root.
    """
    lane = _add_worktree(canonical, "lane-fresh-temp")
    parts = lanes.partitioned_paths(lane)
    assert parts.pytest_basetemp.parent.is_dir()
    assert parts.scratch_dir.is_dir()
    # Round-trips pytest's own creation step for the leaf itself, proving the
    # parent is genuinely walkable and not merely reported as existing.
    parts.pytest_basetemp.mkdir(mode=0o700, exist_ok=True)
    assert parts.pytest_basetemp.is_dir()


def test_park_gives_a_clean_tree_without_touching_refs_stash(canonical: Path) -> None:
    """The affordance that makes `git stash` unnecessary rather than merely banned."""
    lane = _add_worktree(canonical, "lane-park")
    tracked = lane / "docs" / "concepts.yaml"
    tracked.write_text("concepts: [{id: AU-KG.compute.wip}]\n", encoding="utf-8")
    assert lanes.tree_has_uncommitted_work(lane) is True

    parked = lanes.park_worktree(lane)
    assert parked["parked"] is True
    assert parked["ref"] == "refs/lane/lane-park/stash"
    assert lanes.tree_has_uncommitted_work(lane) is False
    # The shared ref every worktree contends for must be untouched.
    assert (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "refs/stash"],
            cwd=str(lane),
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        != 0
    )

    lanes.unpark_worktree(lane)
    assert "AU-KG.compute.wip" in tracked.read_text(encoding="utf-8")
    assert lanes.tree_has_uncommitted_work(lane) is True


def test_two_lanes_park_independently(canonical: Path) -> None:
    one = _add_worktree(canonical, "lane-park-a")
    two = _add_worktree(canonical, "lane-park-b")
    for tree, marker in ((one, "alpha"), (two, "beta")):
        (tree / "docs" / "concepts.yaml").write_text(
            f"concepts: []  # {marker}\n", encoding="utf-8"
        )
        lanes.park_worktree(tree)
    for tree, marker in ((one, "alpha"), (two, "beta")):
        lanes.unpark_worktree(tree)
        assert marker in (tree / "docs" / "concepts.yaml").read_text(encoding="utf-8")


def test_park_refuses_the_canonical_checkout(canonical: Path) -> None:
    (canonical / "docs" / "concepts.yaml").write_text(
        "concepts: []\n", encoding="utf-8"
    )
    with pytest.raises(lanes.CanonicalCheckoutError):
        lanes.park_worktree(canonical)


def test_unpark_without_a_park_is_a_loud_error(canonical: Path) -> None:
    lane = _add_worktree(canonical, "lane-park-c")
    with pytest.raises(lanes.LaneArbitrationError, match="nothing parked"):
        lanes.unpark_worktree(lane)


# ---------------------------------------------------------------------------
# PARTITION — the pre-commit store's orphan detector discriminates instead of
# treating "a patch file exists" as proof of an incident (D-OB-12)
# ---------------------------------------------------------------------------
def _write_precommit_patch(
    precommit_home: Path, *, epoch: int, pid: int, content: bytes
) -> Path:
    precommit_home.mkdir(parents=True, exist_ok=True)
    patch_path = precommit_home / f"patch{epoch}-{pid}"
    patch_path.write_bytes(content)
    return patch_path


def test_orphaned_patches_is_empty_when_the_store_does_not_exist(
    canonical: Path,
) -> None:
    lane = _add_worktree(canonical, "lane-oc-empty")
    assert lanes.orphaned_precommit_patches(lane) == []


def test_orphan_detector_discriminates_orphaned_from_restored(
    canonical: Path,
) -> None:
    """The real end-to-end shape pre-commit creates: capture a real unstaged
    change into a patch<epoch>-<pid> file exactly as
    ``_unstaged_changes_cleared`` does, discard the change (the crash window),
    and prove the detector calls it ORPHANED — then replay the patch and prove
    the SAME function now calls it "restored". A detector that reports every
    patch file as an incident would pass the first half and fail to
    discriminate on the second; this proves it does not.
    """
    lane = _add_worktree(canonical, "lane-orphan")
    target = lane / "docs" / "concepts.yaml"
    target.write_text(
        "concepts: [{id: AU-KG.compute.orphan-proof}]\n", encoding="utf-8"
    )

    tree = _run(["git", "write-tree"], lane)
    diff = subprocess.run(
        [
            "git",
            "diff-index",
            "--ignore-submodules",
            "--binary",
            "--exit-code",
            "--no-color",
            "--no-ext-diff",
            tree,
            "--",
        ],
        cwd=str(lane),
        capture_output=True,
        check=False,
    ).stdout
    assert diff  # there is a real unstaged change to capture

    precommit_home = lanes.partitioned_paths(lane).precommit_home
    dead_pid = 2**22  # above any live pid on Linux (same convention as the lease test)
    patch_path = _write_precommit_patch(
        precommit_home, epoch=int(time.time()), pid=dead_pid, content=diff
    )

    # Simulate the crash window: pre-commit checked the change away and never
    # reached the `finally: git apply` that restores it.
    _run(["git", "checkout", "--", "."], lane)
    assert "AU-KG.compute.orphan-proof" not in target.read_text(encoding="utf-8")

    report = lanes.orphaned_precommit_patches(lane)
    assert len(report) == 1
    entry = report[0]
    assert entry["path"] == str(patch_path)
    assert entry["state"] == "ORPHANED"
    assert entry["pid"] == dead_pid
    assert "replay" in entry and str(patch_path) in entry["replay"]

    # Now actually replay it, exactly as the reported remedy says.
    _run(["git", "apply", str(patch_path)], lane)
    assert "AU-KG.compute.orphan-proof" in target.read_text(encoding="utf-8")

    report_after = lanes.orphaned_precommit_patches(lane)
    assert len(report_after) == 1
    assert report_after[0]["state"] == "restored"
    assert report_after[0]["path"] == str(patch_path)


def test_orphan_detector_reports_a_live_pid_as_in_progress(canonical: Path) -> None:
    lane = _add_worktree(canonical, "lane-inprogress")
    precommit_home = lanes.partitioned_paths(lane).precommit_home
    patch_path = _write_precommit_patch(
        precommit_home, epoch=int(time.time()), pid=os.getpid(), content=b"junk"
    )
    report = lanes.orphaned_precommit_patches(lane)
    assert len(report) == 1
    assert report[0]["path"] == str(patch_path)
    assert report[0]["state"] == "in-progress"
    assert report[0]["pid"] == os.getpid()


def test_orphan_detector_reports_unparseable_filenames_as_unknown(
    canonical: Path,
) -> None:
    lane = _add_worktree(canonical, "lane-unknown")
    precommit_home = lanes.partitioned_paths(lane).precommit_home
    precommit_home.mkdir(parents=True, exist_ok=True)
    (precommit_home / "patch-not-a-real-name").write_text("junk", encoding="utf-8")
    report = lanes.orphaned_precommit_patches(lane)
    assert len(report) == 1
    assert report[0]["state"] == "unknown"


def test_lane_report_surfaces_only_orphaned_and_unknown_patches(
    canonical: Path,
) -> None:
    """A clean lane's report carries an empty list; a crashed one's is loud."""
    lane = _add_worktree(canonical, "lane-report-clean")
    report = lanes.lane_report(lane)
    assert report["partitioned"]["precommit_home"] == str(
        lanes.partitioned_paths(lane).precommit_home
    )
    assert report["orphaned_precommit_patches"] == []

    precommit_home = lanes.partitioned_paths(lane).precommit_home
    dead_pid = 2**22
    _write_precommit_patch(
        precommit_home,
        epoch=int(time.time()),
        pid=dead_pid,
        content=b"not a real diff",
    )
    dirty_report = lanes.lane_report(lane)
    assert len(dirty_report["orphaned_precommit_patches"]) == 1
    assert dirty_report["orphaned_precommit_patches"][0]["state"] == "ORPHANED"


# ---------------------------------------------------------------------------
# APPEND-ONLY — concurrent writers both survive
# ---------------------------------------------------------------------------
def test_fragment_append_never_touches_another_lanes_file(tmp_path: Path) -> None:
    store = lanes.FragmentStore(root=tmp_path / "frag", key="id")
    store.append({"id": "a", "v": 1}, lane="lane-1")
    store.append({"id": "b", "v": 1}, lane="lane-2")
    before = (store.fragment_for("lane-1")).read_bytes()
    store.append({"id": "c", "v": 1}, lane="lane-2")
    assert store.fragment_for("lane-1").read_bytes() == before
    assert [r["id"] for r in store.fold()] == ["a", "b", "c"]


def test_fold_collapses_restated_records_instead_of_conflicting(
    tmp_path: Path,
) -> None:
    store = lanes.FragmentStore(root=tmp_path / "frag", key="id")
    store.append({"id": "a", "status": "reserved"}, lane="lane-1")
    store.append({"id": "a", "status": "landed"}, lane="lane-2")
    folded = store.fold()
    assert len(folded) == 1 and folded[0]["status"] == "landed"


def _reserve_in(worktree: str, concept_id: str, queue: object) -> None:
    from pathlib import Path as P

    from agent_utilities.governance import concept_allocator as allocator

    try:
        allocator.reserve_concept_id(
            concept_id, session_id="lane", repo_root=P(worktree)
        )
    except ValueError:
        queue.put(("denied", concept_id))  # type: ignore[attr-defined]
    else:
        queue.put(("reserved", concept_id))  # type: ignore[attr-defined]


def test_two_concurrent_reservations_in_two_worktrees_both_survive(
    canonical: Path,
) -> None:
    """The clobber case: two lanes reserving at once must BOTH keep their claim."""
    one = _add_worktree(canonical, "lane-i")
    two = _add_worktree(canonical, "lane-j")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    jobs = [
        ctx.Process(
            target=_reserve_in, args=(str(one), "AU-KG.compute.lane-one", queue)
        ),
        ctx.Process(
            target=_reserve_in, args=(str(two), "AU-KG.compute.lane-two", queue)
        ),
    ]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join(timeout=60)
        assert job.exitcode == 0
    assert sorted(queue.get(timeout=5)[0] for _ in range(2)) == ["reserved", "reserved"]
    # Each lane wrote ONLY its own fragment, and each still holds its own claim.
    assert [r["id"] for r in ca.read_ledger(one)] == ["AU-KG.compute.lane-one"]
    assert [r["id"] for r in ca.read_ledger(two)] == ["AU-KG.compute.lane-two"]
    assert ca.fragment_dir(one).glob("*.yaml") is not None
    assert [p.name for p in sorted(ca.fragment_dir(one).glob("*.yaml"))] == [
        "lane-i.yaml"
    ]
    assert [p.name for p in sorted(ca.fragment_dir(two).glob("*.yaml"))] == [
        "lane-j.yaml"
    ]


def test_same_id_claimed_from_two_worktrees_has_exactly_one_winner(
    canonical: Path,
) -> None:
    """The race the old per-worktree lock could not see, let alone prevent."""
    one = _add_worktree(canonical, "lane-k")
    two = _add_worktree(canonical, "lane-l")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    contended = "AU-KG.compute.contended"
    jobs = [
        ctx.Process(target=_reserve_in, args=(str(path), contended, queue))
        for path in (one, two)
    ]
    for job in jobs:
        job.start()
    for job in jobs:
        job.join(timeout=60)
        assert job.exitcode == 0
    assert sorted(queue.get(timeout=5)[0] for _ in range(2)) == ["denied", "reserved"]


def test_reserving_from_a_worktree_never_writes_the_canonical_tree(
    canonical: Path,
) -> None:
    lane = _add_worktree(canonical, "lane-m")
    before = (canonical / "docs" / "concept_reservations.yaml").read_text(
        encoding="utf-8"
    )
    ca.reserve_concept_id("AU-KG.compute.isolated", session_id="lane", repo_root=lane)
    assert (canonical / "docs" / "concept_reservations.yaml").read_text(
        encoding="utf-8"
    ) == before
    assert not (canonical / "docs" / "concept_reservations.d").exists()


def test_reconcile_sees_a_marker_that_landed_in_the_worktree(canonical: Path) -> None:
    """Evidence item 6: reconcile scanned canonical and refused to flip a branch claim."""
    lane = _add_worktree(canonical, "lane-n")
    concept_id = "AU-KG.compute.branch-marker"
    ca.reserve_concept_id(concept_id, session_id="lane", repo_root=lane)
    (lane / "agent_utilities").mkdir(exist_ok=True)
    (lane / "agent_utilities" / "feature.py").write_text(
        f"# CONCEPT:{concept_id}\n", encoding="utf-8"
    )
    assert ca.reconcile(repo_root=lane)["landed"] == [concept_id]
    assert [r["id"] for r in ca.list_reservations(repo_root=lane, status="landed")] == [
        concept_id
    ]


def test_a_lane_cannot_release_another_lanes_claim(canonical: Path) -> None:
    one = _add_worktree(canonical, "lane-o")
    two = _add_worktree(canonical, "lane-p")
    concept_id = "AU-KG.compute.owned-elsewhere"
    ca.reserve_concept_id(concept_id, session_id="lane", repo_root=one)
    # Make lane-p see lane-o's fragment (as a merge would) without granting it
    # permission to rewrite that fragment.
    target = ca.fragment_dir(two)
    target.mkdir(parents=True, exist_ok=True)
    (target / "lane-o.yaml").write_text(
        (ca.fragment_dir(one) / "lane-o.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reserved by another lane"):
        ca.release_concept_id(concept_id, repo_root=two)


# ---------------------------------------------------------------------------
# The classification is complete and enforced
# ---------------------------------------------------------------------------
def test_every_known_collision_resource_is_classified() -> None:
    classified = {r.name: r.arbitration for r in lanes.resource_rules()}
    assert classified["cargo-target-dir"] is lanes.ArbitrationClass.PARTITION
    assert classified["git-stash"] is lanes.ArbitrationClass.PARTITION
    assert classified["concept-reservations"] is lanes.ArbitrationClass.APPEND_ONLY
    assert classified["deferred-register"] is lanes.ArbitrationClass.APPEND_ONLY
    assert classified["dependency-lock"] is lanes.ArbitrationClass.LEASE
    assert classified["reconciliation-merge"] is lanes.ArbitrationClass.LEASE
    assert classified["epistemic-graph-daemon"] is lanes.ArbitrationClass.LEASE
    assert classified["workspace-scripts"] is lanes.ArbitrationClass.LEASE
    assert classified["canonical-checkout"] is lanes.ArbitrationClass.READ_ONLY
    assert all(r.evidence.strip() for r in lanes.resource_rules())


def test_an_unclassified_resource_is_a_hard_error() -> None:
    with pytest.raises(
        lanes.LaneArbitrationError, match="unclassified shared resource"
    ):
        lanes.resource_class("some-new-shared-thing")


def test_lane_identity_degrades_safely_outside_a_repo(tmp_path: Path) -> None:
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    assert lanes.current_tree(outside) is None
    assert lanes.lane_name(outside) == "local"
    assert lanes.shared_arbitration_dir(outside) is None


def test_lane_report_names_the_canonical_checkout(canonical: Path) -> None:
    lane = _add_worktree(canonical, "lane-q")
    report = lanes.lane_report(lane)
    assert report["is_canonical"] is False
    assert report["canonical_checkout"] == str(canonical)
    assert report["partitioned"]["stash_ref"].startswith("refs/lane/")
    assert set(report["leases"]) == {
        r.name
        for r in lanes.resource_rules()
        if r.arbitration is lanes.ArbitrationClass.LEASE
    }


# ---------------------------------------------------------------------------
# The pre-commit gate itself — enforcement, not just a library that could refuse
# ---------------------------------------------------------------------------
def _guard_module():
    import sys

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts"))
    import check_lane_guard  # type: ignore[import-not-found]

    return check_lane_guard


def test_gate_refuses_a_source_commit_in_the_canonical_checkout(
    canonical: Path,
) -> None:
    guard = _guard_module()
    scope = lanes.lane_scope(canonical)
    refusal = guard._check_canonical(scope, ["agent_utilities/thing.py"])
    assert refusal is not None
    assert "CANONICAL checkout" in refusal and "worktree add" in refusal


def test_gate_allows_a_lane_commit_and_a_canonical_merge(canonical: Path) -> None:
    guard = _guard_module()
    lane = _add_worktree(canonical, "lane-r")
    assert guard._check_canonical(lanes.lane_scope(lane), ["a.py"]) is None
    head = _run(["git", "rev-parse", "HEAD"], canonical)
    (canonical / ".git" / "MERGE_HEAD").write_text(head + "\n", encoding="utf-8")
    assert guard._check_canonical(lanes.lane_scope(canonical), ["a.py"]) is None


def test_gate_refuses_a_hand_edited_generated_ledger_view(canonical: Path) -> None:
    guard = _guard_module()
    lane = _add_worktree(canonical, "lane-s")
    ca.reserve_concept_id("AU-KG.compute.gated", session_id="lane", repo_root=lane)
    view = lane / "docs" / "concept_reservations.yaml"
    assert guard._check_generated_view(lane, ["docs/concept_reservations.yaml"]) is None
    view.write_text(
        view.read_text(encoding="utf-8") + "- {id: hand-edited}\n", encoding="utf-8"
    )
    refusal = guard._check_generated_view(lane, ["docs/concept_reservations.yaml"])
    assert refusal is not None and "GENERATED" in refusal


def test_guard_refusal_names_the_remedy(canonical: Path) -> None:
    """A refusal that does not say what to do instead just gets worked around."""
    with pytest.raises(lanes.CanonicalCheckoutError) as excinfo:
        lanes.require_mutable_tree(canonical, operation="write the ledger")
    message = str(excinfo.value)
    assert "worktree add" in message and str(canonical) in message
    assert os.path.sep in message


# ---------------------------------------------------------------------------
# D-CP-3/D-CP-4 reach — the gate and the cargo binding work from a repo that is
# NOT agent-utilities, proven by invoking the real installed script as a
# subprocess (exactly how another repo's `.pre-commit-config.yaml` calls it),
# not by calling its Python functions directly.
# ---------------------------------------------------------------------------
GUARD_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_lane_guard.py"


def _init_foreign_repo(root: Path, *, with_cargo: bool = False) -> Path:
    """A repo wholly unrelated to agent-utilities — the reach proof needs one."""
    root.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-b", "main"], root)
    _run(["git", "config", "user.email", "foreign@test"], root)
    _run(["git", "config", "user.name", "Foreign Test"], root)
    if with_cargo:
        (root / "Cargo.toml").write_text(
            '[package]\nname = "foreign"\nversion = "0.1.0"\nedition = "2021"\n',
            encoding="utf-8",
        )
        (root / "src").mkdir(exist_ok=True)
        (root / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    else:
        (root / "README.md").write_text("foreign repo\n", encoding="utf-8")
    _run(["git", "add", "-A"], root)
    _run(["git", "commit", "-qm", "base"], root)
    return root


def test_gate_reaches_a_repo_that_is_not_agent_utilities(tmp_path: Path) -> None:
    """The SAME script, invoked from a foreign repo, guards THAT repo — not AU.

    Before this reach fix, ``main()`` resolved ``lanes.current_tree(REPO)``
    where ``REPO`` was hardcoded to wherever the script file itself lives
    (agent-utilities), so a foreign caller would silently guard
    agent-utilities' OWN tree and never see its own staged work at all — a
    no-op gate for every other repo (D-CP-3). It must now resolve the tree
    from cwd, which is what pre-commit always sets to the repo being
    committed.
    """
    foreign = _init_foreign_repo(tmp_path / "foreign")
    lane = _add_worktree(foreign, "foreign-lane")
    (lane / "new.py").write_text("x = 1\n", encoding="utf-8")
    _run(["git", "add", "new.py"], lane)
    env = dict(os.environ)
    env.pop("CARGO_TARGET_DIR", None)
    proc = subprocess.run(
        ["python3", str(GUARD_SCRIPT)],
        cwd=str(lane),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "foreign-lane" in proc.stdout
    assert str(foreign) not in proc.stdout


def test_gate_still_refuses_a_canonical_commit_in_a_foreign_repo(
    tmp_path: Path,
) -> None:
    """Reach cuts both ways: a foreign repo's OWN canonical checkout is guarded too."""
    foreign = _init_foreign_repo(tmp_path / "foreign2")
    (foreign / "new.py").write_text("x = 1\n", encoding="utf-8")
    _run(["git", "add", "new.py"], foreign)
    env = dict(os.environ)
    env.pop("CARGO_TARGET_DIR", None)
    proc = subprocess.run(
        ["python3", str(GUARD_SCRIPT)],
        cwd=str(foreign),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    assert "CANONICAL checkout" in proc.stderr


def test_gate_detects_a_stray_cargo_target_dir_export_in_a_foreign_cargo_repo(
    tmp_path: Path,
) -> None:
    """A global CARGO_TARGET_DIR export is refused loudly — the residual gap, made loud."""
    foreign = _init_foreign_repo(tmp_path / "foreign-cargo", with_cargo=True)
    lane = _add_worktree(foreign, "cargo-lane")
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = "/some/shared/global/target"
    proc = subprocess.run(
        ["python3", str(GUARD_SCRIPT)],
        cwd=str(lane),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 1
    assert "CARGO_TARGET_DIR" in proc.stderr
    assert "target-isolated" in proc.stderr


def test_gate_accepts_the_lanes_own_cargo_target_dir_export(tmp_path: Path) -> None:
    """The lane's OWN partitioned dir, if exported, is not flagged as an override."""
    foreign = _init_foreign_repo(tmp_path / "foreign-cargo-ok", with_cargo=True)
    lane = _add_worktree(foreign, "cargo-lane-ok")
    own_target = lanes.partitioned_paths(lane).cargo_target_dir
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(own_target)
    proc = subprocess.run(
        ["python3", str(GUARD_SCRIPT)],
        cwd=str(lane),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# D-CP-4 — PARTITION binds: a REAL `cargo` invocation lands in the lane's own
# target dir once `write_cargo_partition_config` has run, with no env var set.
# ---------------------------------------------------------------------------
def test_write_cargo_partition_config_refuses_a_non_cargo_tree(tmp_path: Path) -> None:
    plain = _init_foreign_repo(tmp_path / "not-cargo")
    with pytest.raises(lanes.LaneArbitrationError, match="Cargo.toml"):
        lanes.write_cargo_partition_config(plain)


def test_write_cargo_partition_config_refuses_to_clobber_existing_config(
    tmp_path: Path,
) -> None:
    repo = _init_foreign_repo(tmp_path / "cargo-existing", with_cargo=True)
    cargo_dir = repo / ".cargo"
    cargo_dir.mkdir()
    (cargo_dir / "config.toml").write_text(
        '[target.x86_64]\nrustflags = ["-C", "target-cpu=native"]\n', encoding="utf-8"
    )
    with pytest.raises(lanes.LaneArbitrationError, match="already exists"):
        lanes.write_cargo_partition_config(repo)
    forced = lanes.write_cargo_partition_config(repo, force=True)
    assert forced["written"] and forced["appended"]
    content = (cargo_dir / "config.toml").read_text(encoding="utf-8")
    assert "target-cpu=native" in content  # untouched
    assert 'target-dir = "target-isolated"' in content  # appended


@pytest.mark.skipif(
    subprocess.run(["which", "cargo"], capture_output=True, check=False).returncode
    != 0,
    reason="cargo not installed",
)
def test_write_cargo_partition_config_binds_a_real_cargo_invocation(
    tmp_path: Path,
) -> None:
    """The load-bearing proof: `cargo metadata` — a REAL cargo invocation, not a
    unit assertion about the written TOML — resolves ``target_directory`` to
    THIS lane's own partitioned dir, with no env var and no --target-dir flag.
    """
    import json

    repo = _init_foreign_repo(tmp_path / "cargo-real", with_cargo=True)
    lane = _add_worktree(repo, "cargo-real-lane")
    result = lanes.write_cargo_partition_config(lane)
    assert result["written"]
    _run(["git", "add", ".cargo/config.toml"], lane)
    _run(["git", "commit", "-qm", "bind cargo partition"], lane)
    env = dict(os.environ)
    env.pop("CARGO_TARGET_DIR", None)
    proc = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version=1"],
        cwd=str(lane),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    resolved = Path(json.loads(proc.stdout)["target_directory"])
    assert resolved == lanes.partitioned_paths(lane).cargo_target_dir.resolve()
    # And it is genuinely PER-WORKTREE: a second worktree of the same repo
    # resolves to a DIFFERENT directory without writing any new config —
    # the relative path in the committed file does the rest.
    lane2 = _add_worktree(repo, "cargo-real-lane-2")
    _run(["git", "merge", "cargo-real-lane"], lane2)  # pick up the committed config
    proc2 = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version=1"],
        cwd=str(lane2),
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    resolved2 = Path(json.loads(proc2.stdout)["target_directory"])
    assert resolved2 == lanes.partitioned_paths(lane2).cargo_target_dir.resolve()
    assert resolved2 != resolved


# ---------------------------------------------------------------------------
# CLI contract — `lane env` (every lane runs this at startup) exports
# PRE_COMMIT_HOME and is itself the startup orphan-patch detector. Nothing
# covered `lane env` at all before this.
# ---------------------------------------------------------------------------
def test_lane_env_cli_exports_precommit_home_and_orphan_report(
    canonical: Path,
) -> None:
    lane = _add_worktree(canonical, "lane-cli-env")
    args = argparse.Namespace(path=str(lane), lane_action="env")
    result = au_cli._lane(args)

    assert set(result["exports"]) == {
        "CARGO_TARGET_DIR",
        "PYTEST_ADDOPTS",
        "TMPDIR",
        "PRE_COMMIT_HOME",
    }
    assert result["exports"]["PRE_COMMIT_HOME"] == str(
        lanes.partitioned_paths(lane).precommit_home
    )
    assert result["orphaned_precommit_patches"] == []

    precommit_home = lanes.partitioned_paths(lane).precommit_home
    dead_pid = 2**22
    _write_precommit_patch(
        precommit_home,
        epoch=int(time.time()),
        pid=dead_pid,
        content=b"not a real diff",
    )
    dirty = au_cli._lane(args)
    assert len(dirty["orphaned_precommit_patches"]) == 1
    assert dirty["orphaned_precommit_patches"][0]["state"] == "ORPHANED"
