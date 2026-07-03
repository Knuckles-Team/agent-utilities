"""Ingestion TAIL optimizations — the p95/max outlier fixes (perf/ingest-tail).

The median ingestion is healthy; the TAIL blows up on edge cases:

  * a single BIG repo (thousands of files) is one task → one graph → one shard
    writer, pinning a worker for ~13 min (codebase p95=650s/max=797s) — fixed by
    splitting it into K shard-routed sub-tasks (CONCEPT:KG-2.287);
  * a connector / maint task with no per-task bound hangs (456s / 761s) and
    starves a worker — fixed by a per-lane soft timeout that cancels → routes the
    task through the KG-2.113 retry→backoff→dead_letter machinery (CONCEPT:KG-2.286);
  * the profiler reported only per-lane percentiles, so the specific outliers were
    invisible — fixed by surfacing the slowest-N tasks + p99 (CONCEPT:KG-2.288);
  * under heavy ingestion the host worker pool was fully consumed, starving
    interactive/MCP work — fixed by a HARD interactive reservation no ingestion
    lane can spend (CONCEPT:KG-2.289).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _encode_metadata,
)
from agent_utilities.knowledge_graph.core.task_lanes import (
    INTERACTIVE_LANES,
    lane_soft_timeout,
    task_soft_timeout,
)
from agent_utilities.knowledge_graph.core.worker_scheduler import (
    AdmissionPolicy,
    SchedulerConfig,
    WorkerRegistry,
)
from agent_utilities.knowledge_graph.ingestion.repo_split import (
    SPLIT_MIN_FILES,
    plan_repo_split,
    split_graph_suffix,
)


# ── KG-2.287: big-repo split planner ────────────────────────────────────────
class TestPlanRepoSplit:
    def _files(self, root: Path) -> list[Path]:
        files: list[Path] = []
        # one dominant package (split deeper) + several small top-level packages
        for i in range(60):
            files.append(root / "big" / f"sub{i % 10}" / f"f{i}.py")
        for pkg in ("a", "b", "c", "d", "e", "f", "g"):
            for j in range(6):
                files.append(root / pkg / f"f{j}.py")
        return files

    def test_balanced_buckets_cover_all_files_without_duplication(self):
        root = Path("/repo")
        files = self._files(root)
        buckets = plan_repo_split(root, files, 4)
        assert len(buckets) == 4
        flat = [str(p) for b in buckets for p in b]
        # completeness: union == input, no duplicates, no loss.
        assert sorted(flat) == sorted(str(p) for p in files)
        assert len(flat) == len(set(flat)) == len(files)
        # balance: the largest bucket is within ~2x the smallest (LPT schedule).
        sizes = sorted(len(b) for b in buckets)
        assert sizes[-1] <= sizes[0] * 2

    def test_deterministic_regardless_of_input_order(self):
        root = Path("/repo")
        files = self._files(root)
        a = plan_repo_split(root, files, 4)
        b = plan_repo_split(root, list(reversed(files)), 4)

        def _norm(bk: list[list[Path]]) -> list[list[str]]:
            return sorted(sorted(str(p) for p in bucket) for bucket in bk)

        assert _norm(a) == _norm(b)

    def test_single_group_is_not_falsely_split(self):
        # All files share one top-level prefix and no sub-structure → keep whole,
        # so a split never adds tasks for zero shard-spread.
        root = Path("/repo")
        files = [root / "only" / f"f{i}.py" for i in range(40)]
        buckets = plan_repo_split(root, files, 4)
        assert len(buckets) == 1
        assert sorted(str(p) for p in buckets[0]) == sorted(str(p) for p in files)

    def test_k_le_1_or_tiny_input_returns_single_bucket(self):
        root = Path("/repo")
        files = [root / "a" / "f.py", root / "b" / "g.py"]
        assert plan_repo_split(root, files, 1) == [files]
        assert plan_repo_split(root, [files[0]], 4) == [[files[0]]]
        assert plan_repo_split(root, [], 4) == []

    def test_split_graph_suffix(self):
        assert split_graph_suffix(0) == "__s0"
        assert split_graph_suffix(3) == "__s3"


# ── KG-2.286: per-lane soft timeout sizing ──────────────────────────────────
class TestSoftTimeout:
    def test_connector_and_maint_bounds_catch_the_observed_hangs(self):
        # The live tail: a connector hung 456s, a maint tick hung 761s.
        assert task_soft_timeout("connector_sync") <= 456.0
        assert task_soft_timeout("scheduled_job") <= 761.0

    def test_connector_bound_has_huge_headroom_over_its_median(self):
        # connector p50=16ms → a 180s bound is ~10000x headroom (no false kills).
        assert task_soft_timeout("connector_sync") >= 60.0

    def test_codebase_bound_is_generous(self):
        # legitimate codebase work is minutes; its bound must clear the p95.
        assert task_soft_timeout("codebase") >= 1200.0

    def test_interactive_bound_is_tight(self):
        assert task_soft_timeout("conversation") <= 300.0

    def test_unknown_type_falls_to_the_default_lane_bound(self):
        # An unmapped task type routes to DEFAULT_LANE ("maint"), so it inherits
        # the maint bound — never an unbounded run.
        assert task_soft_timeout("nope-not-a-type") == lane_soft_timeout("maint")
        # An unmapped LANE itself uses the module default.
        assert lane_soft_timeout("no-such-lane") == 1800.0


# ── KG-2.286: a hung task hits the timeout → routed to retry/dead_letter ─────
class TestHungTaskTimeout:
    def _mixin(self) -> TaskManagerMixin:
        obj = TaskManagerMixin.__new__(TaskManagerMixin)
        obj._maybe_build_vector_indexes = MagicMock()  # type: ignore[attr-defined]
        return obj

    def test_execute_raises_soft_timeout_when_task_overruns(self):
        obj = self._mixin()

        async def _hang(*_a, **_k):
            await asyncio.sleep(3)  # past the patched 0.05s bound

        obj._run_background_task = _hang  # type: ignore[attr-defined]

        with mock.patch(
            "agent_utilities.knowledge_graph.core.task_lanes.task_soft_timeout",
            return_value=0.05,
        ):
            # A non-heavy type skips the throttle context — keep the test pure.
            with pytest.raises(RuntimeError, match="soft timeout"):
                obj._execute_claimed_task("job-1", Path("/x"), False, "conversation")

    def test_normal_task_completes_through_the_watchdog(self):
        obj = self._mixin()

        async def _ok(*_a, **_k):
            return None

        obj._run_background_task = _ok  # type: ignore[attr-defined]
        # Does not raise; the post-step (index build) runs on the success path.
        obj._execute_claimed_task("job-ok", Path("/x"), False, "conversation")
        obj._maybe_build_vector_indexes.assert_called_once()

    def test_task_failure_propagates_to_the_worker_loop(self):
        # A task that *raises* (not hangs) must surface its real error so the worker
        # loop routes it through _fail_or_retry_task — the watchdog must not swallow it.
        obj = self._mixin()

        async def _boom(*_a, **_k):
            raise ValueError("kaboom")

        obj._run_background_task = _boom  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="kaboom"):
            obj._execute_claimed_task("job-x", Path("/x"), False, "conversation")
        obj._maybe_build_vector_indexes.assert_not_called()

    def test_fail_or_retry_dead_letters_a_repeatedly_timing_out_task(self):
        # Proves the cancel→retry machinery terminates: on the final attempt the
        # task is dead-lettered, not retried forever (CONCEPT:KG-2.113 reuse).
        obj = TaskManagerMixin.__new__(TaskManagerMixin)
        obj.backend = MagicMock()
        # 3rd attempt (attempts already 2, max 3) → dead_letter.
        obj._control_cypher = MagicMock(  # type: ignore[attr-defined]
            return_value=[
                {"meta": _encode_metadata({"attempts": 2, "max_attempts": 3})}
            ]
        )
        statuses: list[str] = []
        obj._update_task_status = lambda jid, status, meta: statuses.append(status)  # type: ignore[attr-defined]
        obj._fail_or_retry_task("job-1", "soft timeout: connector_sync exceeded 180s")
        assert statuses == ["dead_letter"]


# ── KG-2.289: interactive reservation under ingestion saturation ────────────
class TestInteractiveReservation:
    def _policy(self, worker_count=4, reserved=1):
        cfg = SchedulerConfig(
            worker_count=worker_count, reserved=reserved, per_lane_min=1
        )
        reg = WorkerRegistry()
        return AdmissionPolicy(cfg, reg), reg

    def test_queries_is_the_interactive_lane(self):
        assert "queries" in INTERACTIVE_LANES

    def test_floor_is_at_least_one_when_capacity_allows(self):
        pol, _ = self._policy(worker_count=4, reserved=0)
        assert pol.interactive_floor() == 1  # floored at 1 even with reserved=0

    def test_ingestion_cannot_claim_the_last_interactive_slot(self):
        pol, reg = self._policy(worker_count=4, reserved=1)
        reg.start("w0", "ingestion", "codebase")
        reg.start("w1", "ingestion", "codebase")
        reg.start("w2", "ingestion", "document")  # 1 free, floor=1
        assert pol.admit("ingestion", "codebase", {"ingestion": 9}) is False

    def test_reservation_not_relaxed_to_cover_an_uncovered_ingestion_lane(self):
        # The key guarantee: even a *starving* ingestion lane cannot spend the
        # interactive floor (unlike the relaxable hot spare).
        pol, reg = self._policy(worker_count=4, reserved=1)
        reg.start("w0", "maint", "scheduled_job")
        reg.start("w1", "maint", "scheduled_job")
        reg.start("w2", "maint", "scheduled_job")
        assert pol.admit("ingestion", "codebase", {"ingestion": 50}) is False

    def test_interactive_work_may_claim_the_reserved_slot(self):
        pol, reg = self._policy(worker_count=4, reserved=1)
        reg.start("w0", "maint", "scheduled_job")
        reg.start("w1", "maint", "scheduled_job")
        reg.start("w2", "maint", "scheduled_job")
        assert pol.admit("queries", "conversation", {"queries": 1}) is True

    def test_single_worker_pool_does_not_wedge(self):
        # A degenerate 1-worker pool reserves 0 so the lone worker serves all work.
        pol, _ = self._policy(worker_count=1, reserved=1)
        assert pol.interactive_floor() == 0
        assert pol.admit("ingestion", "codebase", {"ingestion": 3}) is True


# ── KG-2.287: _maybe_fanout_codebase fan-out + guards ───────────────────────
class TestCodebaseFanout:
    def _mixin(self, submit_ids: list[str] | None = None) -> TaskManagerMixin:
        obj = TaskManagerMixin.__new__(TaskManagerMixin)
        ids = iter(submit_ids or [f"child-{i}" for i in range(16)])
        obj.submit_task = MagicMock(side_effect=lambda *a, **k: next(ids))  # type: ignore[attr-defined]
        obj._update_task_status = MagicMock()  # type: ignore[attr-defined]
        return obj

    def _patch_env(self, files: list[Path], *, routing=True, shards=4):
        return [
            mock.patch(
                "agent_utilities.knowledge_graph.core.ingest_routing.routing_enabled",
                return_value=routing,
            ),
            mock.patch(
                "agent_utilities.knowledge_graph.enrichment.pipeline.discover_source_files",
                return_value=files,
            ),
            mock.patch(
                "agent_utilities.knowledge_graph.core.worker_scheduler.durable_shard_writers",
                return_value=shards,
            ),
        ]

    def _big_repo(self, root: Path) -> list[Path]:
        # > SPLIT_MIN_FILES files across many sub-packages so it splits.
        files: list[Path] = []
        n = SPLIT_MIN_FILES + 200
        for i in range(n):
            files.append(root / f"pkg{i % 20}" / f"mod{i}.py")
        return files

    def test_large_repo_splits_into_parallel_shard_routed_subtasks(
        self, tmp_path: Path
    ):
        files = self._big_repo(tmp_path)
        obj = self._mixin()
        patches = self._patch_env(files, routing=True, shards=4)
        with patches[0], patches[1], patches[2]:
            fanned = obj._maybe_fanout_codebase("parent-1", tmp_path, {})
        assert fanned is True

        calls = obj.submit_task.call_args_list
        assert len(calls) == 4  # one sub-task per shard writer
        seen_files: set[str] = set()
        suffixes: set[str] = set()
        for i, c in enumerate(calls):
            kw = c.kwargs
            assert kw["skip_dedupe"] is True
            assert kw["task_type"] == "codebase"
            em = kw["extra_meta"]
            assert em["split_child"] is True
            assert em["split_parent"] == "parent-1"
            assert em["route_repo"].endswith(f"__s{em['split_bucket']}")
            suffixes.add(em["route_repo"])
            seen_files.update(em["only_files"])
        # distinct shard graphs + complete, non-overlapping coverage of every file.
        assert len(suffixes) == 4
        assert seen_files == {str(p) for p in files}

        # parent is closed out as a fan-out coordinator, not re-ingested inline.
        _jid, status, meta = obj._update_task_status.call_args.args
        assert status == "completed"
        assert meta["status"] == "fanned_out"
        assert meta["split_buckets"] == 4

    def test_subtask_is_never_resplit(self, tmp_path: Path):
        files = self._big_repo(tmp_path)
        obj = self._mixin()
        patches = self._patch_env(files)
        with patches[0], patches[1], patches[2]:
            # a child carries route_repo/split_child → must NOT fan out again.
            assert (
                obj._maybe_fanout_codebase(
                    "c1", tmp_path, {"route_repo": "x__s0", "split_child": True}
                )
                is False
            )
        obj.submit_task.assert_not_called()

    def test_explicitly_scoped_task_is_not_split(self, tmp_path: Path):
        files = self._big_repo(tmp_path)
        obj = self._mixin()
        patches = self._patch_env(files)
        with patches[0], patches[1], patches[2]:
            assert (
                obj._maybe_fanout_codebase("c1", tmp_path, {"only_files": ["/a/b.py"]})
                is False
            )
        obj.submit_task.assert_not_called()

    def test_no_split_when_routing_disabled(self, tmp_path: Path):
        # Distinct graphs require routing; with routing off a split buys nothing.
        files = self._big_repo(tmp_path)
        obj = self._mixin()
        patches = self._patch_env(files, routing=False)
        with patches[0], patches[1], patches[2]:
            assert obj._maybe_fanout_codebase("p", tmp_path, {}) is False
        obj.submit_task.assert_not_called()

    def test_small_repo_is_not_split(self, tmp_path: Path):
        files = [tmp_path / f"pkg{i % 4}" / f"m{i}.py" for i in range(50)]
        obj = self._mixin()
        patches = self._patch_env(files)
        with patches[0], patches[1], patches[2]:
            assert obj._maybe_fanout_codebase("p", tmp_path, {}) is False
        obj.submit_task.assert_not_called()


# ── KG-2.288: profiler surfaces the slowest-N tasks + p99 ───────────────────
class TestProfilerTail:
    def _mixin(self, rows: list[dict]) -> TaskManagerMixin:
        obj = TaskManagerMixin.__new__(TaskManagerMixin)
        obj._control_cypher = MagicMock(return_value=rows)  # type: ignore[attr-defined]
        return obj

    def _row(self, tid: str, status: str, **meta: object) -> dict:
        return {"id": tid, "status": status, "meta": _encode_metadata(meta)}

    def test_slowest_list_and_p99_surface_the_outliers(self):
        rows = [
            self._row(
                f"job-{i}",
                "completed",
                lane="ingestion",
                type="codebase",
                duration_ms=float(d),
                target=f"/repo/{i}",
            )
            for i, d in enumerate([100, 200, 300, 797000])  # one 13-min outlier
        ]
        rep = self._mixin(rows).profile_report(window_sec=0, group_by="lane")
        slowest = rep["slowest"]
        # the big-repo pin is the #1 slowest task, named with its id + target.
        assert slowest[0]["duration_ms"] == 797000.0
        assert slowest[0]["id"] == "job-3"
        assert slowest[0]["type"] == "codebase"
        assert slowest[0]["target"] == "/repo/3"
        # descending order.
        durs = [t["duration_ms"] for t in slowest]
        assert durs == sorted(durs, reverse=True)
        # p99 is reported per group alongside p95/max.
        ing = rep["groups"]["ingestion"]
        assert "p99_ms" in ing
        assert ing["p99_ms"] <= ing["max_ms"]
        assert ing["max_ms"] == 797000.0

    def test_slowest_is_capped(self):
        rows = [
            self._row(f"job-{i}", "completed", lane="maint", duration_ms=float(i + 1))
            for i in range(50)
        ]
        rep = self._mixin(rows).profile_report(window_sec=0)
        assert len(rep["slowest"]) == 10  # top-N only
