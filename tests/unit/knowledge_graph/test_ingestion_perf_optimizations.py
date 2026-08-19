"""Ingestion performance optimizations (CONCEPT:AU-KG.query.vendor-agnostic-traversal / KG-2.8).

Covers the delta/throttle work that keeps a large-repo (re-)ingest cheap and
prevents bulk ingest from saturating the engine:

  * pre-hash skip — unchanged files never reach the parse RPC;
  * git-aware delta — only changed source files (any language) are enriched on re-ingest;
  * deep_analysis gating — recursive fan-out is capped while bulk ingest drains.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.enrichment.extractors.code_test import (
    IncompleteParse,
    extract_source_files,
)
from agent_utilities.knowledge_graph.enrichment.pipeline import (
    EnrichmentPipeline,
    logical_file_identity,
    make_batch_parse_fn,
)
from agent_utilities.knowledge_graph.ingestion.engine import (
    _changed_source_files,
    _git_head_sha,
)


# ── #2: pre-hash skip avoids the parse RPC for unchanged files ──────────────
class TestPreHashSkip:
    def _pipe(self, parse_calls: list[str], hash_seen: dict[str, str]):
        backend = MagicMock()

        def parse_fn(file_path: str, source: bytes) -> dict:
            parse_calls.append(file_path)
            return {}  # no symbols — we only care about the call count

        return EnrichmentPipeline(backend, parse_fn, hash_seen=hash_seen)

    def test_unchanged_file_not_parsed_on_reingest(self, tmp_path: Path):
        f = tmp_path / "mod.py"
        f.write_text("def a():\n    return 1\n")

        parse_calls: list[str] = []
        hash_seen: dict[str, str] = {}

        # First pass: file is new → parsed once, hash recorded.
        s1 = self._pipe(parse_calls, hash_seen).enrich_files([f])
        assert s1.files_parsed == 1
        assert s1.files_skipped_unchanged == 0
        assert len(parse_calls) == 1

        # Second pass (same hash_seen, unchanged content): MUST skip the parse RPC.
        s2 = self._pipe(parse_calls, hash_seen).enrich_files([f])
        assert s2.files_parsed == 0
        assert s2.files_skipped_unchanged == 1
        assert len(parse_calls) == 1  # no new parse call

    def test_changed_file_is_reparsed(self, tmp_path: Path):
        f = tmp_path / "mod.py"
        f.write_text("def a():\n    return 1\n")
        parse_calls: list[str] = []
        hash_seen: dict[str, str] = {}

        self._pipe(parse_calls, hash_seen).enrich_files([f])
        f.write_text("def a():\n    return 2\n")  # content changed
        s2 = self._pipe(parse_calls, hash_seen).enrich_files([f])

        assert s2.files_parsed == 1
        assert s2.files_skipped_unchanged == 0
        assert len(parse_calls) == 2


# ── logical identity (CONCEPT:AU-KG.ingest.logical-identity) ────────────────
class TestLogicalIdentity:
    """The persisted artifact identity is repository-relative and stable
    across checkout roots — not the caller's absolute path (NE-006/AU-INGEST).
    Ingesting the same commit from a worktree, a container mount, or the
    canonical checkout must land on the identical identity, and a checkout
    rename must not perturb it. These tests exercise ``enrich_files``'s
    ``source_root``-driven normalization end to end; they extend, not
    replace, ``TestPreHashSkip`` and the exact-acknowledgement guarantees
    (coverage/duplicate/verified-empty/watermark-ordering) above."""

    @staticmethod
    def _pipe(hash_seen: dict[str, str] | None = None, **kwargs):
        backend = MagicMock()
        return EnrichmentPipeline(
            backend,
            lambda file_path, source: {},
            hash_seen=hash_seen if hash_seen is not None else {},
            **kwargs,
        )

    def test_logical_file_identity_is_repo_relative_posix(self, tmp_path: Path):
        root = tmp_path / "repo"
        (root / "pkg").mkdir(parents=True)
        f = root / "pkg" / "mod.py"
        f.write_text("x = 1\n")
        assert logical_file_identity(f.resolve(), root.resolve()) == "pkg/mod.py"

    def test_source_root_that_is_a_single_file(self, tmp_path: Path):
        f = tmp_path / "solo.py"
        f.write_text("def a():\n    return 1\n")
        pipe = self._pipe()
        summary = pipe.enrich_files([f], source_root=f)
        assert summary.files_parsed == 1
        assert list(pipe._hash_seen) == ["solo.py"]

    def test_same_file_two_roots_real_and_symlinked_same_identity(self, tmp_path: Path):
        real_root = tmp_path / "real_repo"
        (real_root / "pkg").mkdir(parents=True)
        f = real_root / "pkg" / "mod.py"
        f.write_text("def a():\n    return 1\n")
        link_root = tmp_path / "link_repo"
        link_root.symlink_to(real_root)
        link_f = link_root / "pkg" / "mod.py"

        hash_seen1: dict[str, str] = {}
        self._pipe(hash_seen1).enrich_files([f], source_root=real_root)

        hash_seen2: dict[str, str] = {}
        self._pipe(hash_seen2).enrich_files([link_f], source_root=link_root)

        assert set(hash_seen1) == set(hash_seen2) == {"pkg/mod.py"}
        assert hash_seen1["pkg/mod.py"] == hash_seen2["pkg/mod.py"]

    def test_identity_unchanged_when_checkout_directory_renamed(self, tmp_path: Path):
        root = tmp_path / "checkout"
        (root / "pkg").mkdir(parents=True)
        (root / "pkg" / "mod.py").write_text("def a():\n    return 1\n")

        hash_seen1: dict[str, str] = {}
        self._pipe(hash_seen1).enrich_files([root / "pkg" / "mod.py"], source_root=root)

        renamed_root = tmp_path / "checkout-renamed"
        root.rename(renamed_root)

        hash_seen2: dict[str, str] = {}
        self._pipe(hash_seen2).enrich_files(
            [renamed_root / "pkg" / "mod.py"], source_root=renamed_root
        )

        assert hash_seen1 == hash_seen2 == {"pkg/mod.py": hash_seen1["pkg/mod.py"]}

    def test_path_escaping_root_fails_closed(self, tmp_path: Path):
        root = tmp_path / "repo"
        root.mkdir()
        outside = tmp_path / "outside.py"
        outside.write_text("def leaked():\n    pass\n")

        pipe = self._pipe()
        with pytest.raises(IncompleteParse):
            pipe.enrich_files([outside], source_root=root)
        assert pipe._hash_seen == {}

    def test_symlink_escape_fails_closed(self, tmp_path: Path):
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret = outside_dir / "secret.py"
        secret.write_text("def leaked():\n    pass\n")

        repo = tmp_path / "repo"
        repo.mkdir()
        link = repo / "vendored.py"
        link.symlink_to(secret)

        pipe = self._pipe()
        with pytest.raises(IncompleteParse):
            pipe.enrich_files([link], source_root=repo)
        assert pipe._hash_seen == {}

    def test_legacy_no_source_root_uses_self_describing_absolute_identity(
        self, tmp_path: Path
    ):
        """``source_root=None`` is the deprecated legacy mode: the identity
        is the resolved absolute path, which always starts with ``/`` and so
        can never collide with, or be mistaken for, a logical (always
        relative) identity."""
        f = tmp_path / "mod.py"
        f.write_text("def a():\n    return 1\n")
        pipe = self._pipe()
        pipe.enrich_files([f])  # no source_root
        assert list(pipe._hash_seen) == [str(f)]
        assert str(f).startswith("/")

    def test_old_absolute_path_watermark_is_reingested_not_falsely_skipped(
        self, tmp_path: Path
    ):
        """Migration behaviour: a ``hash_seen`` dict inherited from before
        this change (keyed by absolute path) does not falsely skip a file
        under the new logical-identity scheme — the stale key simply never
        matches, so the file is (harmlessly) re-parsed and re-hashed under
        its new logical key. A file is never skipped as "unchanged" without
        having actually been verified under the new identity."""
        import hashlib

        root = tmp_path / "repo"
        root.mkdir()
        f = root / "mod.py"
        f.write_text("def a():\n    return 1\n")

        legacy_hash_seen = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()}
        parse_calls: list[str] = []

        def parse_fn(file_path: str, source: bytes) -> dict:
            parse_calls.append(file_path)
            return {}

        pipe = EnrichmentPipeline(MagicMock(), parse_fn, hash_seen=legacy_hash_seen)
        summary = pipe.enrich_files([f], source_root=root)

        assert summary.files_parsed == 1  # re-parsed, not skipped as "unchanged"
        assert summary.files_skipped_unchanged == 0
        assert parse_calls == ["mod.py"]
        assert "mod.py" in pipe._hash_seen
        # The stale legacy key is left alone (not purged) — harmless, and
        # proves the fail-safe behaviour is "identities don't match", not a
        # migration rewrite that could get the mapping wrong.
        assert str(f) in pipe._hash_seen

    def test_exact_coverage_duplicate_and_verified_empty_hold_under_logical_identity(
        self, tmp_path: Path
    ):
        """Re-proves the exact-parser-acknowledgement guarantees (exact
        coverage, duplicate-identity rejection, unrequested-identity
        rejection, verified-empty-vs-never-acknowledged) with the
        request/response keyed on the NEW logical identity, not an absolute
        path."""
        root = tmp_path / "repo"
        root.mkdir()
        busy = root / "busy.py"
        quiet = root / "quiet.py"
        busy.write_text("def fn():\n    pass\n")
        quiet.write_text("PLACEHOLDER = 1\n")  # no symbols -> verified-empty

        def index_sym(file_path: str, name: str = "fn") -> dict:
            return {
                "node_id": f"symbol:{file_path}:{name}",
                "node_type": "SYMBOL",
                "properties": {
                    "symbol_type": "Function",
                    "name": name,
                    "line": "1",
                    "ast_hash": "h",
                    "file_path": file_path,
                    "is_test": "false",
                },
            }

        seen_requests: list[list[str]] = []

        def index_fn(files):
            seen_requests.append([fp for fp, _ in files])
            return {
                "nodes": [index_sym("busy.py")],  # logical identity, not absolute
                "edges": [],
                "files_parsed": 2,
            }

        pipe = EnrichmentPipeline(MagicMock(), lambda *a: {}, index_fn=index_fn)
        summary = pipe.enrich_files([busy, quiet], source_root=root)

        assert summary.files_parsed == 2
        assert seen_requests == [["busy.py", "quiet.py"]]
        assert set(pipe._hash_seen) == {"busy.py", "quiet.py"}  # both acknowledged

        # Duplicate identity: two distinct Path objects resolving to the same
        # real file under source_root are still rejected before any RPC.
        dup_pipe = EnrichmentPipeline(MagicMock(), lambda *a: {})
        with pytest.raises(IncompleteParse):
            dup_pipe.enrich_files([busy, root / "busy.py"], source_root=root)
        assert dup_pipe._hash_seen == {}

        # Unrequested identity in the engine response is still rejected, and
        # nothing is persisted for the whole batch (watermark ordering).
        def bad_index_fn(files):
            return {
                "nodes": [index_sym("not-requested.py")],
                "edges": [],
                "files_parsed": 2,
            }

        bad_pipe = EnrichmentPipeline(MagicMock(), lambda *a: {}, index_fn=bad_index_fn)
        with pytest.raises(IncompleteParse):
            bad_pipe.enrich_files([busy, quiet], source_root=root)
        assert bad_pipe._hash_seen == {}


# ── #3: git-aware delta file selection ──────────────────────────────────────
def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    try:
        _git(repo, "init", "-q")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git not available")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.py").write_text("def a():\n    return 1\n")
    (repo / "b.py").write_text("def b():\n    return 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


class TestGitDelta:
    def test_head_sha_for_git_and_non_git(self, git_repo: Path, tmp_path: Path):
        sha = _git_head_sha(str(git_repo))
        assert sha and len(sha) == 40
        assert _git_head_sha(str(tmp_path / "not-a-repo")) is None

    def test_changed_source_files_returns_only_modified_any_language(
        self, git_repo: Path
    ):
        first = _git_head_sha(str(git_repo))
        (git_repo / "a.py").write_text("def a():\n    return 99\n")  # modify a.py
        (git_repo / "c.py").write_text("def c():\n    return 3\n")  # add c.py
        (git_repo / "W.java").write_text("class W {}\n")  # add Java
        (git_repo / "lib.rs").write_text("pub fn f() {}\n")  # add Rust
        _git(git_repo, "add", "-A")
        _git(git_repo, "commit", "-q", "-m", "change")

        changed = _changed_source_files(str(git_repo), first)
        assert changed is not None
        names = sorted(p.name for p in changed)
        # All languages are caught now — not just .py (b.py unchanged → excluded).
        assert names == ["W.java", "a.py", "c.py", "lib.rs"]

    def test_no_source_changes_yields_empty_list(self, git_repo: Path):
        first = _git_head_sha(str(git_repo))
        (git_repo / "README.md").write_text("# docs\n")  # non-source change only
        _git(git_repo, "add", "-A")
        _git(git_repo, "commit", "-q", "-m", "docs")

        changed = _changed_source_files(str(git_repo), first)
        assert (
            changed == []
        )  # functional git, but no source changed → near-empty re-ingest


# ── #5: deep_analysis gating during bulk ingest ─────────────────────────────
class TestBulkIngestGate:
    def _mixin(self, rows: list[dict]):
        from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin

        obj = TaskManagerMixin.__new__(TaskManagerMixin)
        obj._ingest_work_item_index = MagicMock(
            return_value={str(i): row for i, row in enumerate(rows)}
        )  # type: ignore[attr-defined]
        return obj

    @staticmethod
    def _item(task_type: str) -> dict:
        return {
            "status": "ready",
            "metadata": {"type": task_type, "target": "opaque-ref"},
        }

    def test_active_when_codebase_task_present(self):
        rows = [self._item("codebase"), self._item("document")]
        assert self._mixin(rows)._bulk_ingest_active() is True

    def test_inactive_when_no_codebase_task(self):
        rows = [self._item("document"), self._item("deep_analysis")]
        assert self._mixin(rows)._bulk_ingest_active() is False

    def test_query_failure_fails_closed(self):
        obj = self._mixin([])
        obj._ingest_work_item_index = MagicMock(side_effect=RuntimeError("engine down"))
        with pytest.raises(RuntimeError, match="engine down"):
            obj._bulk_ingest_active()


# ── per-lane/stage profiling (CONCEPT:AU-OS.observability.per-lane-latency-metrics) ──────────────────────────────
class TestProfileReport:
    def _mixin(self, rows: list[dict]):
        from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin

        obj = TaskManagerMixin.__new__(TaskManagerMixin)
        native_status = {
            "completed": "succeeded",
            "failed": "failed",
            "dead_letter": "dead_letter",
        }
        obj._ingest_work_item_index = MagicMock(
            return_value={
                str(i): {
                    "status": native_status[row["status"]],
                    "metadata": row["meta"],
                }
                for i, row in enumerate(rows)
            }
        )  # type: ignore[attr-defined]
        obj._control_cypher = MagicMock(return_value=[])  # type: ignore[attr-defined]
        return obj

    def _row(self, status: str, **meta: object) -> dict:
        return {"status": status, "meta": meta}

    def test_groups_by_lane_with_percentiles_and_parallelism(self):
        rows = [
            self._row(
                "completed",
                lane="ingestion",
                type="document",
                duration_ms=100,
                started_at="2026-06-22T00:00:00+00:00",
                completed_at="2026-06-22T00:00:00.100+00:00",
                tokens=10,
            ),
            self._row(
                "completed",
                lane="ingestion",
                type="document",
                duration_ms=300,
                started_at="2026-06-22T00:00:00+00:00",
                completed_at="2026-06-22T00:00:00.300+00:00",
                tokens=30,
            ),
            self._row(
                "failed",
                lane="research",
                type="research_paper_fetch",
                duration_ms=50,
                started_at="2026-06-22T00:00:00+00:00",
                completed_at="2026-06-22T00:00:00.050+00:00",
            ),
        ]
        rep = self._mixin(rows).profile_report(window_sec=0, group_by="lane")
        assert rep["group_by"] == "lane"
        assert set(rep["groups"]) == {"ingestion", "research"}
        ing = rep["groups"]["ingestion"]
        assert ing["count"] == 2 and ing["completed"] == 2
        assert ing["p50_ms"] == 200.0  # interpolated midpoint of [100, 300]
        assert ing["max_ms"] == 300.0
        assert ing["tokens"] == 40
        assert rep["groups"]["research"]["failed"] == 1
        # total task ms (450) over wall span (300ms) → 1.5x pipelining
        assert rep["parallelism_factor"] == 1.5

    def test_work_item_read_failure_fails_closed(self):
        obj = self._mixin([])
        obj._ingest_work_item_index = MagicMock(side_effect=RuntimeError("down"))
        with pytest.raises(RuntimeError, match="down"):
            obj.profile_report()


# ── batched parse (CONCEPT:EG-KG.compute.graph-compute-engine): extractor + pipeline routing ───────────
class TestBatchExtract:
    def test_order_and_hashes_preserved(self):
        import hashlib

        files = [("a.py", "def a(): pass\n"), ("b.py", "def b(): pass\n")]
        calls: dict = {}

        def batch_parse_fn(raw: list[tuple[str, bytes]]) -> list[dict]:
            calls["n"] = len(raw)
            calls["count"] = calls.get("count", 0) + 1
            return [{} for _ in raw]

        out = extract_source_files(files, batch_parse_fn)
        assert [r.file_path for r in out] == ["a.py", "b.py"]
        for (fp, src), r in zip(files, out, strict=True):
            want = hashlib.sha256(src.encode("utf-8", "surrogatepass")).hexdigest()
            assert r.content_hash == want
        assert calls == {"n": 2, "count": 1}  # ONE batch call for both files

    def test_missing_slot_is_rejected_not_silently_degraded(self):
        """CONCEPT:AU-KG.ingest.exact-parser-acknowledgement (landed in
        071ea4e6e): a response with fewer result slots than requested files
        must raise, never silently pad the missing slot as an empty
        successful result — this test previously asserted the pre-fix
        degrade behavior and was stale against that landed contract."""
        files = [("a.py", "x = 1\n"), ("b.py", "y = 2\n")]
        with pytest.raises(IncompleteParse):
            extract_source_files(files, lambda raw: [{}])  # only 1 result for 2

    def test_batch_exception_is_rejected_not_silently_degraded(self):
        """Same contract, RPC-failure variant (see test above)."""

        def boom(raw):
            raise RuntimeError("engine down")

        with pytest.raises(IncompleteParse):
            extract_source_files([("a.py", "x = 1\n")], boom)


class TestMakeBatchParseFn:
    class _GC:
        def __init__(self):
            self.calls: list[int] = []

        def parse_files(self, files):
            self.calls.append(len(files))
            return [{} for _ in files]

    def test_chunks_by_env(self, monkeypatch):
        monkeypatch.setenv("KG_PARSE_BATCH", "2")
        gc = self._GC()
        fn = make_batch_parse_fn(gc)
        assert fn is not None
        out = fn([("a", b"1"), ("b", b"2"), ("c", b"3")])
        assert len(out) == 3
        assert gc.calls == [2, 1]  # 3 files → chunks of 2 + 1


class TestPipelineBatchRouting:
    def _pipe(self, batch_fn, parse_calls, hash_seen):
        backend = MagicMock()

        def parse_fn(fp: str, src: bytes) -> dict:
            parse_calls.append(fp)
            return {}

        return EnrichmentPipeline(
            backend, parse_fn, hash_seen=hash_seen, batch_parse_fn=batch_fn
        )

    def test_uses_batch_when_available(self, tmp_path: Path):
        f1, f2 = tmp_path / "a.py", tmp_path / "b.py"
        f1.write_text("def a(): pass\n")
        f2.write_text("def b(): pass\n")
        batch_calls: list[list[str]] = []

        def batch_fn(raw):
            batch_calls.append([fp for fp, _ in raw])
            return [{} for _ in raw]

        parse_calls: list[str] = []
        s = self._pipe(batch_fn, parse_calls, {}).enrich_files([f1, f2])
        assert s.files_parsed == 2
        assert len(batch_calls) == 1 and len(batch_calls[0]) == 2  # ONE batch RPC
        assert parse_calls == []  # per-file parse_fn NOT used

    def test_per_file_when_no_batch(self, tmp_path: Path):
        f1 = tmp_path / "a.py"
        f1.write_text("def a(): pass\n")
        parse_calls: list[str] = []
        s = self._pipe(None, parse_calls, {}).enrich_files([f1])
        assert s.files_parsed == 1 and parse_calls == [str(f1)]
