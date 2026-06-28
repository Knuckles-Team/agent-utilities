"""Ingestion performance optimizations (CONCEPT:KG-2.7 / KG-2.8).

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
    extract_source_files,
)
from agent_utilities.knowledge_graph.enrichment.pipeline import (
    EnrichmentPipeline,
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
        # :Task status/metadata is the CONTROL plane, so _bulk_ingest_active reads
        # via _control_cypher (CONCEPT:KG-2.148), not the data-plane query_cypher.
        obj._control_cypher = MagicMock(return_value=rows)  # type: ignore[attr-defined]
        return obj

    def _meta(self, task_type: str) -> str:
        from agent_utilities.knowledge_graph.core.engine_tasks import _encode_metadata

        return _encode_metadata({"type": task_type, "target": "x"})

    def test_active_when_codebase_task_present(self):
        rows = [{"meta": self._meta("codebase")}, {"meta": self._meta("document")}]
        assert self._mixin(rows)._bulk_ingest_active() is True

    def test_inactive_when_no_codebase_task(self):
        rows = [{"meta": self._meta("document")}, {"meta": self._meta("deep_analysis")}]
        assert self._mixin(rows)._bulk_ingest_active() is False

    def test_query_failure_degrades_to_false(self):
        obj = self._mixin([])
        obj._control_cypher = MagicMock(side_effect=RuntimeError("engine down"))
        assert obj._bulk_ingest_active() is False


# ── per-lane/stage profiling (CONCEPT:OS-5.55) ──────────────────────────────
class TestProfileReport:
    def _mixin(self, rows: list[dict]):
        from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin

        obj = TaskManagerMixin.__new__(TaskManagerMixin)
        # profile_report issues two control queries: the :Task scan (rows) then the
        # off-queue :ProfileSpan scan (none here). Returning ``rows`` for BOTH would
        # double-count every task, so the span query yields [].
        obj._control_cypher = MagicMock(side_effect=[rows, []])  # type: ignore[attr-defined]
        return obj

    def _row(self, status: str, **meta: object) -> dict:
        from agent_utilities.knowledge_graph.core.engine_tasks import _encode_metadata

        return {"status": status, "meta": _encode_metadata(meta)}

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

    def test_empty_on_cypher_failure(self):
        obj = self._mixin([])
        obj._control_cypher = MagicMock(side_effect=RuntimeError("down"))
        assert obj.profile_report() == {}


# ── batched parse (CONCEPT:KG-2.16): extractor + pipeline routing ───────────
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

    def test_missing_slot_degrades_to_empty(self):
        files = [("a.py", "x = 1\n"), ("b.py", "y = 2\n")]
        out = extract_source_files(files, lambda raw: [{}])  # only 1 result for 2
        assert len(out) == 2
        assert out[1].code == [] and out[1].tests == []

    def test_batch_exception_degrades(self):
        def boom(raw):
            raise RuntimeError("engine down")

        out = extract_source_files([("a.py", "x = 1\n")], boom)
        assert len(out) == 1 and out[0].file_path == "a.py"


class TestMakeBatchParseFn:
    class _GC:
        supports_batch_parse = True

        def __init__(self):
            self.calls: list[int] = []

        def parse_files(self, files):
            self.calls.append(len(files))
            return [{} for _ in files]

    def test_returns_none_when_unsupported(self):
        gc = self._GC()
        gc.supports_batch_parse = False
        assert make_batch_parse_fn(gc) is None

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
