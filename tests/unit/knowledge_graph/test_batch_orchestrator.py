#!/usr/bin/python
"""Enterprise-scale repository batch ingestion (CONCEPT:KG-2.19 / KG-2.49).

Covers idempotent prefilter (manifest-hit skip), crash-resume re-skip, archived
filtering, and in-flight backpressure capping.
"""

import pytest

from agent_utilities.knowledge_graph.ingestion.batch_orchestrator import (
    RepoBatchIngestor,
    RepoRef,
)
from agent_utilities.knowledge_graph.ingestion.manifest import DeltaManifest

pytestmark = pytest.mark.concept("KG-2.19")


class _FakeEngine:
    """Engine double: records submit_task calls; reports a fixed in-flight count."""

    def __init__(self, inflight=0):
        self.submitted: list[tuple] = []
        self._inflight = inflight
        self.backend = None  # forces DeltaManifest into sqlite mode

    def submit_task(
        self, target_path, is_codebase, provenance, task_type=None, skip_dedupe=False
    ):
        job = f"job-{len(self.submitted)}"
        self.submitted.append((target_path, is_codebase, task_type, provenance))
        return job

    def query_cypher(self, q):
        if "count(t)" in q:
            return [{"c": self._inflight}]
        return []


def _ref(name, sha="sha1", archived=False):
    return RepoRef(
        vcs="gitlab",
        full_path=f"group/{name}",
        clone_path=f"/cache/gitlab/group/{name}",
        head_sha=sha,
        archived=archived,
    )


def _ingestor(tmp_path, engine, **kw):
    manifest = DeltaManifest(db_path=str(tmp_path / "m.db"))
    return RepoBatchIngestor(engine, manifest=manifest, **kw)


def test_submits_changed_repos(tmp_path):
    engine = _FakeEngine()
    ing = _ingestor(tmp_path, engine, inflight_target=100)
    prog = ing.submit_batch([_ref("a"), _ref("b")])
    assert prog.enumerated == 2
    assert prog.submitted == 2
    assert len(engine.submitted) == 2
    # provenance carries the vcs + head sha
    assert engine.submitted[0][3]["vcs"] == "gitlab"
    assert engine.submitted[0][3]["head_sha"] == "sha1"


def test_unchanged_repo_is_skipped_on_rerun(tmp_path):
    engine = _FakeEngine()
    ing = _ingestor(tmp_path, engine, inflight_target=100)
    ing.submit_batch([_ref("a", sha="x")])
    assert len(engine.submitted) == 1
    # Re-run with the SAME head sha → manifest hit → no new submit.
    prog2 = ing.submit_batch([_ref("a", sha="x")])
    assert prog2.skipped_unchanged == 1
    assert prog2.submitted == 0
    assert len(engine.submitted) == 1


def test_moved_head_resubmits(tmp_path):
    engine = _FakeEngine()
    ing = _ingestor(tmp_path, engine, inflight_target=100)
    ing.submit_batch([_ref("a", sha="x")])
    prog2 = ing.submit_batch([_ref("a", sha="y")])  # HEAD moved
    assert prog2.submitted == 1
    assert len(engine.submitted) == 2


def test_crash_resume_reskips_recorded(tmp_path):
    # A fresh ingestor over the SAME manifest db re-skips already-submitted repos.
    engine1 = _FakeEngine()
    ing1 = _ingestor(tmp_path, engine1, inflight_target=100)
    ing1.submit_batch([_ref("a"), _ref("b")])
    engine2 = _FakeEngine()
    ing2 = _ingestor(tmp_path, engine2, inflight_target=100)
    prog = ing2.submit_batch([_ref("a"), _ref("b"), _ref("c")])
    assert prog.skipped_unchanged == 2  # a, b already recorded
    assert prog.submitted == 1  # only c is new
    assert engine2.submitted[0][0].endswith("/c")


def test_archived_filtered(tmp_path):
    engine = _FakeEngine()
    ing = _ingestor(tmp_path, engine, inflight_target=100)
    prog = ing.submit_batch([_ref("a"), _ref("z", archived=True)])
    assert prog.skipped_archived == 1
    assert prog.submitted == 1


def test_backpressure_caps_submits(tmp_path):
    # Queue already at the target depth → nothing submitted, all deferred.
    engine = _FakeEngine(inflight=10)
    ing = _ingestor(tmp_path, engine, inflight_target=10)
    prog = ing.submit_batch([_ref("a"), _ref("b"), _ref("c")])
    assert prog.submitted == 0
    assert prog.deferred_backpressure == 3
    assert engine.submitted == []


def test_status_aggregates_task_counts(tmp_path):
    class _StatusEngine(_FakeEngine):
        def query_cypher(self, q):
            if "count(t) AS c" in q and "t.status AS s" in q:
                return [{"s": "pending", "c": 3}, {"s": "completed", "c": 5}]
            return super().query_cypher(q)

    engine = _StatusEngine()
    ing = _ingestor(tmp_path, engine)
    assert ing.status() == {"pending": 3, "completed": 5}
