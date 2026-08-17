#!/usr/bin/python
"""Enterprise-scale repository batch ingestion (CONCEPT:EG-KG.query.wire-protocol / KG-2.49).

Covers idempotent prefilter (manifest-hit skip), crash-resume re-skip, archived
filtering, and in-flight backpressure capping.

The bulk-prefilter watermark is recorded ONLY on a VERIFIED completion --
never by ``submit_batch`` itself, which only enqueues a task and proves
nothing about whether it will ever run or succeed
(CONCEPT:AU-KG.ingest.exact-parser-acknowledgement). The completion side
lives in ``ingestion/engine.py``'s ``_run_codebase_structural`` (a separate
module/class), so these tests simulate it directly via
``manifest.record(graph_name, _CATEGORY, clone_path, head_sha)`` -- exactly
the call the real structural ingest makes after a verified, fully-
acknowledged parse -- to prove the prefilter skip requires that completion
signal, not mere submission.
"""

import pytest

from agent_utilities.knowledge_graph.ingestion.batch_orchestrator import (
    _CATEGORY,
    RepoBatchIngestor,
    RepoRef,
)
from agent_utilities.knowledge_graph.ingestion.manifest import DeltaManifest

pytestmark = pytest.mark.concept("EG-KG.query.wire-protocol")


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

    def ingest_queue_depth(self):
        return self._inflight

    def list_tasks(self):
        return {}


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


def _complete(ing: RepoBatchIngestor, ref: RepoRef) -> None:
    """Simulate the structural ingest's VERIFIED-completion watermark record
    -- the real call this makes lives in ``ingestion/engine.py``'s
    ``_run_codebase_structural``, a separate module/class this unit test
    does not exercise end-to-end."""
    ing.manifest.record(ing.graph_name, _CATEGORY, ref.clone_path, ref.head_sha)


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


def test_unchanged_repo_is_skipped_on_rerun_after_verified_completion(tmp_path):
    engine = _FakeEngine()
    ing = _ingestor(tmp_path, engine, inflight_target=100)
    ref = _ref("a", sha="x")
    ing.submit_batch([ref])
    assert len(engine.submitted) == 1
    _complete(ing, ref)  # the structural ingest verified/acknowledged the parse
    # Re-run with the SAME head sha, ingest already verified complete →
    # manifest hit → no new submit.
    prog2 = ing.submit_batch([ref])
    assert prog2.skipped_unchanged == 1
    assert prog2.submitted == 0
    assert len(engine.submitted) == 1


def test_submission_alone_never_advances_the_watermark(tmp_path):
    """KNOWN-BAD (pre-fix regression): submitting a task used to record the
    bulk-prefilter watermark immediately -- before the ingest had even
    started, let alone been verified. A repo whose ingest then crashed or
    was rejected would silently read back as "already done" forever. Proves
    submission ALONE (no completion signal) never causes a skip: the SAME
    head_sha is resubmitted every time until something actually completes."""
    engine = _FakeEngine()
    ing = _ingestor(tmp_path, engine, inflight_target=100)
    ref = _ref("a", sha="x")
    ing.submit_batch([ref])
    assert len(engine.submitted) == 1

    # No completion was ever recorded (simulates a crash/failed/rejected
    # ingest) -- a resumed batch run must resubmit, never skip.
    prog2 = ing.submit_batch([ref])
    assert prog2.skipped_unchanged == 0
    assert prog2.submitted == 1
    assert len(engine.submitted) == 2

    prog3 = ing.submit_batch([ref])
    assert prog3.skipped_unchanged == 0
    assert prog3.submitted == 1
    assert len(engine.submitted) == 3


def test_moved_head_resubmits(tmp_path):
    engine = _FakeEngine()
    ing = _ingestor(tmp_path, engine, inflight_target=100)
    ref_x = _ref("a", sha="x")
    ing.submit_batch([ref_x])
    _complete(ing, ref_x)
    prog2 = ing.submit_batch([_ref("a", sha="y")])  # HEAD moved
    assert prog2.submitted == 1
    assert len(engine.submitted) == 2


def test_crash_resume_reskips_only_verified_complete_repos(tmp_path):
    # A fresh ingestor over the SAME manifest db re-skips repos whose ingest
    # already completed and verified -- NOT merely-submitted ones.
    engine1 = _FakeEngine()
    ing1 = _ingestor(tmp_path, engine1, inflight_target=100)
    ref_a, ref_b = _ref("a"), _ref("b")
    ing1.submit_batch([ref_a, ref_b])
    _complete(ing1, ref_a)
    _complete(ing1, ref_b)
    engine2 = _FakeEngine()
    ing2 = _ingestor(tmp_path, engine2, inflight_target=100)
    prog = ing2.submit_batch([ref_a, ref_b, _ref("c")])
    assert prog.skipped_unchanged == 2  # a, b verified complete
    assert prog.submitted == 1  # only c is new
    assert engine2.submitted[0][0].endswith("/c")


def test_crash_resume_resubmits_merely_submitted_not_completed_repos(tmp_path):
    """The crash-resume companion to test_submission_alone_never_advances_
    the_watermark: a repo that was submitted before a crash, but whose
    ingest never completed/verified, must be resubmitted by the resumed
    run -- not silently treated as done."""
    engine1 = _FakeEngine()
    ing1 = _ingestor(tmp_path, engine1, inflight_target=100)
    ref_a, ref_b = _ref("a"), _ref("b")
    ing1.submit_batch([ref_a, ref_b])
    _complete(ing1, ref_a)  # only "a" actually finished before the crash

    engine2 = _FakeEngine()
    ing2 = _ingestor(tmp_path, engine2, inflight_target=100)
    prog = ing2.submit_batch([ref_a, ref_b, _ref("c")])
    assert prog.skipped_unchanged == 1  # only a
    assert prog.submitted == 2  # b (never verified) and c (new)
    resubmitted = {t[0] for t in engine2.submitted}
    assert any(p.endswith("/b") for p in resubmitted)
    assert any(p.endswith("/c") for p in resubmitted)


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
        def list_tasks(self):
            return {
                "pending": [object(), object(), object()],
                "completed": [object(), object(), object(), object(), object()],
            }

    engine = _StatusEngine()
    ing = _ingestor(tmp_path, engine)
    assert ing.status() == {"pending": 3, "completed": 5}
