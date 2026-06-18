"""ScholarX RSS research-feed loop — the flagship unified-scheduler consumer
(CONCEPT:KG-2.114).

Grades incoming RSS items (keyword + novelty), skips already-seen items via a
DeltaManifest seen-set, and enqueues a grade-prioritized full-paper fetch only
for the high-graded ones. These pin: the seen-set skips on the 2nd tick, the
grade decides queue-vs-marginal-vs-reject, and the fetch task's priority bucket
is derived from the grade (best paper fetched first).
"""

from __future__ import annotations

import pytest


class _Engine:
    def __init__(self):
        self.backend = None  # DeltaManifest → SQLite mode (pointed at tmp below)
        self.submitted: list[dict] = []
        self.nodes: dict[str, dict] = {}

    def submit_task(self, **kw):
        self.submitted.append(kw)
        return kw.get("job_id", "job-x")

    def query_cypher(self, q, params=None):
        return []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"type": node_type, **(properties or {})}


_FEED = [
    {"id": "arxiv:2601.0001", "title": "HIGH", "abstract": "", "authors": [],
     "url": "http://x/1", "pdf_url": "http://x/1.pdf"},
    {"id": "arxiv:2601.0002", "title": "MARG", "abstract": "", "authors": [],
     "url": "http://x/2", "pdf_url": ""},
    {"id": "arxiv:2601.0003", "title": "LOW", "abstract": "", "authors": [],
     "url": "http://x/3", "pdf_url": ""},
]


def _score(_self, title, abstract, extra_keywords=None):
    return ({"HIGH": 7.0, "MARG": 1.5, "LOW": 0.0}.get(title, 0.0), ["d"])


@pytest.fixture
def _patched(monkeypatch, tmp_path):
    from agent_utilities.automation.research_pipeline import ResearchPipelineRunner
    from agent_utilities.knowledge_graph.ingestion import manifest as _m
    from agent_utilities.knowledge_graph.research.loop_controller import LoopController

    async def _fetch(self, runner, limit):
        return list(_FEED)[:limit]

    async def _marginal(self, *a, **k):
        return "article:marg"

    # Deterministic grading; novel (no demotion); cheap marginal ingest stub.
    monkeypatch.setattr(ResearchPipelineRunner, "score_paper", _score)
    monkeypatch.setattr(
        ResearchPipelineRunner, "_paper_novelty", lambda self, t, a: 1.0
    )
    monkeypatch.setattr(ResearchPipelineRunner, "ingest_paper_marginal", _marginal)
    monkeypatch.setattr(LoopController, "_fetch_rss_feed", _fetch)
    # Isolate the seen-set DB to a tmp file.
    monkeypatch.setattr(
        _m.DeltaManifest,
        "_default_db_path",
        staticmethod(lambda: str(tmp_path / "manifest.db")),
    )
    return LoopController


def test_feed_screen_grades_and_enqueues_by_priority(_patched):
    ctrl = _patched(_Engine())
    rep = ctrl.run_rss_feed_screen()
    assert rep["feed_items"] == 3
    assert rep["graded"] == 3
    assert rep["queued_full"] == 1  # HIGH only
    assert rep["ingested_marginal"] == 1  # MARG
    assert rep["rejected"] == 1  # LOW
    # The fetch task is enqueued with a grade-derived priority bucket.
    fetches = [s for s in ctrl.engine.submitted if s["task_type"] == "research_paper_fetch"]
    assert len(fetches) == 1
    assert fetches[0]["priority"] == 0  # 7.0 >= 2*relevant(3.0) → most urgent
    assert fetches[0]["extra_meta"]["paper"]["id"] == "arxiv:2601.0001"


def test_feed_screen_skips_already_seen_on_second_tick(_patched):
    eng = _Engine()
    ctrl = _patched(eng)
    first = ctrl.run_rss_feed_screen()
    assert first["graded"] == 3 and first["seen_skipped"] == 0
    # Second tick over the same feed: everything was examined → all skipped.
    second = ctrl.run_rss_feed_screen()
    assert second["seen_skipped"] == 3
    assert second["graded"] == 0 and second["queued_full"] == 0


def test_research_feed_schedule_registered_default_on(monkeypatch):
    from agent_utilities.core import schedule_engine as se
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )
    from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin

    inst = TaskManagerMixin.__new__(TaskManagerMixin)  # type: ignore[type-abstract]
    inst.backend = EpistemicGraphBackend()
    inst._register_maintenance_schedules()
    specs = {s.name: s for s in se._load_all(inst)}
    rf = specs["research_feed"]
    assert rf.enabled and rf.payload == {"kind": "research_feed"}
    assert rf.trigger == "interval"
