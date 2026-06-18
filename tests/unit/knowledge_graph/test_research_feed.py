"""Unified feed research path — grade + prioritized enqueue (CONCEPT:KG-2.114/2.121).

The scholarx-only ``run_rss_feed_screen`` was collapsed into the one world-model
gate: research items from ANY feed (native RSS, ScholarX, FreshRSS-arXiv) go through
``grade_and_enqueue_paper`` — keyword+novelty grade → prioritized ``research_paper_fetch``
(best-graded first) / abstract-only / reject. These pin that shared decision and the
``research_feed`` schedule registration.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.research.feed_grading import grade_and_enqueue_paper


class _Engine:
    def __init__(self):
        self.backend = None
        self.submitted: list[dict] = []

    def submit_task(self, **kw):
        self.submitted.append(kw)
        return kw.get("job_id", "job-x")

    def query_cypher(self, q, params=None):
        return []

    def add_node(self, node_id, node_type, properties=None):
        pass


@pytest.fixture
def _patched(monkeypatch):
    """Deterministic grading: HIGH→7.0, MARG→1.5, LOW→0.0; novel; marginal no-op."""
    from agent_utilities.automation.research_pipeline import ResearchPipelineRunner

    async def _marginal(self, *a, **k):
        return "article:marg"

    monkeypatch.setattr(
        ResearchPipelineRunner,
        "score_paper",
        lambda self, t, a, extra_keywords=None: (
            {"HIGH": 7.0, "MARG": 1.5, "LOW": 0.0}.get(t, 0.0),
            ["d"],
        ),
    )
    monkeypatch.setattr(ResearchPipelineRunner, "_paper_novelty", lambda self, t, a: 1.0)
    monkeypatch.setattr(ResearchPipelineRunner, "ingest_paper_marginal", _marginal)


def _paper(pid, title):
    return {"id": pid, "title": title, "abstract": "", "authors": [], "url": f"http://x/{pid}"}


def test_high_grade_enqueues_prioritized_fetch(_patched):
    eng = _Engine()
    res = grade_and_enqueue_paper(eng, _paper("arxiv:1", "HIGH"))
    assert res["tier"] == "queued_full"
    assert res["bucket"] == 0  # 7.0 >= 2*relevant(3.0) → most urgent
    fetch = eng.submitted[-1]
    assert fetch["task_type"] == "research_paper_fetch"
    assert fetch["priority"] == 0
    assert fetch["extra_meta"]["paper"]["id"] == "arxiv:1"
    assert fetch["skip_dedupe"] is False  # queue target-dedup handles re-grades


def test_marginal_grade_ingests_abstract_only(_patched):
    eng = _Engine()
    res = grade_and_enqueue_paper(eng, _paper("arxiv:2", "MARG"))
    assert res["tier"] == "ingested_marginal"
    assert not eng.submitted  # no fetch task


def test_low_grade_rejected(_patched):
    eng = _Engine()
    res = grade_and_enqueue_paper(eng, _paper("arxiv:3", "LOW"))
    assert res["tier"] == "rejected"
    assert not eng.submitted


def test_gate_routes_research_item_through_grade_and_enqueue(_patched, monkeypatch):
    """A research SourceDocument flowing through the world-model gate enqueues a fetch."""
    from agent_utilities.automation import worldmodel_pipeline as wm
    from agent_utilities.protocols.source_connectors.base import SourceDocument

    eng = _Engine()
    eng.graph = None  # _is_known short-circuits → treat as unknown
    doc = SourceDocument(
        id="arxiv:2601.0009",
        title="HIGH",
        text="abstract",
        metadata={
            "record": {
                "id": "arxiv:2601.0009",
                "origin": {"streamId": "scholarx:arxiv", "htmlUrl": "http://a/9"},
                "canonical": [{"href": "http://arxiv.org/abs/2601.0009"}],
            },
            "source_system": "rss",
        },
    )
    report = wm.WorldModelPipelineRunner(engine=eng).run_gated_ingest([doc])
    assert report.research == 1
    assert eng.submitted and eng.submitted[-1]["task_type"] == "research_paper_fetch"


def test_research_feed_schedule_registered_default_on():
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
