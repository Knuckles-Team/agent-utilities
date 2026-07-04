#!/usr/bin/python
"""Research cohort — batch ingest + self-polling barrier (CONCEPT:AU-KG.coordination.research-cohort-barrier)."""

import pytest

from agent_utilities.knowledge_graph.research.cohort import (
    SYNTHESIZE_TASK_TYPE,
    cohort_ready,
    cohort_status,
    create_cohort,
    finalize_cohort,
)

pytestmark = pytest.mark.concept("AU-KG.coordination.research-cohort-barrier")


class _Graph:
    def __init__(self, nodes):
        self._n = nodes

    def nodes(self, data=False):
        return list(self._n.items()) if data else list(self._n)


class _FakeEngine:
    """Records submitted tasks + nodes; serves the cohort status query."""

    def __init__(self):
        self.tasks: dict = {}
        self._nodes: dict = {}
        self.graph = _Graph(self._nodes)
        self._n = 0

    def submit_task(
        self,
        target,
        is_codebase,
        provenance,
        task_type=None,
        extra_meta=None,
        job_id=None,
        skip_dedupe=False,
        **_kw,
    ):
        if not job_id:
            self._n += 1
            job_id = f"job-{self._n}"
        meta = {
            "target": target,
            "type": task_type or ("codebase" if is_codebase else "document"),
        }
        if extra_meta:
            meta.update(extra_meta)
        self.tasks[job_id] = {
            "status": "pending",
            "meta": meta,
            "source_url": (provenance or {}).get("source_url"),
        }
        return job_id

    def add_node(self, node_id, node_type=None, properties=None, **_kw):
        self._nodes[node_id] = {
            **self._nodes.get(node_id, {}),
            **(properties or {}),
            "type": node_type,
            "id": node_id,
        }

    def _control_cypher(self, cypher, params=None):
        from agent_utilities.knowledge_graph.core.engine_tasks import _encode_metadata

        if "t.status as s" in cypher:
            return [
                {"s": t["status"], "meta": _encode_metadata(t["meta"])}
                for t in self.tasks.values()
            ]
        return []

    def set_status(self, job_id, status):
        self.tasks[job_id]["status"] = status


def test_create_cohort_fans_out_and_tags():
    eng = _FakeEngine()
    out = create_cohort(
        eng,
        papers=["https://arxiv.org/abs/2606.18381", "https://arxiv.org/abs/2606.18508"],
        repos=["/oss/SproutRAG"],
        goal="evolve retrieval",
    )
    cid = out["cohort_id"]
    assert out["papers"] == 2 and out["repos"] == 1 and len(out["members"]) == 3

    # cohort node created in 'ingesting' state
    assert eng._nodes[cid]["status"] == "ingesting"
    assert eng._nodes[cid]["member_count"] == 3

    members = [
        t
        for t in eng.tasks.values()
        if t["meta"].get("cohort_id") == cid
        and t["meta"]["type"] != SYNTHESIZE_TASK_TYPE
    ]
    # papers → research_paper_fetch (Article nodes, KG-2.194); repos → codebase
    assert sorted(t["meta"]["type"] for t in members) == [
        "codebase",
        "research_paper_fetch",
        "research_paper_fetch",
    ]
    # each paper task carries a paper dict with the parsed arxiv id + url
    paper = next(t for t in members if t["meta"]["type"] == "research_paper_fetch")
    p = paper["meta"]["paper"]
    assert p["id"] == "2606.18381" and p["url"].startswith("https://arxiv.org")
    # exactly one barrier gate, tagged but not a member
    assert eng.tasks[f"{cid}:synth"]["meta"]["type"] == SYNTHESIZE_TASK_TYPE


def test_readiness_terminal_poison_member_and_deadline():
    eng = _FakeEngine()
    cid = create_cohort(eng, papers=["u1", "u2"], repos=[])["cohort_id"]

    ready, st = cohort_ready(eng, cid, deadline_unix=0.0)
    assert not ready and st["total"] == 2 and st["pending"] == 2

    # one completes, one FAILS — both terminal → cohort still proceeds (no wedge)
    eng.set_status(f"{cid}:p0", "completed")
    eng.set_status(f"{cid}:p1", "failed")
    ready, st = cohort_ready(eng, cid, deadline_unix=0.0)
    assert ready and st["completed"] == 1 and st["failed"] == 1 and st["terminal"] == 2


def test_deadline_forces_ready_with_pending_members():
    eng = _FakeEngine()
    cid = create_cohort(eng, papers=["u1"], repos=[])["cohort_id"]
    # still pending, but a long-past deadline (epoch 1) forces readiness
    ready, _ = cohort_ready(eng, cid, deadline_unix=1.0)
    assert ready


def test_empty_cohort_is_trivially_ready():
    eng = _FakeEngine()
    cid = create_cohort(eng, papers=[], repos=[])["cohort_id"]
    ready, st = cohort_ready(eng, cid, deadline_unix=0.0)
    assert ready and st["total"] == 0


def test_cohort_source_ids_from_task_provenance():
    """Provenance (KG-2.192): the cohort's source node ids are recovered from member
    tasks' recorded article_id — failures (no article_id) contribute nothing."""
    from agent_utilities.knowledge_graph.research.cohort import cohort_source_ids

    eng = _FakeEngine()
    cid = create_cohort(eng, papers=["2606.1", "2606.2"], repos=[])["cohort_id"]
    eng.tasks[f"{cid}:p0"]["status"] = "completed"
    eng.tasks[f"{cid}:p0"]["meta"]["article_id"] = "article:scholarx:arxiv-2606.1"
    eng.tasks[f"{cid}:p1"]["status"] = "failed"  # no article_id recorded
    assert cohort_source_ids(eng, cid) == {"article:scholarx:arxiv-2606.1"}


def test_finalize_is_scoped_and_marks_synthesized(monkeypatch):
    eng = _FakeEngine()
    cid = create_cohort(eng, papers=["2606.18381"], repos=[])["cohort_id"]
    # simulate the research_paper_fetch handler: completed + recorded article_id
    eng.tasks[f"{cid}:p0"]["status"] = "completed"
    eng.tasks[f"{cid}:p0"]["meta"]["article_id"] = "article:arxiv-2606.18381"

    captured: dict = {}

    def fake_pass(engine, *, force=False, restrict_to=None, matrix_node_id="", **_kw):
        captured.update(force=force, restrict_to=restrict_to, node=matrix_node_id)
        return {
            "auto_satisfied": 1,
            "related": 1,
            "open_gaps": 1,
            "synergy_bundles": 0,
            "feature_matrix": {
                "node_id": matrix_node_id,
                "counts": {"total": 1, "novel": 1},
            },
        }

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.research.loop_controller.run_assimilation_pass",
        fake_pass,
    )

    finalize_cohort(eng, cid)
    # SCOPED to the cohort's article id (provenance → restrict_to), and a cohort-OWN
    # matrix node (never clobbers the ecosystem feature_matrix:latest)
    assert captured["restrict_to"] == {"article:arxiv-2606.18381"}
    assert captured["node"] == f"feature_matrix:{cid}"
    assert captured["force"] is True
    assert eng._nodes[cid]["status"] == "synthesized"

    cs = cohort_status(eng, cid)
    assert cs["status"] == "synthesized"
    assert cs["feature_matrix"]["total"] == 1
    assert cs["members"]["completed"] == 1
