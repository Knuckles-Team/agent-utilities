#!/usr/bin/python
"""Research cohort — batch ingest + self-polling barrier (CONCEPT:KG-2.172)."""

import pytest

from agent_utilities.knowledge_graph.research.cohort import (
    SYNTHESIZE_TASK_TYPE,
    cohort_ready,
    cohort_status,
    create_cohort,
    finalize_cohort,
)

pytestmark = pytest.mark.concept("KG-2.172")


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
    assert sorted(t["meta"]["type"] for t in members) == [
        "codebase",
        "content_url",
        "content_url",
    ]
    # papers ride the real URL on source_url (content_url Path()-mangles target)
    paper = next(t for t in members if t["meta"]["type"] == "content_url")
    assert paper["source_url"].startswith("https://arxiv.org")
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


def test_finalize_runs_assimilate_and_marks_synthesized(monkeypatch):
    eng = _FakeEngine()
    cid = create_cohort(eng, papers=["u1"], repos=[])["cohort_id"]
    eng.set_status(f"{cid}:p0", "completed")

    captured: dict = {}

    def fake_pass(engine, *, force=False, **_kw):
        captured["force"] = force
        return {
            "auto_satisfied": 3,
            "related": 2,
            "open_gaps": 5,
            "synergy_bundles": 1,
            "feature_matrix": {
                "node_id": "feature_matrix:latest",
                "counts": {"total": 7, "novel": 5, "bundles": 1},
            },
        }

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.research.loop_controller.run_assimilation_pass",
        fake_pass,
    )

    res = finalize_cohort(eng, cid)
    assert captured["force"] is True
    assert res["feature_matrix"]["counts"]["total"] == 7
    assert eng._nodes[cid]["status"] == "synthesized"

    # the unified status surface reads it back (members + matrix counts)
    cs = cohort_status(eng, cid)
    assert cs["status"] == "synthesized"
    assert cs["feature_matrix"]["total"] == 7
    assert cs["members"]["completed"] == 1
