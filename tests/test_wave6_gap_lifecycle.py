"""Wave-6 unified Gap → SDD → Implement → Promote → Close lifecycle wiring tests.

Proves the spine the ADR (`reports/wave6/ADR-unified-gap-sdd-evolution-lifecycle.md`)
specifies: ONE canonical :Gap that every discovery track folds into, threaded to a
:SpecProposal, closed on publish, with a single provenance chain traversal.

@pytest.mark.concept("AU-AHE.harness.canonical-gap-lifecycle")
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "unit"))

from fleet_autonomy_fakes import FakeEngine  # noqa: E402

from agent_utilities.knowledge_graph.research import gaps  # noqa: E402
from agent_utilities.knowledge_graph.research.gaps import (  # noqa: E402
    get_gap,
    open_gaps,
    resolve_gaps_for_loop,
    submit_gap,
)
from agent_utilities.knowledge_graph.research.spec_proposals import (  # noqa: E402
    get_spec,
    persist_spec_proposal,
    review_spec,
)

pytestmark = pytest.mark.concept("AU-AHE.harness.canonical-gap-lifecycle")


class LifecycleEngine(FakeEngine):
    """FakeEngine + add_edge + the id-lookup / label-scan / edge-walk cyphers the
    Wave-6 lifecycle uses. Backend-agnostic reads work against this in-memory double.
    """

    def add_edge(self, src, dst, rel_type, properties=None):  # noqa: D401
        self.edges.append((src, dst, rel_type))

    def query_cypher(self, query, params=None):
        params = params or {}
        # node-by-id (get_gap / get_spec / load_proposal)
        if "WHERE n.id = $id" in query:
            node = self.nodes.get(params.get("id"))
            return [{"n": dict(node)}] if node else []
        # label scan (open_gaps / list_specs): MATCH (n:<Label>) RETURN n
        m = re.search(r"MATCH \(n:(\w+)\) RETURN n", query)
        if m:
            lbl = m.group(1)
            return [{"n": dict(v)} for v in self.nodes.values() if v.get("type") == lbl]
        # projected label scan (artifact_evolution_summary): MATCH (v:<Label>) RETURN v.id AS id, ...
        m2 = re.search(r"MATCH \(v:(\w+)\) RETURN v\.", query)
        if m2:
            lbl = m2.group(1)
            fields = (
                "id",
                "status",
                "artifact_kind",
                "skill_id",
                "prompt_id",
                "parent_hash",
                "benchmark_score",
                "reward",
                "timestamp",
            )
            return [
                {f: v.get(f) for f in fields}
                for v in self.nodes.values()
                if v.get("type") == lbl
            ]
        # RESOLVES edge walk: (l)-[:RESOLVES]->(g:Gap) WHERE l.id = $id RETURN g.id
        if "[:RESOLVES]->" in query:
            lid = params.get("id")
            return [
                {"id": dst}
                for (s, dst, r) in self.edges
                if s == lid
                and r == "RESOLVES"
                and self.nodes.get(dst, {}).get("type") == gaps.GAP_LABEL
            ]
        return super().query_cypher(query, params)


class _Draft:
    """Minimal SpecDraft-shaped object for persist_spec_proposal."""

    def __init__(self, title, target_file="", concept_ids=None):
        self.title = title
        self.target_codebase = "agent-utilities"
        self.problem = "p"
        self.approach = "a"
        self.value = "v"
        self.concept_ids = concept_ids or []
        self.value_score = 1.0
        self.target_file = target_file


# ---------------------------------------------------------------------------
# W6.1 — the canonical :Gap + discovery-track folds
# ---------------------------------------------------------------------------


def test_submit_gap_persists_canonical_gap_and_lease():
    eng = LifecycleEngine()
    gap = submit_gap(
        eng, source="failure", signature="sig-1", statement="thing broke", severity=0.9
    )
    assert gap and gap["id"] == "gap:failure:sig-1"
    node = get_gap(eng, gap["id"])
    assert node["status"] == gaps.STATUS_OPEN
    assert node["severity"] == 0.9
    # High severity → expedited bucket 0.
    assert gaps.severity_to_bucket(0.9) == 0
    # Given a WorkItem/lease (goal_loop) so it is schedulable, not a transient dict.
    assert any(n.get("type") == "WorkItem" for n in eng.nodes.values())


def test_production_failure_track_folds_into_canonical_gap():
    from agent_utilities.knowledge_graph.adaptation.failure_analyzer import (
        FailurePattern,
        file_gap_topic,
    )

    eng = LifecycleEngine()
    pattern = FailurePattern(
        signature="db-timeout",
        name="ingest",
        kind="error",
        anomaly_type="latency",
        count=5,
        trace_ids=["t1", "t2"],
    )

    # In-memory ChangeEnvelope adapter (the native envelope kernel is absent in this
    # verification venv); the canonical-gap fold runs the same either way.
    def _writer(entities, relationships):
        for e in entities:
            eng.add_node(
                e["id"],
                e.get("node_type", "Concept"),
                properties={k: v for k, v in e.items() if k not in ("id", "node_type")},
            )
        for r in relationships:
            eng.add_edge(r["source"], r["target"], r["relationship"])
        return {"ok": True}

    topic = file_gap_topic(eng, pattern, graph_writer=_writer)
    # The signature is sanitized (hashed) by _safe_pattern; the fold produces a
    # canonical gap under the failure source regardless.
    assert topic and topic["gap_id"].startswith("gap:failure:")
    assert get_gap(eng, topic["gap_id"])["status"] == gaps.STATUS_OPEN
    assert get_gap(eng, topic["gap_id"])["source"] == "failure"


def test_skill_coverage_track_folds_into_canonical_gap():
    from agent_utilities.knowledge_graph.adaptation.skill_evolver import (
        SkillGap,
        submit_skill_gap,
    )

    eng = LifecycleEngine()
    sg = SkillGap(
        task_text="transcode a video file",
        similarity_score=0.1,
        suggested_name="transcode-video",
    )
    gap = submit_skill_gap(eng, sg)
    assert gap and gap["id"] == "gap:skill:transcode-video"
    # Far from any skill (low similarity) → high severity.
    assert gap["severity"] >= 0.85


# ---------------------------------------------------------------------------
# W6.2 — the loop authors a first-class DSTDD Spec+Tasks via SDDManager
# ---------------------------------------------------------------------------


def test_sddmanager_authors_typed_spec_and_tasks_from_draft(tmp_path):
    from agent_utilities.models import Spec, Tasks
    from agent_utilities.sdd import SDDManager

    mgr = SDDManager(tmp_path)
    draft = _Draft(
        "Add retrieval cache",
        target_file="agent_utilities/retrieval/cache.py",
        concept_ids=["c:1"],
    )
    spec_path = mgr.author_from_draft(draft)
    # One writer, first-class DSTDD artifacts (not a raw kg-distilled prose file).
    assert spec_path.exists() and spec_path.name == "spec.md"
    assert (spec_path.parent / "tasks.md").exists()
    # The one spec model round-trips (title/user stories via markdown).
    loaded = mgr.load(Spec, "add-retrieval-cache")
    assert loaded and loaded.title == "Add retrieval cache"
    assert loaded.user_stories and loaded.user_stories[0].title == "Add retrieval cache"
    # Tasks carry the D3 target_file so code-synthesis has a single-file target.
    tasks = mgr.load(Tasks, "add-retrieval-cache")
    assert tasks and len(tasks.tasks) == 4
    assert tasks.tasks[1].file_paths == ["agent_utilities/retrieval/cache.py"]


# ---------------------------------------------------------------------------
# W6.8 — the 4th track: code-correctness/security-audit detector
# ---------------------------------------------------------------------------


def test_audit_detector_files_canonical_gap_per_finding():
    from agent_utilities.harness.audit_gap_detector import AuditGapDetector

    eng = LifecycleEngine()
    # A code unit already ingested into the KG, carrying a real durability defect.
    eng.add_node(
        "code:pkg::save",
        "CodeUnit",
        properties={
            "name": "save",
            "file_path": "pkg/store.py",
            "source": "def save(conn, row):\n    conn.execute(SQL, row)\n"
            "    # NOTE: never conn.commit()\n",
        },
    )

    def fake_review(_prompt):
        return (
            '[{"finding_class":"transaction-durability","severity":"critical",'
            '"statement":"save() executes an INSERT but never commits, so the write '
            'is silently lost."}]'
        )

    filed = AuditGapDetector(eng, review_fn=fake_review).detect(limit=5)
    assert len(filed) == 1
    gap = filed[0]
    assert gap["source"] == "audit"
    assert gap["id"].startswith("gap:audit:transaction-durability")
    # critical → expedited bucket 0.
    assert gap["priority_bucket"] == 0
    # An audit gap starts OPEN like any other track, ready for the SAME lifecycle.
    assert get_gap(eng, gap["id"])["status"] == gaps.STATUS_OPEN


def test_audit_scan_is_opt_in_and_off_by_default(monkeypatch):
    from agent_utilities.core.config import config
    from agent_utilities.harness.audit_gap_detector import run_audit_gap_scan

    eng = LifecycleEngine()
    monkeypatch.setattr(config, "kg_loop_audit", False, raising=False)
    assert run_audit_gap_scan(eng)["skipped"] is True
    # Nothing filed when off — a non-opted-in deployment is unaffected.
    assert not [g for g in eng.nodes.values() if g.get("type") == "Gap"]


# ---------------------------------------------------------------------------
# W6.1/W6.6 — gap → spec SPECIFIED_BY chain hop
# ---------------------------------------------------------------------------


def test_persist_spec_proposal_threads_gap_specified_by():
    eng = LifecycleEngine()
    gap = submit_gap(eng, source="research", signature="feat-x", statement="add x")
    sid = persist_spec_proposal(eng, _Draft("Add X"), gap_id=gap["id"])
    assert sid == "spec_proposal:add-x"
    assert (gap["id"], sid, "SPECIFIED_BY") in eng.edges
    # The gap is now 'specified' (a spec is in flight).
    assert get_gap(eng, gap["id"])["status"] == gaps.STATUS_SPECIFIED
    # The spec carries gap_id so the develop step can close the origin gap.
    assert get_spec(eng, sid)["gap_id"] == gap["id"]


# ---------------------------------------------------------------------------
# W6.5 — close the loop: RESOLVES edge + gap resolved on publish
# ---------------------------------------------------------------------------


def test_approve_binds_develop_loop_with_resolves_edge():
    eng = LifecycleEngine()
    gap = submit_gap(eng, source="research", signature="feat-y", statement="add y")
    sid = persist_spec_proposal(eng, _Draft("Add Y"), gap_id=gap["id"])
    out = review_spec(eng, sid, "approve")
    loop_id = out["develop_loop"]["id"]
    assert loop_id == f"loop:develop:{sid}"
    # (develop-Loop)-[:RESOLVES]->(:Gap) written, and gap_id stamped on the loop.
    assert (loop_id, gap["id"], "RESOLVES") in eng.edges
    assert eng.nodes[loop_id]["gap_id"] == gap["id"]


def test_resolve_gaps_for_loop_walks_edge_and_closes_gap():
    eng = LifecycleEngine()
    gap = submit_gap(eng, source="research", signature="feat-z", statement="add z")
    sid = persist_spec_proposal(eng, _Draft("Add Z"), gap_id=gap["id"])
    out = review_spec(eng, sid, "approve")
    loop_id = out["develop_loop"]["id"]
    resolved = resolve_gaps_for_loop(eng, loop_id)
    assert resolved == [gap["id"]]
    assert get_gap(eng, gap["id"])["status"] == gaps.STATUS_RESOLVED
    # A resolved gap drops out of the open backlog.
    assert gap["id"] not in {g["id"] for g in open_gaps(eng)}


def test_develop_spec_closes_gap_on_publish(monkeypatch):
    from agent_utilities.knowledge_graph.research import spec_proposals

    eng = LifecycleEngine()
    gap = submit_gap(eng, source="research", signature="feat-w", statement="add w")
    sid = persist_spec_proposal(eng, _Draft("Add W"), gap_id=gap["id"])
    review_spec(eng, sid, "approve")
    # Monkeypatch the governed publish so this unit stays git-free; the closure is what we test.
    # develop_spec imports governed_publish from change_publisher at call time, so patch
    # it there.
    from agent_utilities.knowledge_graph.research import change_publisher

    monkeypatch.setattr(
        change_publisher, "governed_publish", lambda *a, **k: {"status": "published"}
    )
    res = spec_proposals.develop_spec(eng, sid)
    assert res["status"] == "published"
    assert res["gap_id"] == gap["id"]
    assert get_gap(eng, gap["id"])["status"] == gaps.STATUS_RESOLVED
