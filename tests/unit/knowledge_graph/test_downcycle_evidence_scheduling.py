"""7.4 — downcycle scheduling for deep research / comparative analysis.

Two things this file proves:

1. **Live wiring** — the deep-research stages (``_run_intake_papers``,
   ``_run_breadth``) genuinely enter ``PriorityClass.BACKGROUND_INGESTION``
   on the real call path (reusing ``core.resource_priority`` — no second
   scheduler; the gate-level yield mechanics themselves are already covered
   by ``tests/unit/core/test_resource_priority.py``).
2. **Yield-and-resume-exactly-once** — an evolution run interrupted mid-flight
   (a "pause" signal, the same shape a live user query would raise) yields
   IMMEDIATELY (before the next step starts, never mid-step) and, on resume,
   continues from its checkpointed memento exactly once with no duplicate
   writes — reusing the EXISTING ``run_loop``/``WorkItem`` checkpoint/lease
   machinery (``orchestration/work_item.py``), proven here specifically for
   iterations that record :class:`~agent_utilities.knowledge_graph.research.
   evidence.Evidence` (lane 7.1), so a resumed run never double-records the
   evidence an already-committed iteration produced.
"""

from __future__ import annotations

import asyncio

from agent_utilities.core.resource_priority import PriorityClass, current_priority
from agent_utilities.knowledge_graph.research.evidence import (
    Evidence,
    EvidenceChannel,
    EvidenceOutcome,
    record_evidence,
)
from agent_utilities.knowledge_graph.research.loop_controller import LoopController
from agent_utilities.knowledge_graph.research.loops import claim_loop, submit_loop
from agent_utilities.orchestration import work_item as wi
from tests.unit.knowledge_graph.test_loops import LoopEngine, _authority


class _StubEngine:
    """Minimal engine for the standalone _run_breadth/_run_intake_papers calls."""

    def __init__(self) -> None:
        self.backend = object()

    def query_cypher(self, q, params=None):
        return []


def _submit(engine: LoopEngine, *, loop_id: str) -> dict:
    with _authority():
        return submit_loop(
            engine,
            "bounded objective",
            loop_id=loop_id,
            kind="develop",
            validation_cmd="validate",
        )


# ---------------------------------------------------------------------------
# 1. Live wiring — the stages genuinely enter BACKGROUND_INGESTION
# ---------------------------------------------------------------------------


def test_run_breadth_executes_under_background_ingestion_priority(monkeypatch):
    import agent_utilities.core.workspace_config as wc
    import agent_utilities.knowledge_graph.assimilation as assim
    from agent_utilities.knowledge_graph.assimilation.breadth_ingest import (
        BreadthReport,
    )

    monkeypatch.delenv("KG_BREADTH_LIBRARY_ROOTS", raising=False)
    monkeypatch.delenv("KG_BREADTH_REPO_ROOTS", raising=False)
    monkeypatch.setattr(wc, "workspace_project_roots", lambda *a, **k: ["/eco/repo-a"])

    observed: dict = {}

    def fake_run(engine, *, library_roots=None, repo_roots=None, **kw):
        observed["priority"] = current_priority()
        return BreadthReport()

    monkeypatch.setattr(assim, "run_breadth_ingest", fake_run)

    assert current_priority() is None  # untagged outside the stage
    LoopController(_StubEngine())._run_breadth()
    assert observed["priority"] is PriorityClass.BACKGROUND_INGESTION
    # the scope is exited again afterwards — never leaks into the caller
    assert current_priority() is None


def test_run_assimilate_satisfy_stage_executes_under_background_ingestion_priority(
    monkeypatch,
):
    """(D-71-7) ``_run_assimilate``'s own comparative-analysis LLM-capacity call
    (``ConceptMatcher.satisfy``, "embedding-recall + LLM-judge" per its own
    docstring) enters ``BACKGROUND_INGESTION`` too -- not just the two stages
    lane 7.1 originally wired."""
    import agent_utilities.knowledge_graph.assimilation as assim
    from agent_utilities.knowledge_graph.assimilation.concept_matcher import (
        MatchReport,
    )

    class _Graph:
        def __init__(self):
            self._n: dict = {}

        def nodes(self, data=False):
            return list(self._n.items()) if data else list(self._n)

        def add_node(self, nid, attrs):
            self._n[nid] = attrs

        def out_edges(self, nid, data=False):
            return []

        def in_edges(self, nid, data=False):
            return []

    class _Engine:
        def __init__(self):
            self.graph = _Graph()
            self.backend = None

        def add_node(self, nid, node_type, properties=None, ephemeral=False):
            self.graph.add_node(nid, {**(properties or {}), "type": node_type})

        def link_nodes(self, src, dst, rel_type, properties=None, ephemeral=False):
            pass

    observed: dict = {}

    def fake_satisfy(self, engine, *, feature_types, concept_types, restrict_to=None):
        observed["priority"] = current_priority()
        return MatchReport(features=0, concepts=0)

    monkeypatch.setattr(assim.ConceptMatcher, "satisfy", fake_satisfy)
    monkeypatch.setattr(assim, "dedup_features", lambda engine: None)
    monkeypatch.setattr(assim, "enrich_concepts", lambda engine: None)
    monkeypatch.setattr(
        assim,
        "synergy_bundles",
        lambda engine, restrict_to=None: type("S", (), {"bundles": []})(),
    )
    monkeypatch.setattr(assim, "rank_features", lambda engine, feature_ids=None: [])

    assert current_priority() is None  # untagged outside the stage
    LoopController(_Engine())._run_assimilate(force=True)
    assert observed["priority"] is PriorityClass.BACKGROUND_INGESTION
    # the scope is exited again afterwards — never leaks into the caller
    assert current_priority() is None


def test_run_intake_papers_executes_under_background_ingestion_priority(monkeypatch):
    from types import SimpleNamespace

    import agent_utilities.automation.research_pipeline as rp

    observed: dict = {}

    class _FakeRunner:
        def __init__(self, engine=None, **kw):
            pass

        async def run_daily_pipeline(self, papers=None):
            observed["priority"] = current_priority()
            return SimpleNamespace(
                papers_discovered=0,
                papers_relevant=0,
                papers_marginal=0,
                papers_already_known=0,
                owl_inferences=0,
                errors=[],
            )

    monkeypatch.setattr(rp, "ResearchPipelineRunner", _FakeRunner)

    LoopController(_StubEngine())._run_intake_papers(papers=[{"id": "p1"}])
    assert observed["priority"] is PriorityClass.BACKGROUND_INGESTION
    assert current_priority() is None


# ---------------------------------------------------------------------------
# 2. Yield-and-resume-exactly-once, tied to Evidence dedup
# ---------------------------------------------------------------------------


def test_pause_signal_yields_before_the_next_step_ever_starts():
    """A "pause" (the shape a live user query's interrupt takes) is honored
    BEFORE the next iteration's step function runs — never mid-step — so no
    partial/duplicate work is ever in flight when the run yields."""
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:pause-immediate")
    calls = 0

    def develop(_cmd, _cwd):
        nonlocal calls
        calls += 1
        return False, f"try {calls}"

    with _authority():
        result = asyncio.run(
            LoopController(engine, develop_runner=develop).run_loop(
                loop,
                max_iterations=10,
                desired_state=lambda: "pause",
            )
        )
    assert result["status"] == "paused"
    assert result["interrupted"] is True
    assert calls == 0  # yielded before the FIRST step ever ran
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    # The WorkItem itself goes back to "ready" (re-claimable for resume) — the
    # LOGICAL "paused" status lives on the Loop/Concept node (``result["status"]``
    # above), not the WorkItem row; no step was ever checkpointed.
    assert item is not None and item["status"] == "ready"
    assert item.get("checkpoint_id") is None


def test_interrupted_run_resumes_from_memento_exactly_once_no_duplicate_evidence():
    """Simulate a run interrupted mid-flight after 2 iterations each recorded
    one Evidence node (standing in for downcycle research work done as part
    of that checkpointed step) — the SAME expired-lease crash scenario
    ``test_expired_lease_resumes_after_last_fenced_checkpoint`` already
    proves resumes correctly, reused here to additionally prove the resume
    never DUPLICATES the Evidence an already-committed iteration wrote.
    The resumed run must continue from iteration 3 (not re-run 1/2) and end
    with EXACTLY 3 distinct Evidence node ids — never more."""
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:crash-resume")
    item_id = wi.loop_work_item_id(loop["id"])
    recorded_ids: list[str] = []

    def _record(n: int) -> None:
        ev = Evidence(
            channel=EvidenceChannel.RESEARCH_FINDING,
            subject_id=f"iteration-{n}",
            outcome=EvidenceOutcome.PROPOSED,
            signal=0.0,
            confidence=1.0,
            occurred_at=f"iteration-{n}",
        )
        recorded_ids.append(record_evidence(engine, ev))

    # -- simulate "2 iterations already ran and each recorded its evidence,
    # then the process crashed" — the exact fixture shape
    # test_expired_lease_resumes_after_last_fenced_checkpoint uses: claim,
    # checkpoint through iteration 2, then an expired lease. --
    with _authority():
        assert claim_loop(engine, loop["id"])
        claim = wi.current_work_item_claim(engine, item_id)
        assert claim is not None
        _record(1)
        _record(2)
        assert wi.checkpoint_work_item(engine, item_id, claim, "checkpoint:iteration:2")
    engine.nodes[item_id]["lease_expires_at"] = 0.0
    assert len(set(recorded_ids)) == 2

    calls = 0

    def develop(_cmd, _cwd):
        nonlocal calls
        calls += 1
        _record(2 + calls)
        return True, "done"

    # -- resume: must continue from iteration 3, NOT re-run 1/2 (which would
    # be visible as extra develop_runner calls / extra recorded evidence). --
    with _authority():
        result = asyncio.run(
            LoopController(engine, develop_runner=develop).run_loop(
                loop, max_iterations=10
            )
        )
    assert result["status"] == "completed"
    assert result["iterations"] == 3
    assert calls == 1  # only iteration 3 ran — 1/2 were NOT re-executed
    assert len(recorded_ids) == 3
    assert len(set(recorded_ids)) == 3  # every id distinct — no duplicate content
    final_item = wi.get_work_item(engine, item_id)
    assert final_item["checkpoint_id"] == "checkpoint:iteration:3"
    assert final_item["status"] == "succeeded"
