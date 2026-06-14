"""Search-distillation harvester + corpus collapse guard (OS-5.36, SAFE-1.4).

The reasoning router's verified high-scoring results become a collapse-guarded SFT /
preference corpus (test-time compute → training data), so recursive distillation has
clean data to consume without quietly degenerating.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.corpus_collapse_guard import CorpusCollapseGuard
from agent_utilities.harness.search_distillation import (
    SearchDistillationHarvester,
    SFTRow,
)
from agent_utilities.knowledge_graph.core.reasoner import (
    ReasonerRouter,
    ReasoningResult,
    ReasoningTask,
)

pytestmark = pytest.mark.concept("OS-5.36")


class _Engine:
    def __init__(self):
        self.nodes = {}

    def add_node(self, nid, ntype, properties=None):
        self.nodes[nid] = {"id": nid, "type": ntype, **(properties or {})}


# ── SAFE-1.4 collapse guard ──────────────────────────────────────────


@pytest.mark.concept("SAFE-1.4")
class TestCollapseGuard:
    def test_rejects_duplicates(self):
        g = CorpusCollapseGuard()
        assert g.admit("k1")[0] is True
        ok, reason = g.admit("k1")
        assert ok is False and "duplicate" in reason

    def test_caps_synthetic_fraction(self):
        g = CorpusCollapseGuard(synthetic_cap=0.5)
        assert g.admit("human", synthetic=False)[0] is True  # 0/1
        assert g.admit("syn1", synthetic=True)[0] is True  # 1/2 = 0.5, not > cap
        ok, reason = g.admit("syn2", synthetic=True)  # 2/3 = 0.67 > 0.5
        assert ok is False and "synthetic fraction" in reason

    def test_embedding_novelty_floor(self):
        g = CorpusCollapseGuard(min_novelty=0.1, synthetic_cap=1.0)  # isolate the novelty gate
        assert g.admit("a", embedding=[1.0, 0.0])[0] is True
        ok, reason = g.admit("b", embedding=[1.0, 0.001])  # ~identical direction
        assert ok is False and "novelty" in reason
        assert g.admit("c", embedding=[0.0, 1.0])[0] is True  # orthogonal ⇒ novel

    def test_is_collapsing_on_synthetic_saturation(self):
        g = CorpusCollapseGuard(synthetic_cap=1.0)
        for i in range(5):
            g.admit(f"k{i}", synthetic=True)
        assert g.is_collapsing() is True  # 100% synthetic
        assert g.diagnostics()["synthetic_fraction"] == 1.0


# ── OS-5.36 harvester ────────────────────────────────────────────────


class _Task:
    def __init__(self, goal, payload=None, embedding=None):
        self.goal = goal
        self.payload = payload or {}
        self.embedding = embedding


class TestHarvester:
    def test_high_score_result_is_distilled(self):
        eng = _Engine()
        h = SearchDistillationHarvester(eng, min_score=0.8)
        row = h.harvest_result(_Task("q1"), ReasoningResult(answer="A1", reasoner="deductive", score=1.0))
        assert isinstance(row, SFTRow) and row.completion == "A1" and row.source == "deductive"
        assert any(n["type"] == "SyntheticCorpus" for n in eng.nodes.values())
        assert len(h.corpus()) == 1

    def test_low_score_not_distilled(self):
        h = SearchDistillationHarvester(min_score=0.8)
        assert h.harvest_result(_Task("q"), ReasoningResult("a", "gen", score=0.3)) is None
        assert h.corpus() == []

    def test_duplicate_rejected_by_guard(self):
        h = SearchDistillationHarvester(min_score=0.5)
        h.harvest_result(_Task("q"), ReasoningResult("same", "d", 1.0))
        assert h.harvest_result(_Task("q"), ReasoningResult("same", "d", 1.0)) is None  # dup
        assert len(h.corpus()) == 1

    def test_best_of_k_yields_sft_and_preference_pairs(self):
        h = SearchDistillationHarvester(min_score=0.8)
        rows, pairs = h.harvest_candidates("q", [("best", 1.0), ("mid", 0.6), ("worst", 0.2)])
        assert len(rows) == 1 and rows[0].completion == "best"
        assert {(p.chosen, p.rejected) for p in pairs} == {("best", "mid"), ("best", "worst")}


# ── live router wiring (OS-5.36 producer) ────────────────────────────


class TestRouterDistillation:
    def test_router_distills_a_win(self):
        eng = _Engine()
        harvester = SearchDistillationHarvester(eng, min_score=0.8)
        router = ReasonerRouter(harvester=harvester)
        router.register(_StubWinner())
        result = router.reason(ReasoningTask(goal="derive", tags=("shared",)))
        assert result.score == 1.0 and result.trace.get("distilled") is True
        assert len(harvester.corpus()) == 1

    def test_router_without_harvester_unchanged(self):
        router = ReasonerRouter()
        router.register(_StubWinner())
        result = router.reason(ReasoningTask(goal="x", tags=("shared",)))
        assert result.score == 1.0 and "distilled" not in result.trace


class _StubWinner:
    name = "winner"
    capability_tags = ("shared",)

    def reason(self, task):
        return ReasoningResult(answer="WIN", reasoner=self.name, score=1.0)
