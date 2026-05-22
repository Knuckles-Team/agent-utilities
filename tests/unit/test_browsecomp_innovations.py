"""Tests for BrowseComp-Plus innovation transfer components.

Covers all 5 components:
- Component 1: Fixed Corpus Evaluation (EvaluationCorpus, CorpusManager)
- Component 2: Adaptive Reasoning Effort (get_budget, estimate_query_complexity)
- Component 3: Disentangled Evaluation (evaluate_disentangled, compute_ndcg)
- Component 4: Hard Negative Mining (HardNegativeMiner)
- Component 5: Citation Tracker (extract_citations, evaluate_citations)
"""


import pytest

# ── Component 1: Fixed Corpus Evaluation ──────────────────────────────────


class TestEvaluationCorpus:
    """Tests for the EvaluationCorpus model and CorpusManager."""

    def test_corpus_model_creation(self):
        from agent_utilities.knowledge_graph.retrieval.evaluation_corpus import (
            CorpusQuery,
            EvaluationCorpus,
        )

        corpus = EvaluationCorpus(
            corpus_id="test-corpus",
            name="Test Corpus",
            document_ids=["doc-1", "doc-2", "doc-3"],
            queries=[
                CorpusQuery(
                    query="What is X?",
                    answer="Y",
                    gold_doc_ids=["doc-1"],
                ),
            ],
            document_count=3,
            query_count=1,
        )
        assert corpus.corpus_id == "test-corpus"
        assert corpus.name == "Test Corpus"
        assert len(corpus.document_ids) == 3
        assert len(corpus.queries) == 1
        assert corpus.frozen is False
        assert corpus.frozen_at is None

    def test_corpus_query_model(self):
        from agent_utilities.knowledge_graph.retrieval.evaluation_corpus import (
            CorpusQuery,
        )

        q = CorpusQuery(
            query="Compare A and B",
            answer="A differs from B in X",
            gold_doc_ids=["doc-1", "doc-2"],
            difficulty="hard",
        )
        assert q.difficulty == "hard"
        assert len(q.gold_doc_ids) == 2

    def test_corpus_default_not_frozen(self):
        from agent_utilities.knowledge_graph.retrieval.evaluation_corpus import (
            EvaluationCorpus,
        )

        corpus = EvaluationCorpus(corpus_id="c1", name="test", document_ids=[])
        assert corpus.frozen is False

    def test_corpus_serialization(self):
        from agent_utilities.knowledge_graph.retrieval.evaluation_corpus import (
            EvaluationCorpus,
        )

        corpus = EvaluationCorpus(
            corpus_id="c1",
            name="test",
            document_ids=["d1"],
        )
        data = corpus.model_dump(mode="json")
        assert data["corpus_id"] == "c1"
        assert data["document_ids"] == ["d1"]


# ── Component 2: Adaptive Reasoning Effort ────────────────────────────────


class TestReasoningEffort:
    """Tests for continuous reasoning effort budget scaling."""

    def test_get_budget_zero(self):
        from agent_utilities.harness.reasoning_effort import get_budget

        b = get_budget(0.0)
        assert b.effort == 0.0
        assert b.max_search_calls == 1
        assert b.max_retrieval_depth == 1
        assert b.enable_decomposition is False

    def test_get_budget_one(self):
        from agent_utilities.harness.reasoning_effort import get_budget

        b = get_budget(1.0)
        assert b.effort == 1.0
        assert b.max_search_calls >= 7
        assert b.max_retrieval_depth >= 3
        assert b.context_window >= 20
        assert b.enable_decomposition is True
        assert b.max_decomposition_subtasks >= 4

    def test_get_budget_half(self):
        from agent_utilities.harness.reasoning_effort import get_budget

        b = get_budget(0.5)
        assert 0.4 <= b.effort <= 0.6
        assert b.max_search_calls >= 2
        assert b.enable_decomposition is True

    def test_get_budget_clamping(self):
        from agent_utilities.harness.reasoning_effort import get_budget

        b_low = get_budget(-0.5)
        assert b_low.effort == 0.0
        b_high = get_budget(2.0)
        assert b_high.effort == 1.0

    def test_get_budget_monotonic(self):
        """Higher effort should produce higher search call budgets."""
        from agent_utilities.harness.reasoning_effort import get_budget

        budgets = [get_budget(e / 10) for e in range(11)]
        search_calls = [b.max_search_calls for b in budgets]
        # Should be non-decreasing
        for i in range(1, len(search_calls)):
            assert search_calls[i] >= search_calls[i - 1]

    def test_decomposition_threshold(self):
        """Decomposition should activate at effort >= 0.3."""
        from agent_utilities.harness.reasoning_effort import get_budget

        assert get_budget(0.2).enable_decomposition is False
        assert get_budget(0.3).enable_decomposition is True
        assert get_budget(0.5).enable_decomposition is True

    def test_estimate_empty_query(self):
        from agent_utilities.harness.reasoning_effort import (
            estimate_query_complexity,
        )

        assert estimate_query_complexity("") == 0.0
        assert estimate_query_complexity("   ") == 0.0

    def test_estimate_simple_query(self):
        from agent_utilities.harness.reasoning_effort import (
            estimate_query_complexity,
        )

        score = estimate_query_complexity("What is Python?")
        assert 0.0 <= score <= 0.3

    def test_estimate_complex_query(self):
        from agent_utilities.harness.reasoning_effort import (
            estimate_query_complexity,
        )

        score = estimate_query_complexity(
            "Compare and contrast the differences between X and Y, "
            "analyzing the impact of each approach across multiple "
            "domains? Also, evaluate the historical evolution."
        )
        assert score >= 0.3

    def test_estimate_always_in_range(self):
        from agent_utilities.harness.reasoning_effort import (
            estimate_query_complexity,
        )

        queries = [
            "hi",
            "x" * 1000,
            "why? why? why? why? why?",
            "compare, contrast, analyze, evaluate, investigate, diagnose",
        ]
        for q in queries:
            score = estimate_query_complexity(q)
            assert 0.0 <= score <= 1.0, f"Out of range for: {q!r}"


# ── Component 3: Disentangled Evaluation & nDCG ──────────────────────────


class TestNDCG:
    """Tests for nDCG computation on the RetrievalQualityGate."""

    def _make_gate(self):
        from unittest.mock import MagicMock

        from agent_utilities.knowledge_graph.retrieval.retrieval_quality import (
            RetrievalQualityGate,
        )

        engine = MagicMock()
        return RetrievalQualityGate(engine)

    def test_ndcg_perfect(self):
        gate = self._make_gate()
        results = [{"id": "d1"}, {"id": "d2"}, {"id": "d3"}]
        gold = {"d1", "d2", "d3"}
        score = gate.compute_ndcg(results, gold, k=3)
        assert abs(score - 1.0) < 0.01

    def test_ndcg_empty_results(self):
        gate = self._make_gate()
        assert gate.compute_ndcg([], {"d1"}) == 0.0

    def test_ndcg_empty_gold(self):
        gate = self._make_gate()
        assert gate.compute_ndcg([{"id": "d1"}], set()) == 0.0

    def test_ndcg_no_matches(self):
        gate = self._make_gate()
        results = [{"id": "d1"}, {"id": "d2"}]
        gold = {"d3"}
        assert gate.compute_ndcg(results, gold) == 0.0

    def test_ndcg_partial(self):
        gate = self._make_gate()
        # Gold at rank 2, not at rank 1
        results = [{"id": "d1"}, {"id": "d2"}, {"id": "d3"}]
        gold = {"d2"}
        score = gate.compute_ndcg(results, gold, k=3)
        # Should be < 1.0 since gold is not at rank 1
        assert 0.0 < score < 1.0


# ── Component 4: Hard Negative Mining ─────────────────────────────────────


class TestHardNegativeMiner:
    """Tests for the hard negative miner."""

    def test_model_creation(self):
        from agent_utilities.knowledge_graph.retrieval.hard_negative_miner import (
            HardNegative,
        )

        hn = HardNegative(
            doc_id="d1",
            triggering_subquery="sub-q",
            original_query="full query",
            relevance_score=0.8,
        )
        assert hn.doc_id == "d1"
        assert hn.relevance_score == 0.8

    def test_penalty_application(self):
        from unittest.mock import MagicMock

        from agent_utilities.knowledge_graph.retrieval.hard_negative_miner import (
            HardNegativeMiner,
        )

        retriever = MagicMock()
        miner = HardNegativeMiner(retriever, penalty_factor=0.5)

        results = [
            {"id": "d1", "_score": 1.0},
            {"id": "d2", "_score": 0.9},
            {"id": "d3", "_score": 0.8},
        ]
        penalized = miner.apply_penalties(results, {"d2"})
        # d2 should have reduced score
        d2 = next(r for r in penalized if r["id"] == "d2")
        assert d2["_score"] == pytest.approx(0.45)
        assert d2["_hard_negative"] is True

    def test_penalty_no_negatives(self):
        from unittest.mock import MagicMock

        from agent_utilities.knowledge_graph.retrieval.hard_negative_miner import (
            HardNegativeMiner,
        )

        retriever = MagicMock()
        miner = HardNegativeMiner(retriever)
        results = [{"id": "d1", "_score": 1.0}]
        same = miner.apply_penalties(results, set())
        assert same[0]["_score"] == 1.0

    def test_env_var_gating(self):
        from unittest.mock import MagicMock

        from agent_utilities.knowledge_graph.retrieval.hard_negative_miner import (
            HardNegativeMiner,
        )

        retriever = MagicMock()
        miner = HardNegativeMiner(retriever)
        # Without env var, mine() should return empty
        result = miner.mine("complex query")
        assert result == []


# ── Component 5: Citation Tracker ─────────────────────────────────────────


class TestCitationTracker:
    """Tests for citation extraction and evaluation."""

    def _tracker(self):
        from agent_utilities.harness.citation_tracker import CitationTracker

        return CitationTracker()

    def test_extract_kg_refs(self):
        tracker = self._tracker()
        text = "Based on [KG:node-123] and [source:node-456], we conclude..."
        citations = tracker.extract_citations(text)
        ids = {c.source_id for c in citations}
        assert "node-123" in ids
        assert "node-456" in ids
        assert all(c.citation_type == "kg_node" for c in citations)

    def test_extract_concept_refs(self):
        tracker = self._tracker()
        text = "This implements CONCEPT:KG-2.3 and CONCEPT:AHE-3.1"
        citations = tracker.extract_citations(text)
        ids = {c.source_id for c in citations}
        assert "KG-2.3" in ids
        assert "AHE-3.1" in ids

    def test_extract_urls(self):
        tracker = self._tracker()
        text = "See https://arxiv.org/abs/2508.06600 for details."
        citations = tracker.extract_citations(text)
        urls = [c for c in citations if c.citation_type == "url"]
        assert len(urls) >= 1

    def test_extract_arxiv_ids(self):
        tracker = self._tracker()
        text = "Refer to arXiv: 2508.06600 and 2401.12345"
        citations = tracker.extract_citations(text)
        arxiv = [c for c in citations if c.citation_type == "arxiv"]
        ids = {c.source_id for c in arxiv}
        assert "2508.06600" in ids
        assert "2401.12345" in ids

    def test_extract_file_refs(self):
        tracker = self._tracker()
        text = "See file:///home/user/doc.md for context."
        citations = tracker.extract_citations(text)
        files = [c for c in citations if c.citation_type == "file"]
        assert len(files) == 1

    def test_evaluate_perfect_precision(self):
        tracker = self._tracker()
        from agent_utilities.harness.citation_tracker import Citation

        citations = [
            Citation(source_id="d1", citation_type="kg_node", raw_text="[KG:d1]"),
            Citation(source_id="d2", citation_type="kg_node", raw_text="[KG:d2]"),
        ]
        report = tracker.evaluate_citations(citations, {"d1", "d2"})
        assert report.precision == 1.0

    def test_evaluate_precision_with_hallucination(self):
        tracker = self._tracker()
        from agent_utilities.harness.citation_tracker import Citation

        citations = [
            Citation(source_id="d1", citation_type="kg_node", raw_text="x"),
            Citation(source_id="d3", citation_type="kg_node", raw_text="x"),
        ]
        report = tracker.evaluate_citations(citations, {"d1", "d2"})
        assert report.precision == 0.5
        assert "d3" in report.hallucinated_citations

    def test_evaluate_recall(self):
        tracker = self._tracker()
        from agent_utilities.harness.citation_tracker import Citation

        citations = [
            Citation(source_id="d1", citation_type="kg_node", raw_text="x"),
        ]
        report = tracker.evaluate_citations(citations, {"d1", "d2"})
        assert report.recall == 0.5
        assert "d2" in report.uncited_evidence

    def test_evaluate_empty_citations(self):
        tracker = self._tracker()
        report = tracker.evaluate_citations([], {"d1"})
        assert report.total_citations == 0
        assert report.precision == 0.0

    def test_deduplication(self):
        tracker = self._tracker()
        text = "[KG:node-1] and again [KG:node-1]"
        citations = tracker.extract_citations(text)
        assert len(citations) == 1
