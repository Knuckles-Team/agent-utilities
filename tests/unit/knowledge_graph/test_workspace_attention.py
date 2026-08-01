#!/usr/bin/python
from __future__ import annotations

"""Tests for CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Global Workspace Attention."""

import pytest

from agent_utilities.graph.workspace_attention import (
    Proposal,
    WorkspaceAttention,
    reset_workspace_attention_telemetry,
    workspace_attention_telemetry,
)


class FakeEngine:  # type: ignore
    """Minimal mock engine for GWT tests."""

    def __init__(self):
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )

        self.graph = GraphComputeEngine(backend_type="rust")
        self.backend = None

    def _upsert_node(self, label, node_id, props):
        self.last_upserted = (label, node_id, props)


class _LocalGraph:
    """A truly in-process graph (unlike the rust daemon, which shares state)."""

    def __init__(self):
        self._n: dict = {}

    def add_node(self, node_id, **attrs):
        self._n[node_id] = attrs

    def add_edge(self, src, dst, **attrs):  # links are irrelevant to the read path
        pass

    def nodes(self, data=False):
        return list(self._n.items()) if data else list(self._n)

    def __contains__(self, key):
        return key in self._n


class LocalEngine:
    """Isolated engine instance — no shared backend, for mismatch testing."""

    def __init__(self):
        self.graph = _LocalGraph()
        self.backend = None

    def _upsert_node(self, *a, **k):  # pragma: no cover
        pass


class TestProposalCollection:
    def test_collect_proposals_creates_scored_proposals(self):
        gwt = WorkspaceAttention(max_broadcast_slots=3)

        outputs = {
            "spec:gitlab": "Here are 5 GitLab projects...",
            "spec:jira": "Found 3 Jira tickets...",
            "spec:search": "No results found.",
        }

        proposals = gwt.collect_proposals(outputs, query="list all projects")

        assert len(proposals) == 3
        # All proposals should have composite scores
        for p in proposals:
            assert 0.0 <= p.composite_score <= 1.0

    def test_proposals_sorted_by_composite_score(self):
        gwt = WorkspaceAttention()

        outputs = {
            "spec:a": "Detailed relevant answer about projects and tasks",
            "spec:b": "Unrelated response about weather",
        }

        proposals = gwt.collect_proposals(outputs, query="list all projects")

        assert proposals[0].composite_score >= proposals[1].composite_score

    def test_empty_outputs_returns_empty(self):
        gwt = WorkspaceAttention()

        proposals = gwt.collect_proposals({}, query="test")
        assert proposals == []


class TestWinnerSelection:
    def test_select_winners_limits_to_max_slots(self):
        gwt = WorkspaceAttention(max_broadcast_slots=2)

        proposals = [
            Proposal(
                specialist_id=f"spec:{i}",
                output=f"Output {i}",
                composite_score=0.9 - i * 0.1,
            )
            for i in range(5)
        ]

        winners = gwt.select_winners(proposals)
        assert len(winners) == 2
        assert winners[0].composite_score > winners[1].composite_score

    def test_select_winners_returns_all_if_fewer_than_max(self):
        gwt = WorkspaceAttention(max_broadcast_slots=10)

        proposals = [
            Proposal(specialist_id="spec:1", output="Output", composite_score=0.9),
        ]

        winners = gwt.select_winners(proposals)
        assert len(winners) == 1


class TestKGBroadcast:
    def test_broadcast_creates_proposal_nodes(self):
        gwt = WorkspaceAttention()
        engine = FakeEngine()  # type: ignore

        winners = [
            Proposal(
                specialist_id="spec:gitlab",
                specialist_name="GitLab Agent",
                output="Found 5 projects",
                relevance_score=0.9,
                confidence_score=0.8,
                track_record_score=0.85,
                composite_score=0.87,
            ),
        ]

        node_ids = gwt.broadcast_to_kg(winners, engine, task_id="task:1")  # type: ignore[arg-type]

        assert len(node_ids) == 1
        # Check node was added to NetworkX
        assert node_ids[0] in engine.graph

    def test_broadcast_uses_engine_from_constructor(self):
        engine = FakeEngine()  # type: ignore
        gwt = WorkspaceAttention(engine)  # engine via constructor, not arg
        assert gwt.engine is engine
        winners = [
            Proposal(specialist_id="spec:a", output="x", composite_score=0.5),
        ]
        node_ids = gwt.broadcast_to_kg(winners)  # no engine arg → uses self.engine
        assert len(node_ids) == 1

    def test_broadcast_without_engine_is_noop(self):
        gwt = WorkspaceAttention()  # no engine anywhere
        node_ids = gwt.broadcast_to_kg(
            [Proposal(specialist_id="s", output="o", composite_score=0.4)]
        )
        assert node_ids == []


class TestAttentionScoreReadback:
    """The GWT loop: broadcast (write) → get_attention_score (read)."""

    def test_get_attention_score_reads_back_broadcast(self):
        engine = FakeEngine()  # type: ignore
        gwt = WorkspaceAttention(engine)
        winners = gwt.select_and_broadcast(
            {
                "spec:gitlab": "Here are 5 gitlab projects with details",
                "spec:weather": "It is sunny today, unrelated",
            },
            query="list gitlab projects",
        )
        assert winners  # at least one winner broadcast
        top = winners[0]
        score = gwt.get_attention_score(top.specialist_id)
        assert score is not None
        assert score == pytest.approx(top.composite_score)

    def test_get_attention_score_none_without_history(self):
        engine = FakeEngine()  # type: ignore
        gwt = WorkspaceAttention(engine)
        assert gwt.get_attention_score("spec:never-seen") is None

    def test_get_attention_score_none_without_engine(self):
        gwt = WorkspaceAttention()
        assert gwt.get_attention_score("spec:x") is None

    def test_select_and_broadcast_empty_outputs(self):
        engine = FakeEngine()  # type: ignore
        gwt = WorkspaceAttention(engine)
        assert gwt.select_and_broadcast({}, query="q") == []


class TestExecutorWiringRegression:
    """Locks the latent bug: executor imported a non-existent module and called a
    method that did not exist (both silently swallowed)."""

    def test_executor_imports_the_real_module(self):
        import inspect

        import agent_utilities.graph.executor as ex

        src = inspect.getsource(ex)
        # The dead import path must be gone…
        assert "knowledge_graph.workspace_attention" not in src
        # …replaced by the real sibling module.
        assert "from .workspace_attention import WorkspaceAttention" in src

    def test_get_attention_score_exists_on_class(self):
        # The method the executor calls must actually exist now.
        assert callable(getattr(WorkspaceAttention, "get_attention_score", None))


class TestGwtTelemetry:
    """Surface the write-but-never-read (engine-mismatch) failure mode."""

    def setup_method(self):
        reset_workspace_attention_telemetry()

    def teardown_method(self):
        reset_workspace_attention_telemetry()

    def test_healthy_loop_has_hits_and_no_mismatch(self):
        engine = LocalEngine()
        wa = WorkspaceAttention(engine)
        winners = wa.select_and_broadcast(
            {
                "spec:gitlab": "Here are 5 gitlab projects with details",
                "spec:weather": "sunny and unrelated",
            },
            query="list gitlab projects",
        )
        # Same engine for read → hits, no mismatch.
        assert wa.get_attention_score(winners[0].specialist_id) is not None
        t = workspace_attention_telemetry()
        assert t["broadcasts_written"] >= 1
        assert t["attention_hits"] >= 1
        assert t["suspected_engine_mismatch"] is False

    def test_engine_mismatch_is_detected(self):
        WorkspaceAttention(LocalEngine()).select_and_broadcast(
            {
                "spec:a": "relevant detailed answer about projects",
                "spec:b": "another relevant answer about tasks",
            },
            query="projects and tasks",
        )
        reader = WorkspaceAttention(LocalEngine())  # isolated, different engine
        # Reads against the wrong engine never resolve a broadcast.
        for _ in range(4):
            assert reader.get_attention_score("spec:a") is None
        t = workspace_attention_telemetry()
        assert t["broadcasts_written"] >= 1
        assert t["attention_hits"] == 0
        assert t["suspected_engine_mismatch"] is True

    def test_strict_mode_raises_on_mismatch(self, monkeypatch):
        import agent_utilities.graph.workspace_attention as wa_mod

        monkeypatch.setattr(wa_mod, "_STRICT", True)
        WorkspaceAttention(LocalEngine()).select_and_broadcast(
            {
                "spec:a": "relevant detailed answer about projects",
                "spec:b": "another relevant answer about tasks",
            },
            query="projects",
        )
        reader = WorkspaceAttention(LocalEngine())  # wrong engine
        with pytest.raises(AssertionError):
            for _ in range(4):
                reader.get_attention_score("spec:a")


class TestConfidenceExtraction:
    def test_extract_confidence_from_text(self):
        gwt = WorkspaceAttention()

        assert gwt._extract_confidence("Confidence: 0.85") == 0.85
        assert gwt._extract_confidence("I'm 90% sure about this") == 0.9
        assert gwt._extract_confidence("No confidence signal here") == 0.5

    def test_extract_confidence_clamped(self):
        gwt = WorkspaceAttention()

        # Values >1 should clamp to 1.0
        assert gwt._extract_confidence("Confidence: 1.5") == 1.0
        # Negative values don't match the regex pattern [\\d.]+, so they
        # correctly fall through to the default 0.5 (not parseable)
        assert gwt._extract_confidence("Confidence: -0.5") == 0.5


class TestRelevanceScoring:
    def test_keyword_relevance_fallback(self):
        gwt = WorkspaceAttention()

        # Without engine (no embeddings), uses keyword overlap
        score = gwt._score_relevance(
            output="The gitlab project list shows 5 items",
            query="list gitlab projects",
            query_embedding=None,
            engine=None,
        )
        assert score > 0.0

    def test_irrelevant_output_scores_low(self):
        gwt = WorkspaceAttention()

        score = gwt._score_relevance(
            output="The weather today is sunny and warm",
            query="list gitlab projects",
            query_embedding=None,
            engine=None,
        )
        # Should be lower than a relevant output
        relevant_score = gwt._score_relevance(
            output="Here are the gitlab projects with their details",
            query="list gitlab projects",
            query_embedding=None,
            engine=None,
        )
        assert relevant_score >= score


class TestTrackRecordScoring:
    def test_default_track_record(self):
        gwt = WorkspaceAttention()

        score = gwt._score_track_record("spec:unknown", memory_retriever=None)  # type: ignore[call-arg]
        assert score == 0.5  # Neutral default


class TestWeightConfiguration:
    def test_custom_weights(self):
        gwt = WorkspaceAttention(
            relevance_weight=0.7,
            track_record_weight=0.2,
            confidence_weight=0.1,
        )
        assert gwt.w_relevance == 0.7
        assert gwt.w_track_record == 0.2
        assert gwt.w_confidence == 0.1
