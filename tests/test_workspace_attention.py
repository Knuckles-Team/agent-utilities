#!/usr/bin/python
"""Tests for CONCEPT:ORCH-1.2 — Global Workspace Attention."""

from __future__ import annotations

from agent_utilities.graph.workspace_attention import (
    Proposal,
    WorkspaceAttention,
)


class FakeEngine:  # type: ignore
    """Minimal mock engine for GWT tests."""

    def __init__(self):
        import networkx as nx

        self.graph = nx.MultiDiGraph()
        self.backend = None

    def _upsert_node(self, label, node_id, props):
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

        score = gwt._score_track_record("spec:unknown", self_model=None)  # type: ignore[call-arg]
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
