"""Tests for CONCEPT:KG-2.8 — Retrieval Quality Gate & CONCEPT:KG-2.9 — Context Provenance."""

import pytest
import time

from agent_utilities.knowledge_graph.retrieval_quality import (
    ContextProvenanceRecord,
    RetrievalFailureMode,
    RetrievalQualityGate,
    RetrievalQualityReport,
)


@pytest.fixture
def mock_engine():
    """Minimal mock engine for quality gate tests."""
    import networkx as nx

    class _MockEngine:
        def __init__(self):
            self.graph = nx.MultiDiGraph()
            self.backend = None

    return _MockEngine()


@pytest.fixture
def gate(mock_engine):
    return RetrievalQualityGate(mock_engine, min_relevance_threshold=0.6)


# ── Quality Report Computation ─────────────────────────────────────────


class TestQualityReport:
    """Tests for assess_quality() metric computation."""

    def test_empty_results_returns_low_relevance_failure(self, gate):
        report = gate.assess_quality([], query="test query")
        assert not report.gate_passed
        assert RetrievalFailureMode.LOW_RELEVANCE_TOPK in report.failure_modes_detected
        assert report.total_candidates == 0
        assert report.composite_quality == 0.0

    def test_all_high_scores_passes_gate(self, gate):
        results = [
            {"id": "n1", "_score": 0.9},
            {"id": "n2", "_score": 0.85},
            {"id": "n3", "_score": 0.8},
        ]
        report = gate.assess_quality(results, query="relevant query")
        assert report.gate_passed
        assert report.above_threshold == 3
        assert report.context_precision == 1.0
        assert report.mean_reciprocal_rank == 1.0
        assert report.composite_quality > 0.7

    def test_all_low_scores_fails_gate(self, gate):
        results = [
            {"id": "n1", "_score": 0.2},
            {"id": "n2", "_score": 0.15},
            {"id": "n3", "_score": 0.1},
        ]
        report = gate.assess_quality(results, query="irrelevant query")
        assert not report.gate_passed
        assert RetrievalFailureMode.LOW_RELEVANCE_TOPK in report.failure_modes_detected
        assert report.above_threshold == 0

    def test_mixed_scores_computes_precision(self, gate):
        results = [
            {"id": "n1", "_score": 0.9},
            {"id": "n2", "_score": 0.3},
            {"id": "n3", "_score": 0.1},
        ]
        report = gate.assess_quality(results)
        assert report.above_threshold == 1
        assert report.context_precision == pytest.approx(1 / 3, abs=0.01)
        assert report.mean_reciprocal_rank == 1.0  # First result is above threshold


# ── Failure Mode Detection ─────────────────────────────────────────────


class TestFailureModes:
    """Tests for the 5-mode failure taxonomy detection."""

    def test_drift_detected(self, gate):
        """Drift: top result is good but the rest are very low."""
        results = [
            {"id": "n1", "_score": 0.8},
            {"id": "n2", "_score": 0.1},
            {"id": "n3", "_score": 0.05},
            {"id": "n4", "_score": 0.1},
        ]
        report = gate.assess_quality(results)
        assert RetrievalFailureMode.DRIFT in report.failure_modes_detected

    def test_stale_index_detected(self, gate):
        """Stale index: majority of results have old timestamps."""
        old_ts = "2025-01-01T00:00:00Z"
        results = [
            {"id": f"n{i}", "_score": 0.7, "timestamp": old_ts}
            for i in range(10)
        ]
        report = gate.assess_quality(results)
        assert RetrievalFailureMode.STALE_INDEX in report.failure_modes_detected

    def test_context_truncation_detected(self, gate):
        """Truncation: many results above threshold (>80%, >10 results)."""
        results = [
            {"id": f"n{i}", "_score": 0.8} for i in range(15)
        ]
        report = gate.assess_quality(results)
        assert RetrievalFailureMode.CONTEXT_TRUNCATION in report.failure_modes_detected

    def test_inter_agent_propagation_detected(self, gate):
        """Inter-agent: upstream provenance shows low quality."""
        upstream = [
            ContextProvenanceRecord(
                source_agent="upstream_agent",
                retrieval_quality_score=0.2,
                failure_modes=[RetrievalFailureMode.LOW_RELEVANCE_TOPK],
            )
        ]
        results = [{"id": "n1", "_score": 0.7}]
        report = gate.assess_quality(results, upstream_provenance=upstream)
        assert RetrievalFailureMode.INTER_AGENT_PROPAGATION in report.failure_modes_detected


# ── Gate Filtering ─────────────────────────────────────────────────────


class TestGateFiltering:
    """Tests for gate_results() filtering behavior."""

    def test_gate_filters_below_threshold(self, gate):
        results = [
            {"id": "n1", "_score": 0.9},
            {"id": "n2", "_score": 0.3},
        ]
        filtered, report = gate.gate_results(results)
        assert len(filtered) == 1
        assert filtered[0]["id"] == "n1"
        assert report.gate_passed

    def test_gate_returns_all_on_failure(self, gate):
        """When gate fails, returns empty list."""
        results = [
            {"id": "n1", "_score": 0.1},
            {"id": "n2", "_score": 0.05},
        ]
        filtered, report = gate.gate_results(results)
        assert filtered == []
        assert not report.gate_passed

    def test_gate_disabled_passes_everything(self, mock_engine, monkeypatch):
        """When gate is disabled, all results pass through."""
        import agent_utilities.knowledge_graph.retrieval_quality as rq_module

        monkeypatch.setattr(rq_module, "_GATE_ENABLED", False)
        gate = RetrievalQualityGate(mock_engine)
        results = [{"id": "n1", "_score": 0.1}]
        filtered, report = gate.gate_results(results)
        assert len(filtered) == 1
        assert report.gate_passed


# ── Temporal Freshness ─────────────────────────────────────────────────


class TestTemporalFreshness:
    """Tests for Ebbinghaus-style temporal freshness scoring."""

    def test_fresh_node_score_is_one(self, gate):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        assert gate.temporal_freshness_score({"timestamp": now}) == pytest.approx(
            1.0, abs=0.1
        )

    def test_no_timestamp_assumes_fresh(self, gate):
        assert gate.temporal_freshness_score({}) == 1.0

    def test_old_node_decays(self, gate):
        old_ts = "2024-01-01T00:00:00Z"
        score = gate.temporal_freshness_score({"timestamp": old_ts})
        assert score < 0.5  # Should be significantly decayed


# ── Context Provenance ─────────────────────────────────────────────────


class TestContextProvenance:
    """Tests for CONCEPT:KG-2.9 provenance tracking."""

    def test_create_provenance_record(self, gate):
        report = RetrievalQualityReport(
            composite_quality=0.85,
            failure_modes_detected=[],
            total_candidates=10,
            mean_relevance_score=0.75,
        )
        record = gate.create_provenance_record("agent_1", report)
        assert record.source_agent == "agent_1"
        assert record.retrieval_quality_score == 0.85
        assert record.mean_relevance == 0.75
        assert record.failure_modes == []

    def test_provenance_record_serializes(self):
        record = ContextProvenanceRecord(
            source_agent="test",
            retrieval_quality_score=0.5,
            failure_modes=[RetrievalFailureMode.DRIFT],
        )
        data = record.model_dump()
        assert data["source_agent"] == "test"
        assert "drift" in data["failure_modes"]

    def test_report_latency_tracking(self, gate):
        results = [{"id": "n1", "_score": 0.9}]
        report = gate.assess_quality(results)
        assert report.latency_ms >= 0.0
