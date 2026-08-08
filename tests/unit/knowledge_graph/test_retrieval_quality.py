"""Tests for CONCEPT:AU-KG.research.research-pipeline-runner — Retrieval Quality Gate & CONCEPT:AU-KG.research.research-pipeline-runner — Context Provenance."""

import time

import pytest

from agent_utilities.knowledge_graph.retrieval.retrieval_quality import (
    ContextProvenanceRecord,
    RetrievalFailureMode,
    RetrievalQualityGate,
    RetrievalQualityReport,
)


@pytest.fixture
def mock_engine():
    """Minimal mock engine for quality gate tests."""
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    class _MockEngine:
        def __init__(self):
            self.graph = GraphComputeEngine(backend_type="rust")
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
            {"id": f"n{i}", "_score": 0.7, "timestamp": old_ts} for i in range(10)
        ]
        report = gate.assess_quality(results)
        assert RetrievalFailureMode.STALE_INDEX in report.failure_modes_detected

    def test_context_truncation_detected(self, gate):
        """Truncation: many results above threshold (>80%, >10 results)."""
        results = [{"id": f"n{i}", "_score": 0.8} for i in range(15)]
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
        assert (
            RetrievalFailureMode.INTER_AGENT_PROPAGATION
            in report.failure_modes_detected
        )


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

    def test_keyword_discover_result_graded_against_its_own_threshold(self, gate):
        """D-EMB-6/D-GS27-6: the engine-native ``discover()`` keyword-overlap
        score is NOT a cosine similarity and must not be graded against the
        vector-calibrated 0.6 default — a real single-keyword-out-of-many match
        (e.g. composite ~0.02-0.1) previously always failed LOW_RELEVANCE_TOPK.
        """
        results = [
            {"id": "n1", "_score": 0.15, "_fallback": "keyword_discover"},
            {"id": "n2", "_score": 0.05, "_fallback": "keyword_discover"},
        ]
        filtered, report = gate.gate_results(results)
        assert [r["id"] for r in filtered] == ["n1"]
        assert report.gate_passed

    def test_keyword_discover_threshold_is_below_the_vector_default(self, gate):
        assert gate._keyword_discover_threshold < gate._threshold
        assert gate._result_threshold({"_fallback": "keyword_discover"}) == (
            gate._keyword_discover_threshold
        )

    def test_gate_disabled_passes_everything(self, mock_engine, monkeypatch):
        """When gate is disabled, all results pass through."""
        import agent_utilities.knowledge_graph.retrieval.retrieval_quality as rq_module

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
    """Tests for CONCEPT:AU-KG.research.research-pipeline-runner provenance tracking."""

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


# ── D-EGD-5: index-empty vs results-poor distinction ────────────────────


class _FakeBackendWithCounts:
    """Minimal backend stub answering the gate's two population-sample
    Cypher queries deterministically."""

    def __init__(self, total: int, embedded: int) -> None:
        self._total = total
        self._embedded = embedded
        self.calls = 0

    def execute(self, query: str, *_args, **_kwargs):
        self.calls += 1
        if "n.embedding IS NOT NULL" in query:
            return [{"c": self._embedded}]
        return [{"c": self._total}]


class TestSparseIndexDetection:
    """A failing retrieval against a near-empty index must be tagged
    SPARSE_INDEX (an ingestion problem), distinct from LOW_RELEVANCE_TOPK
    against a populated index (a query/coverage problem) — conflating the two
    cost the program real time reading composite=0.05 as a retrieval bug when
    the index was 0.5% populated (D-PERF-5/D-EGD-5)."""

    def test_empty_results_against_sparse_index_tags_sparse_index(self, mock_engine):
        mock_engine.backend = _FakeBackendWithCounts(total=26680, embedded=136)
        gate = RetrievalQualityGate(mock_engine, min_relevance_threshold=0.6)

        report = gate.assess_quality([], query="test query")

        assert not report.gate_passed
        assert RetrievalFailureMode.LOW_RELEVANCE_TOPK in report.failure_modes_detected
        assert RetrievalFailureMode.SPARSE_INDEX in report.failure_modes_detected
        assert report.index_population_ratio == pytest.approx(136 / 26680)

    def test_low_scores_against_populated_index_does_not_tag_sparse_index(
        self, mock_engine
    ):
        """The same LOW_RELEVANCE_TOPK failure, but the index itself is
        healthy (99% populated) — this is a genuine query/coverage miss, not
        an ingestion gap, and must NOT be mislabeled SPARSE_INDEX."""
        mock_engine.backend = _FakeBackendWithCounts(total=1000, embedded=990)
        gate = RetrievalQualityGate(mock_engine, min_relevance_threshold=0.6)

        results = [{"id": "n1", "_score": 0.1}, {"id": "n2", "_score": 0.05}]
        report = gate.assess_quality(results, query="test query")

        assert not report.gate_passed
        assert RetrievalFailureMode.LOW_RELEVANCE_TOPK in report.failure_modes_detected
        assert RetrievalFailureMode.SPARSE_INDEX not in report.failure_modes_detected
        assert report.index_population_ratio == pytest.approx(0.99)

    def test_passing_retrieval_never_samples_population(self, mock_engine):
        """The population sample only runs on the failing path — a healthy
        retrieval must not pay for an extra engine round-trip."""
        backend = _FakeBackendWithCounts(total=1000, embedded=10)
        mock_engine.backend = backend
        gate = RetrievalQualityGate(mock_engine, min_relevance_threshold=0.6)

        results = [{"id": "n1", "_score": 0.9}]
        report = gate.assess_quality(results, query="test query")

        assert report.gate_passed
        assert backend.calls == 0
        assert report.index_population_ratio is None

    def test_population_sample_is_cached_within_ttl(self, mock_engine):
        """A burst of failing queries samples the engine at most once per TTL
        window, not once per query — the engine is already contended."""
        backend = _FakeBackendWithCounts(total=26680, embedded=136)
        mock_engine.backend = backend
        gate = RetrievalQualityGate(mock_engine, min_relevance_threshold=0.6)

        for _ in range(5):
            gate.assess_quality([], query="test query")

        # One SAMPLE (total + embedded count queries = 2 backend.execute
        # calls) for the whole 5-query burst, not 2 per query.
        assert backend.calls == 2

    def test_population_cache_does_not_leak_across_gate_instances(self, mock_engine):
        """Regression guard: the cache must be keyed per-gate-instance, not by
        id(engine) — id() reuse after garbage collection could otherwise leak
        one engine's ratio into an unrelated gate's report."""
        sparse_backend = _FakeBackendWithCounts(total=26680, embedded=136)
        mock_engine.backend = sparse_backend
        sparse_gate = RetrievalQualityGate(mock_engine, min_relevance_threshold=0.6)
        sparse_report = sparse_gate.assess_quality([], query="q1")
        assert RetrievalFailureMode.SPARSE_INDEX in sparse_report.failure_modes_detected

        class _HealthyEngine:
            pass

        healthy_engine = _HealthyEngine()
        healthy_engine.backend = _FakeBackendWithCounts(total=1000, embedded=990)
        healthy_gate = RetrievalQualityGate(healthy_engine, min_relevance_threshold=0.6)
        healthy_report = healthy_gate.assess_quality([], query="q2")

        assert (
            RetrievalFailureMode.SPARSE_INDEX
            not in healthy_report.failure_modes_detected
        )
        assert healthy_report.index_population_ratio == pytest.approx(0.99)
