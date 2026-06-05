"""CONCEPT:KG-2.18 — live-path integration: recall telemetry is actually invoked by retrieval.

Reachable != invoked. This test exercises the EXISTING `plan_and_retrieve` hot path and asserts the
evidence-weighting telemetry is populated as a side effect — and that the usage half closes the loop.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import HybridRetriever


class _FakeReport:
    gate_passed = True


class _FakeBackend:
    def __init__(self):
        self.writes = []

    def execute(self, q, params=None):
        self.writes.append((q, params))
        return []


class _FakeEngine:
    def __init__(self):
        self.backend = _FakeBackend()


class _FakeRetriever:
    plan_and_retrieve = HybridRetriever.plan_and_retrieve
    usage_telemetry = HybridRetriever.usage_telemetry
    record_answer_usage = HybridRetriever.record_answer_usage

    def __init__(self, results):
        self._results = results
        self.engine = _FakeEngine()
        self._report = _FakeReport()

    @property
    def last_quality_report(self):
        return self._report

    def retrieve_hybrid(self, query, **kw):
        return list(self._results)


@pytest.mark.concept(id="KG-2.18")
def test_plan_and_retrieve_records_recall_on_live_path():
    r = _FakeRetriever([{"id": "n1", "_score": 0.9}, {"id": "n2", "_score": 0.8}])
    r.plan_and_retrieve("a real query", mode="standard")
    # The retrieval populated recall telemetry as a side effect (the integration that was missing).
    assert r.usage_telemetry._recalled.get("n1") == 1
    assert r.usage_telemetry._recalled.get("n2") == 1


@pytest.mark.concept(id="KG-2.18")
def test_record_answer_usage_closes_loop_and_persists_trust():
    r = _FakeRetriever([{"id": "n1", "_score": 0.9}, {"id": "n2", "_score": 0.8}])
    r.plan_and_retrieve("q", mode="standard")
    lineage = r.record_answer_usage(["n1"], query="q")
    # n1 was used → higher trust than the merely-recalled n2; trust persisted to the backend.
    assert r.usage_telemetry.trust("n1") > r.usage_telemetry.trust("n2")
    assert any("trust_score" in w[0] for w in r.engine.backend.writes)
    assert lineage["used_ids"] == ["n1"] and lineage["context_hash"]
