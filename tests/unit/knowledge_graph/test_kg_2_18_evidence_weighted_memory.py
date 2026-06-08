"""CONCEPT:KG-2.18 — Evidence-Weighted Memory.

Covers the Bayesian trust score, recall→usage telemetry + usage rate, the generation lineage
record, and trust persistence to a fake backend.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.retrieval.retrieval_quality import (
    TRUST_PRIOR,
    UsageTelemetry,
    bayesian_trust,
    build_lineage,
)


@pytest.mark.concept(id="KG-2.18")
def test_bayesian_trust_prior_and_movement():
    assert bayesian_trust(0, 0) == TRUST_PRIOR  # unseen → prior
    high = bayesian_trust(9, 10)
    low = bayesian_trust(0, 10)
    assert high > TRUST_PRIOR > low
    # Clamps helpful to total.
    assert 0.0 <= bayesian_trust(99, 1) <= 1.0


@pytest.mark.concept(id="KG-2.18")
def test_usage_telemetry_records_and_rates():
    t = UsageTelemetry()
    t.record_recall(["a", "b", "c"])
    t.record_recall(["a"])  # a recalled twice
    t.record_usage(["a"])  # only a used
    assert t.trust("a") > t.trust("b")  # used > merely-recalled
    assert t.usage_rate() == pytest.approx(
        1 / 3, abs=1e-3
    )  # 1 of 3 distinct nodes used
    s = t.summary()
    assert s["recalled_nodes"] == 3 and s["used_nodes"] == 1


@pytest.mark.concept(id="KG-2.18")
def test_build_lineage_stable_hash():
    a = build_lineage("q1", ["n2", "n1"], used_ids=["n1"], model="m")
    b = build_lineage("q1", ["n1", "n2"], used_ids=["n1"], model="m")
    assert a.context_hash == b.context_hash  # order-independent
    assert a.retrieved_ids == ["n2", "n1"] and a.used_ids == ["n1"]
    c = build_lineage("q2", ["n1", "n2"])
    assert c.context_hash != a.context_hash  # query change → new hash


class _FakeBackend:
    def __init__(self):
        self.writes = []

    def execute(self, q, params=None):
        self.writes.append((q, params))
        return []


class _FakeEngine:
    def __init__(self):
        self.backend = _FakeBackend()


@pytest.mark.concept(id="KG-2.18")
def test_flush_to_engine_persists_trust_score():
    t = UsageTelemetry()
    t.record_recall(["a", "b"])
    t.record_usage(["a"])
    eng = _FakeEngine()
    written = t.flush_to_engine(eng)
    assert written == 2
    assert all("trust_score" in w[0] for w in eng.backend.writes)


@pytest.mark.concept(id="KG-2.18")
def test_flush_no_backend_is_noop():
    t = UsageTelemetry()
    t.record_recall(["a"])

    class _NoBackend:
        backend = None

    assert t.flush_to_engine(_NoBackend()) == 0
