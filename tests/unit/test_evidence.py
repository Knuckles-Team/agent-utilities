"""The unified Evidence resource (CONCEPT:AU-KG.evolution.unified-evidence-resource, lane 7.1).

@pytest.mark.concept("AU-KG.evolution.unified-evidence-resource")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.research.candidate_insight import CandidateInsight
from agent_utilities.knowledge_graph.research.evidence import (
    Evidence,
    EvidenceChannel,
    EvidenceOutcome,
    candidates_from_evidence,
    evidence_lineage,
    from_candidate_insight,
    from_health_anomaly,
    from_optimization_result,
    from_outcome_evaluation,
    from_process_signal,
    gather_evidence,
    record_evidence,
)

pytestmark = pytest.mark.concept("AU-KG.evolution.unified-evidence-resource")


class _RecordingEngine:
    """Captures add_node/add_edge calls; optionally serves canned query_cypher rows."""

    def __init__(self, rows: dict[str, list[dict]] | None = None) -> None:
        self.nodes: dict[str, dict] = {}
        self.add_node_calls: list[tuple[str, str, dict]] = []
        self.edges: list[tuple[str, str, str]] = []
        self._rows = rows or {}

    def add_node(self, node_id, label, properties=None):
        properties = properties or {}
        self.nodes[node_id] = properties
        self.add_node_calls.append((node_id, label, properties))

    def add_edge(self, source, target, rel_type, **kwargs):
        self.edges.append((source, target, rel_type))

    def query_cypher(self, query, params=None):
        if "OutcomeEvaluation" in query:
            return self._rows.get("execution_trace", [])
        if "HealthAnomaly" in query:
            return self._rows.get("graph_health", [])
        if "EvolutionEvidence" in query and "Claim" not in query:
            return self._rows.get("evidence_lookup", [])
        # Checked BEFORE the plain "Claim"+"DERIVED_FROM" claims-query match below:
        # "DERIVED_FROM_RESEARCH" also contains the substring "DERIVED_FROM", so the
        # more specific proposals-query markers must win first.
        if "DERIVED_FROM_RESEARCH" in query or "SATISFIED_BY" in query:
            return self._rows.get("proposals", [])
        if "Claim" in query and "DERIVED_FROM" in query:
            return self._rows.get("claims", [])
        if "CapabilityRatchetResult" in query:
            return self._rows.get("capability_ratchet", [])
        return []


# ---------------------------------------------------------------------------
# Evidence core behavior
# ---------------------------------------------------------------------------


def test_evidence_id_is_stable_content_addressed():
    e1 = Evidence(
        channel=EvidenceChannel.EXECUTION_TRACE,
        subject_id="trace-1",
        outcome=EvidenceOutcome.FAILURE,
        signal=0.0,
        confidence=1.0,
        occurred_at="2026-01-01T00:00:00Z",
    )
    e2 = Evidence(
        channel=EvidenceChannel.EXECUTION_TRACE,
        subject_id="trace-1",
        outcome=EvidenceOutcome.FAILURE,
        signal=0.0,
        confidence=1.0,
        occurred_at="2026-01-01T00:00:00Z",
    )
    assert e1.evidence_id == e2.evidence_id


def test_evidence_id_differs_on_different_subject():
    e1 = Evidence(
        channel=EvidenceChannel.EXECUTION_TRACE,
        subject_id="trace-1",
        outcome=EvidenceOutcome.FAILURE,
        signal=0.0,
        confidence=1.0,
        occurred_at="t",
    )
    e2 = Evidence(
        channel=EvidenceChannel.EXECUTION_TRACE,
        subject_id="trace-2",
        outcome=EvidenceOutcome.FAILURE,
        signal=0.0,
        confidence=1.0,
        occurred_at="t",
    )
    assert e1.evidence_id != e2.evidence_id


def test_signal_and_confidence_are_clamped_into_0_1():
    ev = Evidence(
        channel=EvidenceChannel.GRAPH_HEALTH,
        subject_id="s",
        outcome=EvidenceOutcome.ANOMALOUS,
        signal=5.0,
        confidence=-3.0,
    )
    assert ev.signal == 1.0
    assert ev.confidence == 0.0


def test_empty_subject_id_rejected():
    with pytest.raises(ValueError):
        Evidence(
            channel=EvidenceChannel.GRAPH_HEALTH,
            subject_id="",
            outcome=EvidenceOutcome.ANOMALOUS,
            signal=0.5,
            confidence=0.5,
        )


def test_channel_and_outcome_accept_plain_strings():
    ev = Evidence(
        channel="graph_health",
        subject_id="s",
        outcome="anomalous",
        signal=0.5,
        confidence=0.5,
    )
    assert ev.channel is EvidenceChannel.GRAPH_HEALTH
    assert ev.outcome is EvidenceOutcome.ANOMALOUS


# ---------------------------------------------------------------------------
# to_candidate_insight — only claim-worthy outcomes route into governance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outcome", [EvidenceOutcome.FAILURE, EvidenceOutcome.DEGRADED, EvidenceOutcome.ANOMALOUS]
)
def test_claim_worthy_outcomes_produce_a_candidate_insight(outcome):
    ev = Evidence(
        channel=EvidenceChannel.GRAPH_HEALTH,
        subject_id="entity-1",
        outcome=outcome,
        signal=0.8,
        confidence=0.8,
    )
    cand = ev.to_candidate_insight()
    assert cand is not None
    assert isinstance(cand, CandidateInsight)
    assert cand.finding_type == "EvidenceSignal"
    assert cand.confidence == 0.8


@pytest.mark.parametrize("outcome", [EvidenceOutcome.SUCCESS, EvidenceOutcome.PROPOSED])
def test_non_claim_worthy_outcomes_produce_no_candidate(outcome):
    ev = Evidence(
        channel=EvidenceChannel.GRAPH_HEALTH,
        subject_id="entity-1",
        outcome=outcome,
        signal=0.8,
        confidence=0.8,
    )
    assert ev.to_candidate_insight() is None


def test_candidates_from_evidence_only_returns_claim_worthy():
    evs = [
        Evidence(
            channel=EvidenceChannel.EXECUTION_TRACE,
            subject_id="t1",
            outcome=EvidenceOutcome.SUCCESS,
            signal=1.0,
            confidence=1.0,
        ),
        Evidence(
            channel=EvidenceChannel.EXECUTION_TRACE,
            subject_id="t2",
            outcome=EvidenceOutcome.FAILURE,
            signal=0.0,
            confidence=1.0,
        ),
    ]
    cands = candidates_from_evidence(evs)
    assert len(cands) == 1
    assert cands[0].finding_id == evs[1].evidence_id


# ---------------------------------------------------------------------------
# Per-channel adapters — real native shapes, never fabricated
# ---------------------------------------------------------------------------


def test_from_outcome_evaluation_degraded_is_never_success():
    props = {
        "id": "oe-1",
        "trace_id": "trace-1",
        "status": "degraded",
        "success": False,
        "reward": 0.25,
        "timestamp": "2026-01-01T00:00:00Z",
    }
    ev = from_outcome_evaluation(props)
    assert ev.channel is EvidenceChannel.EXECUTION_TRACE
    assert ev.outcome is EvidenceOutcome.DEGRADED
    assert ev.signal == pytest.approx(0.25)
    assert ev.confidence == 1.0  # a directly-observed status, maximal confidence
    assert ev.source_node_id == "oe-1"


def test_from_outcome_evaluation_success_is_success():
    props = {
        "id": "oe-2",
        "trace_id": "trace-2",
        "status": "completed",
        "success": True,
        "reward": 1.0,
        "timestamp": "t",
    }
    ev = from_outcome_evaluation(props)
    assert ev.outcome is EvidenceOutcome.SUCCESS


def test_from_optimization_result_error_status_is_failure():
    ev = from_optimization_result("extraction", {"status": "error", "error_code": "x"})
    assert ev.channel is EvidenceChannel.OPTIMIZATION_SIGNAL
    assert ev.outcome is EvidenceOutcome.FAILURE
    assert ev.signal == 0.0


def test_from_optimization_result_optimized_status_is_success():
    ev = from_optimization_result("routing", {"status": "optimized"})
    assert ev.outcome is EvidenceOutcome.SUCCESS


def test_from_optimization_result_no_data_is_proposed_not_fabricated():
    ev = from_optimization_result("concept_match", {"status": "no_data"})
    assert ev.outcome is EvidenceOutcome.PROPOSED
    assert ev.signal == 0.0


def test_from_health_anomaly_confidence_matches_saturated_zscore():
    props = {
        "id": "ha-1",
        "entity": "capability:foo",
        "signal": "latency",
        "kind": "spike",
        "zscore": 10.0,  # saturates at |z|/5 clamped to 1.0
        "observedAt": "t",
    }
    ev = from_health_anomaly(props)
    assert ev.channel is EvidenceChannel.GRAPH_HEALTH
    assert ev.outcome is EvidenceOutcome.ANOMALOUS
    assert ev.confidence == 1.0
    assert ev.signal == 1.0


def test_from_candidate_insight_denied_is_failure():
    cand = CandidateInsight(
        finding_type="Anomaly", finding_id="f1", statement="s", confidence=0.9
    )
    ev = from_candidate_insight(cand, governance_valid=True, action_decision="deny")
    assert ev.channel is EvidenceChannel.RESEARCH_FINDING
    assert ev.outcome is EvidenceOutcome.FAILURE


def test_from_candidate_insight_promoted_is_success():
    cand = CandidateInsight(
        finding_type="Anomaly", finding_id="f2", statement="s", confidence=0.9
    )
    ev = from_candidate_insight(
        cand, governance_valid=True, action_decision="allow", promoted=True
    )
    assert ev.outcome is EvidenceOutcome.SUCCESS


def test_from_candidate_insight_unpromoted_but_not_denied_is_proposed():
    cand = CandidateInsight(
        finding_type="Anomaly", finding_id="f3", statement="s", confidence=0.9
    )
    ev = from_candidate_insight(cand, governance_valid=True, action_decision="allow")
    assert ev.outcome is EvidenceOutcome.PROPOSED


def test_from_process_signal_is_always_proposed_never_fabricated_signal():
    ev = from_process_signal(
        {
            "mode": "ocel_2.0",
            "tenant": "t1",
            "content_hash": "abc",
            "idempotency_key": "idem-1",
            "node_count": 42,
            "relationship_count": 7,
        }
    )
    assert ev.channel is EvidenceChannel.PROCESS_SIGNAL
    assert ev.outcome is EvidenceOutcome.PROPOSED
    assert ev.signal == 0.0
    assert ev.subject_id == "idem-1"


# ---------------------------------------------------------------------------
# record_evidence / gather_evidence — dedup + best-effort writeback
# ---------------------------------------------------------------------------


def test_record_evidence_upserts_and_links_derived_from():
    engine = _RecordingEngine()
    ev = Evidence(
        channel=EvidenceChannel.GRAPH_HEALTH,
        subject_id="entity-1",
        outcome=EvidenceOutcome.ANOMALOUS,
        signal=0.5,
        confidence=0.5,
        source_node_id="ha-1",
    )
    node_id = record_evidence(engine, ev)
    assert node_id == ev.evidence_id
    assert node_id in engine.nodes
    assert (node_id, "ha-1", "DERIVED_FROM") in engine.edges


def test_record_evidence_same_evidence_twice_upserts_not_duplicates():
    engine = _RecordingEngine()
    ev = Evidence(
        channel=EvidenceChannel.GRAPH_HEALTH,
        subject_id="entity-1",
        outcome=EvidenceOutcome.ANOMALOUS,
        signal=0.5,
        confidence=0.5,
        occurred_at="fixed-ts",
    )
    id1 = record_evidence(engine, ev)
    id2 = record_evidence(engine, ev)
    assert id1 == id2
    assert len(engine.add_node_calls) == 2  # two upserts, same node id
    assert len(engine.nodes) == 1  # deduplicated: one logical node


def test_record_evidence_returns_none_on_engine_failure():
    class _BrokenEngine:
        def add_node(self, *a, **k):
            raise RuntimeError("boom")

    ev = Evidence(
        channel=EvidenceChannel.GRAPH_HEALTH,
        subject_id="entity-1",
        outcome=EvidenceOutcome.ANOMALOUS,
        signal=0.5,
        confidence=0.5,
    )
    assert record_evidence(_BrokenEngine(), ev) is None


def test_record_evidence_none_engine_is_a_noop():
    ev = Evidence(
        channel=EvidenceChannel.GRAPH_HEALTH,
        subject_id="entity-1",
        outcome=EvidenceOutcome.ANOMALOUS,
        signal=0.5,
        confidence=0.5,
    )
    assert record_evidence(None, ev) is None


def test_gather_evidence_reads_both_queryable_channels_and_records_them():
    engine = _RecordingEngine(
        rows={
            "execution_trace": [
                {
                    "id": "oe-1",
                    "trace_id": "trace-1",
                    "status": "failed",
                    "success": False,
                    "reward": 0.0,
                    "timestamp": "t",
                }
            ],
            "graph_health": [
                {
                    "id": "ha-1",
                    "entity": "capability:foo",
                    "signal": "latency",
                    "kind": "spike",
                    "zscore": 6.0,
                    "observedAt": "t",
                }
            ],
        }
    )
    gathered = gather_evidence(engine)
    channels = {ev.channel for ev in gathered}
    assert channels == {EvidenceChannel.EXECUTION_TRACE, EvidenceChannel.GRAPH_HEALTH}
    # both got recorded (upserted) as :EvolutionEvidence nodes
    assert len(engine.nodes) == 2


def test_gather_evidence_respects_channel_filter():
    engine = _RecordingEngine(
        rows={
            "execution_trace": [
                {
                    "id": "oe-1",
                    "trace_id": "trace-1",
                    "status": "failed",
                    "success": False,
                    "reward": 0.0,
                    "timestamp": "t",
                }
            ],
            "graph_health": [
                {
                    "id": "ha-1",
                    "entity": "capability:foo",
                    "zscore": 6.0,
                    "observedAt": "t",
                }
            ],
        }
    )
    gathered = gather_evidence(engine, channels=(EvidenceChannel.EXECUTION_TRACE,))
    assert all(ev.channel is EvidenceChannel.EXECUTION_TRACE for ev in gathered)
    assert len(gathered) == 1


def test_gather_evidence_degrades_to_empty_on_query_failure():
    class _BrokenEngine:
        def query_cypher(self, *a, **k):
            raise RuntimeError("engine unavailable")

        def add_node(self, *a, **k):
            raise AssertionError("should never be called")

    assert gather_evidence(_BrokenEngine()) == []


# ---------------------------------------------------------------------------
# evidence_lineage (7.6) — negative results stay queryable, never deleted
# ---------------------------------------------------------------------------


def test_evidence_lineage_not_found_reports_found_false():
    engine = _RecordingEngine()
    result = evidence_lineage(engine, "evolution_evidence:missing")
    assert result["found"] is False
    assert result["chain"] == []


def test_evidence_lineage_walks_evidence_to_claim_to_proposal():
    engine = _RecordingEngine(
        rows={
            "evidence_lookup": [
                {
                    "id": "evolution_evidence:abc",
                    "channel": "graph_health",
                    "outcome": "anomalous",
                    "signal": 0.8,
                    "subject_id": "capability:foo",
                    "occurred_at": "t",
                }
            ],
            "claims": [
                {
                    "id": "claim:insight:xyz",
                    "claim_text": "capability foo is anomalous",
                    "status": "proposal",
                    "confidence": 0.8,
                    "is_verified": False,
                }
            ],
            "proposals": [
                {"id": "sdd_feature:foo", "status": "open"},
            ],
        }
    )
    result = evidence_lineage(engine, "evolution_evidence:abc")
    assert result["found"] is True
    stages = [row["stage"] for row in result["chain"]]
    assert stages == ["evidence", "claim_proposal", "proposal"]


def test_evidence_lineage_walks_one_more_hop_to_the_capability_ratchet_verdict():
    """(D-71-5) the chain now also surfaces a recorded ``CapabilityRatchetResult``
    for a proposal — the test/rollout half of trace -> evaluation -> proposal ->
    test -> rollout, not just the pre-promotion proposal status."""
    engine = _RecordingEngine(
        rows={
            "evidence_lookup": [
                {
                    "id": "evolution_evidence:abc",
                    "channel": "graph_health",
                    "outcome": "anomalous",
                    "signal": 0.8,
                    "subject_id": "capability:foo",
                    "occurred_at": "t",
                }
            ],
            "claims": [
                {
                    "id": "claim:insight:xyz",
                    "claim_text": "capability foo is anomalous",
                    "status": "proposal",
                    "confidence": 0.8,
                    "is_verified": False,
                }
            ],
            "proposals": [
                {"id": "sdd_feature:foo", "status": "implemented"},
            ],
            "capability_ratchet": [
                {
                    "id": "capability_ratchet:1",
                    "result": "pass",
                    "recommendation": "promote",
                    "recorded_at": "t2",
                },
            ],
        }
    )
    result = evidence_lineage(engine, "evolution_evidence:abc")
    stages = [row["stage"] for row in result["chain"]]
    assert stages == ["evidence", "claim_proposal", "proposal", "capability_ratchet"]
    ratchet_row = result["chain"][-1]
    assert ratchet_row["proposal_id"] == "sdd_feature:foo"
    assert ratchet_row["result"] == "pass"


def test_evidence_lineage_preserves_rejected_claims_not_deleted():
    """A denied/rejected claim keeps its real ``status`` (e.g. "rejected") in
    the chain — this function never filters negative results out."""
    engine = _RecordingEngine(
        rows={
            "evidence_lookup": [
                {
                    "id": "evolution_evidence:abc",
                    "channel": "graph_health",
                    "outcome": "anomalous",
                    "signal": 0.8,
                    "subject_id": "capability:foo",
                    "occurred_at": "t",
                }
            ],
            "claims": [
                {
                    "id": "claim:insight:xyz",
                    "claim_text": "capability foo is anomalous",
                    "status": "rejected",
                    "confidence": 0.8,
                    "is_verified": False,
                }
            ],
        }
    )
    result = evidence_lineage(engine, "evolution_evidence:abc")
    claim_rows = [r for r in result["chain"] if r["stage"] == "claim_proposal"]
    assert len(claim_rows) == 1
    assert claim_rows[0]["status"] == "rejected"
