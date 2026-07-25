"""CandidateInsight — mined finding → reviewable Claim (CONCEPT:AU-KG.evolution.
insight-engine-closed-loop, workstream C4).

@pytest.mark.concept("AU-KG.evolution.insight-engine-closed-loop")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.research.candidate_insight import (
    CONFIDENCE_FLOOR,
    CandidateInsight,
    _claim_evidence_ids,
    _claim_metadata,
    _is_ops_causal_claim,
    candidates_from_anomalies,
    candidates_from_association_rules,
    candidates_from_mine_discovery,
    candidates_from_predicted_edges,
)

pytestmark = pytest.mark.concept("AU-KG.evolution.insight-engine-closed-loop")


# ---------------------------------------------------------------------------
# CandidateInsight core behavior
# ---------------------------------------------------------------------------


def test_clears_floor_true_above_and_false_below():
    high = CandidateInsight(
        finding_type="AssociationRule",
        finding_id="f1",
        statement="X ⇒ Y",
        confidence=CONFIDENCE_FLOOR + 0.1,
    )
    low = CandidateInsight(
        finding_type="AssociationRule",
        finding_id="f2",
        statement="X ⇒ Y",
        confidence=CONFIDENCE_FLOOR - 0.1,
    )
    assert high.clears_floor is True
    assert low.clears_floor is False


def test_confidence_is_clamped_into_0_1():
    over = CandidateInsight(
        finding_type="Anomaly", finding_id="f1", statement="s", confidence=5.0
    )
    under = CandidateInsight(
        finding_type="Anomaly", finding_id="f2", statement="s", confidence=-3.0
    )
    assert over.confidence == 1.0
    assert under.confidence == 0.0


def test_unknown_finding_type_rejected():
    with pytest.raises(ValueError):
        CandidateInsight(
            finding_type="NotARealType",
            finding_id="f1",
            statement="s",
            confidence=0.9,
        )


def test_to_claim_node_is_never_pre_verified():
    """A mined finding is NEVER self-verifying — verification/promotion is the
    governance + action-policy gate's job, not the constructor's, regardless of
    how high the mined confidence is."""
    cand = CandidateInsight(
        finding_type="AssociationRule",
        finding_id="f1",
        statement="X ⇒ Y",
        confidence=1.0,
    )
    claim = cand.to_claim_node()
    assert claim.is_verified is False
    assert claim.confidence == 1.0
    assert claim.claim_text == "X ⇒ Y"
    assert claim.id == "claim:insight:f1"


def test_to_evidence_bundle_never_fabricates_confidence():
    cand = CandidateInsight(
        finding_type="Anomaly",
        finding_id="f1",
        statement="capability X is divergent",
        confidence=0.7,
        payload={"anomaly_score": 3.5},
    )
    bundle = cand.to_evidence_bundle()
    assert bundle.confidence == 0.7
    assert bundle.evidence_spans[0]["anomaly_score"] == 3.5
    assert bundle.claims[0]["text"] == "capability X is divergent"


def test_claim_id_is_content_addressed_and_stable():
    """Re-mining the identical finding must upsert the same Claim, not duplicate
    it — mirroring the idempotency discipline loop_controller uses elsewhere."""
    result = {
        "examples": [
            {
                "antecedent": ["concept:a"],
                "consequent": ["capability:z"],
                "confidence": 0.9,
                "lift": 1.2,
            }
        ]
    }
    first = candidates_from_association_rules(result)[0]
    second = candidates_from_association_rules(result)[0]
    assert first.claim_id == second.claim_id


# ---------------------------------------------------------------------------
# Per-finding-type extraction
# ---------------------------------------------------------------------------


def test_association_rule_confidence_passes_through_directly():
    result = {
        "examples": [
            {
                "antecedent": ["concept:cA", "concept:cB"],
                "consequent": ["capability:capZ"],
                "confidence": 0.92,
                "lift": 1.67,
            }
        ]
    }
    candidates = candidates_from_association_rules(result)
    assert len(candidates) == 1
    assert candidates[0].confidence == 0.92
    assert candidates[0].finding_type == "AssociationRule"
    assert candidates[0].clears_floor is True


def test_anomaly_confidence_is_a_saturating_transform_of_z_score():
    result = {
        "examples": [
            {"capability": "cap:weak", "covered_concepts": 1.0, "anomaly_score": 1.0},
            {"capability": "cap:strong", "covered_concepts": 0.0, "anomaly_score": 6.0},
        ]
    }
    candidates = {c.payload["capability"]: c for c in candidates_from_anomalies(result)}
    assert candidates["cap:weak"].confidence == pytest.approx(0.2)
    assert candidates["cap:weak"].clears_floor is False
    assert candidates["cap:strong"].confidence == 1.0  # saturated
    assert candidates["cap:strong"].clears_floor is True


def test_anomaly_without_capability_id_is_skipped():
    result = {"examples": [{"covered_concepts": 0.0, "anomaly_score": 9.0}]}
    assert candidates_from_anomalies(result) == []


def test_predicted_edge_uses_first_available_score_field():
    result = {
        "examples": [
            {"src": "concept:a", "dst": "concept:b", "score": 0.81},
            {"source": "concept:c", "target": "concept:d", "confidence": 0.5},
        ]
    }
    candidates = candidates_from_predicted_edges(result)
    assert candidates[0].confidence == 0.81
    assert candidates[1].confidence == 0.5


def test_predicted_edge_with_no_score_field_never_fabricates_confidence():
    """No calibrated score in the payload ⇒ confidence 0.0 (unverified), never a
    guessed mid-range default — mirrors EvidenceBundle's no-fabrication contract."""
    result = {"examples": [{"src": "concept:a", "dst": "concept:b"}]}
    candidates = candidates_from_predicted_edges(result)
    assert candidates[0].confidence == 0.0
    assert candidates[0].clears_floor is False


# ---------------------------------------------------------------------------
# Top-level fan-out
# ---------------------------------------------------------------------------


def test_candidates_from_mine_discovery_fans_out_all_three_kinds():
    mine_result = {
        "association_rules": {
            "examples": [
                {
                    "antecedent": ["concept:a"],
                    "consequent": ["capability:z"],
                    "confidence": 0.95,
                    "lift": 2.0,
                }
            ]
        },
        "anomalies": {
            "examples": [
                {"capability": "cap:x", "covered_concepts": 0.0, "anomaly_score": 6.0}
            ]
        },
        "predicted_edges": {
            "examples": [{"source": "concept:a", "target": "concept:c", "score": 0.9}]
        },
        "errors": [],
    }
    candidates = candidates_from_mine_discovery(mine_result)
    assert {c.finding_type for c in candidates} == {
        "AssociationRule",
        "Anomaly",
        "PredictedEdge",
    }
    assert all(c.clears_floor for c in candidates)


def test_candidates_from_mine_discovery_degrades_cleanly_on_missing_sections():
    assert candidates_from_mine_discovery(None) == []
    assert candidates_from_mine_discovery({}) == []
    assert candidates_from_mine_discovery({"errors": ["boom"]}) == []


# ---------------------------------------------------------------------------
# B17 bridge helpers — _claim_metadata / _is_ops_causal_claim /
# _claim_evidence_ids (CONCEPT:AU-KG.enrichment.ops-causal-graph /
# CONCEPT:AU-KG.enrichment.cross-layer-incident-correlation). Live here (not
# ops_causal_graph.py) specifically so recognizing an ops-causal Claim never
# pulls in formal_reasoning_core's numeric-kernel dependency.
# ---------------------------------------------------------------------------


def test_claim_metadata_accepts_a_nested_dict():
    assert _claim_metadata({"metadata": {"finding_type": "OpsCausalFinding"}}) == {
        "finding_type": "OpsCausalFinding"
    }


def test_claim_metadata_accepts_a_json_string():
    """Some write paths round-trip ``metadata`` as a JSON string (see
    ``knowledge_graph/core/engine.py``'s explicit ``json.dumps`` convention)
    rather than a nested map — this must be tolerated too."""
    assert _claim_metadata({"metadata": '{"finding_type": "OpsCausalFinding"}'}) == {
        "finding_type": "OpsCausalFinding"
    }


def test_claim_metadata_degrades_to_empty_dict_on_anything_else():
    assert _claim_metadata({}) == {}
    assert _claim_metadata({"metadata": None}) == {}
    assert _claim_metadata({"metadata": "not json"}) == {}
    assert _claim_metadata({"metadata": "[1, 2]"}) == {}  # valid JSON, not a dict
    assert _claim_metadata("not even a dict") == {}


def test_is_ops_causal_claim_true_for_the_opt_in_as_claim_shape():
    """``domain="ops_causal"`` — the ``as_claim=true`` persist path in
    ``ops_causal_tools.py``."""
    assert _is_ops_causal_claim({"domain": "ops_causal", "metadata": {}}) is True


def test_is_ops_causal_claim_true_for_the_governed_materialize_claims_shape():
    """``domain="mining_flywheel"`` (shared with every other mined-finding
    family) BUT ``metadata.finding_type == "OpsCausalFinding"`` — the default
    ``materialize_claims=true`` persist path via ``CandidateInsight.
    to_claim_node``/``candidates_from_ops_causal_root_cause``."""
    assert (
        _is_ops_causal_claim(
            {
                "domain": "mining_flywheel",
                "metadata": {"finding_type": "OpsCausalFinding"},
            }
        )
        is True
    )


def test_is_ops_causal_claim_false_for_other_mining_flywheel_families():
    """A same-``domain`` claim from a DIFFERENT mined-finding family (e.g.
    AssociationRule) must not be mistaken for an ops-causal one."""
    assert (
        _is_ops_causal_claim(
            {
                "domain": "mining_flywheel",
                "metadata": {"finding_type": "AssociationRule"},
            }
        )
        is False
    )
    assert _is_ops_causal_claim({"domain": "second_brain", "metadata": {}}) is False
    assert _is_ops_causal_claim({}) is False
    assert _is_ops_causal_claim(None) is False


def test_claim_evidence_ids_unions_source_ids_and_extracted_from():
    ids = _claim_evidence_ids({"source_ids": ["a", "b"], "extracted_from": "c"})
    assert ids == {"a", "b", "c"}


def test_claim_evidence_ids_tolerates_missing_fields():
    assert _claim_evidence_ids({}) == set()
    assert _claim_evidence_ids({"source_ids": None, "extracted_from": ""}) == set()
