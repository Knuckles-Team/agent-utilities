"""Unit tests for ``ClassificationClaim`` (U-47, GOC-86) — the model itself,
the idempotent key, the structural method/status invariants, the promotion
lifecycle FSM, cross-source identity proposals, and malformed-input
tolerance, all in isolation from any live engine.

@pytest.mark.concept("AU-KG.ontology.classification-claim-multi-category")
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.ontology.classification_claims import (
    APPLIES_TO,
    CANDIDATE_METHODS,
    DECLARES,
    DERIVED_FROM,
    DETERMINISTIC_METHODS,
    HAS_PROVENANCE,
    TYPED_RELATIONS,
    ClassificationClaim,
    ClassificationPromotionLedger,
    IllegalClaimTransition,
    claim_from_raw,
    claim_id_for,
    propose_cross_source_identity,
    query_categories,
    query_claim_history,
    query_claims,
    record_claim,
    resolve_claim_evidence,
)

pytestmark = pytest.mark.concept("AU-KG.ontology.classification-claim-multi-category")


class _StubEngine:
    """Round-trips nodes/edges through ``query_cypher`` — same convention as
    ``research/claim_flywheel.py``'s own test double
    (``_FlywheelStubEngine``), extended with a Fragment/Artifact join for the
    redaction test and an ``edges`` list for relation assertions."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(
        self, node_id: str, node_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def link_nodes(
        self,
        source: str,
        target: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self.edges.append((source, target, rel_type))

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        if "ClassificationClaimLifecycleEvent" in query:
            rows = [
                n
                for n in self.nodes.values()
                if n.get("type") == "ClassificationClaimLifecycleEvent"
            ]
            cid = params.get("id")
            if cid is not None:
                rows = [r for r in rows if r.get("claim_id") == cid]
            return rows
        if "ClassificationClaim" in query and "Fragment" not in query:
            rows = [
                n for n in self.nodes.values() if n.get("type") == "ClassificationClaim"
            ]
            sid = params.get("sid")
            if sid is not None:
                rows = [r for r in rows if r.get("subject_id") == sid]
            status = params.get("status")
            if status is not None:
                rows = [r for r in rows if r.get("status") == status]
            category = params.get("category")
            if category is not None:
                rows = [r for r in rows if r.get("category") == category]
            return rows
        if "Fragment" in query and "Artifact" in query:
            fid = params.get("fid")
            frag = self.nodes.get(fid)
            if not frag or frag.get("type") != "Fragment":
                return []
            artifact_id = frag.get("artifact_id")
            artifact = self.nodes.get(artifact_id, {})
            return [
                {
                    "id": frag["id"],
                    "text": frag.get("text"),
                    "address": frag.get("address"),
                    "classification": artifact.get("classification"),
                }
            ]
        return []

    def add_fragment(self, fragment_id: str, artifact_id: str, text: str) -> None:
        self.nodes[fragment_id] = {
            "id": fragment_id,
            "type": "Fragment",
            "artifact_id": artifact_id,
            "text": text,
            "address": fragment_id,
        }

    def add_artifact(self, artifact_id: str, classification: str = "internal") -> None:
        self.nodes[artifact_id] = {
            "id": artifact_id,
            "type": "Artifact",
            "classification": classification,
        }


# ---------------------------------------------------------------------------
# claim_id_for — idempotence
# ---------------------------------------------------------------------------


def test_claim_id_is_deterministic_and_evidence_order_independent():
    a = claim_id_for(
        "art:1", "code", "python", "observed", ["frag:1", "frag:2"], "sha:abc"
    )
    b = claim_id_for(
        "art:1", "code", "python", "observed", ["frag:2", "frag:1"], "sha:abc"
    )
    assert a == b


def test_claim_id_changes_when_source_snapshot_changes():
    a = claim_id_for("art:1", "code", "python", "observed", ["frag:1"], "sha:abc")
    b = claim_id_for("art:1", "code", "python", "observed", ["frag:1"], "sha:def")
    assert a != b


def test_claim_id_is_case_insensitive_on_value():
    a = claim_id_for("art:1", "code", "Python", "observed", ["frag:1"], "sha:abc")
    b = claim_id_for("art:1", "code", "python", "observed", ["frag:1"], "sha:abc")
    assert a == b


# ---------------------------------------------------------------------------
# Structural method/status invariants (Authority #3, #4)
# ---------------------------------------------------------------------------


def test_declare_always_promoted():
    claim = ClassificationClaim.declare(
        subject_id="art:1",
        category="code",
        value="python",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
    )
    assert claim.method == "declared"
    assert claim.status == "promoted"
    assert claim.is_deterministic
    assert claim.is_active_fact


def test_observe_always_promoted():
    claim = ClassificationClaim.observe(
        subject_id="art:1",
        category="code",
        value="python",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
    )
    assert claim.status == "promoted"


def test_propose_always_candidate():
    claim = ClassificationClaim.propose(
        subject_id="art:1",
        category="security-critical",
        value="true",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
        method="generated",
        policy_approved=True,
    )
    assert claim.method == "generated"
    assert claim.status == "candidate"
    assert not claim.is_deterministic
    assert not claim.is_active_fact  # a candidate is NOT yet a policy fact


def test_generated_without_policy_approval_refused():
    with pytest.raises(PermissionError):
        ClassificationClaim.propose(
            subject_id="art:1",
            category="security-critical",
            value="true",
            evidence_refs=["frag:1"],
            source_snapshot="sha:abc",
            method="generated",
            policy_approved=False,
        )


def test_derived_does_not_require_policy_approval():
    claim = ClassificationClaim.propose(
        subject_id="art:1",
        category="cross_source_identity",
        value="art:2",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
        method="derived",
    )
    assert claim.status == "candidate"


def test_a_deterministic_claim_cannot_be_constructed_at_candidate_status():
    """The type-level proof that a guess can never be indistinguishable from
    a fact: constructing a 'declared'/'observed' claim at any status other
    than promoted/superseded is structurally impossible."""
    with pytest.raises(ValueError, match="deterministic ground truth"):
        ClassificationClaim(
            claim_id=claim_id_for(
                "art:1", "code", "python", "declared", ["frag:1"], "sha:abc"
            ),
            subject_id="art:1",
            category="code",
            value="python",
            method="declared",
            status="candidate",
            evidence_refs=("frag:1",),
            source_snapshot="sha:abc",
        )


def test_a_generated_claim_cannot_be_read_as_declared_or_observed():
    """A candidate is never indistinguishable from a fact: filtering by
    method never confuses the two populations."""
    declared = ClassificationClaim.declare(
        subject_id="art:1",
        category="code",
        value="python",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
    )
    generated = ClassificationClaim.propose(
        subject_id="art:1",
        category="security-critical",
        value="true",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
        method="generated",
        policy_approved=True,
    )
    assert declared.method in DETERMINISTIC_METHODS
    assert generated.method in CANDIDATE_METHODS
    assert declared.method not in CANDIDATE_METHODS
    assert generated.method not in DETERMINISTIC_METHODS
    # a promoted, active-fact read never returns the still-candidate one
    assert declared.is_active_fact and not generated.is_active_fact


def test_claim_requires_evidence():
    with pytest.raises(ValueError, match="evidence"):
        ClassificationClaim.declare(
            subject_id="art:1",
            category="code",
            value="python",
            evidence_refs=[],
            source_snapshot="sha:abc",
        )


def test_claim_id_must_match_content_or_raise():
    with pytest.raises(ValueError, match="claim_id"):
        ClassificationClaim(
            claim_id="classification_claim:not-the-real-hash",
            subject_id="art:1",
            category="code",
            value="python",
            method="declared",
            status="promoted",
            evidence_refs=("frag:1",),
            source_snapshot="sha:abc",
        )


# ---------------------------------------------------------------------------
# with_status FSM
# ---------------------------------------------------------------------------


def test_with_status_full_lifecycle():
    claim = ClassificationClaim.propose(
        subject_id="art:1",
        category="skill-resource",
        value="true",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
        method="generated",
        policy_approved=True,
    )
    reviewed = claim.with_status("reviewed")
    assert reviewed.status == "reviewed"
    promoted = reviewed.with_status("promoted", reviewer="alice")
    assert promoted.status == "promoted"
    assert promoted.reviewer == "alice"
    assert promoted.is_active_fact
    superseded = promoted.with_status(
        "superseded", superseded_by="classification_claim:next"
    )
    assert superseded.status == "superseded"
    assert superseded.superseded_by == "classification_claim:next"


def test_illegal_transition_candidate_to_promoted_directly_refused():
    claim = ClassificationClaim.propose(
        subject_id="art:1",
        category="skill-resource",
        value="true",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
        method="generated",
        policy_approved=True,
    )
    with pytest.raises(IllegalClaimTransition):
        claim.with_status("promoted")


def test_superseded_is_terminal():
    claim = ClassificationClaim.declare(
        subject_id="art:1",
        category="code",
        value="python",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
    )
    superseded = claim.with_status(
        "superseded", superseded_by="classification_claim:next"
    )
    with pytest.raises(IllegalClaimTransition):
        superseded.with_status("promoted")


def test_deterministic_claim_only_ever_advances_to_superseded():
    claim = ClassificationClaim.observe(
        subject_id="art:1",
        category="code",
        value="python",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
    )
    with pytest.raises(IllegalClaimTransition):
        claim.with_status("reviewed")
    # the only legal move for a deterministic (already-promoted) claim
    assert claim.with_status("superseded").status == "superseded"


# ---------------------------------------------------------------------------
# ClassificationPromotionLedger — persisted lifecycle, promoted + rejected
# ---------------------------------------------------------------------------


def test_ledger_promote_persists_status_and_audit_trail():
    engine = _StubEngine()
    ledger = ClassificationPromotionLedger(engine)
    claim = ClassificationClaim.propose(
        subject_id="art:1",
        category="skill-resource",
        value="true",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
        method="generated",
        policy_approved=True,
    )
    record_claim(engine, claim)
    reviewed = ledger.review(claim)
    promoted = ledger.promote(reviewed, reviewer="bob", reason="matches skill registry")

    stored = query_claims(engine, "art:1", status="promoted")
    assert [c.claim_id for c in stored] == [promoted.claim_id]
    history = ledger.history(claim.claim_id)
    assert [e["to_status"] for e in history] == ["reviewed", "promoted"]


def test_ledger_reject_never_resurfaces_as_active_fact():
    engine = _StubEngine()
    ledger = ClassificationPromotionLedger(engine)
    claim = ClassificationClaim.propose(
        subject_id="art:1",
        category="deprecated-api",
        value="true",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
        method="generated",
        policy_approved=True,
    )
    record_claim(engine, claim)
    reviewed = ledger.review(claim)
    rejected = ledger.reject(reviewed, reviewer="bob", reason="false positive")

    assert rejected.status == "rejected"
    assert not rejected.is_active_fact
    assert query_claims(engine, "art:1", status="promoted") == []
    # retained for audit — the rejected claim is still readable by id
    all_claims = query_claims(engine, "art:1")
    assert rejected.claim_id in {c.claim_id for c in all_claims}


def test_ledger_supersede_retains_old_claim_and_links_new_one():
    engine = _StubEngine()
    ledger = ClassificationPromotionLedger(engine)
    old = ClassificationClaim.observe(
        subject_id="art:1",
        category="code",
        value="python2",
        evidence_refs=["frag:1"],
        source_snapshot="sha:v1",
    )
    record_claim(engine, old)
    new = ClassificationClaim.observe(
        subject_id="art:1",
        category="code",
        value="python3",
        evidence_refs=["frag:2"],
        source_snapshot="sha:v2",
    )
    updated_old, recorded_new = ledger.supersede(old, new)

    assert updated_old.status == "superseded"
    assert updated_old.superseded_by == new.claim_id
    history = query_claim_history(engine, "art:1", "code")
    ids = {c.claim_id for c in history}
    assert old.claim_id in ids and new.claim_id in ids
    assert (new.claim_id, old.claim_id, DERIVED_FROM) in engine.edges


# ---------------------------------------------------------------------------
# record_claim wiring — typed relations
# ---------------------------------------------------------------------------


def test_record_claim_wires_applies_to_and_provenance():
    engine = _StubEngine()
    claim = ClassificationClaim.declare(
        subject_id="art:1",
        category="ownership",
        value="team-x",
        evidence_refs=["frag:1", "frag:2"],
        source_snapshot="sha:abc",
    )
    record_claim(engine, claim)
    assert (claim.claim_id, "art:1", APPLIES_TO) in engine.edges
    assert (claim.claim_id, "frag:1", HAS_PROVENANCE) in engine.edges
    assert (claim.claim_id, "frag:2", HAS_PROVENANCE) in engine.edges
    assert ("art:1", claim.claim_id, DECLARES) in engine.edges


def test_record_claim_extra_relations_are_validated():
    engine = _StubEngine()
    claim = ClassificationClaim.declare(
        subject_id="art:1",
        category="policy-implementation",
        value="std:42",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
    )
    with pytest.raises(ValueError, match="TYPED_RELATIONS"):
        record_claim(engine, claim, extra_relations=[("NOT_A_RELATION", "std:42")])
    record_claim(engine, claim, extra_relations=[("IMPLEMENTS", "std:42")])
    assert (claim.claim_id, "std:42", "IMPLEMENTS") in engine.edges
    assert "IMPLEMENTS" in TYPED_RELATIONS


# ---------------------------------------------------------------------------
# Cross-source identity proposals (W06)
# ---------------------------------------------------------------------------


def test_cross_source_identity_proposal_is_evidence_bearing():
    proposal = propose_cross_source_identity(
        artifact_a_id="artifact:github:foo",
        artifact_b_id="artifact:gitlab:foo-mirror",
        evidence_refs=["frag:content-match-1"],
        source_snapshot="sha:abc",
        extractor_ref="identity-resolver@1",
    )
    assert proposal.category == "cross_source_identity"
    assert proposal.method == "derived"
    assert proposal.status == "candidate"  # never a shortcut straight to promoted
    assert proposal.value == "artifact:gitlab:foo-mirror"
    assert proposal.evidence_refs == ("frag:content-match-1",)


def test_cross_source_identity_proposal_refuses_self_identity():
    with pytest.raises(ValueError, match="DISTINCT"):
        propose_cross_source_identity(
            artifact_a_id="artifact:x",
            artifact_b_id="artifact:x",
            evidence_refs=["frag:1"],
            source_snapshot="sha:abc",
            extractor_ref="r",
        )


def test_cross_source_identity_proposal_refuses_bare_name_match():
    """There is no name-only parameter — evidence is structurally required."""
    with pytest.raises(ValueError, match="evidence"):
        propose_cross_source_identity(
            artifact_a_id="artifact:x",
            artifact_b_id="artifact:y",
            evidence_refs=[],
            source_snapshot="sha:abc",
            extractor_ref="r",
        )


def test_cross_source_identity_promotion_wires_derived_from():
    engine = _StubEngine()
    ledger = ClassificationPromotionLedger(engine)
    proposal = propose_cross_source_identity(
        artifact_a_id="artifact:a",
        artifact_b_id="artifact:b",
        evidence_refs=["frag:1"],
        source_snapshot="sha:abc",
        extractor_ref="r",
    )
    record_claim(engine, proposal)
    reviewed = ledger.review(proposal)
    promoted = ledger.promote(reviewed, reviewer="carol")
    assert promoted.status == "promoted"
    assert query_categories(engine, "artifact:a") == {"cross_source_identity"}


# ---------------------------------------------------------------------------
# Malformed source tolerance
# ---------------------------------------------------------------------------


def test_claim_from_raw_drops_missing_required_field():
    assert (
        claim_from_raw(
            {"category": "code", "value": "x", "evidence_refs": ["f1"]},
            source_snapshot="sha:1",
        )
        is None
    )


def test_claim_from_raw_drops_empty_evidence_refs():
    raw = {"subject_id": "art:1", "category": "code", "value": "x", "evidence_refs": []}
    assert claim_from_raw(raw, source_snapshot="sha:1") is None


def test_claim_from_raw_drops_malformed_evidence_shape():
    raw = {
        "subject_id": "art:1",
        "category": "code",
        "value": "x",
        "evidence_refs": "not-a-list",
    }
    assert claim_from_raw(raw, source_snapshot="sha:1") is None


def test_claim_from_raw_drops_unknown_method():
    raw = {
        "subject_id": "art:1",
        "category": "code",
        "value": "x",
        "evidence_refs": ["f1"],
        "method": "hallucinated",
    }
    assert claim_from_raw(raw, source_snapshot="sha:1") is None


def test_claim_from_raw_drops_generated_without_policy_approval():
    raw = {
        "subject_id": "art:1",
        "category": "code",
        "value": "x",
        "evidence_refs": ["f1"],
        "method": "generated",
    }
    assert claim_from_raw(raw, source_snapshot="sha:1", policy_approved=False) is None


def test_claim_from_raw_builds_a_well_formed_deterministic_claim():
    raw = {
        "subject_id": "art:1",
        "category": "code",
        "value": "python",
        "evidence_refs": ["f1"],
        "method": "observed",
    }
    claim = claim_from_raw(raw, source_snapshot="sha:1")
    assert claim is not None
    assert claim.status == "promoted"


def test_claim_from_raw_never_raises_on_garbage_input():
    for garbage in (None, {}, {"subject_id": None}, {"evidence_refs": {"a": 1}}):
        try:
            result = claim_from_raw(garbage or {}, source_snapshot="sha:1")
        except Exception as exc:  # pragma: no cover - the assertion below fails first
            pytest.fail(f"claim_from_raw raised on malformed input {garbage!r}: {exc}")
        assert result is None


# ---------------------------------------------------------------------------
# Private-content redaction (evidence resolution)
# ---------------------------------------------------------------------------


def test_resolve_claim_evidence_redacts_below_clearance():
    engine = _StubEngine()
    engine.add_artifact("art:secret", classification="restricted")
    engine.add_fragment(
        "frag:secret-1", "art:secret", "the private password is hunter2"
    )
    claim = ClassificationClaim.observe(
        subject_id="art:secret",
        category="code",
        value="python",
        evidence_refs=["frag:secret-1"],
        source_snapshot="sha:abc",
    )
    redacted = resolve_claim_evidence(engine, claim, viewer_clearance="internal")
    assert redacted[0]["redacted"] is True
    assert redacted[0]["text"] is None

    cleared = resolve_claim_evidence(engine, claim, viewer_clearance="restricted")
    assert cleared[0]["redacted"] is False
    assert cleared[0]["text"] == "the private password is hunter2"


def test_resolve_claim_evidence_fails_closed_on_unresolved_fragment():
    engine = _StubEngine()
    claim = ClassificationClaim.observe(
        subject_id="art:1",
        category="code",
        value="python",
        evidence_refs=["frag:missing"],
        source_snapshot="sha:abc",
    )
    result = resolve_claim_evidence(engine, claim, viewer_clearance="restricted")
    assert result[0]["redacted"] is True
    assert result[0]["text"] is None
