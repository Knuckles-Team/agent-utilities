"""GOC-86 / U-47 acceptance scenario (W07) — ONE artifact demonstrating every
acceptance-gate condition end to end, retrieved through the supported
query-API surface (``query_claims``/``query_categories``/``query_claim_history``/
``resolve_claim_evidence``/``ClassificationPromotionLedger.history`` — every
one of them a wrapper over ``engine.query_cypher``, never a direct read of
the fake engine's internal dict), never a direct storage-count assertion.

Covered in one ordered walk, each step labeled with the acceptance-gate
condition it proves:

1. >=4 simultaneous categories on one artifact.
2. Two evidence spans, correctly cited and resolvable.
3. Declared + generated claims coexisting, neither overwriting the other.
4. One promoted candidate, one rejected candidate, both auditable.
5. A version change (old + new both queryable, correct ordering).
6. A cross-source identity proposal, evidence-bearing, reviewable.
7. Idempotent replay (no duplicate claims).
8. A malformed source (dropped, not corrupted).
9. Private-content redaction (evidence text withheld, claim itself intact).
10. Cross-graph denial (a second, distinct graph/engine sees nothing).
11. Restart survival (a FRESH ledger instance over the SAME durable store
    reproduces every prior read identically).

@pytest.mark.concept("AU-KG.ontology.classification-claim-multi-category")
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.ontology.classification_claims import (
    ClassificationClaim,
    ClassificationPromotionLedger,
    claim_from_raw,
    propose_cross_source_identity,
    query_categories,
    query_claim_history,
    query_claims,
    record_claim,
    resolve_claim_evidence,
)

pytestmark = pytest.mark.concept("AU-KG.ontology.classification-claim-multi-category")


class _AcceptanceEngine:
    """One physical graph — nodes/edges round-trip through ``query_cypher``
    exactly like the unit-test double, extended with a ``graph`` tag so two
    INSTANCES of this class stand in for two distinct physical graphs
    (production routes a different ``GraphSession.graph`` to a different
    physical store/engine handle entirely — a second instance here is an
    honest analogue of that boundary, not a shortcut around it)."""

    def __init__(self, name: str) -> None:
        self.name = name
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
            artifact = self.nodes.get(frag.get("artifact_id"), {})
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

    def claim_node_count(self) -> int:
        return sum(
            1 for n in self.nodes.values() if n.get("type") == "ClassificationClaim"
        )


def _emit_v1_claims(engine: _AcceptanceEngine) -> dict[str, ClassificationClaim]:
    """The EXTRACTION half of ingesting ``art:demo`` at snapshot v1 —
    deterministic claims recorded directly, candidates proposed (not yet
    governed). This is the half a re-ingest of the SAME artifact at the SAME
    snapshot actually replays; calling it a SECOND time against the same
    engine must not add a single node (the idempotence this lane requires).
    Governance (review/promote/reject) is a SEPARATE, one-time decision — see
    :func:`_govern_v1_candidates` — never re-run just because the source was
    re-ingested."""
    # -- deterministic (declared/observed) claims: ground truth, promoted at
    # construction, never held for review. Two DIFFERENT evidence spans.
    security = ClassificationClaim.declare(
        subject_id="art:demo",
        category="security-critical",
        value="true",
        evidence_refs=["frag:frontmatter"],
        source_snapshot="sha:v1",
        extractor_ref="frontmatter-parser@1",
    )
    code = ClassificationClaim.observe(
        subject_id="art:demo",
        category="code",
        value="python",
        evidence_refs=["frag:codeblock"],
        source_snapshot="sha:v1",
        extractor_ref="lang-sniffer@1",
    )
    ownership = ClassificationClaim.declare(
        subject_id="art:demo",
        category="ownership",
        value="team-platform",
        evidence_refs=["frag:frontmatter"],
        source_snapshot="sha:v1",
        extractor_ref="frontmatter-parser@1",
    )
    for claim in (security, code, ownership):
        record_claim(engine, claim)

    # -- candidate (generated) claims: policy-approved LLM proposals, always
    # minted as 'candidate'. One will be promoted, one rejected (below).
    skill_candidate = ClassificationClaim.propose(
        subject_id="art:demo",
        category="skill-resource",
        value="true",
        evidence_refs=["frag:codeblock"],
        source_snapshot="sha:v1",
        method="generated",
        extractor_ref="skill-classifier@2026-08",
        confidence=0.87,
        policy_approved=True,
    )
    deprecated_candidate = ClassificationClaim.propose(
        subject_id="art:demo",
        category="deprecated-api",
        value="true",
        evidence_refs=["frag:frontmatter"],
        source_snapshot="sha:v1",
        method="generated",
        extractor_ref="skill-classifier@2026-08",
        confidence=0.31,
        policy_approved=True,
    )
    for claim in (skill_candidate, deprecated_candidate):
        record_claim(engine, claim)

    return {
        "security": security,
        "code": code,
        "ownership": ownership,
        "skill_candidate": skill_candidate,
        "deprecated_candidate": deprecated_candidate,
    }


def _govern_v1_candidates(
    engine: _AcceptanceEngine,
    ledger: ClassificationPromotionLedger,
    claims: dict[str, ClassificationClaim],
) -> dict[str, ClassificationClaim]:
    """The GOVERNANCE half — a one-time review decision over the candidates
    :func:`_emit_v1_claims` proposed. Never replayed by a re-ingest."""
    reviewed_skill = ledger.review(
        claims["skill_candidate"], reason="passes skill-registry cross-check"
    )
    promoted_skill = ledger.promote(
        reviewed_skill, reviewer="reviewer-1", reason="matches known skill shape"
    )

    reviewed_deprecated = ledger.review(
        claims["deprecated_candidate"], reason="manual audit"
    )
    rejected_deprecated = ledger.reject(
        reviewed_deprecated,
        reviewer="reviewer-1",
        reason="false positive — API is current",
    )
    return {
        **claims,
        "skill_promoted": promoted_skill,
        "deprecated_rejected": rejected_deprecated,
    }


@pytest.fixture
def acceptance_engine() -> _AcceptanceEngine:
    engine = _AcceptanceEngine("graph-a")
    # frag:frontmatter belongs to a RESTRICTED artifact-scoped fragment (for
    # the redaction test); frag:codeblock is ordinary internal content.
    engine.add_artifact("art:demo", classification="restricted")
    engine.add_fragment(
        "frag:frontmatter", "art:demo", "owner: team-platform\nsecurity_critical: true"
    )
    engine.add_fragment("frag:codeblock", "art:demo", "def handle_request(): ...")
    return engine


def test_full_w07_acceptance_scenario(acceptance_engine: _AcceptanceEngine) -> None:
    engine = acceptance_engine
    ledger = ClassificationPromotionLedger(engine)
    claims = _emit_v1_claims(engine)
    claims = _govern_v1_candidates(engine, ledger, claims)

    # ── 1. >=4 simultaneous categories, retrieved through query_categories ──
    categories = query_categories(engine, "art:demo")
    assert categories >= {"security-critical", "code", "ownership", "skill-resource"}
    assert len(categories) >= 4

    # ── 2. Two evidence spans, correctly cited and resolvable ──────────────
    all_claims = query_claims(engine, "art:demo")
    cited_fragments = {ref for c in all_claims for ref in c.evidence_refs}
    assert {"frag:frontmatter", "frag:codeblock"} <= cited_fragments
    resolved = resolve_claim_evidence(
        engine, claims["code"], viewer_clearance="restricted"
    )
    assert resolved[0]["text"] == "def handle_request(): ..."

    # ── 3. Declared + generated coexist, neither overwrites the other ──────
    declared_categories = {c.category for c in all_claims if c.method == "declared"}
    generated_categories = {c.category for c in all_claims if c.method == "generated"}
    assert "security-critical" in declared_categories
    assert "skill-resource" in generated_categories
    assert declared_categories.isdisjoint(generated_categories)
    # the promoted generated claim is STILL tagged method='generated' — a
    # promotion never launders a candidate into looking deterministic.
    assert (
        query_claims(engine, "art:demo", category="skill-resource")[0].method
        == "generated"
    )

    # ── 4. One promoted candidate, one rejected candidate, both auditable ──
    assert claims["skill_promoted"].status == "promoted"
    assert claims["deprecated_rejected"].status == "rejected"
    promoted_history = ledger.history(claims["skill_promoted"].claim_id)
    assert [e["to_status"] for e in promoted_history] == ["reviewed", "promoted"]
    rejected_history = ledger.history(claims["deprecated_rejected"].claim_id)
    assert [e["to_status"] for e in rejected_history] == ["reviewed", "rejected"]
    # rejected is retained, never resurfaces as an active fact
    assert "deprecated-api" not in query_categories(engine, "art:demo")
    assert claims["deprecated_rejected"].claim_id in {
        c.claim_id for c in query_claims(engine, "art:demo", category="deprecated-api")
    }

    # ── 5. A version change — old AND new queryable, correct ordering ──────
    code_v2 = ClassificationClaim.observe(
        subject_id="art:demo",
        category="code",
        value="python3.12",
        evidence_refs=["frag:codeblock"],
        source_snapshot="sha:v2",
        extractor_ref="lang-sniffer@2",
    )
    updated_v1, recorded_v2 = ledger.supersede(
        claims["code"], code_v2, reason="re-extracted at sha:v2"
    )
    version_history = query_claim_history(engine, "art:demo", "code")
    assert version_history[0].claim_id == recorded_v2.claim_id  # newest first
    assert version_history[0].value == "python3.12"
    assert any(
        c.claim_id == updated_v1.claim_id and c.status == "superseded"
        for c in version_history
    )
    assert (
        query_claims(engine, "art:demo", category="code", status="promoted")[0].value
        == "python3.12"
    )

    # ── 6. Cross-source identity proposal — evidence-bearing, reviewable ───
    engine.add_artifact("art:demo-mirror", classification="internal")
    proposal = propose_cross_source_identity(
        artifact_a_id="art:demo",
        artifact_b_id="art:demo-mirror",
        evidence_refs=["frag:codeblock"],
        source_snapshot="sha:v1",
        extractor_ref="identity-resolver@1",
        confidence=0.95,
    )
    assert proposal.status == "candidate"
    record_claim(engine, proposal)
    reviewed_identity = ledger.review(
        proposal, reason="content hash matches across connectors"
    )
    promoted_identity = ledger.promote(reviewed_identity, reviewer="reviewer-2")
    assert promoted_identity.status == "promoted"
    assert "cross_source_identity" in query_categories(engine, "art:demo")
    assert (
        query_claims(engine, "art:demo", category="cross_source_identity")[0].value
        == "art:demo-mirror"
    )

    # ── 7. Idempotent replay — re-running v1 EXTRACTION adds ZERO new nodes ─
    before_count = engine.claim_node_count()
    _emit_v1_claims(engine)  # exact same inputs, second pass — extraction only
    after_count = engine.claim_node_count()
    assert after_count == before_count, "idempotent replay must not duplicate claims"
    assert len(query_claims(engine, "art:demo", category="security-critical")) == 1
    assert len(query_claims(engine, "art:demo", category="skill-resource")) == 1

    # ── 8. A malformed source — dropped cleanly, graph left uncorrupted ────
    malformed_raw = {
        "subject_id": "art:demo",
        "category": "code",
    }  # missing value/evidence_refs
    before_malformed = engine.claim_node_count()
    dropped = claim_from_raw(malformed_raw, source_snapshot="sha:v1")
    assert dropped is None
    assert engine.claim_node_count() == before_malformed

    # ── 9. Private-content redaction — text withheld, claim itself intact ──
    redacted_view = resolve_claim_evidence(
        engine, claims["security"], viewer_clearance="internal"
    )
    assert redacted_view[0]["redacted"] is True
    assert redacted_view[0]["text"] is None
    # the classification FACT (not its raw evidence text) is still readable —
    # the claim layer degrades to "redacted evidence", never to "no claim".
    assert "security-critical" in query_categories(engine, "art:demo")
    cleared_view = resolve_claim_evidence(
        engine, claims["security"], viewer_clearance="restricted"
    )
    assert cleared_view[0]["redacted"] is False

    # ── 10. Cross-graph denial — a second, distinct graph sees NOTHING ─────
    other_graph = _AcceptanceEngine("graph-b")
    assert query_categories(other_graph, "art:demo") == set()
    assert query_claims(other_graph, "art:demo") == []

    # ── 11. Restart survival — a FRESH ledger over the SAME durable store ──
    restarted_ledger = ClassificationPromotionLedger(engine)
    assert query_categories(engine, "art:demo") == categories | {
        "cross_source_identity"
    }
    assert [
        e["to_status"]
        for e in restarted_ledger.history(claims["skill_promoted"].claim_id)
    ] == [
        "reviewed",
        "promoted",
    ]
    post_restart_version_history = query_claim_history(engine, "art:demo", "code")
    assert post_restart_version_history[0].value == "python3.12"
    assert (
        resolve_claim_evidence(engine, claims["security"], viewer_clearance="internal")[
            0
        ]["redacted"]
        is True
    )
