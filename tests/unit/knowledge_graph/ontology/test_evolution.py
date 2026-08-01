"""Ontology evolution as a governed proposal lifecycle (CONCEPT:AU-KG.ontology.evolution-governed-loop,
program item 7.5): detect -> validate/classify -> shadow-replay -> review gate
-> promote/rollback, with negative results staying queryable.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent_utilities.knowledge_graph.ontology import evolution
from agent_utilities.knowledge_graph.ontology.lifecycle import (
    OntologyLifecycle,
    reset_registry,
)
from agent_utilities.orchestration.action_policy import ActionDecision, ActionRequest

pytestmark = pytest.mark.concept("AU-KG.ontology.evolution-governed-loop")

PETS_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/pets#> .
<http://example.org/pets> a owl:Ontology .
ex:Animal a owl:Class .
ex:Dog a owl:Class ; rdfs:subClassOf ex:Animal .
ex:hasOwner a owl:ObjectProperty .
"""

PETS_TTL_ADDITIVE = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/pets#> .
<http://example.org/pets> a owl:Ontology .
ex:Animal a owl:Class .
ex:Dog a owl:Class ; rdfs:subClassOf ex:Animal .
ex:Cat a owl:Class ; rdfs:subClassOf ex:Animal .
ex:hasOwner a owl:ObjectProperty .
"""

PETS_TTL_BREAKING = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://example.org/pets#> .
<http://example.org/pets> a owl:Ontology .
ex:Animal a owl:Class .
"""

BAD_TTL = "this is @@ not <<< valid turtle"


@pytest.fixture(autouse=True)
def _clean_registries():
    reset_registry()
    evolution.reset_proposal_registry()
    yield
    reset_registry()
    evolution.reset_proposal_registry()


def _load_baseline():
    """Host the v1 ontology directly (bypassing the governed proposal path,
    the way a first-ever import would) so later tests have a baseline to
    diff a proposal against."""
    OntologyLifecycle(engine=None).load(
        PETS_TTL, source_type="text", iri="http://example.org/pets", version="1.0.0"
    )


# ── classify_change / next_semver — deterministic, not LLM-judged ──────────


def test_classify_change_detects_additive():
    baseline = {"classes": ["A"], "properties": []}
    candidate = {"classes": ["A", "B"], "properties": []}
    result = evolution.classify_change(baseline, candidate)
    assert result["kind"] == "additive"
    assert result["added_classes"] == ["B"]
    assert result["removed_classes"] == []


def test_classify_change_detects_breaking():
    baseline = {"classes": ["A", "B"], "properties": []}
    candidate = {"classes": ["A"], "properties": []}
    result = evolution.classify_change(baseline, candidate)
    assert result["kind"] == "breaking"
    assert result["removed_classes"] == ["B"]


def test_classify_change_unchanged():
    baseline = {"classes": ["A"], "properties": ["p"]}
    result = evolution.classify_change(baseline, baseline)
    assert result["kind"] == "unchanged"


@pytest.mark.parametrize(
    ("prior", "kind", "expected"),
    [
        ("1.2.3", "breaking", "2.0.0"),
        ("1.2.3", "additive", "1.3.0"),
        ("1.2.3", "unchanged", "1.2.4"),
        ("0.0.0", "additive", "0.1.0"),
    ],
)
def test_next_semver(prior, kind, expected):
    assert evolution.next_semver(prior, kind) == expected


# ── compare_against_standards (program item 2, D-75-8) — deterministic name
# collision against the bundled authoritative-standards corpus, never an LLM
# judgment ────────────────────────────────────────────────────────────────


def test_bundled_standard_vocabulary_is_nonempty_and_carries_known_standard_terms():
    """The bundled ontology.ttl genuinely absorbs BFO/PROV-O/Schema.org/SKOS
    alignment targets (verified directly against the file), not a stand-in
    corpus -- 'creativework' (schema:CreativeWork) and 'activity'
    (prov:Activity) must both be present."""
    vocab = evolution._bundled_standard_vocabulary()
    assert "creativework" in vocab
    assert "activity" in vocab


def test_compare_against_standards_flags_a_colliding_class_name():
    candidate = {
        "classes": ["http://example.org/pets#CreativeWork", "http://example.org/pets#Dog"],
        "properties": [],
    }
    flags = evolution.compare_against_standards(candidate)
    assert len(flags) == 1
    assert flags[0]["term"] == "http://example.org/pets#CreativeWork"
    assert flags[0]["kind"] == "class"
    assert flags[0]["local_name"] == "creativework"


def test_compare_against_standards_flags_nothing_for_a_novel_name():
    candidate = {"classes": ["http://example.org/pets#Dog"], "properties": []}
    assert evolution.compare_against_standards(candidate) == []


def test_propose_flags_a_standards_collision_without_forcing_review():
    """A standards collision is advisory (flagged for a reviewer), never a
    forced review on its own -- classify_change/validate_graph/replay
    regressions remain the ONLY things that force requires_review=True."""
    result = evolution.propose_ontology_change(
        None,
        None,
        PETS_TTL,  # defines :Animal, :Dog (rdfs:subClassOf), :hasOwner -- no collision
        iri="http://example.org/pets",
        source_type="text",
    )
    proposal = result["proposal"]
    assert proposal["standards_alignment"]["checked"] is True
    assert proposal["standards_alignment"]["flags"] == []


def test_propose_standards_alignment_flags_present_for_a_colliding_candidate():
    collide_ttl = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://example.org/docs#> .
<http://example.org/docs> a owl:Ontology .
ex:CreativeWork a owl:Class .
"""
    result = evolution.propose_ontology_change(
        None, None, collide_ttl, iri="http://example.org/docs", source_type="text"
    )
    proposal = result["proposal"]
    flags = proposal["standards_alignment"]["flags"]
    assert any(f["local_name"] == "creativework" for f in flags)
    # advisory only — a pure additive+colliding change still isn't forced to review
    assert proposal["requires_review"] is False


# ── propose — detect/align/decide/shadow, never touches the active ontology ─


def test_propose_first_version_is_additive_against_empty_baseline():
    result = evolution.propose_ontology_change(
        None, None, PETS_TTL, iri="http://example.org/pets", source_type="text"
    )
    assert result["status"] == "ok"
    proposal = result["proposal"]
    assert proposal["classification"]["kind"] == "additive"
    assert proposal["version"] == "0.1.0"
    assert proposal["prior_version"] is None
    assert proposal["status"] == evolution.STATUS_PENDING_REVIEW
    # Nothing was hosted -- proposing never auto-merges into the active ontology.
    assert OntologyLifecycle(engine=None).list_ontologies()["count"] == 0


def test_propose_rejects_malformed_candidate():
    result = evolution.propose_ontology_change(
        None, None, BAD_TTL, iri="http://example.org/pets", source_type="text"
    )
    assert result["status"] == "rejected"
    assert result["errors"]


def test_propose_additive_change_against_hosted_baseline():
    _load_baseline()
    result = evolution.propose_ontology_change(
        None,
        None,
        PETS_TTL_ADDITIVE,
        iri="http://example.org/pets",
        source_type="text",
    )
    proposal = result["proposal"]
    assert proposal["classification"]["kind"] == "additive"
    assert proposal["classification"]["added_classes"] == [
        "http://example.org/pets#Cat"
    ]
    assert proposal["prior_version"] == "1.0.0"
    assert proposal["version"] == "1.1.0"
    assert proposal["requires_review"] is False


def test_propose_breaking_change_forces_review():
    _load_baseline()
    result = evolution.propose_ontology_change(
        None,
        None,
        PETS_TTL_BREAKING,
        iri="http://example.org/pets",
        source_type="text",
    )
    proposal = result["proposal"]
    assert proposal["classification"]["kind"] == "breaking"
    assert proposal["version"] == "2.0.0"
    assert proposal["requires_review"] is True


def test_propose_stores_evidence_refs_and_is_listable():
    result = evolution.propose_ontology_change(
        None,
        None,
        PETS_TTL,
        iri="http://example.org/pets",
        source_type="text",
        evidence_refs=["doc:123#span=4-9"],
        proposer="schema_discovery",
        reason="observed 'ex:Dog' in 12 sampled records",
    )
    proposal_id = result["proposal"]["proposal_id"]
    fetched = evolution.get_proposal(None, None, proposal_id)
    assert fetched["evidence_refs"] == ["doc:123#span=4-9"]
    assert fetched["proposer"] == "schema_discovery"
    listed = evolution.list_proposals(None, None)
    assert any(p["proposal_id"] == proposal_id for p in listed)


# ── review ───────────────────────────────────────────────────────────────


def test_review_records_decision():
    result = evolution.propose_ontology_change(
        None, None, PETS_TTL, iri="http://example.org/pets", source_type="text"
    )
    proposal_id = result["proposal"]["proposal_id"]
    reviewed = evolution.review_ontology_proposal(
        None, None, proposal_id, approve=True, reviewer="alice", notes="looks fine"
    )
    assert reviewed["proposal"]["status"] == evolution.STATUS_APPROVED
    assert reviewed["proposal"]["decision"]["reviewer"] == "alice"


def test_review_unknown_proposal_errors():
    result = evolution.review_ontology_proposal(
        None, None, "does-not-exist", approve=True, reviewer="alice"
    )
    assert "error" in result


# ── shadow-graph GC on rejection (CONCEPT:AU-KG.ontology.shadow-graph-gc, D-75-7) ──


def test_reject_discards_the_shadow_graph():
    """A rejected proposal is never coming back for promotion (promote() reads
    the stored `turtle`, never the shadow graph) -- rejecting it tears down its
    scratch shadow graph immediately rather than leaking it until an engine
    idle-sweep eventually reclaims it."""
    with patch(
        "agent_utilities.knowledge_graph.ontology.evolution.materialize_shadow",
        return_value=("ontology:tenant__shadow__proposal-1", {"loaded_to_engine": True}),
    ):
        result = evolution.propose_ontology_change(
            None, None, PETS_TTL, iri="http://example.org/pets", source_type="text"
        )
    proposal_id = result["proposal"]["proposal_id"]
    assert result["proposal"]["shadow_graph"] == "ontology:tenant__shadow__proposal-1"

    with patch(
        "agent_utilities.knowledge_graph.ontology.evolution.discard_shadow",
        return_value={"dropped": True},
    ) as mock_discard:
        reviewed = evolution.review_ontology_proposal(
            None, None, proposal_id, approve=False, reviewer="bob"
        )

    mock_discard.assert_called_once_with(None, "ontology:tenant__shadow__proposal-1")
    assert reviewed["proposal"]["shadow_discard"] == {"dropped": True}


def test_approve_does_not_discard_the_shadow_graph():
    """An APPROVED (not yet promoted) proposal keeps its shadow -- only
    rejection (a dead end) or promotion (which discards it on success, see
    promote_ontology_proposal) tears it down."""
    with patch(
        "agent_utilities.knowledge_graph.ontology.evolution.materialize_shadow",
        return_value=("ontology:tenant__shadow__proposal-2", {"loaded_to_engine": True}),
    ):
        result = evolution.propose_ontology_change(
            None, None, PETS_TTL, iri="http://example.org/pets", source_type="text"
        )
    proposal_id = result["proposal"]["proposal_id"]

    with patch(
        "agent_utilities.knowledge_graph.ontology.evolution.discard_shadow"
    ) as mock_discard:
        reviewed = evolution.review_ontology_proposal(
            None, None, proposal_id, approve=True, reviewer="alice"
        )

    mock_discard.assert_not_called()
    assert "shadow_discard" not in reviewed["proposal"]


# ── promote — gated by the SAME action_policy decision point as every other
# evolution proposal; nothing auto-merges ──────────────────────────────────


def test_promote_ontology_proposal_default_never_auto():
    """SAFETY-CRITICAL: even a clean, reviewer-approved, additive proposal is
    held for the action_policy gate by default -- promotion is NEVER automatic."""
    result = evolution.propose_ontology_change(
        None, None, PETS_TTL, iri="http://example.org/pets", source_type="text"
    )
    proposal_id = result["proposal"]["proposal_id"]
    evolution.review_ontology_proposal(
        None, None, proposal_id, approve=True, reviewer="alice"
    )
    promoted = evolution.promote_ontology_proposal(None, None, proposal_id)
    assert promoted["status"] == "held"
    assert promoted["proposal"]["status"] in (
        evolution.STATUS_QUEUED_FOR_APPROVAL,
        evolution.STATUS_PENDING_REVIEW,
    )
    # The canonical ontology was NOT modified.
    assert OntologyLifecycle(engine=None).list_ontologies()["count"] == 0


def test_promote_unknown_proposal_errors():
    result = evolution.promote_ontology_proposal(None, None, "does-not-exist")
    assert "error" in result


def test_promote_rejected_proposal_is_refused():
    result = evolution.propose_ontology_change(
        None, None, PETS_TTL, iri="http://example.org/pets", source_type="text"
    )
    proposal_id = result["proposal"]["proposal_id"]
    evolution.review_ontology_proposal(
        None, None, proposal_id, approve=False, reviewer="alice", notes="not ready"
    )
    promoted = evolution.promote_ontology_proposal(None, None, proposal_id)
    assert "error" in promoted
    assert OntologyLifecycle(engine=None).list_ontologies()["count"] == 0


def test_promote_when_gate_allows_actually_activates_and_versions():
    """With the gate mocked to allow (simulating a granted approval), promotion
    DOES flip the proposal into the hosted/active ontology set via the SAME
    versioned OntologyLifecycle.update() path -- proving promotion is real,
    not merely gated."""
    result = evolution.propose_ontology_change(
        None, None, PETS_TTL, iri="http://example.org/pets", source_type="text"
    )
    proposal_id = result["proposal"]["proposal_id"]

    allow = ActionDecision(
        decision="allow",
        tier="auto",
        request=ActionRequest(kind="ontology_proposal_promotion", target=proposal_id),
        reason="test override",
    )

    class _StubPolicy:
        def decide(self, request):
            return allow

    with patch(
        "agent_utilities.orchestration.action_policy.get_action_policy",
        return_value=_StubPolicy(),
    ):
        promoted = evolution.promote_ontology_proposal(None, None, proposal_id)

    assert promoted["status"] == "ok"
    assert promoted["proposal"]["status"] == evolution.STATUS_PROMOTED
    assert promoted["proposal"]["active"] is True
    hosted = OntologyLifecycle(engine=None).list_ontologies()
    assert hosted["count"] == 1
    assert hosted["ontologies"][0]["version"] == "0.1.0"

    # Idempotent re-promotion of an already-promoted proposal is a safe no-op.
    again = evolution.promote_ontology_proposal(None, None, proposal_id)
    assert again["status"] == "ok"
    assert again.get("idempotent") is True


def test_rollback_requires_promoted_status():
    result = evolution.propose_ontology_change(
        None, None, PETS_TTL, iri="http://example.org/pets", source_type="text"
    )
    proposal_id = result["proposal"]["proposal_id"]
    rolled_back = evolution.rollback_ontology_proposal(None, None, proposal_id)
    assert "error" in rolled_back


def test_rollback_after_promote_reactivates_prior_version():
    _load_baseline()  # v1.0.0, active
    proposed = evolution.propose_ontology_change(
        None,
        None,
        PETS_TTL_ADDITIVE,
        iri="http://example.org/pets",
        source_type="text",
    )
    proposal_id = proposed["proposal"]["proposal_id"]

    allow = ActionDecision(
        decision="allow",
        tier="auto",
        request=ActionRequest(kind="ontology_proposal_promotion", target=proposal_id),
    )

    class _StubPolicy:
        def decide(self, request):
            return allow

    with patch(
        "agent_utilities.orchestration.action_policy.get_action_policy",
        return_value=_StubPolicy(),
    ):
        promoted = evolution.promote_ontology_proposal(None, None, proposal_id)
    assert promoted["proposal"]["version"] == "1.1.0"

    lc = OntologyLifecycle(engine=None)
    active = lc.list_ontologies(active_only=True)["ontologies"]
    assert active[0]["version"] == "1.1.0"

    rolled_back = evolution.rollback_ontology_proposal(None, None, proposal_id)
    assert rolled_back["status"] == "ok"
    assert rolled_back["proposal"]["status"] == evolution.STATUS_ROLLED_BACK

    active_after = lc.list_ontologies(active_only=True)["ontologies"]
    assert len(active_after) == 1
    assert active_after[0]["version"] == "1.0.0"


# ── negative results stay queryable (CONCEPT:AU-KG.ontology.negative-results-queryable) ─


def test_rejected_and_held_proposals_remain_queryable():
    rejected_src = evolution.propose_ontology_change(
        None, None, PETS_TTL, iri="http://example.org/pets", source_type="text"
    )
    rejected_id = rejected_src["proposal"]["proposal_id"]
    evolution.review_ontology_proposal(
        None, None, rejected_id, approve=False, reviewer="bob"
    )

    listed = evolution.list_proposals(None, None, status=evolution.STATUS_REJECTED)
    assert any(p["proposal_id"] == rejected_id for p in listed)
    fetched = evolution.get_proposal(None, None, rejected_id)
    assert fetched["status"] == evolution.STATUS_REJECTED
    assert fetched["decision"]["approved"] is False
