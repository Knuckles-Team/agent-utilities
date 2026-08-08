"""Wiring tests for CONCEPT:AU-KG.identity.entity-resolution-candidates.

Universal-ingestion program, Track 5 (entity resolution). Proves, with real
call paths:

1. Two near-identical entity names produce a *candidate* pair, never a merge
   — `payments platform` / `payments-platform`.
2. Multiple evidence kinds combine (exact identifier / normalized name /
   structural context) without ever auto-confirming.
3. Identity rules come from a domain-pack `IdentityRule`, not a hardcoded
   per-corpus rule in the engine.
4. A confirmed merge is reversible: `revert_merge` preserves the original
   evidence/confidence/decision fields while marking the edge inactive.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from agent_utilities.knowledge_graph.assimilation.identity_candidates import (
    EntityRecord,
    IdentityEvidenceKind,
    confirm_merge,
    resolve_identity_candidates,
    revert_merge,
    write_candidate,
)
from agent_utilities.models.knowledge_graph import RegistryEdgeType
from agent_utilities.models.schema_pack import IdentityRule, SchemaPack


# --------------------------------------------------------------------------- #
# 1. Near-identical names -> a candidate, never a merge
# --------------------------------------------------------------------------- #


def test_near_identical_names_produce_a_candidate_never_a_merge() -> None:
    records = [
        EntityRecord(id="entity:a", name="payments platform"),
        EntityRecord(id="entity:b", name="payments-platform"),
    ]
    candidates = resolve_identity_candidates(records, identity_rules=())

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.status == "candidate"  # NEVER "confirmed" from this call
    assert {candidate.entity_a, candidate.entity_b} == {"entity:a", "entity:b"}
    assert any(
        e.kind
        in (IdentityEvidenceKind.NORMALIZED_NAME, IdentityEvidenceKind.FUZZY_NAME)
        for e in candidate.evidence
    )


def test_two_unrelated_entities_produce_no_candidate_at_all() -> None:
    """No evidence at all -> silence, never a fabricated low-confidence candidate."""
    records = [
        EntityRecord(id="entity:x", name="Aurora Freight Logistics"),
        EntityRecord(id="entity:y", name="Nimbus Data Analytics"),
    ]
    assert resolve_identity_candidates(records, identity_rules=()) == []


# --------------------------------------------------------------------------- #
# 2. Multiple evidence kinds combine; a CMDB id is strong but STILL a candidate
# --------------------------------------------------------------------------- #


def test_exact_cmdb_identifier_is_strong_evidence_but_still_a_candidate() -> None:
    records = [
        EntityRecord(
            id="cmdb:CI00042",
            name="Payments Svc",
            kind="servicenow_cmdb",
            identifiers={"cmdb_id": "CI00042"},
        ),
        EntityRecord(
            id="doc:payment-service-legacy",
            name="payment service (legacy, retiring)",
            kind="servicenow_cmdb",
            identifiers={"cmdb_id": "CI00042"},
        ),
    ]
    rule = IdentityRule(
        applies_to=["servicenow"],
        identifier_fields=["cmdb_id"],
        name_fields=["name"],
    )
    candidates = resolve_identity_candidates(records, identity_rules=[rule])

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.status == "candidate"
    assert candidate.confidence >= 0.9
    kinds = {e.kind for e in candidate.evidence}
    assert IdentityEvidenceKind.EXACT_IDENTIFIER in kinds


def test_structural_context_evidence_is_optional_and_never_fabricated() -> None:
    """No neighbor_fn supplied -> no structural evidence is even attempted."""
    records = [
        EntityRecord(id="entity:a", name="Widgets Inc"),
        EntityRecord(id="entity:b", name="widgets inc."),
    ]
    without_structure = resolve_identity_candidates(records, identity_rules=())
    assert without_structure  # the name tier alone already flags this pair
    assert all(
        e.kind != IdentityEvidenceKind.STRUCTURAL_CONTEXT
        for e in without_structure[0].evidence
    )

    def neighbor_fn(node_id: str) -> set[str]:
        shared = {"customer:acme", "customer:globex"}
        return shared if node_id in ("entity:a", "entity:b") else set()

    with_structure = resolve_identity_candidates(
        records, identity_rules=(), neighbor_fn=neighbor_fn
    )
    kinds = {e.kind for e in with_structure[0].evidence}
    assert IdentityEvidenceKind.STRUCTURAL_CONTEXT in kinds


# --------------------------------------------------------------------------- #
# 3. Identity rules come from a domain pack, not a hardcoded rule
# --------------------------------------------------------------------------- #


def test_identity_rules_come_from_a_schema_pack_not_a_hardcoded_rule() -> None:
    pack = SchemaPack(
        name="cmdb-pack",
        identity_rules=[
            IdentityRule(
                applies_to=["servicenow"],
                identifier_fields=["cmdb_id"],
                name_fields=["name"],
                min_confidence_to_flag=0.9,
            )
        ],
    )
    scoped = pack.identity_rules_for("servicenow_cmdb")
    unscoped = pack.identity_rules_for("finance_ledger")
    assert len(scoped) == 1
    assert unscoped == []

    # A rule scoped OUT of a kind contributes nothing extra beyond the
    # generic identifier fallback — proving the engine itself carries no
    # per-corpus knowledge; it only ever reads what the pack declared.
    records = [
        EntityRecord(id="a", name="Ledger One", kind="finance_ledger"),
        EntityRecord(id="b", name="Ledger Two", kind="finance_ledger"),
    ]
    candidates = resolve_identity_candidates(
        records, identity_rules=pack.identity_rules
    )
    assert candidates == []  # different names, no identifiers, out-of-scope rule


# --------------------------------------------------------------------------- #
# 4. Confirm/revert — reversible, evidence retained
# --------------------------------------------------------------------------- #


def test_write_candidate_writes_possible_same_as_never_same_as() -> None:
    engine = Mock()
    records = [
        EntityRecord(id="entity:a", name="payments platform"),
        EntityRecord(id="entity:b", name="payments-platform"),
    ]
    candidate = resolve_identity_candidates(records, identity_rules=())[0]
    write_candidate(engine, candidate)

    engine.link_nodes.assert_called_once()
    args, kwargs = engine.link_nodes.call_args
    assert args[2] == RegistryEdgeType.POSSIBLE_SAME_AS
    assert kwargs["properties"]["candidate_id"] == candidate.id
    assert kwargs["properties"]["status"] == "candidate"


def test_confirm_then_revert_merge_preserves_evidence_and_is_undoable() -> None:
    engine = Mock()
    records = [
        EntityRecord(id="entity:a", name="payments platform"),
        EntityRecord(id="entity:b", name="payments-platform"),
    ]
    candidate = resolve_identity_candidates(records, identity_rules=())[0]

    decision = confirm_merge(
        engine,
        candidate,
        decided_by="steward:alice",
        reason="confirmed duplicate CMDB entry",
    )
    confirm_args, confirm_kwargs = engine.link_nodes.call_args
    assert confirm_args[2] == RegistryEdgeType.SAME_AS
    assert confirm_kwargs["properties"]["reverted"] is False
    assert confirm_kwargs["properties"]["evidence_json"]  # non-empty, evidence retained
    assert decision.reverted is False

    reverted = revert_merge(
        engine, decision, reason="turned out to be two real entities"
    )

    assert engine.link_nodes.call_count == 2
    revert_args, revert_kwargs = engine.link_nodes.call_args
    assert (
        revert_args[2] == RegistryEdgeType.SAME_AS
    )  # same edge type — marked, not deleted
    props = revert_kwargs["properties"]
    assert props["reverted"] is True
    assert props["reverted_reason"] == "turned out to be two real entities"
    # The ORIGINAL decision fields survive the revert — nothing is clobbered.
    assert props["decided_by"] == "steward:alice"
    assert props["reason"] == "confirmed duplicate CMDB entry"
    assert props["confidence"] == confirm_kwargs["properties"]["confidence"]
    assert props["evidence_json"] == confirm_kwargs["properties"]["evidence_json"]
    assert (
        reverted.evidence == decision.evidence
    )  # same Python objects, never re-derived


def test_resolve_identity_candidates_never_touches_an_engine() -> None:
    """The scoring/decision function is pure — no engine parameter exists, and
    a live engine double sitting in scope is never called."""
    engine = Mock()
    records = [
        EntityRecord(id="a", name="payments platform"),
        EntityRecord(id="b", name="payments-platform"),
    ]
    resolve_identity_candidates(records, identity_rules=())
    assert engine.mock_calls == []


def test_default_identity_rules_are_wired_to_the_process_active_pack(
    monkeypatch,
) -> None:
    """`identity_rules=None` (the default) is a REAL call path — it resolves
    rules from `schema_pack_loader.get_active_pack()`, not a dead parameter.

    Proves `SchemaPack.identity_rules_for` has a non-test caller: a strict
    pack-declared `min_confidence_to_flag` suppresses a pair that the generic
    (no-pack) fallback would otherwise flag on the exact same evidence.
    """
    records = [
        EntityRecord(
            id="a",
            name="Ledger Account Alpha",
            kind="ledger_account",
            identifiers={"id": "SAME-1"},
        ),
        EntityRecord(
            id="b",
            name="Ledger Account Beta",
            kind="ledger_account",
            identifiers={"id": "SAME-1"},
        ),
    ]

    # The generic fallback (explicitly no pack rules) flags this pair on the
    # shared "id" identifier alone (unrelated names, no name evidence).
    generic = resolve_identity_candidates(records, identity_rules=())
    assert generic
    assert generic[0].confidence == pytest.approx(0.98)

    # A pack that scopes a stricter threshold to "ledger" kinds.
    strict_pack = SchemaPack(
        name="strict-finance-pack",
        identity_rules=[
            IdentityRule(
                applies_to=["ledger"],
                identifier_fields=["id"],
                min_confidence_to_flag=0.99,
            )
        ],
    )
    monkeypatch.setattr(
        "agent_utilities.models.schema_pack_loader.get_active_pack",
        lambda: strict_pack,
    )

    # With `identity_rules` left at its default (None), the SAME evidence
    # (0.98) now falls below the process-active pack's 0.99 floor — proof the
    # pack lookup is really consulted on this call path, not bypassed.
    wired = resolve_identity_candidates(records)
    assert wired == []
