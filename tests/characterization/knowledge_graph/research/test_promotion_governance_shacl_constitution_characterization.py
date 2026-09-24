"""Characterization tests for ``PromotionGovernanceValidator._check_shacl`` and
``._check_constitution`` (CX-AU-09).

CCN at time of writing: ``_check_shacl`` 12, ``_check_constitution`` 12
(``agent_utilities/knowledge_graph/research/promotion_governance.py``). Both
are already exercised directly (as private methods) by the pre-existing
tests/unit/test_promotion_governance.py in this repo -- that file establishes
the convention this one follows.

Per the two-commit discipline, this file must be added and pass GREEN
against the UNMODIFIED promotion_governance.py before any refactor commit,
and must not change during the refactor commit that follows.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent_utilities.knowledge_graph.enrichment.orchestration import TeamSpec
from agent_utilities.knowledge_graph.research.auto_merge import MergePolicy
from agent_utilities.knowledge_graph.research.promotion_governance import (
    PromotionGovernanceValidator,
)


def _strong_team() -> TeamSpec:
    return TeamSpec(
        name="Resolver Team",
        goal="Address open KG topics about retrieval quality",
        lead="Lead",
        members=["Researcher", "Validator"],
        description="A complete, well-formed team proposal.",
    )


def _policy(**kw) -> MergePolicy:
    return MergePolicy(enabled=True, **kw)


class _RuleEngine:
    def __init__(self, rule_rows=None, raise_on_query=False):
        self.rule_rows = rule_rows or []
        self._raise = raise_on_query

    def query_cypher(self, query, params=None):
        if self._raise:
            raise RuntimeError("kg unavailable")
        return self.rule_rows


# ─────────────────────────────────────────────────────────────────────────
# _check_shacl
# ─────────────────────────────────────────────────────────────────────────


class _ShaclEngine:
    """Committed-SHACL stand-in answering with seeded violation messages."""

    def __init__(self, violations=(), raises=None):
        self.violations = list(violations)
        self.raises = raises

    def shacl_validate_committed(self, data_graph="", *, data_triples=()):
        if self.raises is not None:
            raise self.raises
        results = [SimpleNamespace(message=message) for message in self.violations]
        return SimpleNamespace(conforms=not results, results=results)


def test_shacl_without_committed_authority_holds() -> None:
    v = PromotionGovernanceValidator(None, policy=_policy())
    check = v._check_shacl(_strong_team())
    assert check.passed is False
    assert check.reason == "committed EG SHACL authority unavailable"
    assert check.name == "shacl"


def test_shacl_conforming_report_passes() -> None:
    v = PromotionGovernanceValidator(_ShaclEngine(), policy=_policy())
    check = v._check_shacl(_strong_team())
    assert check.passed is True
    assert check.reason == "conforms"


def test_shacl_violation_messages_join_up_to_three() -> None:
    # The first three violation messages appear in the joined reason; the cap is
    # [:3], not [:1] or unlimited.
    messages = ["no context", "no decision", "no authority", "fourth"]
    v = PromotionGovernanceValidator(_ShaclEngine(messages), policy=_policy())
    check = v._check_shacl({"type": "ArchitectureDecisionRecord", "name": "x"})
    assert check.passed is False
    assert check.reason == "no context; no decision; no authority"


def test_shacl_exception_during_validation_holds_not_passes() -> None:
    # OBSERVED: any exception anywhere in the SHACL path degrades to a FAILING
    # check (cannot prove conformance -> hold).
    engine = _ShaclEngine(raises=RuntimeError("engine exploded"))
    v = PromotionGovernanceValidator(engine, policy=_policy())
    check = v._check_shacl(_strong_team())
    assert check.passed is False
    assert "validation error" in check.reason


def test_shacl_adr_without_its_fields_violates_three_committed_constraints(
    engine_graph,
) -> None:
    v = PromotionGovernanceValidator(engine_graph, policy=_policy())
    check = v._check_shacl({"type": "ArchitectureDecisionRecord", "name": "x"})
    assert check.passed is False
    assert "context" in check.reason
    assert "decision" in check.reason
    assert "authority" in check.reason


# ─────────────────────────────────────────────────────────────────────────
# _check_constitution
# ─────────────────────────────────────────────────────────────────────────


def test_constitution_no_engine_is_not_applicable() -> None:
    v = PromotionGovernanceValidator(None, policy=_policy())
    check = v._check_constitution(_strong_team())
    assert check.passed is True
    assert "no engine" in check.reason


def test_constitution_unqueryable_engine_is_not_applicable() -> None:
    class _NoQuery:
        pass

    v = PromotionGovernanceValidator(_NoQuery(), policy=_policy())
    check = v._check_constitution(_strong_team())
    assert check.passed is True
    assert "not queryable" in check.reason


def test_constitution_matching_forbid_rule_blocks_with_id_and_target() -> None:
    eng = _RuleEngine(
        rule_rows=[
            {"id": "rule:1", "kind": "forbid", "target": "retrieval", "active": True}
        ]
    )
    v = PromotionGovernanceValidator(eng, policy=_policy())
    check = v._check_constitution(_strong_team())
    assert check.passed is False
    assert "rule:1" in check.reason
    assert "retrieval" in check.reason


def test_constitution_inactive_rule_is_skipped() -> None:
    eng = _RuleEngine(
        rule_rows=[
            {"id": "r1", "kind": "forbid", "target": "retrieval", "active": False}
        ]
    )
    v = PromotionGovernanceValidator(eng, policy=_policy())
    assert v._check_constitution(_strong_team()).passed is True


def test_constitution_non_forbid_kind_is_skipped_even_if_target_matches() -> None:
    eng = _RuleEngine(
        rule_rows=[{"id": "r1", "kind": "allow", "target": "retrieval", "active": True}]
    )
    v = PromotionGovernanceValidator(eng, policy=_policy())
    assert v._check_constitution(_strong_team()).passed is True


def test_constitution_non_dict_row_is_skipped() -> None:
    eng = _RuleEngine(rule_rows=["not-a-dict"])
    v = PromotionGovernanceValidator(eng, policy=_policy())
    assert v._check_constitution(_strong_team()).passed is True


def test_constitution_kind_and_target_matching_are_case_insensitive() -> None:
    eng = _RuleEngine(
        rule_rows=[
            {"id": "r1", "kind": "FORBID", "target": "RETRIEVAL", "active": True}
        ]
    )
    v = PromotionGovernanceValidator(eng, policy=_policy())
    assert v._check_constitution(_strong_team()).passed is False


def test_constitution_second_forbid_kind_synonym_also_blocks() -> None:
    # _FORBID_KINDS includes "prohibit" as well as "forbid".
    eng = _RuleEngine(
        rule_rows=[
            {"id": "r1", "kind": "prohibit", "target": "retrieval", "active": True}
        ]
    )
    v = PromotionGovernanceValidator(eng, policy=_policy())
    assert v._check_constitution(_strong_team()).passed is False


def test_constitution_query_exception_is_not_applicable_not_a_match() -> None:
    eng = _RuleEngine(raise_on_query=True)
    v = PromotionGovernanceValidator(eng, policy=_policy())
    check = v._check_constitution(_strong_team())
    assert check.passed is True
    assert "not queryable" in check.reason


def test_constitution_first_matching_rule_wins_message() -> None:
    eng = _RuleEngine(
        rule_rows=[
            {"id": "r1", "kind": "forbid", "target": "retrieval", "active": True},
            {"id": "r2", "kind": "forbid", "target": "retrieval", "active": True},
        ]
    )
    v = PromotionGovernanceValidator(eng, policy=_policy())
    check = v._check_constitution(_strong_team())
    assert check.passed is False
    assert "r1" in check.reason
    assert "r2" not in check.reason
