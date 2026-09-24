"""Characterization tests for ``PromotionGovernanceValidator._check_constitution``
(CX-AU-09); the ``_check_shacl`` half was retired with local SHACL (EH-380).

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


# _check_shacl characterizations were removed (EH-380). They pinned the
# retired local-shapes contract (a shapes_path, pyshacl, and "not applicable"
# passes), which 43197d7c6 replaced with EG's committed-GraphSchema authority
# (``engine.shacl_validate_committed``, fail closed when absent). The current
# contract is covered by tests/unit/test_promotion_governance.py::TestShaclRule.


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
