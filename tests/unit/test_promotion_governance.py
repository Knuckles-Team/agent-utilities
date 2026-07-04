"""Production promotion-governance validator (CONCEPT:AU-AHE.harness.promotion-governance-validator).

Covers the four governance rules (MergePolicy thresholds, SHACL shapes,
recorded regression-gate verdicts, constitution forbid rules) pass/fail paths,
the merger integration (GovernedAutoMerger now builds the production validator
by DEFAULT when an engine exists), and the regression-gate verdict recording
added to the failure analyzer's gate.

@pytest.mark.concept("AU-AHE.harness.promotion-governance-validator")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.orchestration import TeamSpec
from agent_utilities.knowledge_graph.research.auto_merge import (
    GovernedAutoMerger,
    MergePolicy,
)
from agent_utilities.knowledge_graph.research.promotion_governance import (
    PromotionGovernanceValidator,
)

pytestmark = pytest.mark.concept("AU-AHE.harness.promotion-governance-validator")


def _strong_team() -> TeamSpec:
    return TeamSpec(
        name="Resolver Team",
        goal="Address open KG topics about retrieval quality",
        lead="Lead",
        members=["Researcher", "Validator"],
        description="A complete, well-formed team proposal.",
    )


def _weak_team() -> TeamSpec:
    return TeamSpec(name="bare", goal="", lead="", members=[])


class _Engine:
    """Fake engine: seedable RegressionGateResult + governance-rule rows."""

    def __init__(self, gate_rows=None, rule_rows=None):
        self.gate_rows = gate_rows or []
        self.rule_rows = rule_rows or []
        self.nodes = {}
        self.backend = object()

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def query_cypher(self, query, params=None):
        if "RegressionGateResult" in query:
            pid = (params or {}).get("pid")
            return [r for r in self.gate_rows if r.get("proposal_id") in (None, pid)]
        if "ConstitutionRule" in query:
            return self.rule_rows
        return []


def _policy(**kw) -> MergePolicy:
    return MergePolicy(enabled=True, **kw)


# ---------------------------------------------------------------------------
# Rule: MergePolicy thresholds
# ---------------------------------------------------------------------------


class TestMergePolicyRule:
    def test_strong_proposal_passes(self):
        v = PromotionGovernanceValidator(None, policy=_policy())
        check = v._check_merge_policy(_strong_team())
        assert check.passed is True

    def test_low_quality_fails(self):
        v = PromotionGovernanceValidator(None, policy=_policy())
        check = v._check_merge_policy(_weak_team())
        assert check.passed is False
        assert "quality" in check.reason

    def test_missing_goal_fails_even_with_explicit_score(self):
        v = PromotionGovernanceValidator(None, policy=_policy())
        check = v._check_merge_policy({"name": "x", "quality_score": 0.99})
        assert check.passed is False
        assert "goal" in check.reason


# ---------------------------------------------------------------------------
# Rule: SHACL governance shapes
# ---------------------------------------------------------------------------


class TestShaclRule:
    def test_team_spec_conforms_vacuously(self):
        # No :Team shape exists in governance.shapes.ttl ⇒ conforms.
        v = PromotionGovernanceValidator(None, policy=_policy())
        check = v._check_shacl(_strong_team())
        assert check.passed is True

    def test_agent_without_name_violates_agent_shape(self):
        pytest.importorskip("pyshacl")
        v = PromotionGovernanceValidator(None, policy=_policy())
        # :AgentShape requires a name — a nameless Agent proposal must fail.
        check = v._check_shacl({"type": "Agent", "goal": "do things"})
        assert check.passed is False

    def test_named_agent_conforms(self):
        pytest.importorskip("pyshacl")
        v = PromotionGovernanceValidator(None, policy=_policy())
        check = v._check_shacl({"type": "Agent", "name": "researcher", "goal": "g"})
        assert check.passed is True

    def test_missing_shapes_file_not_applicable(self):
        v = PromotionGovernanceValidator(
            None, policy=_policy(), shapes_path="/nonexistent/shapes.ttl"
        )
        check = v._check_shacl(_strong_team())
        assert check.passed is True
        assert "not found" in check.reason


# ---------------------------------------------------------------------------
# Rule: recorded regression-gate verdict
# ---------------------------------------------------------------------------


class TestRegressionGateRule:
    def test_recorded_hold_blocks(self):
        eng = _Engine(gate_rows=[{"result": "hold", "timestamp": "2026-01-01"}])
        v = PromotionGovernanceValidator(eng, policy=_policy())
        check = v._check_regression_gate(_strong_team(), "proposal:Resolver Team")
        assert check.passed is False
        assert "hold" in check.reason

    def test_latest_recorded_pass_allows(self):
        eng = _Engine(
            gate_rows=[
                {"result": "hold", "timestamp": "2026-01-01"},
                {"result": "pass", "timestamp": "2026-01-02"},
            ]
        )
        v = PromotionGovernanceValidator(eng, policy=_policy())
        check = v._check_regression_gate(_strong_team(), "proposal:Resolver Team")
        assert check.passed is True

    def test_no_record_defers_to_live_check(self):
        v = PromotionGovernanceValidator(_Engine(), policy=_policy())
        check = v._check_regression_gate(_strong_team(), "proposal:Resolver Team")
        assert check.passed is True
        assert "no recorded" in check.reason


# ---------------------------------------------------------------------------
# Rule: constitution forbid rules
# ---------------------------------------------------------------------------


class TestConstitutionRule:
    def test_matching_forbid_rule_blocks(self):
        eng = _Engine(
            rule_rows=[
                {
                    "id": "rule:1",
                    "kind": "forbid",
                    "target": "retrieval",
                    "active": True,
                }
            ]
        )
        v = PromotionGovernanceValidator(eng, policy=_policy())
        check = v._check_constitution(_strong_team())
        assert check.passed is False
        assert "rule:1" in check.reason

    def test_inactive_or_unrelated_rules_pass(self):
        eng = _Engine(
            rule_rows=[
                {"id": "r1", "kind": "forbid", "target": "retrieval", "active": False},
                {"id": "r2", "kind": "forbid", "target": "blockchain", "active": True},
                {"id": "r3", "kind": "allow", "target": "retrieval", "active": True},
            ]
        )
        v = PromotionGovernanceValidator(eng, policy=_policy())
        assert v._check_constitution(_strong_team()).passed is True

    def test_unqueryable_rules_not_applicable(self):
        class _NoQuery:
            pass

        v = PromotionGovernanceValidator(_NoQuery(), policy=_policy())
        assert v._check_constitution(_strong_team()).passed is True


# ---------------------------------------------------------------------------
# Full verdict + merger integration
# ---------------------------------------------------------------------------


class TestVerdictAndMergerIntegration:
    def test_full_verdict_valid_for_clean_strong_proposal(self):
        v = PromotionGovernanceValidator(_Engine(), policy=_policy())
        verdict = v.validate(_strong_team())
        assert verdict.valid is True
        assert {c.name for c in verdict.checks} == {
            "merge_policy",
            "shacl",
            "regression_gate",
            "capability_ratchet",
            "constitution",
        }

    def test_merger_builds_production_validator_by_default(self):
        merger = GovernedAutoMerger(engine=_Engine(), policy=_policy())
        assert isinstance(merger._governance_validator, PromotionGovernanceValidator)

    def test_merger_without_engine_keeps_no_validator(self):
        merger = GovernedAutoMerger(engine=None, policy=_policy())
        assert merger._governance_validator is None

    def test_explicit_validator_still_wins(self):
        sentinel = lambda spec: True  # noqa: E731
        merger = GovernedAutoMerger(
            engine=_Engine(), policy=_policy(), governance_validator=sentinel
        )
        assert merger._governance_validator is sentinel

    def test_governed_merge_with_real_validator_promotes_clean_proposal(self):
        promoted = []
        merger = GovernedAutoMerger(
            engine=_Engine(),
            policy=_policy(require_governance_valid=True),
            promoter=lambda spec: promoted.append(spec) or True,
        )
        ev = merger.consider(_strong_team())
        assert ev.governance_valid is True
        assert ev.merged is True
        assert len(promoted) == 1

    def test_recorded_gate_hold_blocks_governed_merge(self):
        # TeamSpec mints its own id ("team:resolver-team") — record against it.
        eng = _Engine(
            gate_rows=[
                {
                    "proposal_id": str(_strong_team().id),
                    "result": "hold",
                    "timestamp": "2026-01-01",
                }
            ]
        )
        merger = GovernedAutoMerger(
            engine=eng,
            policy=_policy(require_governance_valid=True),
            promoter=lambda spec: True,
        )
        ev = merger.consider(_strong_team())
        assert ev.merged is False
        assert "governance/SHACL invalid" in ev.failures

    def test_constitution_forbid_blocks_governed_merge(self):
        eng = _Engine(
            rule_rows=[
                {
                    "id": "rule:ban",
                    "kind": "forbid",
                    "target": "retrieval",
                    "active": True,
                }
            ]
        )
        merger = GovernedAutoMerger(
            engine=eng,
            policy=_policy(require_governance_valid=True),
            promoter=lambda spec: True,
        )
        ev = merger.consider(_strong_team())
        assert ev.merged is False


# ---------------------------------------------------------------------------
# Gate verdicts are RECORDED (failure analyzer side)
# ---------------------------------------------------------------------------


class TestGateRecording:
    def test_regression_check_records_pass_verdict(self):
        from agent_utilities.knowledge_graph.adaptation.failure_analyzer import (
            FailureAnalyzer,
        )

        eng = _Engine()
        analyzer = FailureAnalyzer(eng, trace_backend=None)
        check = analyzer.make_regression_check(
            [{"workflow": "wf", "occurrences": 2, "signature": "s", "id": "g"}]
        )
        assert check(_strong_team()) is True
        recorded = [
            n for n in eng.nodes.values() if n["type"] == "RegressionGateResult"
        ]
        assert len(recorded) == 1
        assert recorded[0]["result"] == "pass"
        assert recorded[0]["proposal_id"] == str(_strong_team().id)
