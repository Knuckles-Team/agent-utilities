"""Production promotion-governance validator (CONCEPT:AU-AHE.harness.promotion-governance-validator).

Covers the four governance rules (MergePolicy thresholds, SHACL shapes,
recorded regression-gate verdicts, constitution forbid rules) pass/fail paths,
the merger integration (GovernedAutoMerger now builds the production validator
by DEFAULT when an engine exists), and the regression-gate verdict recording
added to the failure analyzer's gate.

@pytest.mark.concept("AU-AHE.harness.promotion-governance-validator")
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.research.auto_merge import (
    GovernedAutoMerger,
    MergePolicy,
)
from agent_utilities.knowledge_graph.research.promotion_governance import (
    PromotionGovernanceValidator,
)
from tests.golden_loop_proposal_fixtures import strong_team as _strong_team
from tests.golden_loop_proposal_fixtures import weak_team as _weak_team

pytestmark = pytest.mark.concept("AU-AHE.harness.promotion-governance-validator")


class _Engine:
    """Fake engine: seedable RegressionGateResult + governance-rule rows."""

    def __init__(self, gate_rows=None, rule_rows=None):
        self.gate_rows = gate_rows or []
        self.rule_rows = rule_rows or []
        self.nodes = {}
        self.backend = object()

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def shacl_validate_committed(self, _data_graph_turtle):
        # Governance shapes are EG's committed GraphSchema authority (see
        # PromotionGovernanceValidator's module docstring); this fake always
        # conforms so tests focus on the OTHER governance rules (merge
        # policy, regression gate, constitution) unless they bind their own
        # report, same shape as tests/ontology/test_shacl_gate.py's fakes.
        return SimpleNamespace(conforms=True, results=[])

    def query_cypher(self, query, params=None):
        if "RegressionGateResult" in query:
            pid = (params or {}).get("pid")
            return [r for r in self.gate_rows if r.get("proposal_id") in (None, pid)]
        if "ConstitutionRule" in query:
            return self.rule_rows
        if "governance_rule" in query:
            # ActionPolicy._kg_rules()'s own query shape: {"r": {...}} rows
            # for every governance_rule node with scope='action_policy',
            # same contract tests/unit/fleet_autonomy_fakes.py's FakeEngine
            # implements for test_auto_merge_action_policy.py.
            return [
                {"r": dict(n)}
                for n in self.nodes.values()
                if n.get("type") == "governance_rule"
                and n.get("scope") == "action_policy"
            ]
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

    @pytest.mark.spec("AU-SEMANTIC-R004", "AU-SEMANTIC-R007")
    def test_missing_goal_fails_even_with_explicit_score(self):
        v = PromotionGovernanceValidator(None, policy=_policy())
        check = v._check_merge_policy({"name": "x", "quality_score": 0.99})
        assert check.passed is False
        assert "goal" in check.reason


# ---------------------------------------------------------------------------
# Rule: SHACL governance shapes
# ---------------------------------------------------------------------------


class TestShaclRule:
    """EH-470/EH-473: ``_check_shacl`` validates through the engine's
    committed ``shacl_validate_committed`` surface only — never local
    ``pyshacl`` (see ``PromotionGovernanceValidator._validate_shacl_spec``).
    These bind a fake report instead of ``importorskip("pyshacl")``, the
    same shape ``tests/ontology/test_shacl_gate.py`` uses."""

    @pytest.mark.spec("AU-SEMANTIC-R004", "AU-SEMANTIC-R007")
    def test_agent_without_name_violates_agent_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # :AgentShape requires a name — a nameless Agent proposal must fail.
        eng = _Engine()
        monkeypatch.setattr(
            eng,
            "shacl_validate_committed",
            lambda _data: SimpleNamespace(
                conforms=False,
                results=[SimpleNamespace(message="name is required")],
            ),
        )
        v = PromotionGovernanceValidator(eng, policy=_policy())
        check = v._check_shacl({"type": "Agent", "goal": "do things"})
        assert check.passed is False

    @pytest.mark.spec("AU-SEMANTIC-R004", "AU-SEMANTIC-R007")
    def test_named_agent_conforms(self) -> None:
        v = PromotionGovernanceValidator(_Engine(), policy=_policy())
        check = v._check_shacl({"type": "Agent", "name": "researcher", "goal": "g"})
        assert check.passed is True


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


def _consider_strong_team(engine) -> tuple[list, object]:
    """Consider ``_strong_team()`` with governance required, recording
    promotions. Shared by the real-ActionPolicy merger tests below, which
    differ only in whether ``engine`` carries a tier-relaxing
    ``governance_rule`` override."""
    promoted: list = []
    merger = GovernedAutoMerger(
        engine=engine,
        policy=_policy(require_governance_valid=True),
        promoter=lambda spec: promoted.append(spec) or True,
    )
    return promoted, merger.consider(_strong_team())


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
        """A clean, strong proposal promotes end to end through the REAL,
        default-resolved ActionPolicy (no injected fake) -- but only once
        that policy actually resolves to a receipt-backed ``approve``.

        The shipped default tier for an unconfigured actor is
        ``approval_required`` (-> a ``hold`` disposition, see
        ``test_default_tier_holds_and_does_not_promote`` below), which the
        shared promotion contract (``artifact_promotion.promote()``'s
        ``PromotionOutcome.approved``, mirrored by
        ``GovernedAutoMerger._gate_by_action_policy``) correctly does NOT
        activate. Relax the tier the same way production does -- a
        KG-stored ``governance_rule`` override with ``scope='action_policy'``
        -- so this test exercises a genuinely approved decision through the
        real path, not a loosened gate.
        """
        engine = _Engine()
        engine.add_node(
            "rule:promo-auto",
            "governance_rule",
            properties={
                "scope": "action_policy",
                "kind": "merge_promotion",
                "target": "*",
                "tier": "auto",
            },
        )
        promoted, ev = _consider_strong_team(engine)
        assert ev.governance_valid is True
        assert ev.action_decision["decision"] == "approve"
        assert ev.merged is True
        assert len(promoted) == 1

    def test_default_tier_holds_and_does_not_promote(self):
        """The shipped default tier (``approval_required``, unconfigured
        actor) resolves to a ``hold`` disposition -- NOT an approval -- so
        the lifecycle flip must not proceed, proving ``hold`` cannot
        activate a promotion even for an otherwise-clean, governance-valid
        proposal."""
        promoted, ev = _consider_strong_team(_Engine())
        assert ev.governance_valid is True
        assert ev.action_decision["decision"] == "hold"
        assert ev.merged is False
        assert promoted == []

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

        def graph_writer(entities, relationships):
            assert relationships == []
            for entity in entities:
                row = dict(entity)
                node_id = row.pop("id")
                # D-W2X-1: _record_gate_result builds each entity with the
                # canonical `node_type` key (never the retired bare `type`,
                # see D-W2-4). This fixture's own `graph_writer` popped
                # `type` instead, which KeyError'd and was silently
                # swallowed by _record_gate_result's `except Exception`
                # ("recording must never gate the gate") — so the
                # RegressionGateResult node was never actually written, not
                # because of any production-code defect.
                node_type = row.pop("node_type")
                eng.add_node(node_id, node_type, properties=row)
            return {"status": "success"}

        analyzer = FailureAnalyzer(eng, trace_backend=None, graph_writer=graph_writer)
        check = analyzer.make_regression_check(
            [{"workflow": "wf", "occurrences": 2, "signature": "s", "id": "g"}]
        )
        assert check(_strong_team()) is True
        recorded = [
            n for n in eng.nodes.values() if n["type"] == "RegressionGateResult"
        ]
        assert len(recorded) == 1
        assert recorded[0]["result"] == "pass"
        assert recorded[0]["proposal_id"].startswith("pref_proposal_")
