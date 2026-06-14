"""GovernedAutoMerger's promotion decision consults the OS-5.24 ActionPolicy.

The AHE-3.20 → ActionPolicy adoption (previously a noted follow-up): before the
merger flips a proposal's lifecycle (proposal→active), it decides the reserved
``merge_promotion`` kind for the proposal's id. Decision mapping:

- ``deny`` → promotion blocked, recorded on the evaluation + audit trail;
- ``queue_approval`` → the KG-internal lifecycle flip proceeds while the
  real-world materialization stays approval-gated — the SAME (deduped)
  ``ActionApproval`` the AHE-3.21 publication step queues/consumes;
- ``allow`` / ``allow_notify`` → proceed (the policy emits the notification).

The outer ``KG_GOLDEN_AUTO_MERGE`` gate is unchanged (default False).

@pytest.mark.concept("AHE-3.20")
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "unit"))

from fleet_autonomy_fakes import FakeEngine  # noqa: E402

from agent_utilities.knowledge_graph.research.auto_merge import (  # noqa: E402
    GovernedAutoMerger,
    MergePolicy,
)
from agent_utilities.orchestration.action_policy import (  # noqa: E402
    ActionDecision,
    ActionPolicy,
    ActionRequest,
)

pytestmark = pytest.mark.concept("AHE-3.20")


class _FakePolicy:
    """Recording ActionPolicy double returning a canned decision."""

    def __init__(self, decision: str, *, reason: str = "r", approval_id=None):
        self._decision = decision
        self._reason = reason
        self._approval_id = approval_id
        self.requests: list[ActionRequest] = []

    def decide(self, request: ActionRequest) -> ActionDecision:
        self.requests.append(request)
        return ActionDecision(
            decision=self._decision,
            tier="approval_required",
            request=request,
            reason=self._reason,
            approval_id=self._approval_id,
        )


def _spec() -> dict:
    return {
        "id": "proposal:policy-1",
        "name": "Wire ranking synergy",
        "goal": "Blend synergy signals into rank",
        "lead": "Lead",
        "members": ["Researcher", "Validator"],
        "quality_score": 0.95,
    }


def _merger(policy_double, *, enabled=True, promoted=None, engine=None):
    promoted = promoted if promoted is not None else []
    return GovernedAutoMerger(
        engine,
        policy=MergePolicy(enabled=enabled, require_governance_valid=False),
        promoter=lambda spec: promoted.append(spec) or True,
        action_policy=policy_double,
    )


# ---------------------------------------------------------------------------
# Decision mapping (fake policy)
# ---------------------------------------------------------------------------


class TestDecisionMapping:
    def test_deny_blocks_promotion_and_is_recorded(self):
        promoted: list = []
        policy = _FakePolicy("deny", reason="forbidden by policy")
        merger = _merger(policy, promoted=promoted)
        ev = merger.consider(_spec())
        assert ev.merged is False
        assert promoted == [], "deny must block the lifecycle flip"
        assert ev.publication is None, "a blocked promotion never publishes"
        assert ev.action_decision["decision"] == "deny"
        assert "blocked by action policy (merge_promotion)" in ev.reason
        # The block is on the audit trail, not just the return value.
        records = merger.audit.query(action="loop_engine.auto_merge")
        assert records and records[0].details["action_decision"] == "deny"
        assert records[0].details["merged"] is False

    def test_queue_approval_keeps_ahe321_flow(self):
        promoted: list = []
        policy = _FakePolicy(
            "queue_approval",
            reason="tier requires human approval",
            approval_id="action_approval:abc",
        )
        merger = _merger(policy, promoted=promoted)
        ev = merger.consider(_spec())
        # The KG-internal flip proceeds; the real-world publication is what
        # stays approval-gated (AHE-3.21 semantics, same ActionApproval).
        assert ev.merged is True
        assert promoted
        assert ev.action_decision == {
            "decision": "queue_approval",
            "reason": "tier requires human approval",
            "approval_id": "action_approval:abc",
        }

    def test_allow_proceeds_with_merge_promotion_request(self):
        promoted: list = []
        policy = _FakePolicy("allow")
        merger = _merger(policy, promoted=promoted)
        ev = merger.consider(_spec())
        assert ev.merged is True
        assert promoted
        assert ev.action_decision["decision"] == "allow"
        (request,) = policy.requests
        assert request.kind == "merge_promotion"
        assert request.target == "proposal:policy-1"
        assert request.source == "loop_engine"

    def test_allow_notify_proceeds(self):
        policy = _FakePolicy("allow_notify")
        ev = _merger(policy).consider(_spec())
        assert ev.merged is True
        assert ev.action_decision["decision"] == "allow_notify"

    def test_disabled_outer_gate_never_consults(self):
        """KG_GOLDEN_AUTO_MERGE stays the unchanged outer gate (default off)."""
        policy = _FakePolicy("allow")
        ev = _merger(policy, enabled=False).consider(_spec())
        assert ev.merged is False
        assert policy.requests == [], "no promotion attempt → no policy consult"
        assert ev.action_decision is None

    def test_ineligible_proposal_never_consults(self):
        policy = _FakePolicy("allow")
        merger = _merger(policy)
        ev = merger.consider({"name": "bare", "quality_score": 0.1})
        assert ev.merged is False
        assert policy.requests == []

    def test_outer_gate_default_is_off(self):
        assert MergePolicy.from_env(None).enabled is False

    def test_policy_consult_failure_fails_closed(self):
        class _Boom:
            def decide(self, request):
                raise RuntimeError("policy backend down")

        promoted: list = []
        ev = _merger(_Boom(), promoted=promoted).consider(_spec())
        assert ev.merged is False
        assert promoted == []
        assert ev.action_decision["decision"] == "deny"
        assert "fail closed" in ev.action_decision["reason"]


# ---------------------------------------------------------------------------
# Real ActionPolicy (policy file / shipped default / KG override)
# ---------------------------------------------------------------------------


class TestRealActionPolicy:
    def test_forbidden_tier_from_policy_file_blocks(self, tmp_path):
        policy_file = tmp_path / "policy.yml"
        policy_file.write_text(
            "version: 1\n"
            "defaults: {tier: approval_required}\n"
            "rules:\n"
            "  - {kind: merge_promotion, target: '*', tier: forbidden}\n",
            encoding="utf-8",
        )
        engine = FakeEngine()
        promoted: list = []
        merger = _merger(
            ActionPolicy(engine, policy_path=policy_file),
            promoted=promoted,
            engine=engine,
        )
        ev = merger.consider(_spec())
        assert ev.merged is False
        assert promoted == []
        assert ev.action_decision["decision"] == "deny"
        assert "forbidden" in ev.action_decision["reason"]
        # The decision is on the durable KG ledger.
        decisions = engine.by_type("ActionDecision")
        assert decisions and decisions[0]["kind"] == "merge_promotion"

    def test_shipped_default_queues_one_shared_approval(self):
        """Default tier queues; promotion + publication share ONE approval."""
        engine = FakeEngine()
        merger = GovernedAutoMerger(
            engine,
            policy=MergePolicy(enabled=True, require_governance_valid=False),
            promoter=lambda spec: True,
        )
        ev = merger.consider(_spec())
        assert ev.merged is True
        assert ev.action_decision["decision"] == "queue_approval"
        assert ev.publication["status"] == "approval_queued"
        approvals = engine.by_type("ActionApproval")
        assert len(approvals) == 1, "merger + publisher must dedup to one approval"
        assert approvals[0]["id"] == ev.action_decision["approval_id"]
        assert approvals[0]["id"] == ev.publication["approval_id"]
        assert approvals[0]["target"] == "proposal:policy-1"

    def test_kg_rule_can_relax_promotion_to_auto(self):
        engine = FakeEngine()
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
        merger = GovernedAutoMerger(
            engine,
            policy=MergePolicy(enabled=True, require_governance_valid=False),
            promoter=lambda spec: True,
        )
        ev = merger.consider(_spec())
        assert ev.merged is True
        assert ev.action_decision["decision"] == "allow"
        assert not engine.by_type("ActionApproval")
