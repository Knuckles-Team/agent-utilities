"""GovernedAutoMerger's promotion decision consults the OS-5.24 ActionPolicy.

The AHE-3.20 → ActionPolicy adoption (previously a noted follow-up): before the
merger flips a proposal's lifecycle (proposal→active), it decides the reserved
``merge_promotion`` kind for the proposal's id, through the unified
``artifact_promotion.promote()`` gate (b1017d58c), whose ``PromotionOutcome``
reports the coarser, receipt-backed ``PolicyDisposition`` vocabulary
(5a4dd9a2f, "freeze receipt-backed policy outcomes") rather than the raw
``ActionDecision.decision`` string, and whose ``.approved`` property (eligible
AND disposition is ``APPROVE`` AND the receipt is bound to the exact decided
request) is the ONE shared authority every ``promote()`` caller must honor —
see that function's own docstring and ``_receipt_matches_approval``. Decision
mapping, as it now surfaces on ``ev.action_decision["decision"]``:

- ``deny`` / ``unavailable`` / ``hold`` → promotion blocked, recorded on the
  evaluation + audit trail. ``hold`` (``queue_approval``'s raw tier) is a
  pending human approval, not a grant — it still queues the (deduped)
  ``ActionApproval`` the AHE-3.21 publication step would later consume, but
  the KG-internal lifecycle flip itself does NOT proceed until a human
  actually approves and a fresh decide() comes back ``approve`` with a
  matching receipt. ``unavailable`` additionally covers a policy-consult
  failure and an ``approve`` disposition whose receipt could not be bound;
- ``approve`` (``allow``/``allow_notify``'s raw tier) → proceed.

The outer ``KG_GOLDEN_AUTO_MERGE`` gate is unchanged (default False).

@pytest.mark.concept("AHE-3.20")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.research.auto_merge import (
    GovernedAutoMerger,
    MergePolicy,
)
from agent_utilities.orchestration.action_policy import ActionPolicy
from tests.unit.fleet_autonomy_fakes import FakeEngine
from tests.unit.fleet_autonomy_fakes import FakePolicy as _FakePolicy

pytestmark = pytest.mark.concept("AHE-3.20")


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

    def test_queue_approval_blocks_promotion_pending_human_approval(self):
        """``hold`` is a durable approval WAIT, not a grant -- it must not
        activate anything, exactly like ``deny``/``unavailable``. Only a
        receipt-backed ``approve`` (see ``PromotionOutcome.approved``) may
        flip the lifecycle; this is the ONE shared promotion contract every
        ``artifact_promotion.promote()`` caller honors."""
        promoted: list = []
        policy = _FakePolicy(
            "queue_approval",
            reason="tier requires human approval",
            approval_id="action_approval:abc",
        )
        merger = _merger(policy, promoted=promoted)
        ev = merger.consider(_spec())
        assert ev.merged is False
        assert promoted == [], "hold must block the lifecycle flip"
        assert ev.publication is None, "a blocked promotion never publishes"
        assert ev.action_decision == {
            # "queue_approval" (the raw ActionDecision vocabulary) collapses
            # to "hold" once it crosses the unified promote() gate's
            # PolicyDisposition (5a4dd9a2f).
            "decision": "hold",
            "reason": "tier requires human approval",
            "approval_id": "action_approval:abc",
        }
        assert "blocked by action policy (merge_promotion)" in ev.reason

    def test_allow_proceeds_with_merge_promotion_request(self):
        promoted: list = []
        policy = _FakePolicy("allow")
        merger = _merger(policy, promoted=promoted)
        ev = merger.consider(_spec())
        assert ev.merged is True
        assert promoted
        assert ev.action_decision["decision"] == "approve"
        (request,) = policy.requests
        assert request.kind == "merge_promotion"
        assert request.target == "proposal:policy-1"
        assert request.source == "loop_engine"

    def test_allow_notify_proceeds(self):
        policy = _FakePolicy("allow_notify")
        ev = _merger(policy).consider(_spec())
        assert ev.merged is True
        # "allow"/"allow_notify" both collapse to "approve" under the
        # unified gate's PolicyDisposition (5a4dd9a2f) -- the notification
        # side effect itself is exercised by action_policy's own tests.
        assert ev.action_decision["decision"] == "approve"

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
        # A policy-consult exception reports "unavailable" under the unified
        # gate's PolicyDisposition (5a4dd9a2f) -- distinct from an explicit
        # policy "deny" -- but _gate_by_action_policy blocks on both, so the
        # fail-closed behavior this test proves is unaffected.
        assert ev.action_decision["decision"] == "unavailable"
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

    def test_shipped_default_holds_and_blocks_promotion(self):
        """Default tier (unconfigured actor) queues a human approval but
        does NOT activate the promotion -- ``hold`` is a pending approval,
        not a grant. The approval is still queued on the durable ledger
        (a human can act on it later); this merger call simply never reaches
        ``_execute_promotion``, so nothing publishes yet either."""
        engine = FakeEngine()
        merger = GovernedAutoMerger(
            engine,
            policy=MergePolicy(enabled=True, require_governance_valid=False),
            promoter=lambda spec: True,
        )
        ev = merger.consider(_spec())
        assert ev.merged is False
        assert ev.publication is None, "a blocked promotion never publishes"
        # The shipped default's "approval_required" tier queues (the raw
        # ActionDecision vocabulary's "queue_approval"), which the unified
        # gate's PolicyDisposition (5a4dd9a2f) reports as "hold".
        assert ev.action_decision["decision"] == "hold"
        assert "blocked by action policy (merge_promotion)" in ev.reason
        # ActionPolicy.decide() queues the approval as a side effect of the
        # consult itself, independent of what the caller does with the
        # decision -- a human can review and approve it even though THIS
        # consider() call did not promote.
        approvals = engine.by_type("ActionApproval")
        assert len(approvals) == 1
        assert approvals[0]["id"] == ev.action_decision["approval_id"]
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
        assert ev.action_decision["decision"] == "approve"
        assert not engine.by_type("ActionApproval")
