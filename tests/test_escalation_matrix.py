"""Tests for the HITL Escalation Matrix (CONCEPT:AU-OS.observability.empty-derive-from-effect).

Covers the policy unit behavior AND the live wiring into the Ontology Action
System executor: a high-tier action requires human approval before its handler
runs; a low-tier one does not.

@pytest.mark.concept("AU-OS.observability.empty-derive-from-effect")
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.actions.executor import ActionExecutor
from agent_utilities.knowledge_graph.actions.models import (
    ActionEffect,
    ActionStatus,
    OntologyAction,
)
from agent_utilities.knowledge_graph.actions.registry import ActionRegistry
from agent_utilities.observability.escalation_matrix import (
    EscalationGate,
    EscalationMatrix,
    EscalationOutcome,
    Fallback,
    RiskTier,
    ValueTier,
    classify_risk_tier,
    classify_value_tier,
    make_decision_provider,
)
from agent_utilities.security.permissions_kernel import AgentRole, PermissionsKernel

pytestmark = pytest.mark.concept("AU-OS.observability.empty-derive-from-effect")


# ---------------------------------------------------------------------------
# Matrix / classification unit behavior
# ---------------------------------------------------------------------------


class TestMatrix:
    def test_default_low_risk_low_value_no_approval(self):
        m = EscalationMatrix.default()
        assert m.rule_for(RiskTier.LOW, ValueTier.LOW).require_approval is False

    def test_default_high_risk_requires_approval(self):
        m = EscalationMatrix.default()
        rule = m.rule_for(RiskTier.HIGH, ValueTier.LOW)
        assert rule.require_approval is True
        assert rule.fallback == Fallback.AUTO_DENY

    def test_critical_requires_admin(self):
        m = EscalationMatrix.default()
        rule = m.rule_for(RiskTier.CRITICAL, ValueTier.CRITICAL)
        assert rule.require_approval is True
        assert "admin" in rule.approver_roles

    def test_classify_risk_external_nonidempotent_is_high(self):
        assert classify_risk_tier("external", idempotent=False) == RiskTier.HIGH

    def test_classify_risk_external_idempotent_is_medium(self):
        # An idempotent external call (read-style API screen) is MEDIUM so it is
        # not gated at LOW value — keeps read-style verbs flowing.
        assert classify_risk_tier("external", idempotent=True) == RiskTier.MEDIUM

    def test_classify_risk_explicit_override(self):
        assert (
            classify_risk_tier("read", True, explicit="critical") == RiskTier.CRITICAL
        )

    def test_classify_value_string(self):
        assert classify_value_tier("high") == ValueTier.HIGH
        assert classify_value_tier("garbage") == ValueTier.MEDIUM


# ---------------------------------------------------------------------------
# Gate (sync) behavior
# ---------------------------------------------------------------------------


class TestGate:
    def test_low_tier_not_required(self):
        gate = EscalationGate(persist=False)
        d = gate.evaluate_sync(
            action_name="kg.search",
            actor_id="a",
            effect="read",
            value=ValueTier.LOW,
        )
        assert d.outcome == EscalationOutcome.NOT_REQUIRED
        assert d.allowed is True

    def test_high_tier_no_provider_falls_back_to_deny(self):
        gate = EscalationGate(persist=False)
        d = gate.evaluate_sync(
            action_name="db.drop",
            actor_id="a",
            effect="external",
            value=ValueTier.HIGH,
        )
        assert d.allowed is False
        assert d.outcome in (EscalationOutcome.TIMEOUT, EscalationOutcome.DENIED)

    def test_high_tier_provider_approves(self):
        gate = EscalationGate(persist=False)
        provider = make_decision_provider(
            {
                "db.drop": {
                    "approved": True,
                    "approver": "operator",
                    "approver_role": "operator",
                }
            }
        )
        d = gate.evaluate_sync(
            action_name="db.drop",
            actor_id="a",
            effect="external",
            value=ValueTier.HIGH,
            decision_provider=provider,
        )
        assert d.allowed is True
        assert d.outcome == EscalationOutcome.APPROVED

    def test_unauthorized_approver_role_denied(self):
        gate = EscalationGate(persist=False)
        provider = make_decision_provider(
            {"db.drop": {"approved": True, "approver": "x", "approver_role": "guest"}}
        )
        d = gate.evaluate_sync(
            action_name="db.drop",
            actor_id="a",
            effect="external",
            value=ValueTier.HIGH,
            decision_provider=provider,
        )
        assert d.allowed is False
        assert d.outcome == EscalationOutcome.DENIED

    def test_decision_is_audited(self):
        gate = EscalationGate(persist=False)
        gate.evaluate_sync(
            action_name="db.drop", actor_id="a", effect="external", value="high"
        )
        records = gate.audit.query(action="hitl.escalation")
        assert records and records[0].resource_id == "db.drop"


# ---------------------------------------------------------------------------
# LIVE-PATH: executor consults the matrix before running the handler
# ---------------------------------------------------------------------------


def _registry_with(action: OntologyAction, handler) -> ActionRegistry:
    reg = ActionRegistry()
    reg.register(action, handler)
    return reg


class TestExecutorEscalationLivePath:
    """Wire-first: ActionExecutor.execute consults the EscalationGate."""

    def test_low_tier_action_runs_without_approval_live_path(self):
        ran = {"v": False}

        def handler(_params):
            ran["v"] = True
            return "done"

        action = OntologyAction(
            name="kg.search",
            verb="search",
            required_capability="kg.read",
            produces_effect=ActionEffect.READ,
        )
        kernel = PermissionsKernel()
        ex = ActionExecutor(
            _registry_with(action, handler), kernel=kernel, persist=False
        )
        ex.escalation_gate = EscalationGate(persist=False)
        identity = kernel.issue_identity(
            "agent-1", role=AgentRole.SPECIALIST, capabilities=["kg.read"]
        )

        inv = ex.execute("kg.search", identity, params={})
        # No human needed for a read action → handler actually ran.
        assert inv.status == ActionStatus.SUCCESS
        assert ran["v"] is True

    def test_high_tier_action_blocked_without_approval_live_path(self):
        ran = {"v": False}

        def handler(_params):
            ran["v"] = True
            return "danger"

        action = OntologyAction(
            name="infra.destroy",
            verb="destroy",
            required_capability="infra.admin",
            produces_effect=ActionEffect.EXTERNAL,
            idempotent=False,
            value_tier="high",
        )
        kernel = PermissionsKernel()
        ex = ActionExecutor(
            _registry_with(action, handler), kernel=kernel, persist=False
        )
        ex.escalation_gate = EscalationGate(persist=False)
        identity = kernel.issue_identity(
            "agent-2", role=AgentRole.ADMIN, capabilities=["infra.admin", "admin"]
        )

        # No decision provider → high-tier verb auto-denied by the matrix
        # fallback BEFORE the handler runs (escalation gate sits after authz).
        inv = ex.execute("infra.destroy", identity, params={})
        assert inv.status == ActionStatus.DENIED
        assert "escalation" in inv.result_summary
        assert ran["v"] is False  # handler NEVER ran

    def test_high_tier_action_runs_when_approved_live_path(self):
        ran = {"v": False}

        def handler(_params):
            ran["v"] = True
            return "approved-run"

        action = OntologyAction(
            name="infra.destroy",
            verb="destroy",
            required_capability="infra.admin",
            produces_effect=ActionEffect.EXTERNAL,
            idempotent=False,
            value_tier="high",
        )
        kernel = PermissionsKernel()
        ex = ActionExecutor(
            _registry_with(action, handler), kernel=kernel, persist=False
        )
        ex.escalation_gate = EscalationGate(persist=False)
        identity = kernel.issue_identity(
            "agent-3", role=AgentRole.ADMIN, capabilities=["infra.admin", "admin"]
        )
        provider = make_decision_provider(
            {
                "infra.destroy": {
                    "approved": True,
                    "approver": "admin",
                    "approver_role": "admin",
                }
            }
        )

        inv = ex.execute(
            "infra.destroy", identity, params={}, decision_provider=provider
        )
        assert inv.status == ActionStatus.SUCCESS
        assert ran["v"] is True
