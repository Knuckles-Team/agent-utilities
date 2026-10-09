"""Public API exports AU's retained action-policy decision point (AU-BOUNDARY-R013, R017)."""

from __future__ import annotations

from agent_utilities import api
from agent_utilities.api import messaging
from agent_utilities.orchestration import action_policy as core_action_policy


def test_public_messaging_exports_are_the_canonical_implementation() -> None:
    assert api.ActionPolicy is core_action_policy.ActionPolicy
    assert api.ActionRequest is core_action_policy.ActionRequest
    assert api.ActionDecision is core_action_policy.ActionDecision
    assert api.ActionRule is core_action_policy.ActionRule
    assert api.PolicyDisposition is core_action_policy.PolicyDisposition
    assert api.PolicyReceipt is core_action_policy.PolicyReceipt
    assert api.get_action_policy is core_action_policy.get_action_policy
    assert api.in_maintenance_window is core_action_policy.in_maintenance_window
    assert messaging.ActionPolicy is core_action_policy.ActionPolicy


def test_public_action_policy_decides_through_the_real_authority() -> None:
    policy = api.ActionPolicy()
    request = api.ActionRequest(
        kind="restart_service",
        target="api-messaging-port-test",
        source="test",
    )
    decision = policy.decide(request)
    assert isinstance(decision, api.ActionDecision)
    assert isinstance(decision.disposition, api.PolicyDisposition)
