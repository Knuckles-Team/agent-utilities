"""Current capability, policy-decision, and trace object contracts."""

from agent_utilities.models.knowledge_graph import (
    AgentCapabilityGrantNode,
    AgentPolicyDecisionNode,
    RegistryNodeType,
    TraceNode,
)
from agent_utilities.orchestration import action_policy


def test_reuse_audit_keeps_only_named_grant_and_decision_types() -> None:
    names = {member.name for member in RegistryNodeType}
    assert "AGENT_CAPABILITY_GRANT" in names
    assert "AGENT_POLICY_DECISION" in names
    assert "AGENT_TASK" not in names
    assert "AGENT_LEASE" not in names


def test_work_item_execution_grant_round_trip_and_expiry() -> None:
    grant = AgentCapabilityGrantNode(
        id="grant:opaque",
        name="grant",
        agent_id="agent-ref",
        capability="work_item.execute",
        issuer="issuer-ref",
        granted_at=100.0,
        expires_at=200.0,
    )
    restored = AgentCapabilityGrantNode.model_validate_json(grant.model_dump_json())
    assert restored == grant
    assert restored.is_active(now=150.0)
    assert not restored.is_active(now=250.0)


def test_policy_decision_wraps_existing_work_item_audit() -> None:
    request = action_policy.ActionRequest(
        kind="work_item.execute",
        target="workitem-ref",
        source="agent-dispatch",
        actor_id="agent-ref",
    )
    decision = action_policy.ActionDecision(
        decision="allow_notify",
        tier="auto_notify",
        request=request,
        reason="policy",
        rule_origin="file",
        audit_id="action_decision:opaque",
    )
    node = AgentPolicyDecisionNode.from_action_decision(
        decision, agent_id="agent-ref"
    )
    assert node.id == decision.audit_id
    assert node.kind == "work_item.execute"
    assert node.allowed


def test_trace_extension_round_trip_is_evidence_only() -> None:
    trace = TraceNode(
        id="trace:opaque",
        name="run",
        agent="agent-ref",
        task_id="workitem-ref",
        tool_calls=3,
        outcome="succeeded",
    )
    restored = TraceNode.model_validate_json(trace.model_dump_json())
    assert restored.task_id == "workitem-ref"
    assert restored.tool_calls == 3
    assert restored.outcome == "succeeded"
