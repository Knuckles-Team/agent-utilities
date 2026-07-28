"""Current capability, policy-decision, trace, and fence-shape contracts."""

from agent_utilities.models.knowledge_graph import (
    AgentCapabilityGrantNode,
    AgentPolicyDecisionNode,
    RegistryNodeType,
    TraceNode,
)
from agent_utilities.orchestration import action_policy
from agent_utilities.orchestration import agent_dispatch_worker as worker


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


def test_agent_capability_grant_node_round_trips_json() -> None:
    node = AgentCapabilityGrantNode(
        id="grant:3",
        name="Grant: y",
        agent_id="agent-2",
        capability="tool:search",
        issuer="operator",
        granted_at=10.0,
        expires_at=None,
    )
    restored = AgentCapabilityGrantNode.model_validate_json(node.model_dump_json())
    assert restored == node


def test_agent_policy_decision_node_defaults() -> None:
    node = AgentPolicyDecisionNode(id="action_decision:1", name="PolicyDecision: x")
    assert node.type == RegistryNodeType.AGENT_POLICY_DECISION
    assert node.decision == ""
    assert node.allowed is False


def test_agent_policy_decision_node_allowed_property() -> None:
    allow = AgentPolicyDecisionNode(id="d1", name="d1", decision="allow")
    allow_notify = AgentPolicyDecisionNode(id="d2", name="d2", decision="allow_notify")
    queued = AgentPolicyDecisionNode(id="d3", name="d3", decision="queue_approval")
    denied = AgentPolicyDecisionNode(id="d4", name="d4", decision="deny")
    assert allow.allowed is True
    assert allow_notify.allowed is True
    assert queued.allowed is False
    assert denied.allowed is False


def test_agent_policy_decision_reuses_existing_action_audit() -> None:
    request = action_policy.ActionRequest(
        kind="agent_task.execute",
        target="task-1",
        source="agent-dispatch",
        actor_id="agent-1",
    )
    decision = action_policy.ActionDecision(
        decision="allow_notify",
        tier="auto_notify",
        request=request,
        reason="policy",
        rule_origin="file",
        audit_id="action_decision:opaque",
    )
    node = AgentPolicyDecisionNode.from_action_decision(decision, agent_id="agent-ref")
    assert node.id == decision.audit_id
    assert node.kind == "agent_task.execute"
    assert node.allowed


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
    node = AgentPolicyDecisionNode.from_action_decision(decision, agent_id="agent-ref")
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


class _RaisingLeaseEngine:
    def query_cypher(self, query, params=None):
        if "AgentLease {resource_id: $rid}" in query:
            raise RuntimeError("engine query transport error")
        return []


def test_fence_still_valid_kg_claim_fails_open_on_query_error() -> None:
    engine = _RaisingLeaseEngine()
    claim = {"fence_token": 1, "_claim_backend": "kg"}
    assert worker._fence_still_valid(engine, "task-kg", claim, token="hostA:1") is True

    unmarked_claim = {"fence_token": 1}
    assert (
        worker._fence_still_valid(engine, "task-kg-2", unmarked_claim, token="hostA:1")
        is True
    )


def test_fence_still_valid_engine_native_claim_fails_closed_on_query_error() -> None:
    engine = _RaisingLeaseEngine()
    claim = {"fence_token": 1, "_claim_backend": "engine"}
    assert (
        worker._fence_still_valid(engine, "task-engine", claim, token="hostA:1")
        is False
    )


def test_fence_still_valid_engine_native_claim_fails_closed_with_no_engine() -> None:
    claim = {"fence_token": 1, "_claim_backend": "engine"}
    assert (
        worker._fence_still_valid(None, "task-engine", claim, token="hostA:1") is False
    )
