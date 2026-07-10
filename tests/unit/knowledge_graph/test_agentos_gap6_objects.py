#!/usr/bin/python
from __future__ import annotations

"""Codex Gap-6 Agent-OS named objects: AgentCapabilityGrant, AgentPolicyDecision,
AgentTrace, and the claim->execute->outcome orchestration wiring.

CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Graph-Native Agent-OS Objects

Reuse audit (see class docstrings in ``models/knowledge_graph.py`` for the full
rationale):

* ``AgentCapabilityGrantNode`` — genuinely NEW. The ``AUTHORIZED_FOR`` edge
  (and its ``MATCH`` in ``orchestration/engine.py``'s team synthesis) existed
  with nothing writing it; this is the write/read pair that completes it.
* ``AgentPolicyDecisionNode`` — EXTENDS the existing ``ActionDecision`` audit
  ``action_policy.ActionPolicy._audit()`` already writes (no new node type;
  same persisted label, same rate/blast ledger reads). Formalizes it as a
  typed schema entry for the first time.
* ``AgentTrace`` — NOT a new type at all: it IS the existing ``TraceNode``
  (aliased as ``AgentTraceNode``), extended with ``task_id``/``tool_calls``/
  ``outcome`` fields.
* Observation/Claim/Action (the wiring's writeback) reuse the EXISTING
  ``ObservationNode``/``ClaimNode``/``ActionNode`` — no new types.
* "AgentOutcome" reuses the EXISTING ``OutcomeEvaluationNode`` (already
  carrying ``lease_id``/``dag_id`` from C3/Phase 3a) — no new type.

@pytest.mark.concept("AU-OS.state.cognitive-scheduler-preemption")
"""

import time

import pytest

from agent_utilities.models.knowledge_graph import (
    AgentCapabilityGrantNode,
    AgentPolicyDecisionNode,
    AgentTraceNode,
    RegistryNodeType,
    TraceNode,
)
from agent_utilities.orchestration import action_policy
from agent_utilities.orchestration import agent_dispatch_worker as worker

pytestmark = pytest.mark.concept("AU-OS.state.cognitive-scheduler-preemption")


# ---------------------------------------------------------------------------
# Reuse audit — exactly 2 new enum members; no duplicate types for the rest
# ---------------------------------------------------------------------------


def test_exactly_two_new_node_types_this_phase() -> None:
    names = {m.name for m in RegistryNodeType}
    assert "AGENT_CAPABILITY_GRANT" in names
    assert "AGENT_POLICY_DECISION" in names
    # Reuse audit: AgentTrace/AgentOutcome/AgentObservation/AgentClaim/
    # AgentAction were NOT introduced as second node types — they alias or
    # extend existing ones instead.
    for not_expected in (
        "AGENT_TRACE",
        "AGENT_OUTCOME",
        "AGENT_OBSERVATION",
        "AGENT_CLAIM",
        "AGENT_ACTION",
    ):
        assert not_expected not in names


def test_agent_trace_node_is_the_existing_trace_node_not_a_new_type() -> None:
    assert AgentTraceNode is TraceNode


# ---------------------------------------------------------------------------
# AgentCapabilityGrantNode
# ---------------------------------------------------------------------------


def test_agent_capability_grant_node_defaults() -> None:
    node = AgentCapabilityGrantNode(id="grant:1", name="Grant: x")
    assert node.type == RegistryNodeType.AGENT_CAPABILITY_GRANT
    assert node.agent_id == ""
    assert node.capability == ""
    assert node.issuer == ""
    assert node.granted_at == 0.0
    assert node.expires_at is None
    assert node.revoked is False
    assert node.is_active() is True


def test_agent_capability_grant_node_expiry_and_revocation() -> None:
    node = AgentCapabilityGrantNode(
        id="grant:2",
        name="Grant: agent-1",
        agent_id="agent-1",
        capability="agent_task.execute",
        issuer="system",
        granted_at=1000.0,
        expires_at=2000.0,
    )
    assert node.is_active(now=1500.0) is True
    assert node.is_active(now=2500.0) is False

    revoked = node.model_copy(update={"revoked": True})
    assert revoked.is_active(now=1500.0) is False


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


# ---------------------------------------------------------------------------
# AgentPolicyDecisionNode
# ---------------------------------------------------------------------------


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


def test_agent_policy_decision_node_from_action_decision_reuses_audit_id() -> None:
    """The wrapper keys off the ALREADY-PERSISTED ActionDecision audit_id —
    same graph node, no second write (the reuse-audit's load-bearing property)."""
    req = action_policy.ActionRequest(
        kind="agent_task.execute", target="task-1", source="agent-dispatch", actor_id="agent-1"
    )
    decision = action_policy.ActionDecision(
        decision="allow_notify",
        tier="auto_notify",
        request=req,
        reason="tier auto_notify",
        rule_origin="file",
        audit_id="action_decision:abc123",
    )
    node = AgentPolicyDecisionNode.from_action_decision(decision, agent_id="agent-1")
    assert node.id == "action_decision:abc123"
    assert node.kind == "agent_task.execute"
    assert node.target == "task-1"
    assert node.tier == "auto_notify"
    assert node.decision == "allow_notify"
    assert node.agent_id == "agent-1"
    assert node.allowed is True


# ---------------------------------------------------------------------------
# TraceNode / AgentTraceNode extension (task_id / tool_calls / outcome)
# ---------------------------------------------------------------------------


def test_trace_node_agentos_fields_default() -> None:
    node = TraceNode(id="trace:1", name="run")
    assert node.task_id is None
    assert node.tool_calls == 0
    assert node.outcome is None


def test_trace_node_agentos_fields_explicit_round_trip() -> None:
    node = AgentTraceNode(
        id="trace:2",
        name="run",
        agent="agent-1",
        task_id="dag-1:task:a",
        tool_calls=3,
        outcome="completed",
    )
    restored = TraceNode.model_validate_json(node.model_dump_json())
    assert restored.task_id == "dag-1:task:a"
    assert restored.tool_calls == 3
    assert restored.outcome == "completed"


# ---------------------------------------------------------------------------
# Codex Gap-6 orchestration flow: execute_agent_task_turn
# ---------------------------------------------------------------------------


class _Gap6Engine:
    """Minimal engine double covering AgentTask/AgentLease claim + the
    action_policy ledger reads + AgentCapabilityGrant resolution — enough to
    drive the full claim->execute->outcome wiring end to end."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, node_type, properties=None):
        node = self.nodes.setdefault(node_id, {})
        node["type"] = node_type
        node.update(properties or {})

    def add_edge(self, source, target, rel_type, **properties):
        self.edges.append((source, target, rel_type))

    def by_type(self, node_type: str) -> list[dict]:
        return [n for n in self.nodes.values() if n.get("type") == node_type]

    def query_cypher(self, q, params=None):
        params = params or {}
        if "AgentTask {id: $id}" in q:
            node = self.nodes.get(params.get("id"))
            if node is None:
                return []
            return [
                {
                    "status": node.get("status"),
                    "depends_on_task_ids": node.get("depends_on_task_ids") or [],
                    "dag_id": node.get("dag_id"),
                    "checkpoint_id": node.get("checkpoint_id"),
                }
            ]
        if "AgentLease {resource_id: $rid}" in q:
            leases = [
                n
                for n in self.nodes.values()
                if n.get("type") == "AgentLease"
                and n.get("resource_id") == params.get("rid")
            ]
            leases.sort(key=lambda n: n.get("acquired_at", 0.0), reverse=True)
            if not leases:
                return []
            top = leases[0]
            return [
                {
                    "owner_token": top.get("owner_token"),
                    "lease_expires_at": top.get("lease_expires_at"),
                }
            ]
        if "governance_rule" in q:
            return []  # no KG policy overrides in these tests
        if "ActionDecision {kind:" in q:
            return []  # no rate/blast history yet
        if "AgentCapabilityGrant {capability:" in q:
            agent_id = params.get("agent_id")
            capability = params.get("capability")
            grants = [
                (nid, n)
                for nid, n in self.nodes.items()
                if n.get("type") == "AgentCapabilityGrant"
                and n.get("agent_id") == agent_id
                and n.get("capability") == capability
            ]
            grants.sort(key=lambda pair: pair[1].get("granted_at", 0.0), reverse=True)
            if not grants:
                return []
            gid, top = grants[0]
            return [
                {
                    "id": gid,
                    "issuer": top.get("issuer"),
                    "granted_at": top.get("granted_at"),
                    "expires_at": top.get("expires_at"),
                    "revoked": top.get("revoked"),
                }
            ]
        return []


def test_execute_agent_task_turn_allowed_path_writes_full_provenance() -> None:
    engine = _Gap6Engine()
    engine.add_node("task-1", "AgentTask", properties={"status": "pending"})

    outcome = worker.execute_agent_task_turn(
        engine,
        "task-1",
        agent_id="agent-1",
        executor=lambda claim: f"did work for {claim['task_id']}",
    )

    assert outcome == "completed"

    # Claim -> lease.
    assert engine.by_type("AgentLease")

    # Capability grant (Codex Gap-6) — self-issued + AUTHORIZED_FOR edge.
    grants = engine.by_type("AgentCapabilityGrant")
    assert len(grants) == 1
    assert grants[0]["agent_id"] == "agent-1"
    assert grants[0]["capability"] == "agent_task.execute"
    grant_id = next(nid for nid, n in engine.nodes.items() if n is grants[0])
    assert ("agent-1", grant_id, "AUTHORIZED_FOR") in engine.edges

    # Policy decision audit (Codex Gap-6 formalization over ActionDecision).
    decisions = engine.by_type("ActionDecision")
    assert len(decisions) == 1
    assert decisions[0]["kind"] == "agent_task.execute"
    assert decisions[0]["decision"] == "allow_notify"

    # Writeback: Observation / Claim / Action / AgentOutcome (OutcomeEvaluation) / Trace.
    assert len(engine.by_type("Observation")) == 1
    assert len(engine.by_type("Claim")) == 1
    actions = engine.by_type("Action")
    assert len(actions) == 1
    assert actions[0]["status"] == "completed"
    assert actions[0]["capability_grant_id"] == grant_id

    outcomes = engine.by_type("OutcomeEvaluation")
    assert len(outcomes) == 1
    assert outcomes[0]["reward"] == 1.0
    assert outcomes[0]["lease_id"]
    assert outcomes[0]["dag_id"] == ""

    traces = engine.by_type("Trace")
    assert len(traces) == 1
    assert traces[0]["task_id"] == "task-1"
    assert traces[0]["outcome"] == "completed"

    # Final AgentTask status flip — what fire_ready_agent_tasks/the reconciler
    # already polls to wake TASK_DEPENDS_ON dependents (D23/C3).
    assert engine.nodes["task-1"]["status"] == "completed"


def test_execute_agent_task_turn_reuses_existing_capability_grant() -> None:
    engine = _Gap6Engine()
    engine.add_node("task-2", "AgentTask", properties={"status": "pending"})
    engine.add_node(
        "grant:existing",
        "AgentCapabilityGrant",
        properties={
            "agent_id": "agent-1",
            "capability": "agent_task.execute",
            "issuer": "operator",
            "granted_at": time.time(),
            "expires_at": None,
            "revoked": False,
        },
    )

    worker.execute_agent_task_turn(engine, "task-2", agent_id="agent-1")

    grants = engine.by_type("AgentCapabilityGrant")
    assert len(grants) == 1  # no new grant self-issued — the existing one was reused
    actions = engine.by_type("Action")
    assert actions[0]["capability_grant_id"] == "grant:existing"


def test_execute_agent_task_turn_denied_is_terminal_failed(monkeypatch) -> None:
    engine = _Gap6Engine()
    engine.add_node("task-3", "AgentTask", properties={"status": "pending"})

    class _DenyPolicy:
        def decide(self, request):
            return action_policy.ActionDecision(
                decision=action_policy.DECISION_DENY,
                tier=action_policy.TIER_FORBIDDEN,
                request=request,
                reason="forbidden by policy",
                audit_id="action_decision:deny-1",
            )

    monkeypatch.setattr(action_policy, "get_action_policy", lambda engine=None: _DenyPolicy())

    outcome = worker.execute_agent_task_turn(engine, "task-3", agent_id="agent-1")

    assert outcome == "denied"
    assert engine.nodes["task-3"]["status"] == "failed"
    # No capability grant issued for a denied action.
    assert engine.by_type("AgentCapabilityGrant") == []


def test_execute_agent_task_turn_queued_for_approval_is_blocked_not_terminal(
    monkeypatch,
) -> None:
    engine = _Gap6Engine()
    engine.add_node("task-4", "AgentTask", properties={"status": "pending"})

    class _QueuePolicy:
        def decide(self, request):
            return action_policy.ActionDecision(
                decision=action_policy.DECISION_QUEUE,
                tier=action_policy.TIER_APPROVAL,
                request=request,
                reason="tier requires human approval",
                audit_id="action_decision:queue-1",
                approval_id="action_approval:1",
            )

    monkeypatch.setattr(action_policy, "get_action_policy", lambda engine=None: _QueuePolicy())

    outcome = worker.execute_agent_task_turn(engine, "task-4", agent_id="agent-1")

    assert outcome == "blocked"
    assert engine.nodes["task-4"]["status"] == "blocked"
    assert engine.nodes["task-4"]["status"] not in worker._AGENT_TASK_TERMINAL


def test_execute_agent_task_turn_skips_terminal_task() -> None:
    engine = _Gap6Engine()
    engine.add_node("task-5", "AgentTask", properties={"status": "completed"})
    outcome = worker.execute_agent_task_turn(engine, "task-5", agent_id="agent-1")
    assert outcome == "skipped"
    # No writeback of any kind for a duplicate-delivery skip.
    assert engine.by_type("Observation") == []
    assert engine.by_type("OutcomeEvaluation") == []


def test_execute_agent_task_turn_default_executor_never_fabricates() -> None:
    engine = _Gap6Engine()
    engine.add_node("task-6", "AgentTask", properties={"status": "pending"})
    outcome = worker.execute_agent_task_turn(engine, "task-6", agent_id="")
    assert outcome == "completed"
    actions = engine.by_type("Action")
    assert "acknowledged (no executor bound)" in actions[0]["result"]
    # No agent_id ⇒ no capability grant issued (nothing to authorize).
    assert engine.by_type("AgentCapabilityGrant") == []
