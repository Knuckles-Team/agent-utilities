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
        kind="agent_task.execute",
        target="task-1",
        source="agent-dispatch",
        actor_id="agent-1",
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
                    "lease_epoch": top.get("lease_epoch"),
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

    monkeypatch.setattr(
        action_policy, "get_action_policy", lambda engine=None: _DenyPolicy()
    )

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

    monkeypatch.setattr(
        action_policy, "get_action_policy", lambda engine=None: _QueuePolicy()
    )

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
    """AU-P0-3 fail-closed: no bound executor must NEVER be recorded as a
    successful completion — the task is unroutable, not done."""
    engine = _Gap6Engine()
    engine.add_node("task-6", "AgentTask", properties={"status": "pending"})
    outcome = worker.execute_agent_task_turn(engine, "task-6", agent_id="")
    assert outcome == "unroutable"
    assert engine.nodes["task-6"]["status"] == "unroutable"
    assert outcome in worker._AGENT_TASK_TERMINAL

    actions = engine.by_type("Action")
    assert "no executor bound" in actions[0]["result"]
    assert actions[0]["status"] == "unroutable"

    outcomes = engine.by_type("OutcomeEvaluation")
    assert len(outcomes) == 1
    assert outcomes[0]["reward"] == 0.0
    assert outcomes[0]["reward"] != 1.0

    # No agent_id ⇒ no capability grant issued (nothing to authorize).
    assert engine.by_type("AgentCapabilityGrant") == []


def test_execute_agent_task_turn_routes_claim_through_engine_claim_backend_switch(
    monkeypatch,
) -> None:
    """AU-P0-3 wiring: the live worker's claim step must resolve through
    ``engine_claim.claim_agent_task`` (the backend switch), not call this
    module's KG-only ``claim_agent_task`` directly — otherwise
    ``AGENT_CLAIM_BACKEND=engine`` is inert for real dispatch (the audited
    gap: engine_claim.py existed but only its own unit tests imported it).
    With the backend set to ``engine`` and a live engine claim available, the
    KG claim primitive must never be invoked."""
    from agent_utilities.orchestration import engine_claim

    monkeypatch.setenv("AGENT_CLAIM_BACKEND", "engine")

    kg_claim_calls: list[str] = []
    monkeypatch.setattr(
        engine_claim,
        "_claim_agent_task_kg",
        lambda engine, task_id, **kw: kg_claim_calls.append(task_id) or None,
    )

    engine_claim_calls: list[str] = []

    def _fake_try_engine_claim(task_id, *, token, now, claim_ttl_s):
        engine_claim_calls.append(task_id)
        return {
            "task_id": task_id,
            "lease_id": "lease:engine:live",
            "dag_id": "",
            "checkpoint_id": None,
            "depends_on_task_ids": [],
            "fence_token": 1,
        }

    monkeypatch.setattr(engine_claim, "_try_engine_claim", _fake_try_engine_claim)

    engine = _Gap6Engine()
    engine.add_node("task-9", "AgentTask", properties={"status": "pending"})

    outcome = worker.execute_agent_task_turn(
        engine,
        "task-9",
        agent_id="agent-1",
        executor=lambda claim: "ran via the engine-native claim",
    )

    assert outcome == "completed"
    assert engine_claim_calls == ["task-9"]
    assert kg_claim_calls == []  # the live engine claim was used EXCLUSIVELY
    assert engine.nodes["task-9"]["status"] == "completed"


def test_execute_agent_task_turn_concurrent_claimants_only_one_wins_cas() -> None:
    """AU-P0-3: while A is still executing (task status='running', A's lease
    fresh), a second claimant B racing the same ``:AgentTask`` is turned away
    by the fresh-lease CAS check in ``claim_agent_task`` and never executes —
    exactly the concurrency the queue's at-least-once delivery can trigger."""
    engine = _Gap6Engine()
    engine.add_node("task-7", "AgentTask", properties={"status": "pending"})

    b_claim_attempts: list[dict | None] = []

    def _executor(claim: dict) -> str:
        # Simulate B racing in WHILE A is still executing: A's lease is fresh
        # at this instant, so B's concurrent claim attempt must be rejected.
        b_claim_attempts.append(
            worker.claim_agent_task(
                engine, "task-7", token="hostB:2:agent-dispatch", now=1000.5
            )
        )
        return "did work for A"

    outcome_a = worker.execute_agent_task_turn(
        engine,
        "task-7",
        agent_id="agent-1",
        executor=_executor,
        token="hostA:1:agent-dispatch",
        now=1000.0,
    )

    assert outcome_a == "completed"
    # B's concurrent claim attempt lost the CAS — it got None, so B's own
    # executor (if any) is never invoked; only A's lease exists.
    assert b_claim_attempts == [None]
    leases = engine.by_type("AgentLease")
    assert len(leases) == 1
    assert leases[0]["owner_token"] == "hostA:1:agent-dispatch"


def test_execute_agent_task_turn_stale_lease_fenced_out_at_commit() -> None:
    """AU-P0-3 fencing: A's lease goes stale mid-execution and is re-claimed
    by B (crash-recovery re-claim); when A's belated execution finally tries
    to commit, the fencing-token CAS gate at finalize time REJECTS it — A's
    stale result must never overwrite B's (the newer holder's) state."""
    engine = _Gap6Engine()
    engine.add_node("task-8", "AgentTask", properties={"status": "pending"})

    def _slow_executor(claim: dict) -> str:
        # While A is "still running" its long executor, A's lease expires and
        # B re-claims the now-stale task — B is the new authoritative holder.
        reclaim = worker.claim_agent_task(
            engine,
            "task-8",
            token="hostB:2:agent-dispatch",
            now=1000.0 + worker.CLAIM_TTL_S + 10.0,
        )
        assert reclaim is not None  # B's reclaim succeeds — A's lease is stale
        return "did work for A (stale, should be rejected)"

    outcome_a = worker.execute_agent_task_turn(
        engine,
        "task-8",
        agent_id="agent-1",
        executor=_slow_executor,
        token="hostA:1:agent-dispatch",
        now=1000.0,
    )

    assert outcome_a == "fenced"
    # A's stale commit never landed: the task is left in B's reclaimed state
    # ("running", owned by B), never flipped to A's "completed".
    assert engine.nodes["task-8"]["status"] == "running"
    assert engine.by_type("OutcomeEvaluation") == []
    assert engine.by_type("Observation") == []
    leases = engine.by_type("AgentLease")
    assert len(leases) == 2  # A's original + B's reclaim
    assert leases[-1]["owner_token"] == "hostB:2:agent-dispatch"
    assert leases[-1]["lease_epoch"] == 2


# ---------------------------------------------------------------------------
# L15 — the engine-native claim path must fail CLOSED on a fence-check error
# (the KG best-effort path keeps its existing fail-OPEN posture).
# ---------------------------------------------------------------------------


class _RaisingLeaseEngine(_Gap6Engine):
    """``_Gap6Engine`` whose ``:AgentLease`` fence-check query always raises,
    isolating L15's fail-open/fail-closed split from every other engine
    interaction on the commit path (policy decision, capability grant, ...),
    which all still resolve normally."""

    def query_cypher(self, q, params=None):
        if "AgentLease {resource_id: $rid}" in q:
            raise RuntimeError("engine query transport error")
        return super().query_cypher(q, params)


def test_fence_still_valid_kg_claim_fails_open_on_query_error() -> None:
    """Regression guard: the KG best-effort path is UNCHANGED by L15 — a
    fence-check query error must not block a legitimate commit on this path
    (an audit-read hiccup must never block a legitimate commit)."""
    engine = _RaisingLeaseEngine()
    claim = {"fence_token": 1, "_claim_backend": "kg"}
    assert worker._fence_still_valid(engine, "task-kg", claim, token="hostA:1") is True

    # A claim with no `_claim_backend` marker at all (e.g. a hand-built test
    # fixture, or a pre-L15 caller) must also default to the KG fail-open
    # posture — only an EXPLICIT "engine" marker fails closed.
    unmarked_claim = {"fence_token": 1}
    assert (
        worker._fence_still_valid(engine, "task-kg-2", unmarked_claim, token="hostA:1")
        is True
    )


def test_fence_still_valid_engine_native_claim_fails_closed_on_query_error() -> None:
    """L15: the engine-native claim path must fail CLOSED — a worker that
    cannot confirm it still holds the lease must NOT commit."""
    engine = _RaisingLeaseEngine()
    claim = {"fence_token": 1, "_claim_backend": "engine"}
    assert (
        worker._fence_still_valid(engine, "task-engine", claim, token="hostA:1")
        is False
    )


def test_fence_still_valid_engine_native_claim_fails_closed_with_no_engine() -> None:
    """L15: an engine-native claim with no engine client to verify against
    is likewise unconfirmable — fail CLOSED, not open."""
    claim = {"fence_token": 1, "_claim_backend": "engine"}
    assert (
        worker._fence_still_valid(None, "task-engine", claim, token="hostA:1") is False
    )


def test_execute_agent_task_turn_engine_native_claim_aborts_commit_on_fence_query_error(
    monkeypatch,
) -> None:
    """L15 end-to-end: with ``AGENT_CLAIM_BACKEND=engine`` and the fence-check
    query raising at commit time, the turn must be ABORTED (never
    ``"completed"``) and no writeback may land — an engine-native worker that
    cannot confirm it still holds the lease must not commit, unlike the KG
    best-effort path's fail-open posture."""
    from agent_utilities.orchestration import engine_claim

    monkeypatch.setenv("AGENT_CLAIM_BACKEND", "engine")

    def _fake_try_engine_claim(task_id, *, token, now, claim_ttl_s):
        return {
            "task_id": task_id,
            "lease_id": "lease:engine:live",
            "dag_id": "",
            "checkpoint_id": None,
            "depends_on_task_ids": [],
            "fence_token": 1,
            "_claim_backend": engine_claim.AGENT_CLAIM_BACKEND_ENGINE,
        }

    monkeypatch.setattr(engine_claim, "_try_engine_claim", _fake_try_engine_claim)

    engine = _RaisingLeaseEngine()
    engine.add_node("task-10", "AgentTask", properties={"status": "pending"})

    outcome = worker.execute_agent_task_turn(
        engine,
        "task-10",
        agent_id="agent-1",
        executor=lambda claim: "ran via the engine-native claim",
    )

    assert outcome == "fenced"
    # No writeback landed for the unverifiable commit — the task is left
    # exactly where the (mocked) engine-native claim left it, never flipped
    # to "completed" with a fabricated success outcome.
    assert engine.nodes["task-10"]["status"] == "pending"
    assert engine.by_type("OutcomeEvaluation") == []
    assert engine.by_type("Observation") == []
