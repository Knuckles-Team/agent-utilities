"""The agents-as-data activation layer (CONCEPT:AU-ORCH.dispatch.agents-as-data-activation, ADR-6 / W2.3).

A dormant agent is a durable ``eg-statechart`` ``MachineInstance`` + a per-tenant graph
node (NO thread/connection/heartbeat); an *event* creates a WorkItem (the ADR-5 substrate,
CAS-leased) that a stateless worker claims, runs under the ADR-4 identity chain + W2.4 QoS
lane, and commits. These tests exercise the WHOLE loop against a composed in-memory double
(the canonical :class:`NativeEngine` WorkItem verbs + :class:`FakeStatechartClient` from
``test_work_item``, extended with the ``:AgentInstance`` / ``:AgentMessage`` reads the
activation layer adds). No live epistemic-graph engine is required.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from agent_utilities.core.resource_priority import PriorityClass, current_priority
from agent_utilities.orchestration import agent_activation as aa
from agent_utilities.orchestration import work_item as wi
from agent_utilities.security import delegation as _delegation

# Reuse the canonical, faithful WorkItem + statechart doubles (native lease/fencing
# semantics + a real eg-statechart reference interpreter).
from tests.unit.orchestration.test_work_item import FakeStatechartClient, NativeEngine


class ActivationEngine(NativeEngine):
    """The :class:`NativeEngine` WorkItem/statechart double, extended with the
    ``:AgentInstance`` and ``:AgentMessage`` reads the activation layer performs."""

    def query_cypher(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        q = " ".join(cypher.split())

        if q.startswith("MATCH (a:AgentInstance {id: $id})"):
            node = self.nodes.get(str(params["id"]))
            if node is None or node.get("label") != aa._INSTANCE_LABEL:
                return []
            return [
                {
                    "id": params["id"],
                    "agent_name": node.get("agent_name"),
                    "tenant": node.get("tenant"),
                    "template_def_id": node.get("template_def_id"),
                    "statechart_instance_id": node.get("statechart_instance_id"),
                    "lifecycle_state": node.get("lifecycle_state"),
                    "model_id": node.get("model_id"),
                    "tool_ids": node.get("tool_ids") or [],
                    "subscriptions": node.get("subscriptions") or [],
                    "state_snapshot": node.get("state_snapshot"),
                    "mailbox_depth": node.get("mailbox_depth"),
                    "origin_principal": node.get("origin_principal"),
                }
            ]

        if q.startswith("MATCH (a:AgentInstance)"):
            out = []
            # Snapshot via list(): atomic under the GIL, so a concurrent worker pool
            # adding provenance/message nodes cannot raise "dict changed size".
            for nid, node in list(self.nodes.items()):
                if node.get("label") != aa._INSTANCE_LABEL:
                    continue
                if "a.tenant = $tenant" in q and node.get("tenant") != params.get(
                    "tenant"
                ):
                    continue
                if "a.lifecycle_state = $lifecycle_state" in q and node.get(
                    "lifecycle_state"
                ) != params.get("lifecycle_state"):
                    continue
                if "$subscription IN a.subscriptions" in q and params.get(
                    "subscription"
                ) not in (node.get("subscriptions") or []):
                    continue
                out.append(
                    {
                        "id": nid,
                        "agent_name": node.get("agent_name"),
                        "tenant": node.get("tenant"),
                        "lifecycle_state": node.get("lifecycle_state"),
                        "mailbox_depth": node.get("mailbox_depth"),
                        "subscriptions": node.get("subscriptions") or [],
                    }
                )
            return out

        if "MATCH (m:AgentMessage)" in q:
            iid = params.get("iid")
            only_unconsumed = "m.consumed = false" in q
            out = []
            for mid, node in list(self.nodes.items()):
                if node.get("label") != aa._MESSAGE_LABEL:
                    continue
                if node.get("instance_id") != iid:
                    continue
                if only_unconsumed and node.get("consumed"):
                    continue
                out.append(
                    {
                        "id": mid,
                        "message_ref": node.get("message_ref"),
                        "source": node.get("source"),
                        "correlation_id": node.get("correlation_id"),
                        "created_at": node.get("created_at"),
                    }
                )
            return out

        return super().query_cypher(cypher, params)

    # Convenience accessors for assertions.
    def by_label(self, label: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n.get("label") == label]


@pytest.fixture
def engine() -> ActivationEngine:
    return ActivationEngine()


@pytest.fixture(autouse=True)
def _clear_executor():
    """Every test starts with the default executor (no leaked bindings)."""
    aa.set_activation_executor(None)
    yield
    aa.set_activation_executor(None)


# ── registration: a dormant agent is TWO passive rows, no WorkItem ───────────────────


def test_register_creates_dormant_instance(engine: ActivationEngine) -> None:
    instance_id = aa.register_agent_instance(
        engine,
        agent_name="researcher",
        tenant="tenant-a",
        subscriptions=["data.new"],
        tool_ids=["search", "summarize"],
    )
    node = aa.get_agent_instance(engine, instance_id)
    assert node is not None
    assert node["lifecycle_state"] == aa.STATE_DORMANT
    assert node["agent_name"] == "researcher"
    # The durable lifecycle authority is a real MachineInstance in its initial state.
    sc = engine.statechart.get_state(node["statechart_instance_id"])
    assert sc["configuration"]["active"] == [aa.STATE_DORMANT]
    # Dormant means NO WorkItem exists yet — the instance costs only its two rows.
    assert engine.by_label("WorkItem") == []
    # Registration is idempotent by instance_id.
    assert (
        aa.register_agent_instance(
            engine, agent_name="researcher", tenant="tenant-a", instance_id=instance_id
        )
        == instance_id
    )


def test_list_and_terminate(engine: ActivationEngine) -> None:
    a = aa.register_agent_instance(
        engine, agent_name="a", tenant="t", subscriptions=["e1"]
    )
    aa.register_agent_instance(engine, agent_name="b", tenant="t", subscriptions=["e2"])
    assert {i["id"] for i in aa.list_agent_instances(engine, tenant="t")} == {
        a,
        *[i["id"] for i in aa.list_agent_instances(engine, subscription="e2")],
    }
    assert [i["id"] for i in aa.list_agent_instances(engine, subscription="e1")] == [a]
    assert aa.terminate_agent_instance(engine, a) is True
    assert aa.get_agent_instance(engine, a)["lifecycle_state"] == aa.STATE_TERMINATED


# ── the QoS lane is derived from the activation SOURCE (ADR-6 §5) ─────────────────────


@pytest.mark.parametrize(
    "source,expected",
    [
        (aa.ActivationSource.DIRECT, PriorityClass.INTERACTIVE),
        (aa.ActivationSource.TIMER, PriorityClass.ORCHESTRATION),
        (aa.ActivationSource.ORCHESTRATION, PriorityClass.ORCHESTRATION),
        (aa.ActivationSource.BROKER, PriorityClass.ORCHESTRATION),
        (aa.ActivationSource.CDC, PriorityClass.BACKGROUND_INGESTION),
        ("bogus-source", PriorityClass.ORCHESTRATION),
    ],
)
def test_activation_priority_class_from_source(source, expected) -> None:
    assert aa.activation_priority_class(source) is expected


def test_deliver_activation_creates_workitem_in_the_right_lane(
    engine: ActivationEngine,
) -> None:
    instance_id = aa.register_agent_instance(
        engine, agent_name="watcher", tenant="tenant-a"
    )
    # A direct call ⇒ Interactive ⇒ lowest prio_bucket (claimed first).
    wid = aa.deliver_activation(
        engine,
        instance_id,
        message_ref="msg:hello",
        source=aa.ActivationSource.DIRECT,
    )
    item = wi.get_work_item(engine, wid)
    assert item is not None
    assert item["kind"] == aa.WORK_ITEM_KIND
    assert item["status"] == "ready"
    assert item["payload_ref"] == instance_id
    assert int(item["prio_bucket"]) == PriorityClass.INTERACTIVE.rank
    assert item["metadata"]["priority_class"] == "interactive"
    assert item["metadata"]["agent_instance_id"] == instance_id
    # The message landed in the mailbox and bumped its depth.
    assert aa.get_agent_instance(engine, instance_id)["mailbox_depth"] == 1
    msgs = aa._mailbox_messages(engine, instance_id)
    assert [m["message_ref"] for m in msgs] == ["msg:hello"]


def test_interactive_activation_is_claimed_before_ingest(
    engine: ActivationEngine,
) -> None:
    ingest_inst = aa.register_agent_instance(engine, agent_name="ingest", tenant="t")
    hot_inst = aa.register_agent_instance(engine, agent_name="hot", tenant="t")
    # Submit the low-priority (CDC/ingest) activation FIRST, the interactive one second.
    aa.deliver_activation(
        engine, ingest_inst, message_ref="m:bulk", source=aa.ActivationSource.CDC
    )
    hot_wid = aa.deliver_activation(
        engine, hot_inst, message_ref="m:live", source=aa.ActivationSource.DIRECT
    )
    # claim_next orders by prio_bucket asc → the interactive activation wins despite
    # being submitted later (the backpressure/preemption the QoS lane encodes).
    claim = wi.claim_next(
        engine, resource_class=aa.WORK_ITEM_KIND, queue=aa.WORK_ITEM_KIND
    )
    assert claim is not None
    assert claim["work_item_id"] == hot_wid


def test_deliver_to_unknown_or_terminated_instance_fails_loud(
    engine: ActivationEngine,
) -> None:
    with pytest.raises(aa.AgentActivationError):
        aa.deliver_activation(engine, "agent-instance:nope", message_ref="m")
    inst = aa.register_agent_instance(engine, agent_name="x", tenant="t")
    aa.terminate_agent_instance(engine, inst)
    with pytest.raises(aa.AgentActivationError):
        aa.deliver_activation(engine, inst, message_ref="m")


# ── the E2E DEMO: message → activation → tool call under the identity chain →
#    provenance node visible → worker release (ADR-6 acceptance) ──────────────────────


def test_e2e_message_to_activation_to_provenance_to_release(
    engine: ActivationEngine,
) -> None:
    captured: dict[str, Any] = {}

    def mock_executor(ctx: aa.ActivationContext) -> aa.ActivationResult:
        # Proof the (mock) tool call runs INSIDE the identity chain + the QoS scope.
        deleg = _delegation.current_delegation()
        captured["chain"] = list(deleg.chain) if deleg else []
        captured["priority"] = current_priority()
        captured["messages"] = [m["message_ref"] for m in ctx.messages]
        return aa.ActivationResult(
            outcome="succeeded",
            tool_calls=({"tool": "search", "args_ref": "q:agents-as-data"},),
        )

    aa.set_activation_executor(mock_executor)

    # 1) a dormant agent registered with a real caller principal.
    instance_id = aa.register_agent_instance(
        engine,
        agent_name="researcher",
        tenant="tenant-a",
        origin_principal="user:alice",
        tool_ids=["search"],
    )
    # 2) a message activates it (a direct/interactive call).
    wid = aa.deliver_activation(
        engine,
        instance_id,
        message_ref="msg:find-papers",
        source=aa.ActivationSource.DIRECT,
    )

    # 3) a stateless worker drains exactly one activation.
    stop = threading.Event()
    processed = aa.run_activation_worker_loop(
        engine, stop, tenants=["tenant-a"], max_activations=1
    )
    assert processed == 1

    # 4) the tool call ran under the identity chain + interactive QoS scope. The
    # agent-instance identity is the per-run label (run_id = the WorkItem id, sanitized).
    agent_label = _delegation.agent_instance_label("researcher", wid)
    expected_chain = ["user:alice", agent_label]
    assert captured["messages"] == ["msg:find-papers"]
    assert captured["priority"] is PriorityClass.INTERACTIVE
    assert captured["chain"] == expected_chain

    # 5) provenance is visible: a :RunTrace carrying the delegation chain + a :ToolCall.
    traces = engine.by_label(aa._RUNTRACE_LABEL)
    assert len(traces) == 1
    trace = traces[0]
    assert trace["delegation_chain"] == expected_chain
    assert trace["principal"] == "user:alice"
    assert trace["outcome"] == "succeeded"
    tool_calls = engine.by_label(aa._TOOLCALL_LABEL)
    assert [c["tool"] for c in tool_calls] == ["search"]
    assert tool_calls[0]["delegation_chain"] == expected_chain

    # 6) the WorkItem committed terminally (released) and the instance is dormant again.
    item = wi.get_work_item(engine, wid)
    assert item["status"] == "succeeded"
    assert item["lease_owner"] is None  # lease released on commit
    assert (
        aa.get_agent_instance(engine, instance_id)["lifecycle_state"]
        == aa.STATE_DORMANT
    )
    # The mailbox was drained (depth back to 0, message consumed).
    assert aa.get_agent_instance(engine, instance_id)["mailbox_depth"] == 0
    assert aa._mailbox_messages(engine, instance_id) == []


def test_default_executor_acknowledges_mailbox_and_writes_provenance(
    engine: ActivationEngine,
) -> None:
    # No executor bound → the live default drains + writes one :ToolCall per message.
    instance_id = aa.register_agent_instance(engine, agent_name="ack", tenant="t")
    aa.deliver_activation(engine, instance_id, message_ref="m1", source="timer")
    aa.deliver_activation(engine, instance_id, message_ref="m2", source="timer")
    stop = threading.Event()
    aa.run_activation_worker_loop(engine, stop, tenants=["t"], max_activations=2)
    # Two activations queued but ONE drains the whole mailbox (coalescing): the second
    # finds it empty. Across both runs every message is acknowledged exactly once.
    tool_calls = engine.by_label(aa._TOOLCALL_LABEL)
    acked = sorted(c["args_ref"] for c in tool_calls)
    assert acked == ["m1", "m2"]
    assert (
        aa.get_agent_instance(engine, instance_id)["lifecycle_state"]
        == aa.STATE_DORMANT
    )


# ── heartbeats-only-while-active + dead-worker lease expiry re-queues (ADR-6 §3) ─────


def test_dead_worker_lease_expiry_requeues_activation(engine: ActivationEngine) -> None:
    instance_id = aa.register_agent_instance(engine, agent_name="w", tenant="t")
    wid = aa.deliver_activation(engine, instance_id, message_ref="m", source="direct")

    # A worker claims but then "dies" (never commits): the lease is short.
    dead = wi.claim_specific(
        engine, wid, token="dead-worker", now=100.0, lease_ttl_s=5.0
    )
    assert dead is not None
    item = wi.get_work_item(engine, wid)
    assert item["status"] == "leased"

    # No heartbeat renews it (the dead worker holds no live thread). After the lease
    # expires, a fresh worker's claim_next RECLAIMS it (the ADR-5 native lease recovery
    # — the lease IS the liveness signal, not a heartbeat on the instance).
    live = wi.claim_next(
        engine,
        resource_class=aa.WORK_ITEM_KIND,
        queue=aa.WORK_ITEM_KIND,
        token="live-worker",
        now=200.0,  # well past the 5s lease
    )
    assert live is not None
    assert live["work_item_id"] == wid
    assert live["lease_owner"] == "live-worker"
    # The reclaiming worker runs it to success.
    out = aa.process_one_activation(engine, live, token="live-worker", now=200.0)
    assert out == "committed"
    assert wi.get_work_item(engine, wid)["status"] == "succeeded"


def test_concurrent_activation_of_same_instance_defers_the_second(
    engine: ActivationEngine,
) -> None:
    instance_id = aa.register_agent_instance(engine, agent_name="solo", tenant="t")
    w1 = aa.deliver_activation(engine, instance_id, message_ref="m1", source="direct")
    w2 = aa.deliver_activation(engine, instance_id, message_ref="m2", source="direct")

    # Worker A claims w1 and drives the instance dormant→active, then holds it (executor
    # blocks) while worker B claims w2 and tries to activate the SAME instance.
    a = wi.claim_specific(engine, w1, token="A", now=1.0)
    wi.mark_running(engine, w1, a)
    sc_id = aa.get_agent_instance(engine, instance_id)["statechart_instance_id"]
    engine.statechart.send_event(sc_id, aa.EV_ACTIVATE, {})  # A activates the instance

    b = wi.claim_specific(engine, w2, token="B", now=2.0)
    wi.mark_running(engine, w2, b)
    # B sees the instance already active ⇒ defers w2 (does NOT double-run the instance).
    out_b = aa.process_one_activation(engine, b, token="B", now=2.0)
    assert out_b == "deferred"
    assert wi.get_work_item(engine, w2)["status"] == "ready"  # released for later retry


# ── the statechart template mirrors the eg canonical (structural parity) ─────────────


def test_agent_lifecycle_def_registers_and_instantiates_dormant(
    engine: ActivationEngine,
) -> None:
    def_id = aa.agent_lifecycle_def_id(engine)
    assert def_id  # content-addressed id
    # Idempotent: the same def registers to the same id.
    assert aa.agent_lifecycle_def_id(engine) == def_id
    inst = engine.statechart.instantiate(def_id, context={})
    assert inst["configuration"]["active"] == [aa.STATE_DORMANT]
    # The chart's activation cycle is dormant → active → dormant.
    activate = engine.statechart.send_event(inst["instance_id"], aa.EV_ACTIVATE, {})
    assert activate["fired"] is True
    assert activate["instance"]["configuration"]["active"] == [aa.STATE_ACTIVE]
    deactivate = engine.statechart.send_event(inst["instance_id"], aa.EV_DEACTIVATE, {})
    assert deactivate["instance"]["configuration"]["active"] == [aa.STATE_DORMANT]


def test_delegation_off_runs_legacy_identity_no_chain(
    engine: ActivationEngine, monkeypatch
) -> None:
    # With delegation OFF, no chain is built; provenance records an empty chain but the
    # activation still runs + commits (legacy identity).
    monkeypatch.setattr(
        _delegation, "delegation_mode", lambda: _delegation.DelegationMode.OFF
    )
    instance_id = aa.register_agent_instance(
        engine, agent_name="legacy", tenant="t", origin_principal="user:bob"
    )
    wid = aa.deliver_activation(engine, instance_id, message_ref="m", source="direct")
    stop = threading.Event()
    aa.run_activation_worker_loop(engine, stop, tenants=["t"], max_activations=1)
    assert wi.get_work_item(engine, wid)["status"] == "succeeded"
    trace = engine.by_label(aa._RUNTRACE_LABEL)[0]
    assert trace["delegation_chain"] == []
