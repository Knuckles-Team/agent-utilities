"""Queue-only agent dispatch and WorkItem authority contracts."""

from __future__ import annotations

import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_utilities.orchestration import agent_dispatch
from agent_utilities.orchestration.agent_dispatch import (
    AGENT_TURNS_TOPIC,
    KIND_GOAL_LOOP,
    AgentTurnEnvelope,
)
from agent_utilities.orchestration.agent_dispatch_worker import (
    WorkItemLeaseGuard,
    WorkItemLeaseLost,
    worker_token,
)


def test_envelope_round_trip_carries_references_only() -> None:
    envelope = AgentTurnEnvelope(
        session_id="session-ref",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-ref",
        tenant="tenant-ref",
        prio_bucket=1,
    )
    restored = AgentTurnEnvelope.from_item(envelope.to_item())
    assert restored == envelope
    assert restored.payload_ref == "goal-ref"
    assert not hasattr(restored, "payload")


def test_envelope_job_id_is_full_width_uuid_not_truncated_hex8():
    """AU-P0-3: the prior ``dispatch-{uuid4().hex[:8]}`` kept only 32 bits of
    entropy (~50% collision odds by ~77k ids); job_id must now carry the FULL
    128-bit uuid4 hex — 32 hex chars after the ``dispatch-`` prefix, not 8."""
    seen: set[str] = set()
    for _ in range(200):
        env = AgentTurnEnvelope(session_id="sess-1")
        assert env.job_id.startswith("dispatch-")
        suffix = env.job_id[len("dispatch-") :]
        assert len(suffix) == 32, f"expected a full 32-hex-char uuid4, got {suffix!r}"
        int(suffix, 16)  # must be valid hex
        assert re.fullmatch(r"dispatch-[0-9a-f]{32}", env.job_id)
        seen.add(env.job_id)
    # 200 independently generated full-width ids must all be distinct.
    assert len(seen) == 200


def test_envelope_carries_references_not_bodies():
    env = AgentTurnEnvelope(session_id="s", payload_ref="goal-9")
    assert "objective" not in env.to_item()
    assert env.to_item()["payload_ref"] == "goal-9"


def test_session_partition_key_precedes_tenant() -> None:
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        partition_key_for,
    )

    first = AgentTurnEnvelope(session_id="session-ref", tenant="tenant-a").to_item()
    second = AgentTurnEnvelope(session_id="session-ref", tenant="tenant-b").to_item()
    assert partition_key_for(first) == partition_key_for(second)


def test_sqlite_is_queue_transport_not_inline_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        agent_dispatch, "_sqlite_queue_path", lambda: str(tmp_path / "dispatch.db")
    )
    queue = agent_dispatch.create_dispatch_queue(
        SimpleNamespace(
            task_queue_backend="sqlite",
            queue_backend="sqlite",
            state_db_uri=None,
        )
    )
    queue.put(AgentTurnEnvelope(session_id="session-ref").to_item())
    assert queue.get_queue_size() == 1


def test_kafka_dispatch_uses_fixed_topic_and_group(monkeypatch) -> None:
    captured: dict = {}

    class FakeKafkaQueue:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.kafka_queue_backend.KafkaQueueBackend",
        FakeKafkaQueue,
    )
    agent_dispatch.create_dispatch_queue(
        SimpleNamespace(
            task_queue_backend="kafka",
            queue_backend="kafka",
            state_db_uri=None,
            kafka_bootstrap_servers="broker.invalid:9092",
            agent_turns_partitions=8,
        )
    )
    assert captured["tasks_topic"] == AGENT_TURNS_TOPIC
    assert captured["consumer_group"] == "agent-dispatch"
    assert captured["partitions"] == 8


def test_worker_identity_is_opaque_and_process_stable() -> None:
    first = worker_token()
    assert first == worker_token()
    assert re.fullmatch(r"worker:[0-9a-f]{32}", first)
    assert "/" not in first and "\\" not in first and "@" not in first


def test_dispatch_module_has_no_inline_or_parallel_task_authority() -> None:
    assert not hasattr(agent_dispatch, "resolve_dispatch_backend")
    assert not hasattr(agent_dispatch, "dispatch_queue_enabled")


def test_dispatch_lease_defaults_meet_published_rto() -> None:
    from agent_utilities.core.config import AgentConfig

    contract = yaml.safe_load(
        (Path(__file__).parents[2] / "scripts/scale/workload_contract.yml").read_text()
    )
    config = AgentConfig()
    assert config.agent_dispatch_renew_interval_s < config.agent_dispatch_claim_ttl_s
    assert config.agent_dispatch_claim_ttl_s <= contract["availability"]["rto_seconds"]

    guard = WorkItemLeaseGuard(
        object(),
        "workitem-ref",
        {"work_item_id": "workitem-ref"},
        lease_ttl_s=config.agent_dispatch_claim_ttl_s,
    )
    assert guard.heartbeat_interval_s == config.agent_dispatch_renew_interval_s


def test_dispatch_depth_probe_fails_closed() -> None:
    class BrokenQueue:
        def get_queue_size(self):
            raise ConnectionError("unavailable")

    with pytest.raises(ConnectionError):
        agent_dispatch.dispatch_queue_depth(BrokenQueue())


def test_dispatch_rejects_before_workitem_when_queue_is_full(monkeypatch) -> None:
    from agent_utilities.core.config import config
    from agent_utilities.knowledge_graph.core.queue_backend import MemoryQueueBackend

    queue = MemoryQueueBackend()
    queue.put({"job_id": "already-queued"})
    monkeypatch.setattr(config, "agent_dispatch_max_depth", 1)
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.session.resolve_session",
        lambda **_kwargs: SimpleNamespace(tenant="tenant-ref"),
    )
    with pytest.raises(agent_dispatch.DispatchQueueFull):
        agent_dispatch.enqueue_agent_turn(
            AgentTurnEnvelope(session_id="session-ref"), queue=queue
        )


def test_lease_guard_heartbeats_during_long_execution(monkeypatch) -> None:
    calls: list[float] = []

    def renew(*_args, **_kwargs):
        calls.append(time.monotonic())
        return True

    monkeypatch.setattr(
        "agent_utilities.orchestration.agent_dispatch_worker._fence_still_valid",
        renew,
    )
    guard = WorkItemLeaseGuard(
        object(),
        "workitem-ref",
        {"work_item_id": "workitem-ref"},
        lease_ttl_s=0.09,
        heartbeat_interval_s=0.01,
    )
    with guard:
        time.sleep(0.035)
    assert len(calls) >= 3


def test_lease_guard_rejects_side_effect_after_fence_loss(monkeypatch) -> None:
    outcomes = iter((True, False))
    monkeypatch.setattr(
        "agent_utilities.orchestration.agent_dispatch_worker._fence_still_valid",
        lambda *_args, **_kwargs: next(outcomes),
    )
    guard = WorkItemLeaseGuard(
        object(),
        "workitem-ref",
        {"work_item_id": "workitem-ref"},
        lease_ttl_s=10.0,
    )
    guard.start()
    try:
        with pytest.raises(WorkItemLeaseLost):
            guard.side_effect(lambda: pytest.fail("stale effect executed"))
    finally:
        guard.close()
        agent_dispatch.reset_dispatch_queue_for_tests(None)
    assert resp.status_code == 503
    nodes = list(_sessions._goal_engine().nodes.values())
    assert nodes and nodes[0]["status"] == "failed"


# ── graph_orchestrate dispatch seam ───────────────────────────────────────


class _FakeOrchEngine:
    """Just enough engine surface for Orchestrator.dispatch_task/status."""

    def __init__(self):
        self.graph = SimpleNamespace(nodes={})

    def add_node(self, node_id, node_type, properties=None):
        self.graph.nodes[node_id] = dict(properties or {})

    def query_cypher(self, q, params=None):
        return []


@pytest.fixture
def orchestrate_tool(monkeypatch):
    from agent_utilities.mcp import kg_server

    kg_server.ensure_tools_registered()
    engine = _FakeOrchEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    return kg_server, engine


@pytest.mark.asyncio
async def test_orchestrate_dispatch_inline_returns_legacy_string(orchestrate_tool):
    kg_server, engine = orchestrate_tool
    out = await kg_server._execute_tool(
        "graph_orchestrate", action="dispatch", task="summarize the repo"
    )
    assert out.startswith("Task dispatched. Job ID: orch-")
    job_id = out.rsplit(" ", 1)[-1]
    assert engine.graph.nodes[job_id]["status"] == "pending"


@pytest.mark.asyncio
async def test_orchestrate_dispatch_queue_mode_returns_job_handle(
    orchestrate_tool, fake_queue, monkeypatch
):
    kg_server, engine = orchestrate_tool
    monkeypatch.setattr(agent_dispatch, "dispatch_queue_enabled", lambda *a: True)
    out = await kg_server._execute_tool(
        "graph_orchestrate",
        action="dispatch",
        task="summarize the repo",
        agent_name="librarian",
    )
    handle = json.loads(out)
    assert handle["dispatch"] == "queued"
    assert handle["kind"] == KIND_ORCHESTRATOR_TASK
    job_id = handle["job_id"]
    assert handle["status_url"].endswith(f"/job/{job_id}")
    # Durable Task node is the payload of record; queue carries the reference.
    assert engine.graph.nodes[job_id]["status"] == "pending"
    _, item = fake_queue.get()
    env = AgentTurnEnvelope.from_item(item)
    assert env.payload_ref == job_id
    assert env.session_id == job_id  # bare dispatch: self-scoped session
    assert env.agent_name == "librarian"


# ── per-session execution guard ───────────────────────────────────────────


def test_session_execution_guard_is_mutually_exclusive():
    import threading

    overlaps: list[int] = []
    active = {"n": 0}
    lock = threading.Lock()

    def _work():
        with agent_dispatch.session_execution_guard("sess-x"):
            with lock:
                active["n"] += 1
                overlaps.append(active["n"])
            time.sleep(0.02)
            with lock:
                active["n"] -= 1

    threads = [threading.Thread(target=_work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert max(overlaps) == 1  # never two executors inside one session


def test_session_execution_guard_distinct_sessions_run_concurrently():
    import threading

    started = threading.Barrier(2, timeout=5)

    def _work(sid):
        with agent_dispatch.session_execution_guard(sid):
            started.wait()  # both inside their guards at once → no deadlock

    t1 = threading.Thread(target=_work, args=("sess-1",))
    t2 = threading.Thread(target=_work, args=("sess-2",))
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive()


# ── dispatch worker: claim / execute / writeback ──────────────────────────


@pytest.fixture
def queued_goal(dispatch_db, fake_queue, monkeypatch):
    """A goal enqueued in queue mode, ready for a worker to claim."""
    import asyncio

    monkeypatch.setattr(agent_dispatch, "dispatch_queue_enabled", lambda *a: True)
    resp = asyncio.run(
        _sessions.create_goal(
            _FakeRequest(
                {
                    "objective": "worker goal",
                    "max_iterations": 1,
                    "validation_cmd": "true",
                }
            )
        )
    )
    return json.loads(resp.body)


def test_worker_claims_executes_and_writes_back(dispatch_db, fake_queue, queued_goal):
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    goal_id = queued_goal["goal_id"]

    item_id, payload = fake_queue.get()
    env = AgentTurnEnvelope.from_item(payload)
    outcome = worker.execute_agent_turn(env, token="hostA:1:agent-dispatch")
    assert outcome == "completed"
    fake_queue.ack(item_id)

    node = _goal_node(goal_id)
    assert node["status"] == "completed"  # run_goal_loop wrote back durably
    assert node["total_iterations"] == 1
    sessions = _rows(dispatch_db, "sessions")
    assert sessions[0]["status"] == "completed"
    turns = _rows(dispatch_db, "turns")
    assert any(t["role"] == "assistant" for t in turns)  # iteration turn appended
    assert fake_queue.get_queue_size() == 0


def test_worker_skips_duplicate_delivery_of_finished_goal(
    dispatch_db, fake_queue, queued_goal
):
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    _, payload = fake_queue.get()
    env = AgentTurnEnvelope.from_item(payload)
    assert worker.execute_agent_turn(env) == "completed"
    # Redelivery of the same envelope (at-least-once) is an idempotent skip.
    assert worker.execute_agent_turn(env) == "skipped"


def test_worker_skips_goal_with_fresh_live_claim(dispatch_db, fake_queue, queued_goal):
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    goal_id = queued_goal["goal_id"]
    # A fresh live claim by another worker (status running, recent) → skip.
    _sessions._goal_engine().add_node(
        goal_id,
        "Concept",
        properties={
            "status": "running",
            "owner_host": "hostB:9:agent-dispatch",
            "updated_at": time.time(),
        },
    )
    _, payload = fake_queue.get()
    env = AgentTurnEnvelope.from_item(payload)
    assert worker.execute_agent_turn(env) == "skipped"


def test_crash_requeue_stale_claim_is_reclaimed(dispatch_db, fake_queue, queued_goal):
    """Worker crash mid-turn: the envelope was never acked, the claim goes
    stale, and the redelivered envelope is re-claimed by another worker."""
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    goal_id = queued_goal["goal_id"]
    # Worker A claimed (status=running) then died — claim timestamp far in the past.
    _sessions._goal_engine().add_node(
        goal_id,
        "Concept",
        properties={
            "status": "running",
            "owner_host": "dead:1:agent-dispatch",
            "updated_at": time.time() - 2 * worker.CLAIM_TTL_S,
        },
    )

    # The unacked item is still in the queue (head-until-ack / redelivery).
    assert fake_queue.get_queue_size() == 1
    item_id, payload = fake_queue.get()
    env = AgentTurnEnvelope.from_item(payload)
    outcome = worker.execute_agent_turn(env, token="hostB:2:agent-dispatch")
    assert outcome == "completed"
    fake_queue.ack(item_id)
    assert _goal_node(goal_id)["status"] == "completed"


def test_worker_expires_past_deadline_turn(dispatch_db, fake_queue, queued_goal):
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    _, payload = fake_queue.get()
    payload = dict(payload, deadline_unix=time.time() - 10)
    env = AgentTurnEnvelope.from_item(payload)
    assert worker.execute_agent_turn(env) == "expired"
    node = _goal_node(queued_goal["goal_id"])
    assert node["status"] == "failed"
    assert "deadline" in node["error"].lower()


def test_consumer_loop_processes_and_acks_after(dispatch_db, fake_queue, queued_goal):
    import threading

    from agent_utilities.orchestration import agent_dispatch_worker as worker

    stop = threading.Event()

    # Reuse the populated fake queue; stop the loop once it drains.
    real_get = fake_queue.get

    def _get():
        item = real_get()
        if item is None:
            stop.set()
        return item

    fake_queue.get = _get
    worker.run_dispatch_consumer_loop(fake_queue, stop, idle_sleep_s=0.01)
    assert fake_queue.get_queue_size() == 0  # processed AND acked
    assert _goal_node(queued_goal["goal_id"])["status"] == "completed"


def test_consumer_loop_acks_poison_envelope(dispatch_db, fake_queue):
    """A malformed envelope is logged + acked — it never wedges the loop."""
    import threading

    from agent_utilities.orchestration import agent_dispatch_worker as worker

    fake_queue.put({"job_id": "poison", "kind": "goal_loop"})  # no session_id
    stop = threading.Event()
    real_get = fake_queue.get

    def _get():
        item = real_get()
        if item is None:
            stop.set()
        return item

    fake_queue.get = _get
    worker.run_dispatch_consumer_loop(fake_queue, stop, idle_sleep_s=0.01)
    assert fake_queue.get_queue_size() == 0


def test_two_workers_one_session_execute_serially(dispatch_db, fake_queue, monkeypatch):
    """Two workers, one session: per-session mutual exclusion holds end-to-end."""
    import asyncio
    import threading

    from agent_utilities.orchestration import agent_dispatch_worker as worker

    monkeypatch.setattr(agent_dispatch, "dispatch_queue_enabled", lambda *a: True)
    body = json.loads(
        asyncio.run(
            _sessions.create_goal(_FakeRequest({"objective": "serial goal"}))
        ).body
    )
    session_id = body["session_id"]
    # The same envelope delivered to BOTH workers (at-least-once duplicate).
    env = AgentTurnEnvelope(
        session_id=session_id, kind=KIND_GOAL_LOOP, payload_ref=body["goal_id"]
    )

    active = {"n": 0, "max": 0}
    gate = threading.Lock()

    # Replace the goal-loop body with a fast, deterministic no-op. This test
    # asserts ONLY the per-session mutual-exclusion contract (the
    # ``session_execution_guard`` + claim serialize the two workers); it does NOT
    # exercise the autonomous run loop. Calling the real ``_execute_goal_turn``
    # here ran the full ``LoopController.run_loop`` in each thread, which blocks
    # indefinitely on the engine selector in this unit context (and was the source
    # of the hang/flake — whether it returned at all was timing luck). The no-op
    # records concurrency and returns the same "completed" the real body returns,
    # so the contract assertion is unchanged while the test is deterministic.
    def _tracked(spec):
        with gate:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        try:
            time.sleep(0.05)  # widen the window for a concurrency violation to show
            return "completed"
        finally:
            with gate:
                active["n"] -= 1

    monkeypatch.setattr(worker, "_execute_goal_turn", _tracked)
    outcomes: list[str] = []

    def _run(token):
        outcomes.append(worker.execute_agent_turn(env, token=token))

    t1 = threading.Thread(target=_run, args=("hostA:1:agent-dispatch",))
    t2 = threading.Thread(target=_run, args=("hostB:2:agent-dispatch",))
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert active["max"] == 1  # never concurrent within one session
    assert sorted(outcomes) == ["completed", "skipped"]  # exactly one executed


def test_orchestrator_task_claim_execute_writeback(fake_queue, monkeypatch):
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    class _TaskEngine(_FakeOrchEngine):
        def query_cypher(self, q, params=None):
            node = self.graph.nodes.get((params or {}).get("id"))
            if node is None:
                return []
            return [
                {
                    "s": node.get("status"),
                    "d": node.get("description"),
                    "cu": node.get("claim_unix"),
                }
            ]

        def _update_task_status(self, job_id, status, meta=None):
            node = self.graph.nodes.setdefault(job_id, {})
            node["status"] = status
            node.update(meta or {})

    engine = _TaskEngine()
    engine.add_node(
        "orch-abc", "Task", properties={"status": "pending", "description": "do it"}
    )

    async def _fake_execute_agent(self, **kw):
        return f"ran {kw['task']} as {kw['agent_name'] or 'default'}"

    from agent_utilities.orchestration.manager import Orchestrator

    monkeypatch.setattr(Orchestrator, "execute_agent", _fake_execute_agent)
    monkeypatch.setattr(Orchestrator, "__init__", lambda self, engine: None)

    env = AgentTurnEnvelope(
        session_id="orch-abc",
        kind=KIND_ORCHESTRATOR_TASK,
        payload_ref="orch-abc",
        agent_name="librarian",
    )
    assert worker.execute_agent_turn(env, engine) == "completed"
    node = engine.graph.nodes["orch-abc"]
    assert node["status"] == "completed"
    assert "librarian" in node["result"]
    assert node["executed_by"].endswith(":agent-dispatch")
    # Redelivery is an idempotent skip.
    assert worker.execute_agent_turn(env, engine) == "skipped"


# ── claim_agent_task: generalized :AgentTask/:AgentLease claim (C3/Phase 3a) ──
#
# CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Graph-Native Agent-OS Objects
#
# Mirrors the claim_goal_run / claim_orchestrator_task tests above: same
# stale-claim-aware idempotency contract, generalized from an inline
# ownership stamp on the claimed node to a dedicated :AgentLease node.


class _AgentTaskEngine(_FakeOrchEngine):
    """Minimal engine double for :AgentTask + :AgentLease.

    ``add_node`` MERGEs into any existing node (mirrors the real engine's
    MERGE+SET upsert semantics) — ``claim_agent_task`` writes a partial
    ``{"status": "running"}`` update that must not clobber the task's other
    fields, exactly like ``claim_orchestrator_task``'s partial claim stamp.
    """

    def add_node(self, node_id, node_type, properties=None):
        node = self.graph.nodes.setdefault(node_id, {})
        node["type"] = node_type
        node.update(properties or {})

    def query_cypher(self, q, params=None):
        params = params or {}
        if "AgentTask {id: $id}" in q:
            node = self.graph.nodes.get(params.get("id"))
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
                for n in self.graph.nodes.values()
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
        return []


def test_claim_agent_task_unknown_task_is_skipped():
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    engine = _AgentTaskEngine()
    assert worker.claim_agent_task(engine, "does-not-exist") is None


def test_claim_agent_task_terminal_status_is_duplicate_skip():
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    engine = _AgentTaskEngine()
    engine.add_node("task-1", "AgentTask", properties={"status": "completed"})
    assert worker.claim_agent_task(engine, "task-1") is None


def test_claim_agent_task_claims_and_writes_lease():
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    engine = _AgentTaskEngine()
    engine.add_node(
        "task-1",
        "AgentTask",
        properties={
            "status": "ready",
            "dag_id": "dag-1",
            "depends_on_task_ids": ["dag-1:task:a"],
        },
    )
    claim = worker.claim_agent_task(
        engine, "task-1", token="hostA:1:agent-dispatch", now=1000.0
    )
    assert claim == {
        "task_id": "task-1",
        "lease_id": claim["lease_id"],
        "dag_id": "dag-1",
        "checkpoint_id": None,
        "depends_on_task_ids": ["dag-1:task:a"],
        "fence_token": 1,
        # L15: the KG claim path stamps its own backend marker so
        # `_fence_still_valid` knows the fail-OPEN posture applies to it.
        "_claim_backend": "kg",
    }
    assert claim["lease_id"].startswith("lease:task-1:")
    assert engine.graph.nodes["task-1"]["status"] == "running"

    lease = engine.graph.nodes[claim["lease_id"]]
    assert lease["type"] == "AgentLease"
    assert lease["owner_token"] == "hostA:1:agent-dispatch"
    assert lease["resource_id"] == "task-1"
    assert lease["acquired_at"] == 1000.0
    assert lease["lease_expires_at"] == 1000.0 + worker.CLAIM_TTL_S
    assert lease["lease_epoch"] == 1


def test_claim_agent_task_skips_task_with_fresh_live_lease():
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    engine = _AgentTaskEngine()
    engine.add_node("task-1", "AgentTask", properties={"status": "running"})
    engine.add_node(
        "lease:task-1:aaa",
        "AgentLease",
        properties={
            "owner_token": "hostB:9:agent-dispatch",
            "resource_id": "task-1",
            "acquired_at": 1000.0,
            "lease_expires_at": 1000.0 + worker.CLAIM_TTL_S,
        },
    )
    # A live worker holds a lease that has not yet expired -> skip.
    assert worker.claim_agent_task(engine, "task-1", now=1500.0) is None
    # Not reclaimed: no new lease written, task still 'running' under the
    # original owner.
    assert engine.graph.nodes["task-1"]["status"] == "running"


def test_claim_agent_task_reclaims_stale_lease_crash_recovery():
    """Worker A claimed the task then died before writeback; the lease went
    stale — a redelivered/rescanned task is re-claimed by worker B."""
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    engine = _AgentTaskEngine()
    engine.add_node(
        "task-1", "AgentTask", properties={"status": "running", "dag_id": "dag-1"}
    )
    stale_expiry = 1000.0
    engine.add_node(
        "lease:task-1:dead",
        "AgentLease",
        properties={
            "owner_token": "dead:1:agent-dispatch",
            "resource_id": "task-1",
            "acquired_at": stale_expiry - worker.CLAIM_TTL_S,
            "lease_expires_at": stale_expiry,
        },
    )
    now = stale_expiry + 10.0  # past expiry -> stale, re-claimable
    claim = worker.claim_agent_task(
        engine, "task-1", token="hostB:2:agent-dispatch", now=now
    )
    assert claim is not None
    assert claim["dag_id"] == "dag-1"
    new_lease = engine.graph.nodes[claim["lease_id"]]
    assert new_lease["owner_token"] == "hostB:2:agent-dispatch"
    assert new_lease["lease_expires_at"] == now + worker.CLAIM_TTL_S
    # The stale lease from the dead worker is left as-is (a fresh lease node
    # is written instead of mutating the old one) but no longer wins the
    # "most recent lease" ordering.
    assert (
        engine.graph.nodes["lease:task-1:dead"]["owner_token"]
        == "dead:1:agent-dispatch"
    )


def test_claim_agent_task_default_token_and_now():
    """Omitting token/now falls back to worker_token()/time.time(), matching
    the other two claim helpers' defaulting behavior."""
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    engine = _AgentTaskEngine()
    engine.add_node("task-1", "AgentTask", properties={"status": "pending"})
    claim = worker.claim_agent_task(engine, "task-1")
    assert claim is not None
    lease = engine.graph.nodes[claim["lease_id"]]
    assert lease["owner_token"].endswith(":agent-dispatch")


# ── fleet-visible placement: heartbeats, topology, metrics ────────────────


def test_worker_heartbeat_upserts_and_lists(dispatch_db):
    agent_dispatch.record_dispatch_worker_heartbeat(
        "hostA:1:agent-dispatch:0",
        host="hostA",
        capacity=2,
        active_sessions=["sess-1"],
        queue_backend="SQLiteTaskQueue",
    )
    # Second beat updates in place (no duplicate row).
    agent_dispatch.record_dispatch_worker_heartbeat(
        "hostA:1:agent-dispatch:0",
        host="hostA",
        capacity=2,
        active_sessions=[],
        queue_backend="SQLiteTaskQueue",
    )
    workers = agent_dispatch.list_dispatch_workers()
    assert len(workers) == 1
    w = workers[0]
    assert w["worker_id"] == "hostA:1:agent-dispatch:0"
    assert w["host"] == "hostA"
    assert w["capacity"] == 2
    assert w["active_sessions"] == []
    assert w["queue_backend"] == "SQLiteTaskQueue"


def test_stale_workers_drop_out_of_topology(dispatch_db):
    agent_dispatch.record_dispatch_worker_heartbeat("dead:1:agent-dispatch:0")
    conn = sqlite3.connect(str(dispatch_db))
    conn.execute(
        "UPDATE dispatch_workers SET last_heartbeat = ?",
        (time.time() - 10 * agent_dispatch.WORKER_HEARTBEAT_TTL_S,),
    )
    conn.commit()
    conn.close()
    assert agent_dispatch.list_dispatch_workers() == []


@pytest.mark.asyncio
async def test_fleet_topology_surfaces_dispatch_workers(dispatch_db):
    from agent_utilities.gateway import fleet

    agent_dispatch.record_dispatch_worker_heartbeat(
        "hostB:7:agent-dispatch:0", host="hostB", active_sessions=["sess-9"]
    )
    resp = await fleet.fleet_topology(_FakeRequest({}))
    body = json.loads(resp.body)
    assert body["totals"]["dispatch_workers"] == 1
    assert body["dispatch_workers"][0]["host"] == "hostB"
    assert body["dispatch_workers"][0]["active_sessions"] == ["sess-9"]


def test_dispatch_metrics_registered_on_gateway_registry():
    from agent_utilities.observability import gateway_metrics as gm

    assert "DISPATCH_QUEUE_DEPTH" in gm.__all__
    assert "DISPATCH_TURNS" in gm.__all__
    assert "DISPATCH_WORKERS" in gm.__all__
    # Usable regardless of whether prometheus_client is installed.
    gm.DISPATCH_QUEUE_DEPTH.labels(backend="FakeDispatchQueue").set(3.0)
    gm.DISPATCH_TURNS.labels(outcome="completed").inc()
    gm.DISPATCH_WORKERS.set(2.0)


def test_consumer_loop_heartbeats_into_registry(dispatch_db, fake_queue):
    import threading

    from agent_utilities.orchestration import agent_dispatch_worker as worker

    stop = threading.Event()
    real_get = fake_queue.get

    def _get():
        item = real_get()
        if item is None:
            stop.set()
        return item

    fake_queue.get = _get
    worker.run_dispatch_consumer_loop(
        fake_queue, stop, worker_id="hostC:3:agent-dispatch:0", idle_sleep_s=0.01
    )
    workers = agent_dispatch.list_dispatch_workers()
    assert [w["worker_id"] for w in workers] == ["hostC:3:agent-dispatch:0"]
    assert workers[0]["queue_backend"] == "FakeDispatchQueue"


def test_job_status_reports_executing_worker_and_host(fake_queue, monkeypatch):
    """graph_orchestrate job/{id}: the Task node carries the claim/exec stamps."""
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    class _TaskEngine(_FakeOrchEngine):
        def query_cypher(self, q, params=None):
            node = self.graph.nodes.get((params or {}).get("id"))
            if node is None:
                return []
            return [
                {
                    "s": node.get("status"),
                    "d": node.get("description"),
                    "cu": node.get("claim_unix"),
                }
            ]

        def _update_task_status(self, job_id, status, meta=None):
            node = self.graph.nodes.setdefault(job_id, {})
            node["status"] = status
            node.update(meta or {})

    engine = _TaskEngine()
    engine.add_node(
        "orch-xyz", "Task", properties={"status": "pending", "description": "task"}
    )

    async def _fake_execute_agent(self, **kw):
        return "done"

    from agent_utilities.orchestration.manager import Orchestrator

    monkeypatch.setattr(Orchestrator, "execute_agent", _fake_execute_agent)
    monkeypatch.setattr(Orchestrator, "__init__", lambda self, engine: None)

    env = AgentTurnEnvelope(
        session_id="orch-xyz", kind=KIND_ORCHESTRATOR_TASK, payload_ref="orch-xyz"
    )
    worker.execute_agent_turn(env, engine)

    # The existing job/{id} surface (Orchestrator.get_task_status) reads this node.
    from agent_utilities.orchestration.manager import Orchestrator as RealOrch

    status = RealOrch.get_task_status(SimpleNamespace(engine=engine), "orch-xyz")
    assert status["status"] == "completed"
    assert status["executed_by"].endswith(":agent-dispatch")
    import socket as _socket

    assert engine.graph.nodes["orch-xyz"]["dispatch_host"] == _socket.gethostname()
