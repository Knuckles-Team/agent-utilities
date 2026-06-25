"""Queue-driven agent dispatch (CONCEPT:ORCH-1.45).

Covers the envelope contract, the session-first partition-key precedence, the
inline-default seam (existing behavior byte-for-byte), the queue-mode job
handle on both enqueue seams (``graph_orchestrate action=dispatch`` and the
goal machinery), the dispatch worker's claim/execute/writeback cycle with a
fake queue + fake session store, per-session mutual exclusion, crash-requeue
via stale-claim-aware re-claims, and the worker heartbeat/topology surface.
No broker, no Postgres, no engine daemon required.
"""

from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from agent_utilities.core import sessions as _sessions
from agent_utilities.models.goal import GoalStatus
from agent_utilities.orchestration import agent_dispatch
from agent_utilities.orchestration.agent_dispatch import (
    KIND_GOAL_LOOP,
    KIND_ORCHESTRATOR_TASK,
    AgentTurnEnvelope,
    enqueue_agent_turn,
    resolve_dispatch_backend,
)


class FakeDispatchQueue:
    """Minimal QueueBackend-shaped fake (put/get/ack/get_queue_size)."""

    def __init__(self):
        self.items: list[tuple[int, dict]] = []
        self.acked: list[int] = []
        self._counter = 0

    def put(self, item: dict) -> None:
        self._counter += 1
        self.items.append((self._counter, item))

    def get(self):
        for entry in self.items:
            if entry[0] not in self.acked:
                return entry
        return None

    def ack(self, item_id) -> None:
        self.acked.append(item_id)
        self.items = [e for e in self.items if e[0] != item_id]

    def get_queue_size(self) -> int:
        return len(self.items)


class _GoalEngine:
    """Fake KG engine — the goal Loop-node store (CONCEPT:KG-2.78)."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add_node(self, nid, ntype, properties=None):
        cur = self.nodes.get(nid, {})
        cur.update({"type": ntype, **(properties or {})})
        self.nodes[nid] = cur

    def _row(self, nid, n):
        return {
            "goal_id": nid,
            "session_id": n.get("session_id", ""),
            "status": n.get("status", "pending"),
            "objective": n.get("objective", ""),
            "owner_host": n.get("owner_host", ""),
            "summary": n.get("summary", ""),
            "error": n.get("error", ""),
            "total_iterations": n.get("total_iterations", 0),
            "total_duration_ms": n.get("total_duration_ms", 0),
            "total_tool_calls": n.get("total_tool_calls", 0),
            "iterations_json": n.get("iterations_json", "[]"),
            "validation_cmd": n.get("validation_cmd", ""),
            "max_iterations": n.get("max_iterations", 20),
            "updated_at": n.get("updated_at", 0),
        }

    def query_cypher(self, q, params=None):
        params = params or {}
        if "c.id = $id" in q:
            n = self.nodes.get(params.get("id"))
            return [self._row(params.get("id"), n)] if n else []
        if "c.loop_kind = 'develop'" in q:
            return [
                self._row(i, n)
                for i, n in self.nodes.items()
                if n.get("loop_kind") == "develop"
            ]
        return []


def _goal_node(goal_id: str) -> dict:
    """The goal's KG Loop node (the durable record), via the patched engine."""
    return _sessions._goal_engine().nodes.get(goal_id, {})


@pytest.fixture
def dispatch_db(tmp_path, monkeypatch):
    """Isolated sessions store + the KG goal-node store (the one durable model)."""
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_sessions._SQLITE_DDL)
    conn.commit()
    conn.close()
    eng = _GoalEngine()
    monkeypatch.setattr(_sessions, "_get_db_path", lambda: db)
    monkeypatch.setattr(_sessions, "_goal_engine", lambda: eng)
    monkeypatch.setattr(_sessions, "_rehydrated", False)
    monkeypatch.setattr(_sessions, "active_goals", {})
    monkeypatch.setattr(_sessions, "background_goal_runs", {})
    return db


@pytest.fixture
def fake_queue(monkeypatch):
    q = FakeDispatchQueue()
    agent_dispatch.reset_dispatch_queue_for_tests(q)
    yield q
    agent_dispatch.reset_dispatch_queue_for_tests(None)


class _FakeRequest:
    def __init__(self, body: dict):
        self._body = body
        self.path_params: dict = {}
        self.query_params: dict = {}

    async def json(self) -> dict:
        return self._body


def _rows(db, table):
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]  # nosec B608
    conn.close()
    return rows


# ── envelope contract ─────────────────────────────────────────────────────


def test_envelope_round_trip():
    env = AgentTurnEnvelope(
        session_id="sess-1",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-1",
        tenant="acme",
        prio_bucket="high",  # legacy string coerced through the one normalizer
        deadline_unix=123.0,
    )
    item = env.to_item()
    # session_id rides top-level so the queue can key without decoding bodies.
    assert item["session_id"] == "sess-1"
    # The string priority was normalized to the shared 0..3 bucket (high → 1).
    assert env.prio_bucket == 1
    assert item["prio_bucket"] == 1
    restored = AgentTurnEnvelope.from_item(json.loads(json.dumps(item)))
    assert restored == env
    assert restored.job_id.startswith("dispatch-")


def test_envelope_carries_references_not_bodies():
    env = AgentTurnEnvelope(session_id="s", payload_ref="goal-9")
    assert "objective" not in env.to_item()
    assert env.to_item()["payload_ref"] == "goal-9"


def test_envelope_prio_bucket_unified_with_tasks():
    """Dispatch priority is the SAME 0..3 bucket as a :Task (CONCEPT:KG-2.113).

    The string-only path is gone — every spec (legacy string, int, numeric
    string) is coerced through the single shared normalizer, identical to how
    submit_task / schedules / loops normalize their bucket.
    """
    from agent_utilities.knowledge_graph.core.engine_tasks import _coerce_prio_bucket

    # default
    assert AgentTurnEnvelope(session_id="s").prio_bucket == 2
    # legacy strings → the same buckets the one normalizer yields
    for word in ("critical", "high", "normal", "background", "low"):
        env = AgentTurnEnvelope(session_id="s", prio_bucket=word)  # type: ignore[arg-type]
        assert env.prio_bucket == _coerce_prio_bucket(word)
    # int + numeric string, clamped into [0, 3]
    assert AgentTurnEnvelope(session_id="s", prio_bucket=0).prio_bucket == 0
    assert AgentTurnEnvelope(session_id="s", prio_bucket="1").prio_bucket == 1  # type: ignore[arg-type]
    assert AgentTurnEnvelope(session_id="s", prio_bucket=99).prio_bucket == 3
    # the bucket survives serialization onto the queue item
    assert (
        AgentTurnEnvelope(session_id="s", prio_bucket="high").to_item()["prio_bucket"]  # type: ignore[arg-type]
        == 1
    )


# ── partition key: session beats tenant ──────────────────────────────────


def test_partition_key_session_beats_tenant():
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        partition_key_for,
    )
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    item = AgentTurnEnvelope(session_id="sess-42", tenant="acme").to_item()
    actor = ActorContext("u1", ActorType.HUMAN, tenant_id="acme")
    with use_actor(actor):
        # Per-session serial execution is REQUIRED for turn coherence —
        # session outranks the ambient tenant (CONCEPT:ORCH-1.45).
        assert partition_key_for(item) == "session:sess-42"
    # And without the ambient actor too.
    assert partition_key_for(item) == "session:sess-42"


def test_partition_key_ingest_hierarchy_unchanged_without_session():
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        partition_key_for,
    )

    assert (
        partition_key_for({"job_id": "j", "props": {"full_path": "org/repo"}})
        == "corpus:org/repo"
    )


def test_same_session_turns_share_key_distinct_sessions_spread():
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        partition_key_for,
    )

    a1 = AgentTurnEnvelope(session_id="sess-a").to_item()
    a2 = AgentTurnEnvelope(session_id="sess-a").to_item()
    b = AgentTurnEnvelope(session_id="sess-b").to_item()
    assert partition_key_for(a1) == partition_key_for(a2) != partition_key_for(b)


# ── backend selection ─────────────────────────────────────────────────────


def test_resolve_dispatch_backend_defaults_inline():
    assert resolve_dispatch_backend(SimpleNamespace(agent_dispatch_backend=None)) == (
        "inline"
    )
    assert (
        resolve_dispatch_backend(SimpleNamespace(agent_dispatch_backend=" Queue "))
        == "queue"
    )


def test_resolve_dispatch_backend_rejects_unknown():
    with pytest.raises(ValueError, match="AGENT_DISPATCH_BACKEND"):
        resolve_dispatch_backend(SimpleNamespace(agent_dispatch_backend="celery"))


def test_default_config_is_inline():
    # The live AgentConfig default must preserve current behavior.
    assert resolve_dispatch_backend() == "inline"
    assert agent_dispatch.dispatch_queue_enabled() is False


def test_create_dispatch_queue_sqlite_default(tmp_path, monkeypatch):
    monkeypatch.setattr(
        agent_dispatch, "_sqlite_queue_path", lambda: str(tmp_path / "turns.db")
    )
    q = agent_dispatch.create_dispatch_queue(
        SimpleNamespace(
            task_queue_backend=None, queue_backend="sqlite", state_db_uri=None
        )
    )
    env = AgentTurnEnvelope(session_id="s1")
    q.put(env.to_item())
    assert q.get_queue_size() == 1
    item_id, item = q.get()
    assert AgentTurnEnvelope.from_item(item).session_id == "s1"
    q.ack(item_id)
    assert q.get_queue_size() == 0


def test_create_dispatch_queue_kafka_uses_agent_turns_topic(monkeypatch):
    captured: dict = {}

    class _FakeKafka:
        def __init__(self, **kw):
            captured.update(kw)

    import agent_utilities.knowledge_graph.core.kafka_queue_backend as kqb

    monkeypatch.setattr(kqb, "KafkaQueueBackend", _FakeKafka)
    agent_dispatch.create_dispatch_queue(
        SimpleNamespace(
            task_queue_backend="kafka",
            queue_backend="sqlite",
            state_db_uri=None,
            kafka_bootstrap_servers="broker:9092",
            agent_turns_partitions=4,
        )
    )
    assert captured["tasks_topic"] == agent_dispatch.AGENT_TURNS_TOPIC
    assert captured["consumer_group"] == agent_dispatch.DISPATCH_GROUP
    assert captured["partitions"] == 4
    assert captured["fail_loud"] is True  # explicit selection stays fail-loud


# ── enqueue: job handle ───────────────────────────────────────────────────


def test_enqueue_agent_turn_returns_handle(fake_queue):
    env = AgentTurnEnvelope(session_id="sess-7", kind=KIND_ORCHESTRATOR_TASK)
    handle = enqueue_agent_turn(env)
    assert handle["dispatch"] == "queued"
    assert handle["job_id"] == env.job_id
    assert fake_queue.get_queue_size() == 1
    _, item = fake_queue.get()
    assert item["session_id"] == "sess-7"


# ── goal machinery: inline default unchanged (live path) ─────────────────


@pytest.mark.asyncio
async def test_create_goal_inline_mode_unchanged(dispatch_db):
    """Default config: the goal loop still runs in-process — no queue, no handle."""
    resp = await _sessions.create_goal(
        _FakeRequest({"objective": "inline goal", "max_iterations": 1})
    )
    body = json.loads(resp.body)
    assert body["status"] == "success"
    assert "dispatch" not in body
    goal_id = body["goal_id"]
    run = _sessions.background_goal_runs.get(goal_id)
    assert run is not None  # in-process asyncio task spawned
    assert _sessions.active_goals[goal_id]["status"] == GoalStatus.RUNNING
    sessions = _rows(dispatch_db, "sessions")
    assert sessions[0]["status"] == "running"
    run["task"].cancel()


# ── goal machinery: queue mode returns a handle, executes nowhere ────────


@pytest.mark.asyncio
async def test_create_goal_queue_mode_enqueues_and_returns_handle(
    dispatch_db, fake_queue, monkeypatch
):
    monkeypatch.setattr(agent_dispatch, "dispatch_queue_enabled", lambda *a: True)
    resp = await _sessions.create_goal(
        _FakeRequest(
            {
                "objective": "queued goal",
                "max_iterations": 3,
                "validation_cmd": "true",
            }
        )
    )
    body = json.loads(resp.body)
    assert body["status"] == "success"
    assert body["dispatch"]["dispatch"] == "queued"
    goal_id = body["goal_id"]
    session_id = body["session_id"]

    # Nothing runs in this process.
    assert goal_id not in _sessions.background_goal_runs
    assert _sessions.active_goals[goal_id]["status"] == GoalStatus.PENDING

    # Durable record carries the spec (queue carries only references).
    sessions = _rows(dispatch_db, "sessions")
    assert sessions[0]["status"] == "queued"
    spec = json.loads(sessions[0]["metadata_json"])["goal_spec"]
    assert spec["objective"] == "queued goal"
    assert spec["validation_cmd"] == "true"
    assert spec["max_iterations"] == 3

    node = _goal_node(goal_id)
    assert node["status"] == "pending"
    assert node.get("owner_host", "") == ""  # unowned until a worker claims

    # The envelope is session-keyed and references the goal.
    _, item = fake_queue.get()
    env = AgentTurnEnvelope.from_item(item)
    assert env.session_id == session_id
    assert env.kind == KIND_GOAL_LOOP
    assert env.payload_ref == goal_id


@pytest.mark.asyncio
async def test_create_goal_queue_mode_enqueue_failure_is_loud(dispatch_db, monkeypatch):
    monkeypatch.setattr(agent_dispatch, "dispatch_queue_enabled", lambda *a: True)

    class _Broken:
        def put(self, item):
            raise ConnectionError("broker down")

    agent_dispatch.reset_dispatch_queue_for_tests(_Broken())
    try:
        resp = await _sessions.create_goal(_FakeRequest({"objective": "doomed"}))
    finally:
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
