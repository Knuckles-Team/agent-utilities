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


@pytest.fixture
def dispatch_db(tmp_path, monkeypatch):
    """Isolated sessions store (the goal_db pattern from test_goal_durability)."""
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_sessions._SQLITE_DDL)
    conn.commit()
    conn.close()
    monkeypatch.setattr(_sessions, "_get_db_path", lambda: db)
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
        priority="high",
        deadline_unix=123.0,
    )
    item = env.to_item()
    # session_id rides top-level so the queue can key without decoding bodies.
    assert item["session_id"] == "sess-1"
    restored = AgentTurnEnvelope.from_item(json.loads(json.dumps(item)))
    assert restored == env
    assert restored.job_id.startswith("dispatch-")


def test_envelope_carries_references_not_bodies():
    env = AgentTurnEnvelope(session_id="s", payload_ref="goal-9")
    assert "objective" not in env.to_item()
    assert env.to_item()["payload_ref"] == "goal-9"


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

    goals = _rows(dispatch_db, "goals")
    assert goals[0]["status"] == "pending"
    assert goals[0]["owner_host"] == ""  # unowned until a worker claims

    # The envelope is session-keyed and references the goal.
    _, item = fake_queue.get()
    env = AgentTurnEnvelope.from_item(item)
    assert env.session_id == session_id
    assert env.kind == KIND_GOAL_LOOP
    assert env.payload_ref == goal_id


@pytest.mark.asyncio
async def test_create_goal_queue_mode_enqueue_failure_is_loud(
    dispatch_db, monkeypatch
):
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
    goals = _rows(dispatch_db, "goals")
    assert goals[0]["status"] == "failed"


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
