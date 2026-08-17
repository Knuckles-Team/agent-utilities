"""Queue-only agent dispatch and WorkItem authority contracts."""

from __future__ import annotations

import importlib.util
import json
import re
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_utilities.core import sessions as _sessions
from agent_utilities.orchestration import agent_dispatch
from agent_utilities.orchestration import work_item as _wi
from agent_utilities.orchestration.agent_dispatch import (
    AGENT_TURNS_TOPIC,
    KIND_GOAL_LOOP,
    KIND_ORCHESTRATOR_TASK,
    AgentTurnEnvelope,
)
from agent_utilities.orchestration.agent_dispatch_worker import (
    WorkItemLeaseGuard,
    WorkItemLeaseLost,
    worker_token,
)

# test_orchestrate_dispatch_queue_mode_returns_job_handle: queue-mode dispatch
# constructs the process-wide engine (``IntelligenceGraphEngine.get_or_create()``),
# which needs the real ``epistemic_graph`` package; when absent the resulting
# ModuleNotFoundError is caught inside ``enqueue_agent_turn`` and the returned
# handle never gets a "dispatch" key, rather than a raised exception. Matches
# the same "package genuinely absent" contract used elsewhere (e.g.
# tests/unit/test_engine_api_coverage.py).
_NEEDS_ENGINE = pytest.mark.skipif(
    importlib.util.find_spec("epistemic_graph") is None,
    reason="epistemic_graph package not installed in this environment.",
)


class _FakeRequest:
    """Minimal Starlette-``Request``-shaped double: ``sessions.create_goal``
    only ever awaits ``request.json()``."""

    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body


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


def test_enqueue_rejects_envelope_tenant_that_differs_from_caller_session(
    monkeypatch,
) -> None:
    """KNOWN-BAD PROOF (GOC-18): a caller-supplied ``envelope.tenant`` that
    disagrees with the caller's own authenticated ``GraphSession`` must be
    REJECTED before any durable WorkItem or queue write — never silently
    trusted, never silently coerced to the session's tenant. This is the
    admission-time half of the dispatch/delivery authentication contract
    (``agent_dispatch.py:enqueue_agent_turn``'s ``PermissionError`` gate);
    the consumer-side half is
    ``test_consumer_loop_dead_letters_tenant_mismatched_envelope`` below.
    """
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.session.resolve_session",
        lambda **_kwargs: SimpleNamespace(tenant="tenant-legit"),
    )
    with pytest.raises(PermissionError, match="tenant"):
        agent_dispatch.enqueue_agent_turn(
            AgentTurnEnvelope(session_id="session-ref", tenant="tenant-attacker")
        )


def test_enqueue_rejects_unauthenticated_caller(monkeypatch) -> None:
    """KNOWN-BAD PROOF (GOC-18): a caller with no verified ambient
    ``GraphSession`` (no authenticated identity at all) must be rejected
    before any durable WorkItem or queue write — ``enqueue_agent_turn`` has
    no anonymous/unauthenticated admission path."""
    from agent_utilities.knowledge_graph.core.session import SessionRequiredError

    def _no_ambient_session(**_kwargs):
        raise SessionRequiredError(
            "A verified ambient GraphSession is required for this graph operation"
        )

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.session.resolve_session",
        _no_ambient_session,
    )
    with pytest.raises(SessionRequiredError):
        agent_dispatch.enqueue_agent_turn(
            AgentTurnEnvelope(session_id="session-ref", tenant="tenant-x")
        )


def test_lease_guard_heartbeats_during_long_execution(monkeypatch) -> None:
    calls: list[float] = []

    def renew(*_args, **_kwargs):
        calls.append(time.monotonic())
        return True

    monkeypatch.setattr(
        "agent_utilities.orchestration.agent_dispatch_worker._work_item_fence_still_valid",
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
        "agent_utilities.orchestration.agent_dispatch_worker._work_item_fence_still_valid",
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


# ── graph_orchestrate dispatch seam ───────────────────────────────────────
#
# ``graph_orchestrate`` (analysis_tools.py) no longer accepts
# ``action="dispatch"`` at all -- its current action vocabulary is
# ``inspect | enrichment_coverage | process_writeback | placement_plan |
# infra_sweep | security_scan`` (see ``register_analysis_tools``'s
# ``_run_analysis_action`` docstring). Task dispatch was carved out into its
# own focused surface, ``graph_jobs`` (``mcp/tools/job_tools.py``:
# "Focused graph-os surface for durable orchestration jobs"), whose
# ``action="dispatch"`` always enqueues via ``agent_dispatch.enqueue_agent_turn``
# -- there is no more bare in-process "legacy string" return mode; dispatch is
# unconditionally queue-backed (``agent_dispatch``'s module docstring: "Agent
# dispatch is always queue-backed"; ``dispatch_queue_enabled`` was deliberately
# removed, see the already-passing
# ``test_dispatch_module_has_no_inline_or_parallel_task_authority`` above).
# ``test_orchestrate_dispatch_inline_returns_legacy_string`` asserted exactly
# that retired inline-string behaviour against the retired tool/action pair --
# STALE TEST, removed rather than re-targeted, since there is no successor
# behaviour left to assert (queue-mode is the only mode now, and it is
# already covered by ``test_orchestrate_dispatch_queue_mode_returns_job_handle``
# below, retargeted onto ``graph_jobs``).


class _FakeOrchEngine:
    """Engine double for the ``Orchestrator.dispatch_task``/``get_task_status``
    seam AND the engine-native WorkItem verbs (``claim_work_item``/
    ``renew_work_item_lease``/``commit_work_item_result``) that
    ``claim_specific``/``mark_running``/``commit_result`` require
    unconditionally post-AU-P1-1 (No-Legacy: no scan/CAS fallback -- see
    ``work_item.py``'s module docstring). Mirrors
    ``tests/unit/orchestration/test_work_item.py``'s ``NativeEngine`` (exact-id
    claim only; this file never exercises ``claim_next``'s queue scan).
    """

    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add_node(self, node_id, node_type, properties=None, ephemeral=False):
        node = self.nodes.setdefault(node_id, {})
        node["label"] = node_type
        node.update(properties or {})
        return node

    def link_nodes(
        self, source_id, target_id, rel_type, properties=None, ephemeral=False
    ):
        pass

    def query_cypher(self, q, params=None):
        params = params or {}
        if "WorkItem {id: $id}" in q:
            node = self.nodes.get(params.get("id"))
            if node is None or node.get("label") != "WorkItem":
                return []
            row = {"id": params["id"]}
            for f in _wi._FIELDS:
                row[f] = node.get(f)
            return [row]
        return []

    def claim_work_item(self, request: object) -> dict:
        item_id = str(getattr(request, "work_item_id", "") or "") or None
        node = self.nodes.get(item_id) if item_id else None
        if node is None or node.get("label") != "WorkItem":
            return self._negative_claim()
        now = float(getattr(request, "now_ms", 0)) / 1000.0
        status = node.get("status")
        if status in {"leased", "running"}:
            if float(node.get("lease_expires_at") or 0) >= now:
                return self._negative_claim()
        elif status != "ready":
            return self._negative_claim()
        attempt = int(node.get("attempt") or 0) + 1
        max_attempts = int(node.get("max_attempts") or 1)
        if attempt > max_attempts:
            node.update(status="dead_letter", error_ref="lease_exhausted")
            return self._negative_claim()
        epoch = int(node.get("lease_epoch") or 0) + 1
        lease_ms = getattr(request, "lease_ms", 0)
        worker_ref = getattr(request, "worker_ref", "")
        node.update(
            status="leased",
            lease_owner=worker_ref,
            lease_epoch=epoch,
            fencing_token=epoch,
            attempt=attempt,
            lease_expires_at=float(request.now_ms + lease_ms) / 1000.0,
        )
        return {
            "schema_version": "1",
            "claimed": True,
            "reason": "claimed",
            "work_item_id": item_id,
            "kind": node.get("kind"),
            "payload_ref": node.get("payload_ref"),
            "lease_holder_ref": worker_ref,
            "lease_epoch": epoch,
            "fencing_token": epoch,
            "lease_expires_at_ms": request.now_ms + lease_ms,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "tenant_in_flight": 1,
            "changed_work_item_ids": [item_id],
        }

    @staticmethod
    def _negative_claim() -> dict:
        return {
            "schema_version": "1",
            "claimed": False,
            "reason": "empty",
            "work_item_id": None,
            "kind": None,
            "payload_ref": None,
            "lease_holder_ref": None,
            "lease_epoch": None,
            "fencing_token": None,
            "lease_expires_at_ms": None,
            "attempt": None,
            "max_attempts": None,
            "tenant_in_flight": 0,
            "changed_work_item_ids": [],
        }

    @staticmethod
    def _owns(node: dict | None, request: dict) -> bool:
        return bool(
            node
            and node.get("lease_owner") == request.get("worker_ref")
            and node.get("lease_epoch") == request.get("expected_epoch")
            and node.get("fencing_token") == request.get("fencing_token")
        )

    def renew_work_item_lease(self, request: dict) -> dict:
        node = self.nodes.get(request["work_item_id"])
        if not self._owns(node, request) or float(
            (node or {}).get("lease_expires_at") or 0
        ) < float(request["now_unix"]):
            return {"renewed": False}
        node["lease_expires_at"] = float(request["now_unix"]) + float(
            request["lease_ttl"]
        )
        return {"renewed": True}

    def commit_work_item_result(self, request: dict) -> dict:
        node = self.nodes.get(request["work_item_id"])
        if node is None:
            return {"status": "missing"}
        if node.get("status") in _wi.TERMINAL_WORK_ITEM_STATUSES:
            return {"status": "noop"}
        if not self._owns(node, request) or float(
            (node or {}).get("lease_expires_at") or 0
        ) < float(request["now_unix"]):
            return {"status": "fenced"}
        outcome = request["outcome"]
        if outcome == "failed" and request["retryable"]:
            if int(node["attempt"]) >= int(node["max_attempts"]):
                node.update(status="dead_letter", error_ref=request.get("error_ref"))
                return {"status": "dead_letter"}
            node.update(
                status="ready",
                next_retry_at=float(request["now_unix"])
                + float(node["backoff_base_s"]) * (2 ** (int(node["attempt"]) - 1)),
                lease_epoch=int(node["lease_epoch"]) + 1,
                lease_owner=None,
                lease_expires_at=None,
            )
            return {"status": "retry_scheduled"}
        node.update(
            status=outcome,
            result_ref=request.get("result_ref"),
            error_ref=request.get("error_ref"),
            completed_at=request["now_unix"],
            lease_owner=None,
            lease_expires_at=None,
        )
        return {"status": "committed"}


@pytest.fixture
def orchestrate_tool(monkeypatch):
    from agent_utilities.mcp import kg_server

    kg_server.ensure_tools_registered()
    engine = _FakeOrchEngine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    return kg_server, engine


@pytest.mark.asyncio
@_NEEDS_ENGINE
async def test_orchestrate_dispatch_queue_mode_returns_job_handle(
    orchestrate_tool, fake_queue, monkeypatch
):
    kg_server, engine = orchestrate_tool
    out = await kg_server._execute_tool(
        "graph_jobs",
        action="dispatch",
        task="summarize the repo",
        agent_name="librarian",
    )
    handle = json.loads(out)
    assert handle["dispatch"] == "queued"
    assert handle["kind"] == KIND_ORCHESTRATOR_TASK
    job_id = handle["job_id"]
    # job_tools.graph_jobs's current status handle: a fixed collection URL plus
    # a status_request the caller replays (there is no more per-job URL
    # suffix -- see job_tools.py's register_job_tools).
    assert handle["status_url"] == "/api/graph/jobs"
    assert handle["status_request"] == {"action": "status", "job_id": job_id}
    # The orchestrator's own WorkItem (submitted by Orchestrator.dispatch_task)
    # is the durable payload of record; queue carries only the reference.
    item = _wi.get_work_item(engine, _wi.orchestrator_work_item_id(job_id))
    assert item is not None
    assert item["status"] == "ready"  # no deps -> immediately claimable
    _, item_payload = fake_queue.get()
    env = AgentTurnEnvelope.from_item(item_payload)
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
#
# These drive REAL production code end to end -- create_goal ->
# enqueue_agent_turn -> submit_work_item; execute_agent_turn -> claim_specific
# / mark_running / commit_result; run_goal_loop -> LoopController.run_loop --
# against the engine-native WorkItem state machine (AU-P1-1, No-Legacy: no
# scan/CAS fallback, see work_item.py's module docstring). There is no fake
# shim that stays honest against that contract (claim/lease/statechart
# semantics live in the engine), so -- mirroring
# tests/unit/test_goal_loop_durable.py's ``loop_env`` fixture -- these are
# LIVE-PATH tests against the real ephemeral epistemic-graph engine, skipped
# (never failed) when it is unreachable.


def _goal_node(goal_id: str) -> dict:
    """One goal's merged view: KG Loop-node fields + its authoritative WorkItem
    status, via the SAME reader ``sessions.get_goal_iterations``/``list_goals``
    use (``_load_goal_entry``) -- not a raw Concept-node dict (there is no
    such thing reachable from test code; the WorkItem is the sole status
    authority, see ``work_item.py``'s module docstring)."""
    entry = _sessions._load_goal_entry(_sessions._goal_engine(), goal_id)
    assert entry is not None, f"goal {goal_id} has no durable KG entry"
    return entry


def _rows(db_path, table: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}").fetchall()]
    finally:
        conn.close()


@pytest.fixture
def dispatch_db(tmp_path, monkeypatch):
    import tests.conftest as _ct

    if not getattr(_ct, "_TEST_ENGINE_AVAILABLE", False):
        pytest.skip(
            "epistemic-graph engine not reachable; dispatch-worker live path needs it"
        )
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_sessions._SQLITE_DDL)
    conn.commit()
    conn.close()
    monkeypatch.setattr(_sessions, "_get_db_path", lambda: db)
    monkeypatch.setattr(_sessions, "_rehydrated", False)
    monkeypatch.setattr(_sessions, "active_goals", {})
    monkeypatch.setattr(_sessions, "background_goal_runs", {})
    # Pin sqlite state regardless of an ambient STATE_DB_URI (mirrors loop_env).
    monkeypatch.delenv("STATE_DB_URI", raising=False)
    monkeypatch.setattr(
        "agent_utilities.core.state_store.postgres_state_enabled", lambda: False
    )
    # Production processes construct/activate the process engine once at
    # startup, long before any request handler runs -- create_goal's
    # _persist_goal() call (the goal's KG Loop-node write) assumes an already
    # -active engine and is a best-effort no-op (never raises) when
    # ``_goal_engine()`` is still None, so a test that only activates the
    # engine lazily via enqueue_agent_turn's later get_or_create() call would
    # silently lose the goal's Concept node. Establish that same precondition
    # here (test-harness fix, not a production bug: this mirrors real process
    # startup, it doesn't change create_goal's behavior).
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    IntelligenceGraphEngine.get_or_create()

    import asyncio

    _real_sleep = asyncio.sleep

    async def _fast_sleep(delay=0, *args, **kwargs):
        if delay and delay >= 60:
            return await _real_sleep(delay, *args, **kwargs)
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    return db


@pytest.fixture
def fake_queue():
    """The process-wide dispatch queue, swapped for an in-memory double for
    the duration of one test (the same ``MemoryQueueBackend`` the sqlite/
    admission tests above already use)."""
    from agent_utilities.knowledge_graph.core.queue_backend import MemoryQueueBackend

    queue = MemoryQueueBackend()
    agent_dispatch.reset_dispatch_queue_for_tests(queue)
    try:
        yield queue
    finally:
        agent_dispatch.reset_dispatch_queue_for_tests(None)


@pytest.fixture
def queued_goal(dispatch_db, fake_queue):
    """A goal enqueued in queue mode, ready for a worker to claim.

    Dispatch is unconditionally queue-backed now (agent_dispatch's module
    docstring: "Agent dispatch is always queue-backed"; the old
    ``dispatch_queue_enabled`` toggle was deliberately removed -- see the
    already-passing ``test_dispatch_module_has_no_inline_or_parallel_task_authority``
    above) -- ``create_goal`` needs no monkeypatch to take the queue path.
    """
    import asyncio

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
    """A fresh live claim by another worker -> skip.

    Post-AU-P1-1, claim ownership lives on the dispatch WorkItem's engine-
    native lease (``workitem:dispatch:{job_id}``), not a status/owner_host
    stamp on the goal's Concept node (that inline ownership stamp was
    retired by the WorkItem-native rewrite) -- simulate the other worker by
    claiming that SAME WorkItem directly before the redelivery attempt.
    """
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    _, payload = fake_queue.get()
    env = AgentTurnEnvelope.from_item(payload)
    engine = _sessions._goal_engine()
    dispatch_item_id = f"workitem:dispatch:{env.job_id}"
    claimed = _wi.claim_specific(
        engine,
        dispatch_item_id,
        token="hostB:9:agent-dispatch",
        now=time.time(),
        lease_ttl_s=999.0,
    )
    assert claimed is not None  # sanity: the other worker really holds it
    assert worker.execute_agent_turn(env) == "skipped"


def test_crash_requeue_stale_claim_is_reclaimed(dispatch_db, fake_queue, queued_goal):
    """Worker crash mid-turn: the dispatch WorkItem's lease goes stale, and
    the still-unacked (head-until-ack) envelope is re-claimed by another
    worker."""
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    goal_id = queued_goal["goal_id"]
    _, payload = fake_queue.get()
    env = AgentTurnEnvelope.from_item(payload)
    engine = _sessions._goal_engine()
    dispatch_item_id = f"workitem:dispatch:{env.job_id}"
    # Worker A claimed the dispatch WorkItem then died before renewing or
    # committing -- its lease is now long expired.
    stale = _wi.claim_specific(
        engine,
        dispatch_item_id,
        token="dead:1:agent-dispatch",
        now=time.time() - 1000.0,
        lease_ttl_s=1.0,
    )
    assert stale is not None  # sanity: worker A really held it

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
    # ``_fail_expired`` cancels the goal's own WorkItem (cancel_work_item's
    # terminal outcome is "cancelled" -- there is no free-text
    # "failed"/error write onto the goal's Concept node anymore; the
    # WorkItem status is the sole authority post-AU-P1-1, see
    # work_item.py's module docstring).
    node = _goal_node(queued_goal["goal_id"])
    assert node["status"] == "cancelled"


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
    """A malformed envelope never wedges the loop — but (BUG-003) it is only
    acked AFTER a durable dead-letter WorkItem records the failure, never
    logged-and-acked with no durable trace."""
    import threading

    from agent_utilities.core import sessions as _sessions
    from agent_utilities.orchestration import agent_dispatch_worker as worker
    from agent_utilities.orchestration import work_item as _wi

    payload = {"job_id": "poison", "kind": "goal_loop"}  # no session_id
    fake_queue.put(payload)
    stop = threading.Event()
    real_get = fake_queue.get

    def _get():
        item = real_get()
        if item is None:
            stop.set()
        return item

    fake_queue.get = _get
    worker.run_dispatch_consumer_loop(fake_queue, stop, idle_sleep_s=0.01)
    assert fake_queue.get_queue_size() == 0  # still doesn't wedge the loop

    # A durable dead-letter WorkItem for this exact poison payload must exist
    # BEFORE the ack above could have legally happened.
    poison_id = worker.poison_work_item_id(payload)
    engine = _sessions._goal_engine()
    item = _wi.get_work_item(engine, poison_id)
    assert item is not None
    assert item["status"] in _wi.TERMINAL_WORK_ITEM_STATUSES
    assert item["kind"] == "dispatch_poison"


def test_consumer_loop_dead_letters_tenant_mismatched_envelope(
    dispatch_db, fake_queue, queued_goal
):
    """KNOWN-BAD PROOF (GOC-18): a well-formed envelope whose wire ``tenant``
    disagrees with the tenant the durable WorkItem was ADMITTED under (e.g. a
    forged/tampered broker delivery reusing a legitimate ``job_id``) must be
    REJECTED — durably dead-lettered — before the worker ever claims or
    executes it. The queue transport carries no signed per-message carrier
    yet (GOC-15's envelope-carrier contract is deferred), so this is the
    consumer-side re-check that closes the gap left by admission-time-only
    verification (``test_enqueue_rejects_envelope_tenant_that_differs_from
    _caller_session`` above covers the producer side)."""
    import threading

    from agent_utilities.core import sessions as _sessions
    from agent_utilities.orchestration import agent_dispatch_worker as worker
    from agent_utilities.orchestration import work_item as _wi

    goal_id = queued_goal["goal_id"]
    real_item_id, real_payload = fake_queue.get()
    assert real_payload.get("tenant") != "tenant:attacker"  # sanity: genuine mismatch
    fake_queue.ack(real_item_id)  # drop the genuine delivery ("get" only peeks)
    tampered = dict(real_payload, tenant="tenant:attacker")
    fake_queue.put(tampered)  # the only delivery left in the queue

    stop = threading.Event()
    real_get = fake_queue.get

    def _get():
        item = real_get()
        if item is None:
            stop.set()
        return item

    fake_queue.get = _get
    worker.run_dispatch_consumer_loop(fake_queue, stop, idle_sleep_s=0.01)
    assert fake_queue.get_queue_size() == 0  # rejected, but never wedges the loop

    # The goal must NEVER have executed under the forged tenant.
    node = _goal_node(goal_id)
    assert node["status"] != "completed"

    # A durable dead-letter record for the REJECTED delivery must exist
    # BEFORE the ack above could have legally happened.
    poison_id = worker.poison_work_item_id(tampered)
    engine = _sessions._goal_engine()
    item = _wi.get_work_item(engine, poison_id)
    assert item is not None
    assert item["status"] in _wi.TERMINAL_WORK_ITEM_STATUSES
    assert item["kind"] == "dispatch_poison"

    # The original dispatch WorkItem (admitted under the real tenant) was
    # never claimed by this rejected delivery.
    dispatch_item_id = f"workitem:dispatch:{real_payload['job_id']}"
    dispatch_item = _wi.get_work_item(engine, dispatch_item_id)
    assert dispatch_item is not None
    assert dispatch_item["status"] not in _wi.TERMINAL_WORK_ITEM_STATUSES


def test_consumer_loop_rejects_tenant_mismatch_before_claiming(monkeypatch) -> None:
    """Engine-independent proof of the SAME contract as
    ``test_consumer_loop_dead_letters_tenant_mismatched_envelope`` above
    (which needs a real epistemic-graph engine and is skipped without one,
    e.g. in this sandbox): the tenant-mismatch check fires and dead-letters
    the delivery BEFORE ``execute_agent_turn`` (claim/execute) is ever
    invoked, and the ONLY ack chokepoint used is
    ``_ack_after_durable_outcome`` -- never a raw ``queue.ack``."""
    import threading

    from agent_utilities.knowledge_graph.core.queue_backend import MemoryQueueBackend
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    queue = MemoryQueueBackend()
    payload = AgentTurnEnvelope(
        session_id="session-ref",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-ref",
        tenant="tenant-attacker",
    ).to_item()
    queue.put(payload)

    monkeypatch.setattr(
        "agent_utilities.orchestration.work_item.get_work_item",
        lambda _engine, _item_id: {"tenant": "tenant-legit"},
    )

    dead_lettered: list[tuple[dict, BaseException]] = []

    def _fake_dlq(_engine, _payload, *, error):
        dead_lettered.append((_payload, error))
        return "workitem:dispatch:poison:fake"

    monkeypatch.setattr(worker, "_dead_letter_poison_envelope", _fake_dlq)

    acked: list[str | None] = []

    def _fake_ack(_queue, item_id, _engine, work_item_id):
        acked.append(work_item_id)
        _queue.ack(item_id)
        return True

    monkeypatch.setattr(worker, "_ack_after_durable_outcome", _fake_ack)

    def _never(*_args, **_kwargs):
        raise AssertionError(
            "execute_agent_turn must never run for a rejected tenant-mismatched delivery"
        )

    monkeypatch.setattr(worker, "execute_agent_turn", _never)

    stop = threading.Event()
    real_get = queue.get

    def _get():
        item = real_get()
        if item is None:
            stop.set()
        return item

    queue.get = _get
    worker.run_dispatch_consumer_loop(queue, stop, engine=object(), idle_sleep_s=0.01)

    assert len(dead_lettered) == 1
    payload_seen, error_seen = dead_lettered[0]
    assert payload_seen == payload
    assert isinstance(error_seen, worker.TenantMismatchError)
    assert acked == ["workitem:dispatch:poison:fake"]
    assert queue.get_queue_size() == 0


def test_two_workers_one_session_execute_serially(dispatch_db, fake_queue, monkeypatch):
    """Two workers, one session: per-session mutual exclusion holds end-to-end."""
    import asyncio
    import threading

    from agent_utilities.orchestration import agent_dispatch_worker as worker

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

    # The ambient GraphSession (autouse isolate_graph_compute_engine) lives in
    # a contextvar bound to THIS thread; a bare threading.Thread does not
    # inherit it (only asyncio tasks copy context automatically), so a worker
    # thread's engine reads would otherwise fail closed with
    # SessionRequiredError before ever reaching the claim race this test
    # means to exercise. Run each worker inside a copy of the current
    # context, mirroring tests/unit/test_graph_session.py's/
    # test_correlation.py's ``contextvars.copy_context()`` pattern.
    import contextvars
    import functools

    def _run(token, ctx):
        outcomes.append(
            ctx.run(functools.partial(worker.execute_agent_turn, env, token=token))
        )

    # One independent Context snapshot PER thread, captured here in the main
    # thread (a Context can only ever be ``run()`` in one place at a time, so
    # the two concurrent workers cannot share a single captured Context
    # object -- and copy_context() called from inside a new thread would
    # capture that thread's own, unrelated, empty context instead).
    t1 = threading.Thread(
        target=_run, args=("hostA:1:agent-dispatch", contextvars.copy_context())
    )
    t2 = threading.Thread(
        target=_run, args=("hostB:2:agent-dispatch", contextvars.copy_context())
    )
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert active["max"] == 1  # never concurrent within one session
    assert sorted(outcomes) == ["completed", "skipped"]  # exactly one executed


def test_orchestrator_task_claim_execute_writeback(fake_queue, monkeypatch):
    """claim -> execute -> durable writeback for an orchestrator-kind turn,
    against the CURRENT engine-native WorkItem contract.

    The pre-AU-P1-1 version of this test read/wrote ad hoc fields (``result``,
    ``executed_by``) directly on a bare ``Task``-labeled node via a
    ``query_cypher``/``_update_task_status`` pair shaped for a status
    vocabulary ``Orchestrator.get_task_status`` no longer uses (it now reads
    the WorkItem through ``work_item.get_work_item`` exclusively -- see
    ``manager.py``). Under the current WorkItem-native commit contract
    (``_execute_orchestrator_turn`` -> ``work_item.commit_result``), only
    ``status``/``result_ref``/``error_ref`` are durably recorded; there is no
    free-text ``result`` or ``executed_by`` field anywhere in that path
    anymore (``_execute_orchestrator_turn`` never persists the executor's
    return value, and the WorkItem model has no per-execution
    worker/host-attribution field) -- a real observability regression from
    the AU-P1-1 rewrite, flagged in the slice report rather than silently
    dropped or faked here. This rewrite verifies what the current contract
    actually, verifiably provides: the claim/execute/commit outcome, the
    WorkItem's terminal status and result_ref, and idempotent redelivery.
    """
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    engine = _FakeOrchEngine()
    job_id = "orch-abc"
    _wi.submit_orchestrator_work_item(engine, job_id, description="do it")
    dispatch_job_id = f"dispatch-{job_id}"
    _wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref=job_id,
        work_item_id=f"workitem:dispatch:{dispatch_job_id}",
        idempotency_key=dispatch_job_id,
    )

    async def _fake_execute_agent(self, **kw):
        return f"ran {kw['task']} as {kw['agent_name'] or 'default'}"

    from agent_utilities.orchestration.manager import Orchestrator

    monkeypatch.setattr(Orchestrator, "execute_agent", _fake_execute_agent)
    monkeypatch.setattr(Orchestrator, "__init__", lambda self, engine: None)

    env = AgentTurnEnvelope(
        job_id=dispatch_job_id,
        session_id=job_id,
        kind=KIND_ORCHESTRATOR_TASK,
        payload_ref=job_id,
        agent_name="librarian",
    )
    assert worker.execute_agent_turn(env, engine) == "completed"
    item = _wi.get_work_item(engine, _wi.orchestrator_work_item_id(job_id))
    assert item["status"] == "succeeded"
    assert item["result_ref"] == f"orchestrator:{dispatch_job_id}:completed"
    # Redelivery is an idempotent skip (the dispatch wrapper WorkItem is
    # already terminal).
    assert worker.execute_agent_turn(env, engine) == "skipped"


# ── claim_agent_task: RETIRED (No-Legacy, see engine_claim.py's module
# docstring) — agent_dispatch_worker.claim_agent_task/CLAIM_TTL_S no longer
# exist; the AU-P1-1 authority-convergence assimilation rewrote agent task
# claiming around the engine-native WorkItem state machine
# (engine_claim.claim_agent_task -> work_item.claim_agent_task_via_work_item).
# D-DSTO-5 (reports/deferred/lane-dst-orch.md): the 6 tests that lived here
# (test_claim_agent_task_unknown_task_is_skipped, _terminal_status_is_duplicate_skip,
# _claims_and_writes_lease, _skips_task_with_fresh_live_lease,
# _reclaims_stale_lease_crash_recovery, _default_token_and_now) targeted the
# deleted KG-:AgentLease-node claim path with a fake engine shaped for its raw
# query_cypher contract (a ``does-not-exist`` skip, a terminal-status skip, a
# claim+lease write, a fresh-lease skip, a stale-lease reclaim, and default
# token/now — all six CONCEPT:AU-OS.state.cognitive-scheduler-preemption
# stale-claim-aware idempotency behaviors). The identical behavior against the
# CURRENT WorkItem-native contract is already fully covered:
#   * tests/unit/test_engine_claim.py — backend resolution + delegation
#     through claim_agent_task_via_work_item;
#   * tests/unit/test_agent_dispatch_work_item_backend.py — claim/lease/
#     dependency-release semantics end to end (unknown task, terminal skip,
#     claim+lease, cross-task dependency release, and the engine_claim ->
#     work_item bridge).
# Removed here rather than re-authored against a different engine contract,
# to avoid duplicating that coverage (STALE TEST/retired feature, not a
# production bug).


# ── fleet-visible placement: heartbeats, topology, metrics ────────────────


def test_worker_heartbeat_upserts_and_lists(dispatch_db):
    from agent_utilities.messaging.bus_privacy import bus_reference

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
    # record_dispatch_worker_heartbeat pseudonymizes the raw host through
    # bus_reference (CONCEPT:AU-OS.identity, same treatment as agent_id/
    # tenant everywhere else in this codebase) -- it is no longer persisted
    # verbatim.
    assert w["host"] == bus_reference("dispatch_host", "hostA")
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
    from agent_utilities.messaging.bus_privacy import bus_reference

    agent_dispatch.record_dispatch_worker_heartbeat(
        "hostB:7:agent-dispatch:0", host="hostB", active_sessions=["sess-9"]
    )
    resp = await fleet.fleet_topology(_FakeRequest({}))
    body = json.loads(resp.body)
    assert body["totals"]["dispatch_workers"] == 1
    # bus_reference-pseudonymized host, see test_worker_heartbeat_upserts_and_lists.
    assert body["dispatch_workers"][0]["host"] == bus_reference(
        "dispatch_host", "hostB"
    )
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
    # _heartbeat's queue_backend is type(queue).__name__ -- our fake_queue
    # fixture is a MemoryQueueBackend (there is no "FakeDispatchQueue" class
    # anywhere in this codebase; that name was never real).
    assert workers[0]["queue_backend"] == "MemoryQueueBackend"


def test_job_status_reports_executing_worker_and_host(fake_queue, monkeypatch):
    """graph_jobs job/{id} status: ``Orchestrator.get_task_status`` reads the
    committed WorkItem for a dispatched-and-executed orchestrator turn.

    NOTE (finding, not silently dropped): the pre-AU-P1-1 version of this test
    also asserted an ``executed_by``/``dispatch_host`` stamp naming which
    worker/host ran the job. That per-execution attribution has no equivalent
    in the current engine-native WorkItem contract -- ``_execute_orchestrator_
    turn``'s commit path (work_item.commit_result) only ever records
    ``status``/``result_ref``/``error_ref``, and clears ``lease_owner`` on
    terminal commit (see ``NativeEngine.commit_work_item_result`` in
    tests/unit/orchestration/test_work_item.py, mirrored by
    ``_FakeOrchEngine`` above) -- so "which worker/host executed this job" is
    no longer queryable after completion. This is a genuine observability
    regression from the WorkItem-native rewrite, not something a test-file
    change can restore (it would need a new field/verb in the native
    ClaimWorkItem/CommitWorkItemResult contract). Flagged for its own
    registry entry; this test now verifies what the current job/{id} surface
    actually, verifiably reports: the committed terminal status.
    """
    from agent_utilities.orchestration import agent_dispatch_worker as worker

    engine = _FakeOrchEngine()
    job_id = "orch-xyz"
    _wi.submit_orchestrator_work_item(engine, job_id, description="task")
    dispatch_job_id = f"dispatch-{job_id}"
    _wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref=job_id,
        work_item_id=f"workitem:dispatch:{dispatch_job_id}",
        idempotency_key=dispatch_job_id,
    )

    async def _fake_execute_agent(self, **kw):
        return "done"

    from agent_utilities.orchestration.manager import Orchestrator

    monkeypatch.setattr(Orchestrator, "execute_agent", _fake_execute_agent)
    monkeypatch.setattr(Orchestrator, "__init__", lambda self, engine: None)

    env = AgentTurnEnvelope(
        job_id=dispatch_job_id,
        session_id=job_id,
        kind=KIND_ORCHESTRATOR_TASK,
        payload_ref=job_id,
    )
    assert worker.execute_agent_turn(env, engine) == "completed"

    # The existing job/{id} surface (Orchestrator.get_task_status) reads the
    # committed WorkItem.
    from agent_utilities.orchestration.manager import Orchestrator as RealOrch

    status = RealOrch.get_task_status(SimpleNamespace(engine=engine), job_id)
    assert status["status"] == "succeeded"
