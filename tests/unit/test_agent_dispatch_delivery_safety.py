"""GOC-18 / BUG-003 (P0): dispatch delivery must never ack before durable
failure/DLQ state exists. GOC-18 / BUG-002 (P0): the standalone dispatch
worker must bootstrap a verified actor/GraphSession before consuming.

Hermetic — no live epistemic-graph engine required. Reuses ``_FakeOrchEngine``,
the SAME engine-native WorkItem double ``tests/unit/test_agent_dispatch.py``
already trusts for its orchestrator-kind path (claim_work_item /
renew_work_item_lease / commit_work_item_result against an in-memory node
dict) — see that module's docstring for why a fake is honest here: claim/
lease/statechart semantics are the engine's, and this fake mirrors exactly
the same CAS/fencing/retry-exhaustion contract ``work_item.py`` requires.

The first two tests below (``test_poison_envelope_ack_requires_durable_dead_letter_record``
and ``test_crash_mid_execution_never_acks_without_durable_terminal_state``) are
the FAILING-FIRST reproductions the GOC-18 lane requires for BUG-003: they were
written and run RED against the pre-fix ``agent_dispatch_worker.py`` (verified
by temporarily reverting that file to its pre-fix ``main`` content and running
this file alone), then made to pass by the BUG-003 remediation. They
deliberately use ONLY functions that already existed pre-fix
(``run_dispatch_consumer_loop``, ``load_goal_run``, ``_execute_goal_turn``,
``work_item.get_work_item``) so the red run is a genuine behavioral failure,
not an ImportError against a not-yet-written symbol.
"""

from __future__ import annotations

import threading

import pytest

from agent_utilities.knowledge_graph.core.queue_backend import MemoryQueueBackend
from agent_utilities.orchestration import agent_dispatch_worker as worker
from agent_utilities.orchestration import work_item as _wi
from agent_utilities.orchestration.agent_dispatch import (
    KIND_GOAL_LOOP,
    KIND_ORCHESTRATOR_TASK,
    AgentTurnEnvelope,
)

from .test_agent_dispatch import _FakeOrchEngine


def _drain_loop(queue, engine, **kwargs) -> None:
    """Run the consumer loop until the (pre-populated) queue reports empty."""
    stop = threading.Event()
    real_get = queue.get

    def _get():
        item = real_get()
        if item is None:
            stop.set()
        return item

    queue.get = _get
    worker.run_dispatch_consumer_loop(queue, stop, engine, idle_sleep_s=0.01, **kwargs)


# ── BUG-003, hostile case 1: poison envelope ────────────────────────────


def test_poison_envelope_ack_requires_durable_dead_letter_record():
    """A message that fails ``AgentTurnEnvelope`` validation (poison) must
    NOT be acked until a durable dead-letter WorkItem records the failure.

    Pre-fix: the consumer's outer ``except Exception`` logged the parse
    failure and fell through to an unconditional ``queue.ack(item_id)`` with
    NO durable write anywhere — the message and all evidence of its failure
    vanished together (BUG-003). This asserts the fixed contract.
    """
    engine = _FakeOrchEngine()
    queue = MemoryQueueBackend()
    payload = {"job_id": "poison-1", "kind": "goal_loop"}  # no session_id
    queue.put(payload)

    _drain_loop(queue, engine)

    assert queue.get_queue_size() == 0  # must not wedge the consumer loop

    poison_nodes = [
        node
        for node in engine.nodes.values()
        if node.get("label") == "WorkItem" and node.get("kind") == "dispatch_poison"
    ]
    assert len(poison_nodes) == 1, (
        "a poison envelope must produce exactly one durable dead-letter "
        "WorkItem BEFORE it may be acked (BUG-003: silent message loss)"
    )
    assert poison_nodes[0]["status"] in _wi.TERMINAL_WORK_ITEM_STATUSES


def test_poison_envelope_redelivery_is_idempotent_one_dlq_record():
    """The SAME poison payload delivered twice must not create two DLQ
    records — the durable record is keyed by delivery digest."""
    engine = _FakeOrchEngine()
    queue = MemoryQueueBackend()
    payload = {"job_id": "poison-2", "kind": "goal_loop"}
    queue.put(dict(payload))
    queue.put(dict(payload))  # identical redelivery

    _drain_loop(queue, engine)

    poison_nodes = [
        node
        for node in engine.nodes.values()
        if node.get("label") == "WorkItem" and node.get("kind") == "dispatch_poison"
    ]
    assert len(poison_nodes) == 1


# ── BUG-003, hostile case 2: crash-before-commit ────────────────────────


def test_crash_mid_execution_never_acks_without_durable_terminal_state(monkeypatch):
    """If the executor raises BEFORE ``execute_agent_turn`` reaches its
    ``commit_result`` call, the dispatch WorkItem is left in a non-terminal
    state (``leased``/``running``).

    Pre-fix: the consumer loop's outer ``except Exception`` swallowed this,
    the local ``outcome`` variable stayed at its preset ``"failed"`` default,
    and the loop unconditionally called ``queue.ack`` anyway — the broker
    message vanished while the WorkItem stayed stuck non-terminal forever
    (BUG-003: crash-before-commit). This asserts the fixed ack gate: the
    message is acked only once durable terminal state is confirmed to exist.
    """
    engine = _FakeOrchEngine()
    queue = MemoryQueueBackend()
    job_id = "goal-crash-1"
    dispatch_item_id = f"workitem:dispatch:{job_id}"
    _wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref="goal-x",
        work_item_id=dispatch_item_id,
        idempotency_key=job_id,
    )
    env = AgentTurnEnvelope(
        job_id=job_id,
        session_id="sess-crash-1",
        kind=KIND_GOAL_LOOP,
        payload_ref="goal-x",
    )
    queue.put(env.to_item())

    monkeypatch.setattr(
        worker,
        "load_goal_run",
        lambda *a, **kw: {
            "goal_id": "goal-x",
            "session_id": "sess-crash-1",
            "objective": "x",
            "validation_cmd": "",
            "max_iterations": 1,
            "constraints": [],
        },
    )

    def _boom(spec):
        raise RuntimeError("simulated crash mid execution")

    monkeypatch.setattr(worker, "_execute_goal_turn", _boom)

    _drain_loop(queue, engine)

    item = _wi.get_work_item(engine, dispatch_item_id)
    assert item is not None
    assert item["status"] in _wi.TERMINAL_WORK_ITEM_STATUSES, (
        f"WorkItem left non-terminal ({item['status']!r}) after an executor "
        "crash -- BUG-003: crash-before-commit"
    )
    assert queue.get_queue_size() == 0  # acked only now that durable state exists


def test_crash_mid_orchestrator_turn_still_lands_terminal_before_ack(monkeypatch):
    """Same hostile shape, orchestrator-kind turn: an exception inside
    ``_execute_orchestrator_turn`` that escapes ITS OWN internal commit
    (e.g. the commit call itself raises) must still resolve to a durable
    terminal state before ack, not merely rely on that inner function's own
    happy-path exception handling."""
    engine = _FakeOrchEngine()
    queue = MemoryQueueBackend()
    job_id = "orch-crash-1"
    dispatch_job_id = f"dispatch-{job_id}"
    dispatch_item_id = f"workitem:dispatch:{dispatch_job_id}"
    _wi.submit_orchestrator_work_item(engine, job_id, description="do it")
    _wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref=job_id,
        work_item_id=dispatch_item_id,
        idempotency_key=dispatch_job_id,
    )
    env = AgentTurnEnvelope(
        job_id=dispatch_job_id,
        session_id=job_id,
        kind=KIND_ORCHESTRATOR_TASK,
        payload_ref=job_id,
        agent_name="librarian",
    )
    queue.put(env.to_item())

    def _boom(*_a, **_kw):
        raise RuntimeError("catastrophic executor failure")

    monkeypatch.setattr(worker, "_execute_orchestrator_turn", _boom)

    _drain_loop(queue, engine)

    item = _wi.get_work_item(engine, dispatch_item_id)
    assert item is not None
    assert item["status"] in _wi.TERMINAL_WORK_ITEM_STATUSES
    assert queue.get_queue_size() == 0


# ── the ONE ack chokepoint: known-bad-input demonstrations ─────────────


def test_ack_gate_rejects_ack_before_durable_terminal_state():
    """Direct proof of the ONE ack chokepoint: a non-terminal WorkItem (still
    ``leased``) must never be acked, regardless of what the caller believes
    the outcome was."""
    engine = _FakeOrchEngine()
    work_item_id = "workitem:dispatch:ack-gate-1"
    _wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref="x",
        work_item_id=work_item_id,
        idempotency_key="ack-gate-1",
    )
    _wi.claim_specific(engine, work_item_id, token="w1")  # -> "leased", non-terminal

    acked: list[str] = []
    queue = type("Q", (), {"ack": staticmethod(lambda item_id: acked.append(item_id))})()

    result = worker._ack_after_durable_outcome(queue, "broker-item-1", engine, work_item_id)

    assert result is False
    assert acked == []  # the broker was never told to drop the message


def test_ack_gate_permits_ack_after_durable_terminal_state():
    """Positive control for the same gate: a genuinely terminal WorkItem IS
    acked."""
    engine = _FakeOrchEngine()
    work_item_id = "workitem:dispatch:ack-gate-2"
    _wi.submit_work_item(
        engine,
        kind="agent_turn",
        payload_ref="x",
        work_item_id=work_item_id,
        idempotency_key="ack-gate-2",
    )
    claim = _wi.claim_specific(engine, work_item_id, token="w1")
    _wi.commit_result(engine, work_item_id, claim, outcome="failed", retryable=False)

    acked: list[str] = []
    queue = type("Q", (), {"ack": staticmethod(lambda item_id: acked.append(item_id))})()

    result = worker._ack_after_durable_outcome(queue, "broker-item-2", engine, work_item_id)

    assert result is True
    assert acked == ["broker-item-2"]


def test_ack_gate_rejects_when_no_work_item_exists_at_all():
    """A work_item_id with NO durable record (e.g. a durable write that
    never landed) must never be acked."""
    engine = _FakeOrchEngine()
    acked: list[str] = []
    queue = type("Q", (), {"ack": staticmethod(lambda item_id: acked.append(item_id))})()

    result = worker._ack_after_durable_outcome(
        queue, "broker-item-3", engine, "workitem:dispatch:never-existed"
    )

    assert result is False
    assert acked == []


# ── BUG-002: verified actor/GraphSession bootstrap ──────────────────────


def test_main_bootstraps_verified_actor_and_session_before_first_engine_call(
    monkeypatch,
):
    """The dispatch worker's ``main()`` must bind a verified process actor/
    GraphSession BEFORE its first protected engine call -- exactly like
    ``knowledge_graph.ingest_worker.main()`` already does.

    Pre-fix, ``main()`` goes straight from ``KG_DAEMON_ROLE=client`` to
    ``IntelligenceGraphEngine.get_or_create()`` with NO
    ``acquire_process_identity_token``/``mint_actor_from_token_sync``/
    ``mint_graph_session`` call anywhere -- this test fails against that
    code (``calls`` stays empty of the identity-bootstrap steps).
    """
    from types import SimpleNamespace

    calls: list[str] = []

    def _fake_acquire(config=None):
        calls.append("acquire_process_identity_token")
        return "fake-token"

    def _fake_mint_actor(token):
        calls.append("mint_actor_from_token_sync")
        assert token == "fake-token"
        return SimpleNamespace(authenticated=True)

    class _FakeSession:
        actor = SimpleNamespace(authenticated=True)

        def engine_verified_context(self):
            calls.append("engine_verified_context")
            return {}

    def _fake_mint_session(actor):
        calls.append("mint_graph_session")
        assert actor.authenticated
        return _FakeSession()

    class _FakeEngine:
        def query_cypher(self, *a, **kw):
            calls.append("engine.query_cypher")
            return []

        def claim_work_item(self, *a, **kw):
            return None

    monkeypatch.setattr(
        "agent_utilities.security.request_identity.acquire_process_identity_token",
        _fake_acquire,
    )
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.mint_actor_from_token_sync",
        _fake_mint_actor,
    )
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.mint_graph_session",
        _fake_mint_session,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine.IntelligenceGraphEngine.get_or_create",
        classmethod(lambda cls: _FakeEngine()),
    )
    monkeypatch.setattr(worker, "start_dispatch_worker_pool", lambda **kw: [])
    monkeypatch.setattr("time.sleep", lambda *a, **kw: None)
    monkeypatch.setattr("signal.signal", lambda *a, **kw: None)

    rc = worker.main(["--workers", "1"])

    assert rc == 0
    assert "acquire_process_identity_token" in calls
    assert "mint_actor_from_token_sync" in calls
    assert "mint_graph_session" in calls
    assert "engine.query_cypher" in calls
    assert calls.index("mint_graph_session") < calls.index("engine.query_cypher"), (
        "the process actor/session must be minted BEFORE the first protected "
        "engine call (BUG-002)"
    )


def test_dispatch_worker_pool_fails_closed_without_verified_actor(monkeypatch):
    """Known-bad-input demonstration: ``start_dispatch_worker_pool`` must
    refuse to spawn ANY consumer thread when no authenticated actor/session
    is bound -- mirrors the ingest worker's
    ``_capture_verified_background_session`` fail-closed contract. Simulates
    the "no identity" environment even though the autouse test fixture
    normally binds one, by making actor resolution raise as if unauthenticated.
    """
    from agent_utilities.knowledge_graph.core.session import SessionRequiredError
    from agent_utilities.security.brain_context import IdentityRequiredError

    def _no_actor():
        raise IdentityRequiredError("no actor bound")

    monkeypatch.setattr(
        "agent_utilities.security.brain_context.current_actor", _no_actor
    )

    queue = MemoryQueueBackend()
    with pytest.raises(SessionRequiredError):
        worker.start_dispatch_worker_pool(queue=queue, worker_count=1)


def test_dispatch_worker_pool_accepts_explicit_verified_session(monkeypatch):
    """Positive control: an explicitly-provided, already-verified
    ``background_session`` is accepted (mirrors ``start_ingest_consumer_pool``'s
    ``background_session=`` parameter) without needing to re-derive one from
    ambient context."""
    from types import SimpleNamespace

    captured: dict = {}

    class _FakeActor:
        authenticated = True

    class _FakeSession:
        actor = _FakeActor()

    def _fake_authorized_thread(session, target, *, name, args=()):
        captured["session"] = session
        import threading as _threading

        return _threading.Thread(
            target=lambda: None, name=name, daemon=True
        )

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.engine_tasks._authorized_background_thread",
        _fake_authorized_thread,
    )

    queue = MemoryQueueBackend()
    stop = threading.Event()
    stop.set()
    session = _FakeSession()
    threads = worker.start_dispatch_worker_pool(
        queue=queue,
        worker_count=1,
        stop_event=stop,
        engine=object(),
        background_session=session,
    )
    assert len(threads) == 1
    assert captured["session"] is session
