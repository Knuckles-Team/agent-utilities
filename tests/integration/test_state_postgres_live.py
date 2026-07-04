"""Live Postgres state-store pass (CONCEPT:AU-OS.state.unified-durable-state-externalization / KG-2.54 / OS-5.17).

Runs ONLY when ``STATE_DB_URI`` points at a reachable Postgres (e.g. the
deployed ``kg-backbone_pggraph`` service) — skipped everywhere else, so CI
never requires infrastructure. Exercises the real end-to-end paths the unit
fakes emulate: durable-exec checkpoints, queue SKIP LOCKED claims, advisory
leadership, and the sessions/goals schema.
"""

from __future__ import annotations

import os
import uuid

import pytest

_DSN = os.environ.get("STATE_DB_URI", "")

pytestmark = pytest.mark.skipif(
    not _DSN.startswith(("postgresql://", "postgres://")),
    reason="STATE_DB_URI not set — live Postgres state-store test skipped",
)


def _reachable() -> bool:
    try:
        import psycopg

        with psycopg.connect(_DSN, connect_timeout=3):
            return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _require_reachable(monkeypatch):
    if not _reachable():
        pytest.skip("STATE_DB_URI set but Postgres is unreachable")
    from agent_utilities.core import state_store
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "state_db_uri", _DSN, raising=False)
    state_store.reset_state_store_for_tests()
    yield
    state_store.reset_state_store_for_tests()


def test_live_durable_execution_roundtrip():
    from agent_utilities.orchestration.durable_execution import (
        PostgresCheckpointStore,
        DurableExecutionManager,
    )

    sid = f"live-{uuid.uuid4().hex[:8]}"
    mgr = DurableExecutionManager(sid, store=PostgresCheckpointStore())
    calls = {"n": 0}

    def critical():
        calls["n"] += 1
        return {"ok": True}

    first = mgr.run_durable_action("step", critical, idempotency_key=f"{sid}-k")
    second = DurableExecutionManager(
        sid, store=PostgresCheckpointStore()
    ).run_durable_action("step", critical, idempotency_key=f"{sid}-k")
    assert calls["n"] == 1
    assert first == second == {"ok": True}


def test_live_queue_claims_exclusive():
    from agent_utilities.knowledge_graph.core.postgres_queue_backend import (
        PostgresTaskQueue,
    )

    q1 = PostgresTaskQueue()
    q2 = PostgresTaskQueue()
    marker = f"live-{uuid.uuid4().hex[:8]}"
    q1.put({"job_id": marker})
    seen = []
    for q in (q1, q2):
        item = q.get()
        if item and item[1].get("job_id") == marker:
            seen.append(item)
    assert len(seen) == 1  # never double-claimed
    q1.ack(seen[0][0])


def test_live_leadership_single_winner():
    from agent_utilities.core.leadership import DaemonLeadership

    role = f"live-{uuid.uuid4().hex[:8]}"
    a = DaemonLeadership(role, dsn=_DSN)
    b = DaemonLeadership(role, dsn=_DSN)
    try:
        assert a.is_leader() is True
        assert b.is_leader() is False  # second contender must lose
        a.release()
        assert b.is_leader() is True  # fail-over after release
    finally:
        a.release()
        b.release()


def test_live_sessions_schema_migrates():
    from agent_utilities.core import sessions as _sessions

    conn = _sessions._connect_db()
    try:
        assert conn.dialect == "postgres"
        cur = conn.cursor()
        # goal state lives on the KG Loop node now, not a SQLite/PG goals table
        # (CONCEPT:AU-KG.research.these-properties-carry); only sessions/turns remain in the sessions store.
        for table in ("sessions", "turns"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")  # nosec B608
            assert cur.fetchone()[0] >= 0
    finally:
        conn.close()
