"""Postgres-backed durable state — semantics without a live server (OS-5.16).

Exercises :class:`PostgresCheckpointStore` (durable execution) and
:class:`PostgresTaskQueue` (cross-host KG queue, CONCEPT:KG-2.54) against a
small in-memory emulation of exactly the SQL each backend issues, so the
at-least-once / exactly-once / SKIP-LOCKED-claim semantics are verified in CI
with no infrastructure. A live end-to-end pass runs in
``tests/integration/test_state_postgres_live.py`` when ``STATE_DB_URI`` is set.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any

import pytest

from agent_utilities.orchestration.durable_execution import (
    DurableExecutionManager,
    PostgresCheckpointStore,
)
from agent_utilities.orchestration.resilience import ResiliencePolicy

_FAST = ResiliencePolicy(max_attempts=3, backoff_base_s=0.0, jitter=False)


# ── In-memory emulation of the durable_checkpoints statements ─────────────


class _Cursorish(list):
    def fetchone(self):
        return self[0] if self else None

    def fetchall(self):
        return list(self)


class FakeDurableConn:
    """Emulates the four statements PostgresCheckpointStore issues."""

    def __init__(self, rows: dict):
        self.rows = rows  # (session_id, node_id) -> dict

    def execute(self, sql: str, params: tuple = ()) -> _Cursorish:
        s = " ".join(sql.split())
        if s.startswith("INSERT INTO durable_checkpoints"):
            sid, nid, state, status, key, ts = params
            self.rows[(sid, nid)] = {
                "state": state,
                "status": status,
                "idempotency_key": key,
                "result": self.rows.get((sid, nid), {}).get("result"),
                "updated_at": ts,
            }
            return _Cursorish()
        if s.startswith("UPDATE durable_checkpoints"):
            result, ts, sid, nid = params
            row = self.rows.get((sid, nid))
            if row is not None:
                row.update(status="COMPLETED", result=result, updated_at=ts)
            return _Cursorish()
        if "SELECT node_id, state" in s:
            (sid,) = params
            pending = [
                (nid, r["state"], r["updated_at"])
                for (s_, nid), r in self.rows.items()
                if s_ == sid and r["status"] == "PENDING"
            ]
            pending.sort(key=lambda t: t[2], reverse=True)
            return _Cursorish([(n, st) for n, st, _ in pending[:1]])
        if "SELECT result" in s:
            sid, key = params
            done = [
                (r["result"], r["updated_at"])
                for (s_, _), r in self.rows.items()
                if s_ == sid
                and r["idempotency_key"] == key
                and r["status"] == "COMPLETED"
            ]
            done.sort(key=lambda t: t[1], reverse=True)
            return _Cursorish([(d[0],) for d in done[:1]])
        raise AssertionError(f"unexpected SQL: {s}")


class FakeDurablePool:
    def __init__(self):
        self.rows: dict = {}

    @contextmanager
    def connection(self):
        yield FakeDurableConn(self.rows)


@pytest.fixture
def pg_store(monkeypatch):
    from agent_utilities.core import state_store

    monkeypatch.setattr(state_store, "ensure_state_schema", lambda *a, **k: None)
    return PostgresCheckpointStore(pool=FakeDurablePool())


def test_pg_checkpoint_flow(pg_store):
    mgr = DurableExecutionManager("s1", store=pg_store)
    assert mgr.save_checkpoint("step", {"asset": "BTC"}) == "step"
    resumed = mgr.resume_session()
    assert resumed is not None and resumed["node_id"] == "step"
    assert "BTC" in resumed["state"]
    mgr.mark_completed("step")
    assert mgr.resume_session() is None


def test_pg_idempotency_exactly_once(pg_store):
    mgr = DurableExecutionManager("s2", store=pg_store)
    calls = {"n": 0}

    def critical():
        calls["n"] += 1
        return {"order_id": "abc"}

    first = mgr.run_durable_action("place", critical, idempotency_key="ORD-1")
    second = mgr.run_durable_action("place", critical, idempotency_key="ORD-1")
    assert calls["n"] == 1
    assert first == second == {"order_id": "abc"}


def test_pg_idempotency_across_manager_instances(pg_store):
    # "Restart" = new manager over the same shared store (what a second host sees).
    DurableExecutionManager("s3", store=pg_store).run_durable_action(
        "step", lambda: "done", idempotency_key="K-1"
    )
    reran = {"n": 0}

    def again():
        reran["n"] += 1
        return "second"

    out = DurableExecutionManager("s3", store=pg_store).run_durable_action(
        "step", again, idempotency_key="K-1"
    )
    assert reran["n"] == 0
    assert out == "done"


def test_pg_at_least_once_retries(pg_store):
    mgr = DurableExecutionManager("s4", store=pg_store)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ConnectionError("transient")
        return "ok"

    assert (
        mgr.run_durable_action("f", flaky, idempotency_key="F-1", policy=_FAST) == "ok"
    )
    assert attempts["n"] == 2


def test_default_selection_prefers_sqlite_without_uri(tmp_path):
    from agent_utilities.orchestration.durable_execution import (
        SQLiteCheckpointStore,
        _select_store,
    )

    assert isinstance(_select_store(tmp_path / "x.db"), SQLiteCheckpointStore)
    assert isinstance(_select_store(None), SQLiteCheckpointStore)


def test_sqlite_store_reuses_one_connection(tmp_path):
    from agent_utilities.orchestration.durable_execution import SQLiteCheckpointStore

    a = SQLiteCheckpointStore(tmp_path / "d.db")
    b = SQLiteCheckpointStore(tmp_path / "d.db")
    # connect-per-op fixed: one pooled connection per db file, shared.
    assert a._conn is b._conn


# ── Cross-host task queue (CONCEPT:KG-2.54) ───────────────────────────────


class FakeQueueConn:
    """Emulates the SKIP LOCKED claim/ack statements PostgresTaskQueue issues."""

    def __init__(self, db: dict):
        self.db = db  # {"kg_task_queue": [...], "kg_task_staging": [...]}

    def _table(self, sql: str) -> str:
        return "kg_task_staging" if "kg_task_staging" in sql else "kg_task_queue"

    def execute(self, sql: str, params: tuple = ()) -> _Cursorish:
        s = " ".join(sql.split())
        rows = self.db[self._table(s)]
        if s.startswith("INSERT INTO kg_task_queue"):
            rows.append(
                {"id": self.db["seq"](), "data": params[0], "claimed_at": None}
            )
            return _Cursorish()
        if s.startswith("INSERT INTO kg_task_staging"):
            rows.append(
                {
                    "id": self.db["seq"](),
                    "job_id": params[0],
                    "graph_data": params[1],
                    "claimed_at": None,
                }
            )
            return _Cursorish()
        if s.startswith("UPDATE") and "FOR UPDATE SKIP LOCKED" in s:
            claimer, now, cutoff = params
            for row in sorted(rows, key=lambda r: r["id"]):
                if row["claimed_at"] is None or row["claimed_at"] < cutoff:
                    row["claimed_at"] = now
                    row["claimed_by"] = claimer
                    if "graph_data" in row:
                        return _Cursorish(
                            [(row["id"], row["job_id"], row["graph_data"])]
                        )
                    return _Cursorish([(row["id"], row["data"])])
            return _Cursorish()
        if s.startswith("DELETE"):
            rows[:] = [r for r in rows if r["id"] != params[0]]
            return _Cursorish()
        if s.startswith("SELECT COUNT(*)"):
            return _Cursorish([(len(rows),)])
        raise AssertionError(f"unexpected SQL: {s}")


class FakeQueuePool:
    def __init__(self):
        counter = {"n": 0}

        def seq():
            counter["n"] += 1
            return counter["n"]

        self.db: dict[str, Any] = {
            "kg_task_queue": [],
            "kg_task_staging": [],
            "seq": seq,
        }

    @contextmanager
    def connection(self):
        yield FakeQueueConn(self.db)


@pytest.fixture
def pg_queue(monkeypatch):
    from agent_utilities.core import state_store
    from agent_utilities.knowledge_graph.core.postgres_queue_backend import (
        PostgresTaskQueue,
    )

    pool = FakeQueuePool()
    monkeypatch.setattr(state_store, "state_pool", lambda: pool)
    monkeypatch.setattr(state_store, "ensure_state_schema", lambda *a, **k: None)
    return PostgresTaskQueue()


def test_queue_put_get_ack(pg_queue):
    pg_queue.put({"job_id": "j1"})
    pg_queue.put({"job_id": "j2"})
    assert pg_queue.get_queue_size() == 2
    item = pg_queue.get()
    assert item is not None
    item_id, payload = item
    assert payload == {"job_id": "j1"}  # FIFO
    pg_queue.ack(item_id)
    assert pg_queue.get_queue_size() == 1


def test_queue_claims_are_exclusive_across_consumers(pg_queue):
    # Two hosts polling the same queue must never receive the same item —
    # the claim (SKIP LOCKED + claimed_at stamp) is atomic.
    pg_queue.put({"job_id": "only"})
    first = pg_queue.get()
    second = pg_queue.get()  # second host's poll
    assert first is not None
    assert second is None


def test_queue_visibility_timeout_requeues_dead_claims(pg_queue, monkeypatch):
    import agent_utilities.knowledge_graph.core.postgres_queue_backend as pqb

    pg_queue.put({"job_id": "crashy"})
    claimed = pg_queue.get()
    assert claimed is not None
    # Claimer crashed before ack. Within the window the item stays invisible…
    assert pg_queue.get() is None
    # …after the visibility timeout it becomes claimable again (at-least-once).
    monkeypatch.setattr(pqb, "_VISIBILITY_TIMEOUT_S", 0.0)
    time.sleep(0.01)
    retried = pg_queue.get()
    assert retried is not None
    assert retried[1] == {"job_id": "crashy"}


def test_staged_graph_roundtrip(pg_queue):
    pg_queue.put_staged_graph("job-1", [{"id": "n1"}], [{"s": "a"}])
    got = pg_queue.get_staged_graph()
    assert got is not None
    item_id, job_id, graph = got
    assert job_id == "job-1"
    assert graph["nodes"] == [{"id": "n1"}]
    assert json.loads(json.dumps(graph))  # serializable payload
    pg_queue.ack_staged_graph(item_id)
    assert pg_queue.get_staged_graph() is None
