"""Postgres-backed queue state semantics without a live server (OS-5.16).

Exercises :class:`PostgresTaskQueue` (cross-host KG queue,
CONCEPT:AU-KG.ingest.cross-host-safe-kg) against a
small in-memory emulation of exactly the SQL each backend issues, so the
SKIP-LOCKED claim semantics are verified in CI
with no infrastructure. A live end-to-end pass runs in
``tests/integration/test_state_postgres_live.py`` when ``STATE_DB_URI`` is set.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any

import pytest


class _Cursorish(list):
    def fetchone(self):
        return self[0] if self else None

    def fetchall(self):
        return list(self)


# ── Cross-host task queue (CONCEPT:AU-KG.ingest.cross-host-safe-kg) ───────────────────────────────


class FakeQueueConn:
    """Emulates the SKIP LOCKED claim/ack statements PostgresTaskQueue issues."""

    def __init__(self, db: dict):
        self.db = db  # {"kg_task_queue": [...], "kg_task_staging": [...]}

    def _table(self, sql: str) -> str:
        return "kg_task_staging" if "kg_task_staging" in sql else "kg_task_queue"

    def execute(self, sql: str, params: tuple = ()) -> _Cursorish:
        s = " ".join(sql.split())
        rows = self.db[self._table(s)]
        if "FROM information_schema.columns" in s:
            return _Cursorish([(column,) for column in self.db["schema_columns"]])
        if s.startswith("SELECT pg_advisory_xact_lock"):
            return _Cursorish([(None,)])
        if s.startswith("INSERT INTO kg_task_queue_fairness"):
            tenant_ref, tenant_at, _, group_ref, group_at = params
            self.db["fairness"][(tenant_ref, "")] = tenant_at
            self.db["fairness"][(tenant_ref, group_ref)] = group_at
            return _Cursorish()
        if s.startswith("INSERT INTO kg_task_queue"):
            data, tenant, fairness_group, priority, deadline, enqueued_at = params
            rows.append(
                {
                    "id": self.db["seq"](),
                    "data": data,
                    "tenant_ref": tenant,
                    "fairness_group_ref": fairness_group,
                    "prio_bucket": priority,
                    "deadline_unix": deadline,
                    "enqueued_at": enqueued_at,
                    "claimed_at": None,
                }
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
        if (
            (s.startswith("UPDATE") or s.startswith("WITH candidate"))
            and "FOR UPDATE" in s
            and "SKIP LOCKED" in s
        ):
            if self._table(s) == "kg_task_staging":
                claimer, now, cutoff = params
                ordered = sorted(rows, key=lambda row: row["id"])
            else:
                cutoff, deadline_now, claimer, now = params

                def schedule_key(row):
                    tenant_at = self.db["fairness"].get((row["tenant_ref"], ""), 0.0)
                    group_at = self.db["fairness"].get(
                        (row["tenant_ref"], row["fairness_group_ref"]), 0.0
                    )
                    deadline = row["deadline_unix"]
                    return (
                        0 if deadline is not None and deadline <= deadline_now else 1,
                        row["prio_bucket"],
                        tenant_at,
                        group_at,
                        deadline if deadline is not None else float("inf"),
                        row["enqueued_at"],
                        row["id"],
                    )

                ordered = sorted(rows, key=schedule_key)
            for row in ordered:
                if row["claimed_at"] is None or row["claimed_at"] < cutoff:
                    row["claimed_at"] = now
                    row["claimed_by"] = claimer
                    if "graph_data" in row:
                        return _Cursorish(
                            [
                                (
                                    row["id"],
                                    row["job_id"],
                                    row["graph_data"],
                                    row["claimed_by"],
                                    row["claimed_at"],
                                )
                            ]
                        )
                    return _Cursorish(
                        [
                            (
                                row["id"],
                                row["data"],
                                row["tenant_ref"],
                                row["fairness_group_ref"],
                                row["claimed_by"],
                                row["claimed_at"],
                            )
                        ]
                    )
            return _Cursorish()
        if s.startswith("DELETE FROM kg_task_queue_fairness"):
            tenant_ref, group_ref = params
            if not any(
                row["tenant_ref"] == tenant_ref
                and row["fairness_group_ref"] == group_ref
                for row in self.db["kg_task_queue"]
            ):
                self.db["fairness"].pop((tenant_ref, group_ref), None)
            if not any(
                row["tenant_ref"] == tenant_ref for row in self.db["kg_task_queue"]
            ):
                self.db["fairness"].pop((tenant_ref, ""), None)
            return _Cursorish()
        if s.startswith("DELETE"):
            removed = next(
                (
                    row
                    for row in rows
                    if row["id"] == params[0]
                    and (
                        len(params) == 1
                        or (
                            row.get("claimed_by") == params[1]
                            and row.get("claimed_at") == params[2]
                        )
                    )
                ),
                None,
            )
            if removed is not None:
                rows.remove(removed)
            if removed and "tenant_ref" in removed:
                return _Cursorish(
                    [
                        (
                            removed["tenant_ref"],
                            removed["fairness_group_ref"],
                        )
                    ]
                )
            if removed is not None:
                return _Cursorish([(removed["id"],)])
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
            "fairness": {},
            "schema_columns": (
                "id",
                "data",
                "tenant_ref",
                "fairness_group_ref",
                "prio_bucket",
                "deadline_unix",
                "enqueued_at",
                "claimed_by",
                "claimed_at",
            ),
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


def test_queue_rejects_pre_current_schema(monkeypatch):
    from agent_utilities.core import state_store
    from agent_utilities.knowledge_graph.core.postgres_queue_backend import (
        PostgresTaskQueue,
    )

    pool = FakeQueuePool()
    pool.db["schema_columns"] = ("id", "data", "claimed_by", "claimed_at")
    monkeypatch.setattr(state_store, "state_pool", lambda: pool)
    monkeypatch.setattr(state_store, "ensure_state_schema", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="current scheduling schema"):
        PostgresTaskQueue()


def test_queue_claim_honors_priority_and_deadline(pg_queue):
    future = time.time() + 600.0
    pg_queue.put(
        {
            "job_id": "background",
            "tenant": "tenant-a",
            "session_id": "session-a",
            "prio_bucket": 3,
            "deadline_unix": future - 100,
        }
    )
    pg_queue.put(
        {
            "job_id": "critical-later-deadline",
            "tenant": "tenant-a",
            "session_id": "session-b",
            "prio_bucket": 0,
            "deadline_unix": future,
        }
    )
    pg_queue.put(
        {
            "job_id": "critical-earlier-deadline",
            "tenant": "tenant-a",
            "session_id": "session-c",
            "prio_bucket": 0,
            "deadline_unix": future - 10,
        }
    )
    claimed = pg_queue.get()
    assert claimed is not None
    assert claimed[1]["job_id"] == "critical-earlier-deadline"


def test_queue_claim_rotates_tenants_within_priority(pg_queue):
    for job_id, tenant, session in (
        ("a-1", "tenant-a", "session-a-1"),
        ("a-2", "tenant-a", "session-a-2"),
        ("b-1", "tenant-b", "session-b-1"),
    ):
        pg_queue.put(
            {
                "job_id": job_id,
                "tenant": tenant,
                "session_id": session,
                "prio_bucket": 2,
            }
        )
    first = pg_queue.get()
    assert first is not None and first[1]["job_id"] == "a-1"
    pg_queue.ack(first[0])
    second = pg_queue.get()
    assert second is not None and second[1]["job_id"] == "b-1"


def test_queue_admission_bound_is_atomic(pg_queue):
    assert pg_queue.put_if_below({"job_id": "one"}, 2)
    assert pg_queue.put_if_below({"job_id": "two"}, 2)
    assert not pg_queue.put_if_below({"job_id": "three"}, 2)
    assert pg_queue.get_queue_size() == 2


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
    with pytest.raises(RuntimeError, match="fenced"):
        pg_queue.ack(claimed[0])
    assert pg_queue.get_queue_size() == 1
    pg_queue.ack(retried[0])


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
