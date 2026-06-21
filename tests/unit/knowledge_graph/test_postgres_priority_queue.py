"""KG-2.113 durable priority-ordered claim (mocked PG; live SKIP-LOCKED test deferred).

Verifies the functional contract without a real Postgres: ``put`` persists the
claim-priority bucket and ``get`` claims priority-ordered (lowest bucket first).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch


def _make_queue() -> tuple[object, list[tuple[str, tuple]]]:
    """Build a PostgresTaskQueue with a mocked pool; return (queue, calls)."""
    calls: list[tuple[str, tuple]] = []

    conn = MagicMock()

    def _execute(sql, params=None):
        calls.append((sql, params))
        res = MagicMock()
        # wide enough for both get() (id,data) and get_staged_graph() (id,job_id,data)
        res.fetchone.return_value = (1, '{"x": 1}', '{"nodes": [], "edges": []}')
        return res

    conn.execute.side_effect = _execute

    @contextmanager
    def _connection():
        yield conn

    pool = MagicMock()
    pool.connection.side_effect = _connection

    with (
        patch("agent_utilities.core.state_store.state_pool", return_value=pool),
        patch("agent_utilities.core.state_store.ensure_state_schema"),
    ):
        from agent_utilities.knowledge_graph.core.postgres_queue_backend import (
            PostgresTaskQueue,
        )

        return PostgresTaskQueue(), calls


def test_ddl_has_priority_column_and_index() -> None:
    from agent_utilities.knowledge_graph.core import postgres_queue_backend as m

    assert "priority INTEGER NOT NULL DEFAULT 0" in m._DDL
    assert "(priority, claimed_at NULLS FIRST, id)" in m._DDL
    # additional claim tables (e.g. agent dispatch) inherit the priority column
    ddl = m._queue_table_ddl("agent_dispatch_queue")
    assert "priority INTEGER NOT NULL DEFAULT 0" in ddl


def test_put_persists_priority_bucket() -> None:
    q, calls = _make_queue()
    q.put({"task": "research_paper_fetch", "priority": 2})
    inserts = [c for c in calls if "INSERT INTO" in c[0]]
    assert inserts, "put() should INSERT"
    sql, params = inserts[-1]
    assert "priority" in sql
    assert params[1] == 2  # (data_json, priority)


def test_put_defaults_priority_zero_and_accepts_claim_bucket() -> None:
    q, calls = _make_queue()
    q.put({"task": "x"})  # no priority → 0
    assert calls[-1][1][1] == 0
    q.put({"task": "y", "claim_bucket": 3})  # claim_bucket alias
    assert calls[-1][1][1] == 3


def test_put_bad_priority_falls_back_to_zero() -> None:
    q, calls = _make_queue()
    q.put({"task": "z", "priority": "not-an-int"})
    assert calls[-1][1][1] == 0


def test_get_claims_priority_ordered() -> None:
    q, calls = _make_queue()
    q.get()
    claims = [c for c in calls if "FOR UPDATE SKIP LOCKED" in c[0]]
    assert claims, "get() should issue a SKIP LOCKED claim"
    assert "ORDER BY priority ASC, id ASC" in claims[-1][0]


def test_staging_claim_stays_fifo() -> None:
    q, calls = _make_queue()
    q.get_staged_graph()
    claims = [c for c in calls if "FOR UPDATE SKIP LOCKED" in c[0]]
    assert claims
    # staging is ingest-only (no priority column) → plain FIFO
    assert "ORDER BY id" in claims[-1][0]
    assert "priority ASC" not in claims[-1][0]
