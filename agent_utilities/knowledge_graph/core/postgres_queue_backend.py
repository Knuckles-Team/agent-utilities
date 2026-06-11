"""Cross-host-safe Postgres task queue.

CONCEPT:KG-2.54 — Cross-host-safe KG task queue with atomic SKIP LOCKED claims and visibility-timeout recovery on the shared state store

Drop-in :class:`~agent_utilities.knowledge_graph.core.queue_backend.QueueBackend`
backed by the shared state-store Postgres (``state_db_uri``), replacing the
per-host ``kg_task_queue.db`` SQLite file when durable state is externalized.

Why: the SQLite queue's ``get()`` returns the head row *without removing it*
and relies on a per-process ``threading.Lock`` for claim atomicity — safe on
one host, but two hosts each see and process the same head (double-claim).
Here a claim is one atomic statement::

    UPDATE ... WHERE id = (SELECT id ... FOR UPDATE SKIP LOCKED) RETURNING ...

so N hosts can drain the same queue concurrently and never double-claim.

Visibility-timeout semantics are preserved: a claimed-but-unacked item (the
claimer crashed before ``ack``) becomes claimable again once its claim ages
past ``_VISIBILITY_TIMEOUT_S`` — the same at-least-once recovery the SQLite
head-until-ack behavior provided, made multi-host-correct.
"""

from __future__ import annotations

import json
import logging
import socket
import time
from typing import Any

from .queue_backend import QueueBackend

logger = logging.getLogger(__name__)

# A claimed item whose claim is older than this becomes claimable again
# (its claimer is presumed dead — crashed between get() and ack()).
_VISIBILITY_TIMEOUT_S = 600.0

_DDL = """
CREATE TABLE IF NOT EXISTS kg_task_queue (
    id BIGSERIAL PRIMARY KEY,
    data TEXT NOT NULL,
    claimed_by TEXT,
    claimed_at DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS kg_task_staging (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT,
    graph_data TEXT NOT NULL,
    claimed_by TEXT,
    claimed_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_kg_task_queue_claim
    ON kg_task_queue (claimed_at NULLS FIRST, id);
CREATE INDEX IF NOT EXISTS idx_kg_task_staging_claim
    ON kg_task_staging (claimed_at NULLS FIRST, id);
"""


def _queue_table_ddl(table: str) -> str:
    """DDL for an additional SKIP LOCKED claim queue table.

    CONCEPT:ORCH-1.45 — the agent dispatch queue (``agent_dispatch_queue``)
    reuses this backend with its own table; same columns, same claim contract.
    """
    return f"""
CREATE TABLE IF NOT EXISTS {table} (
    id BIGSERIAL PRIMARY KEY,
    data TEXT NOT NULL,
    claimed_by TEXT,
    claimed_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_{table}_claim
    ON {table} (claimed_at NULLS FIRST, id);
"""  # nosec B608 — table names are module-level constants, never user input


class PostgresTaskQueue(QueueBackend):
    """Multi-host KG task/staging queue on the shared state Postgres."""

    def __init__(self, dsn: str | None = None, *, queue_table: str = "kg_task_queue"):
        from agent_utilities.core.state_store import ensure_state_schema, state_pool

        # Connectivity + schema probe up front so the engine can fall back to
        # the SQLite queue if the state store is unreachable at startup.
        self._pool = state_pool()
        self.queue_table = queue_table
        if queue_table == "kg_task_queue":
            ensure_state_schema("kg_task_queue", _DDL, pool=self._pool)
        else:
            # A non-default queue (e.g. the ORCH-1.45 agent dispatch queue)
            # gets its own table; the staging twin stays ingest-only.
            ensure_state_schema(
                queue_table, _queue_table_ddl(queue_table), pool=self._pool
            )
        self._claimer = f"{socket.gethostname()}:{id(self)}"
        logger.info(
            "PostgresTaskQueue ready (table=%s, cross-host SKIP LOCKED claims)",
            queue_table,
        )

    # ── internal ───────────────────────────────────────────────────────

    def _claim_one(self, table: str, columns: str) -> Any:
        """Atomically claim the oldest available row (SKIP LOCKED)."""
        now = time.time()
        cutoff = now - _VISIBILITY_TIMEOUT_S
        sql = f"""
            UPDATE {table} SET claimed_by = %s, claimed_at = %s
            WHERE id = (
                SELECT id FROM {table}
                WHERE claimed_at IS NULL OR claimed_at < %s
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING {columns}
        """  # nosec B608 — table/columns are module constants, values bound
        with self._pool.connection() as conn:
            row = conn.execute(sql, (self._claimer, now, cutoff)).fetchone()
        return row

    # ── QueueBackend: task submission queue ────────────────────────────

    def put(self, item: dict) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                f"INSERT INTO {self.queue_table} (data) VALUES (%s)",  # nosec B608 — constant table
                (json.dumps(item),),
            )

    def get(self) -> tuple[int, dict] | None:
        row = self._claim_one(self.queue_table, "id, data")
        if row is None:
            return None
        return int(row[0]), json.loads(row[1])

    def ack(self, item_id: int) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                f"DELETE FROM {self.queue_table} WHERE id = %s",  # nosec B608 — constant table
                (item_id,),
            )

    def get_queue_size(self) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM {self.queue_table}"  # nosec B608 — constant table
            ).fetchone()
        return int(row[0]) if row else 0

    # ── QueueBackend: staged-graph queue ───────────────────────────────

    def put_staged_graph(self, job_id: str, nodes: list, edges: list) -> None:
        payload = json.dumps({"nodes": nodes, "edges": edges})
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO kg_task_staging (job_id, graph_data) VALUES (%s, %s)",
                (job_id, payload),
            )

    def get_staged_graph(self) -> tuple[int, str, dict] | None:
        row = self._claim_one("kg_task_staging", "id, job_id, graph_data")
        if row is None:
            return None
        return int(row[0]), row[1], json.loads(row[2])

    def ack_staged_graph(self, item_id: int) -> None:
        with self._pool.connection() as conn:
            conn.execute("DELETE FROM kg_task_staging WHERE id = %s", (item_id,))
