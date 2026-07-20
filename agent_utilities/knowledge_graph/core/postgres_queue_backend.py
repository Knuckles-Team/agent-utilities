"""Cross-host-safe Postgres task queue.

CONCEPT:AU-KG.ingest.cross-host-safe-kg — Cross-host-safe KG task queue with atomic SKIP LOCKED claims and visibility-timeout recovery on the shared state store

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

Task claims use the current scheduling schema: priority first, then tenant /
fairness-group recency, deadline, enqueue time, and id. Queue admission takes a
transaction advisory lock so the configured depth bound cannot be raced by
another Postgres producer.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .queue_backend import QueueBackend

logger = logging.getLogger(__name__)

# A claimed item whose claim is older than this becomes claimable again
# (its claimer is presumed dead — crashed between get() and ack()).
_VISIBILITY_TIMEOUT_S = 600.0
_QUEUE_TABLES = frozenset({"kg_task_queue", "agent_dispatch_queue"})
_CURRENT_TASK_COLUMNS = frozenset(
    {
        "id",
        "data",
        "tenant_ref",
        "fairness_group_ref",
        "prio_bucket",
        "deadline_unix",
        "enqueued_at",
        "claimed_by",
        "claimed_at",
    }
)


@dataclass(frozen=True)
class _PostgresReceipt:
    """Opaque acknowledgement receipt fenced to one visibility claim."""

    item_id: int
    claimed_by: str
    claimed_at: float


_DDL = """
CREATE TABLE IF NOT EXISTS kg_task_queue (
    id BIGSERIAL PRIMARY KEY,
    data TEXT NOT NULL,
    tenant_ref TEXT NOT NULL,
    fairness_group_ref TEXT NOT NULL,
    prio_bucket SMALLINT NOT NULL CHECK (prio_bucket BETWEEN 0 AND 3),
    deadline_unix DOUBLE PRECISION,
    enqueued_at DOUBLE PRECISION NOT NULL,
    claimed_by TEXT,
    claimed_at DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS kg_task_queue_fairness (
    tenant_ref TEXT NOT NULL,
    fairness_group_ref TEXT NOT NULL,
    last_claimed_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (tenant_ref, fairness_group_ref)
);
CREATE TABLE IF NOT EXISTS kg_task_staging (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT,
    graph_data TEXT NOT NULL,
    claimed_by TEXT,
    claimed_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_kg_task_queue_claim
    ON kg_task_queue (
        claimed_at NULLS FIRST, prio_bucket, deadline_unix NULLS LAST,
        enqueued_at, id
    );
CREATE INDEX IF NOT EXISTS idx_kg_task_staging_claim
    ON kg_task_staging (claimed_at NULLS FIRST, id);
"""


def _queue_table_ddl(table: str) -> str:
    """DDL for an additional SKIP LOCKED claim queue table.

    CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch — the agent dispatch queue (``agent_dispatch_queue``)
    reuses this backend with its own table; same columns, same claim contract.
    """
    return f"""
CREATE TABLE IF NOT EXISTS {table} (
    id BIGSERIAL PRIMARY KEY,
    data TEXT NOT NULL,
    tenant_ref TEXT NOT NULL,
    fairness_group_ref TEXT NOT NULL,
    prio_bucket SMALLINT NOT NULL CHECK (prio_bucket BETWEEN 0 AND 3),
    deadline_unix DOUBLE PRECISION,
    enqueued_at DOUBLE PRECISION NOT NULL,
    claimed_by TEXT,
    claimed_at DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS {table}_fairness (
    tenant_ref TEXT NOT NULL,
    fairness_group_ref TEXT NOT NULL,
    last_claimed_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (tenant_ref, fairness_group_ref)
);
CREATE INDEX IF NOT EXISTS idx_{table}_claim
    ON {table} (
        claimed_at NULLS FIRST, prio_bucket, deadline_unix NULLS LAST,
        enqueued_at, id
    );
"""  # nosec B608 — table names are module-level constants, never user input


class PostgresTaskQueue(QueueBackend):
    """Multi-host KG task/staging queue on the shared state Postgres."""

    def __init__(self, dsn: str | None = None, *, queue_table: str = "kg_task_queue"):
        from agent_utilities.core.state_store import ensure_state_schema, state_pool

        if queue_table not in _QUEUE_TABLES:
            raise ValueError("queue_table is not part of the current queue schema")
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
        self._require_current_schema()
        self._claimer = f"queue-worker:{uuid.uuid4().hex}"
        logger.info(
            "PostgresTaskQueue ready (table=%s, cross-host SKIP LOCKED claims)",
            queue_table,
        )

    # ── internal ───────────────────────────────────────────────────────

    def _require_current_schema(self) -> None:
        """Fail at startup when a pre-current queue table is still present."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT column_name
                  FROM information_schema.columns
                 WHERE table_schema = current_schema()
                   AND table_name = %s
                """,
                (self.queue_table,),
            ).fetchall()
        columns = {str(row[0]) for row in rows}
        missing = _CURRENT_TASK_COLUMNS - columns
        if missing:
            raise RuntimeError(
                "Postgres task queue does not match the current scheduling schema"
            )

    def _claim_task(self) -> Any:
        """Claim the scheduled row selected by priority/deadline/fairness."""
        now = time.time()
        cutoff = now - _VISIBILITY_TIMEOUT_S
        sql = f"""
            WITH candidate AS (
                SELECT q.id
                FROM {self.queue_table} AS q
                LEFT JOIN {self.queue_table}_fairness AS tenant_fairness
                  ON tenant_fairness.tenant_ref = q.tenant_ref
                 AND tenant_fairness.fairness_group_ref = ''
                LEFT JOIN {self.queue_table}_fairness AS group_fairness
                  ON group_fairness.tenant_ref = q.tenant_ref
                 AND group_fairness.fairness_group_ref = q.fairness_group_ref
                WHERE q.claimed_at IS NULL OR q.claimed_at < %s
                ORDER BY
                    CASE
                        WHEN q.deadline_unix IS NOT NULL
                         AND q.deadline_unix <= %s THEN 0
                        ELSE 1
                    END,
                    q.prio_bucket ASC,
                    COALESCE(tenant_fairness.last_claimed_at, 0.0) ASC,
                    COALESCE(group_fairness.last_claimed_at, 0.0) ASC,
                    q.deadline_unix ASC NULLS LAST,
                    q.enqueued_at ASC,
                    q.id ASC
                LIMIT 1
                FOR UPDATE OF q SKIP LOCKED
            )
            UPDATE {self.queue_table} AS q
               SET claimed_by = %s, claimed_at = %s
              FROM candidate
             WHERE q.id = candidate.id
            RETURNING q.id, q.data, q.tenant_ref, q.fairness_group_ref,
                      q.claimed_by, q.claimed_at
        """  # nosec B608 — queue_table is a closed constructor vocabulary
        with self._pool.connection() as conn:
            row = conn.execute(sql, (cutoff, now, self._claimer, now)).fetchone()
            if row is not None:
                conn.execute(
                    f"""
                    INSERT INTO {self.queue_table}_fairness
                        (tenant_ref, fairness_group_ref, last_claimed_at)
                    VALUES (%s, '', %s), (%s, %s, %s)
                    ON CONFLICT (tenant_ref, fairness_group_ref) DO UPDATE
                        SET last_claimed_at = EXCLUDED.last_claimed_at
                    """,  # nosec B608 — closed table vocabulary
                    (row[2], now, row[2], row[3], now),
                )
        return row

    def _claim_staged(self) -> Any:
        """Atomically claim the oldest available staged graph."""
        now = time.time()
        cutoff = now - _VISIBILITY_TIMEOUT_S
        sql = """
            UPDATE kg_task_staging SET claimed_by = %s, claimed_at = %s
            WHERE id = (
                SELECT id FROM kg_task_staging
                WHERE claimed_at IS NULL OR claimed_at < %s
                ORDER BY id
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, job_id, graph_data, claimed_by, claimed_at
        """
        with self._pool.connection() as conn:
            return conn.execute(sql, (self._claimer, now, cutoff)).fetchone()

    @staticmethod
    def _schedule_fields(
        item: dict[str, Any],
    ) -> tuple[str, str, int, float | None, float]:
        """Return privacy-safe queue scheduling columns for one envelope."""
        from agent_utilities.knowledge_graph.core.engine_tasks import (
            _coerce_prio_bucket,
        )
        from agent_utilities.security.persistence_privacy import (
            persistence_reference,
        )

        tenant = str(item.get("tenant") or "shared")
        fairness_group = str(
            item.get("fairness_group")
            or item.get("session_id")
            or item.get("partition_ref")
            or tenant
        )
        deadline_raw = item.get("deadline_unix")
        deadline = float(deadline_raw) if deadline_raw is not None else None
        enqueued_at = float(item.get("enqueued_at") or time.time())
        return (
            persistence_reference(
                "tenant", tenant, namespace="postgres-task-scheduler"
            ),
            persistence_reference(
                "fairness_group",
                fairness_group,
                namespace="postgres-task-scheduler",
            ),
            _coerce_prio_bucket(item.get("prio_bucket", item.get("priority", 2))),
            deadline,
            enqueued_at,
        )

    def _insert(self, conn: Any, item: dict[str, Any]) -> None:
        tenant_ref, fairness_ref, priority, deadline, enqueued_at = (
            self._schedule_fields(item)
        )
        conn.execute(
            f"""
            INSERT INTO {self.queue_table}
                (data, tenant_ref, fairness_group_ref, prio_bucket,
                 deadline_unix, enqueued_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,  # nosec B608 — closed table vocabulary
            (
                json.dumps(item),
                tenant_ref,
                fairness_ref,
                priority,
                deadline,
                enqueued_at,
            ),
        )

    # ── QueueBackend: task submission queue ────────────────────────────

    def put(self, item: dict) -> None:
        self.put_many([item])

    def put_many(self, items: list[dict[str, Any]]) -> None:
        with self._pool.connection() as conn:
            for item in items:
                self._insert(conn, item)

    def put_if_below(self, item: dict[str, Any], max_depth: int) -> bool:
        """Atomically admit under ``max_depth`` across all producer hosts."""
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        with self._pool.connection() as conn:
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (self.queue_table,),
            )
            row = conn.execute(
                f"SELECT COUNT(*) FROM {self.queue_table}"  # nosec B608
            ).fetchone()
            if row and int(row[0]) >= max_depth:
                return False
            self._insert(conn, item)
            return True

    def get(self) -> tuple[Any, dict] | None:
        row = self._claim_task()
        if row is None:
            return None
        return (
            _PostgresReceipt(int(row[0]), str(row[4]), float(row[5])),
            json.loads(row[1]),
        )

    def ack(self, item_id: Any) -> None:
        if not isinstance(item_id, _PostgresReceipt):
            raise TypeError("Postgres acknowledgement requires a fenced receipt")
        with self._pool.connection() as conn:
            row = conn.execute(
                f"""
                DELETE FROM {self.queue_table}
                 WHERE id = %s AND claimed_by = %s AND claimed_at = %s
                RETURNING tenant_ref, fairness_group_ref
                """,  # nosec B608 — closed table vocabulary
                (item_id.item_id, item_id.claimed_by, item_id.claimed_at),
            ).fetchone()
            if row is None:
                raise RuntimeError("Postgres acknowledgement was fenced")
            conn.execute(
                f"""
                    DELETE FROM {self.queue_table}_fairness AS f
                     WHERE f.tenant_ref = %s
                       AND (
                            (
                                f.fairness_group_ref = %s
                                AND NOT EXISTS (
                                    SELECT 1 FROM {self.queue_table} AS q
                                     WHERE q.tenant_ref = f.tenant_ref
                                       AND q.fairness_group_ref = f.fairness_group_ref
                                )
                            )
                            OR (
                                f.fairness_group_ref = ''
                                AND NOT EXISTS (
                                    SELECT 1 FROM {self.queue_table} AS q
                                     WHERE q.tenant_ref = f.tenant_ref
                                )
                            )
                       )
                """,  # nosec B608 — closed table vocabulary
                (row[0], row[1]),
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

    def get_staged_graph(self) -> tuple[Any, str, dict] | None:
        row = self._claim_staged()
        if row is None:
            return None
        return (
            _PostgresReceipt(int(row[0]), str(row[3]), float(row[4])),
            row[1],
            json.loads(row[2]),
        )

    def ack_staged_graph(self, item_id: Any) -> None:
        if not isinstance(item_id, _PostgresReceipt):
            raise TypeError("Postgres acknowledgement requires a fenced receipt")
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                DELETE FROM kg_task_staging
                 WHERE id = %s AND claimed_by = %s AND claimed_at = %s
                RETURNING id
                """,
                (item_id.item_id, item_id.claimed_by, item_id.claimed_at),
            ).fetchone()
        if row is None:
            raise RuntimeError("Postgres staged acknowledgement was fenced")
