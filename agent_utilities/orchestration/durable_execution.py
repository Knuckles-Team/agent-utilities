"""
Durable Execution Engine (CONCEPT:ECO-4.0 / ORCH-1.36 / OS-5.16)

Provides fault-tolerant, resumable state execution by persisting execution
checkpoints to a durable, crash-safe store — by default the same embedded
SQLite substrate :mod:`agent_utilities.core.sessions` uses for session/turn
recovery. SQLite is chosen over the L1 graph deliberately: the in-memory
epistemic_graph tier is a *cache* (rebuilt on restart), whereas durable
execution must survive process death. With ``state_db_uri`` set
(CONCEPT:OS-5.16) checkpoints live on the shared Postgres instead, so any
host in the fleet can resume another host's execution.

Durability guarantees:

* **At-least-once**: :meth:`run_durable_action` retries the primary callable
  under a :class:`~agent_utilities.orchestration.resilience.ResiliencePolicy`
  (backoff + ``retry_on``), so transient failures don't drop the action.
* **Exactly-once effects**: each critical action carries an *idempotency key*. A
  completed key short-circuits re-execution and returns the recorded result, so
  a retry — or a crash-and-resume — never applies the same effect twice.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from agent_utilities.orchestration.resilience import (
    DEFAULT_POLICY,
    ResiliencePolicy,
    run_with_resilience_sync,
)

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS durable_checkpoints (
    session_id      TEXT NOT NULL,
    node_id         TEXT NOT NULL,
    state           TEXT,
    status          TEXT DEFAULT 'PENDING',
    idempotency_key TEXT,
    result          TEXT,
    updated_at      TEXT,
    PRIMARY KEY (session_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_durable_idem
    ON durable_checkpoints (session_id, idempotency_key, status);
"""

# Same shape on Postgres; only the upsert placeholders differ.
_PG_DDL = _DDL


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _default_db_path() -> Path:
    """Resolve the durable-execution store path (env-overridable, XDG-friendly)."""
    override = os.environ.get("DURABLE_EXECUTION_DB")
    if override:
        return Path(override)
    base = (
        Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        / "agent_utilities"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base / "durable_execution.db"


class CheckpointStore(Protocol):
    """Backend seam for durable checkpoints (CONCEPT:OS-5.16)."""

    def save_checkpoint(
        self,
        session_id: str,
        node_id: str,
        state_json: str,
        status: str,
        idempotency_key: str,
    ) -> None:
        ...

    def resume_session(self, session_id: str) -> dict[str, Any] | None:
        ...

    def mark_completed(self, session_id: str, node_id: str, result_json: str) -> None:
        ...

    def completed_result_raw(
        self, session_id: str, idempotency_key: str
    ) -> tuple[bool, Any]:
        ...


class SQLiteCheckpointStore:
    """Embedded zero-infra default — ONE pooled connection per db file.

    Previously every operation opened a fresh ``sqlite3.connect``; the store
    now keeps a single long-lived connection per path (``check_same_thread``
    off) serialized by a lock, shared across all manager instances in the
    process.
    """

    _conns: dict[str, tuple[sqlite3.Connection, threading.RLock]] = {}
    _conns_lock = threading.Lock()

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn, self._lock = self._shared_conn(str(db_path))

    @classmethod
    def _shared_conn(cls, key: str) -> tuple[sqlite3.Connection, threading.RLock]:
        with cls._conns_lock:
            entry = cls._conns.get(key)
            if entry is None:
                conn = sqlite3.connect(key, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.executescript(_DDL)
                conn.commit()
                entry = (conn, threading.RLock())
                cls._conns[key] = entry
            return entry

    def save_checkpoint(
        self,
        session_id: str,
        node_id: str,
        state_json: str,
        status: str,
        idempotency_key: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO durable_checkpoints
                    (session_id, node_id, state, status, idempotency_key, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, node_id) DO UPDATE SET
                    state = excluded.state,
                    status = excluded.status,
                    idempotency_key = excluded.idempotency_key,
                    updated_at = excluded.updated_at
                """,
                (session_id, node_id, state_json, status, idempotency_key, _now()),
            )
            self._conn.commit()

    def resume_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT node_id, state FROM durable_checkpoints
                WHERE session_id = ? AND status = 'PENDING'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {"node_id": row["node_id"], "state": row["state"]}

    def mark_completed(self, session_id: str, node_id: str, result_json: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE durable_checkpoints
                SET status = 'COMPLETED', result = ?, updated_at = ?
                WHERE session_id = ? AND node_id = ?
                """,
                (result_json, _now(), session_id, node_id),
            )
            self._conn.commit()

    def completed_result_raw(
        self, session_id: str, idempotency_key: str
    ) -> tuple[bool, Any]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT result FROM durable_checkpoints
                WHERE session_id = ? AND idempotency_key = ? AND status = 'COMPLETED'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (session_id, idempotency_key),
            ).fetchone()
        if row is None:
            return (False, None)
        return (True, row["result"])


class PostgresCheckpointStore:
    """Shared-Postgres checkpoints (CONCEPT:OS-5.16) over the ONE state pool."""

    def __init__(self, pool: Any | None = None):
        from agent_utilities.core.state_store import ensure_state_schema, state_pool

        self._pool = pool if pool is not None else state_pool()
        ensure_state_schema("durable_execution", _PG_DDL, pool=self._pool)

    def save_checkpoint(
        self,
        session_id: str,
        node_id: str,
        state_json: str,
        status: str,
        idempotency_key: str,
    ) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO durable_checkpoints
                    (session_id, node_id, state, status, idempotency_key, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(session_id, node_id) DO UPDATE SET
                    state = excluded.state,
                    status = excluded.status,
                    idempotency_key = excluded.idempotency_key,
                    updated_at = excluded.updated_at
                """,
                (session_id, node_id, state_json, status, idempotency_key, _now()),
            )

    def resume_session(self, session_id: str) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT node_id, state FROM durable_checkpoints
                WHERE session_id = %s AND status = 'PENDING'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {"node_id": row[0], "state": row[1]}

    def mark_completed(self, session_id: str, node_id: str, result_json: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                UPDATE durable_checkpoints
                SET status = 'COMPLETED', result = %s, updated_at = %s
                WHERE session_id = %s AND node_id = %s
                """,
                (result_json, _now(), session_id, node_id),
            )

    def completed_result_raw(
        self, session_id: str, idempotency_key: str
    ) -> tuple[bool, Any]:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT result FROM durable_checkpoints
                WHERE session_id = %s AND idempotency_key = %s
                  AND status = 'COMPLETED'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (session_id, idempotency_key),
            ).fetchone()
        if row is None:
            return (False, None)
        return (True, row[0])


def _select_store(db_path: str | os.PathLike | None) -> CheckpointStore:
    """Backend selection: explicit path → SQLite; else state_db_uri → Postgres;
    else the default per-host SQLite file (zero-infra default preserved)."""
    if db_path is not None:
        return SQLiteCheckpointStore(Path(db_path))
    from agent_utilities.core.state_store import postgres_state_enabled

    if postgres_state_enabled():
        return PostgresCheckpointStore()
    return SQLiteCheckpointStore(_default_db_path())


class DurableExecutionManager:
    """Persistence and resumption of execution checkpoints (backend-selectable)."""

    def __init__(
        self,
        session_id: str,
        db_path: str | os.PathLike | None = None,
        store: CheckpointStore | None = None,
    ):
        self.session_id = session_id
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self._store: CheckpointStore = (
            store if store is not None else _select_store(db_path)
        )

    # ── Checkpointing ────────────────────────────────────────────────────
    def save_checkpoint(
        self,
        node_id: str,
        state: dict[str, Any],
        status: str = "PENDING",
        idempotency_key: str | None = None,
    ) -> str:
        """Persist (upsert) a checkpoint."""
        self._store.save_checkpoint(
            self.session_id,
            node_id,
            json.dumps(state, default=str),
            status,
            idempotency_key or "",
        )
        return node_id

    def resume_session(self) -> dict[str, Any] | None:
        """Resume the most recently updated PENDING checkpoint for this session."""
        return self._store.resume_session(self.session_id)

    def mark_completed(self, node_id: str, result: Any = None) -> None:
        """Mark a checkpoint COMPLETED, recording its result."""
        self._store.mark_completed(
            self.session_id,
            node_id,
            json.dumps(result, default=str) if result is not None else "",
        )

    # ── Idempotency / exactly-once ───────────────────────────────────────
    def _completed_result_for(self, idempotency_key: str) -> tuple[bool, Any]:
        """Return ``(applied, result)`` for a previously completed idempotency key."""
        applied, raw = self._store.completed_result_raw(
            self.session_id, idempotency_key
        )
        if not applied:
            return (False, None)
        try:
            return (True, json.loads(raw) if raw else None)
        except (TypeError, ValueError):
            return (True, raw)

    def run_durable_action(
        self,
        node_id: str,
        action: Callable[[], Any],
        *,
        idempotency_key: str | None = None,
        policy: ResiliencePolicy | None = None,
        state: dict[str, Any] | None = None,
    ) -> Any:
        """Execute ``action`` durably: at-least-once retries, exactly-once effect.

        If ``idempotency_key`` has already completed for this session, ``action``
        is *not* re-run and the recorded result is returned (exactly-once). Else a
        PENDING checkpoint is written, ``action`` runs under ``policy`` retries
        (at-least-once), and on success the checkpoint is marked COMPLETED with
        the result keyed by ``idempotency_key`` (defaulting to ``node_id``).
        """
        key = idempotency_key or node_id
        applied, prior = self._completed_result_for(key)
        if applied:
            logger.info(
                "[durable] idempotency_key=%r already completed; skipping re-exec.",
                key,
            )
            return prior

        self.save_checkpoint(
            node_id, state or {}, status="PENDING", idempotency_key=key
        )
        result = run_with_resilience_sync(action, policy or DEFAULT_POLICY)
        self.mark_completed(node_id, result=result)
        return result
