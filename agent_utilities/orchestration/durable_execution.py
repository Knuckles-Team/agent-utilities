"""
Durable Execution Engine (CONCEPT:ECO-4.0 / ORCH-1.36)

Provides fault-tolerant, resumable state execution by persisting execution
checkpoints to an embedded, crash-safe SQLite store — the same durable substrate
:mod:`agent_utilities.core.sessions` uses for session/turn recovery. SQLite is
chosen over the L1 graph deliberately: the in-memory epistemic_graph tier is a
*cache* (rebuilt on restart), whereas durable execution must survive process
death. The store is a single embedded file (no external infra); production
deployments can additionally mirror checkpoints into the KG for lineage.

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
from typing import Any

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

_init_lock = threading.Lock()
_initialized_paths: set[str] = set()


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


def _ensure_schema(db_path: Path) -> None:
    key = str(db_path)
    if key in _initialized_paths:
        return
    with _init_lock:
        if key in _initialized_paths:
            return
        conn = sqlite3.connect(key)
        try:
            conn.executescript(_DDL)
            conn.commit()
        finally:
            conn.close()
        _initialized_paths.add(key)


class DurableExecutionManager:
    """Persistence and resumption of execution checkpoints (SQLite-backed)."""

    def __init__(self, session_id: str, db_path: str | os.PathLike | None = None):
        self.session_id = session_id
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        _ensure_schema(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── Checkpointing ────────────────────────────────────────────────────
    def save_checkpoint(
        self,
        node_id: str,
        state: dict[str, Any],
        status: str = "PENDING",
        idempotency_key: str | None = None,
    ) -> str:
        """Persist (upsert) a checkpoint."""
        conn = self._connect()
        try:
            conn.execute(
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
                (
                    self.session_id,
                    node_id,
                    json.dumps(state, default=str),
                    status,
                    idempotency_key or "",
                    _now(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return node_id

    def resume_session(self) -> dict[str, Any] | None:
        """Resume the most recently updated PENDING checkpoint for this session."""
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT node_id, state FROM durable_checkpoints
                WHERE session_id = ? AND status = 'PENDING'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (self.session_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {"node_id": row["node_id"], "state": row["state"]}

    def mark_completed(self, node_id: str, result: Any = None) -> None:
        """Mark a checkpoint COMPLETED, recording its result."""
        conn = self._connect()
        try:
            conn.execute(
                """
                UPDATE durable_checkpoints
                SET status = 'COMPLETED', result = ?, updated_at = ?
                WHERE session_id = ? AND node_id = ?
                """,
                (
                    json.dumps(result, default=str) if result is not None else "",
                    _now(),
                    self.session_id,
                    node_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Idempotency / exactly-once ───────────────────────────────────────
    def _completed_result_for(self, idempotency_key: str) -> tuple[bool, Any]:
        """Return ``(applied, result)`` for a previously completed idempotency key."""
        conn = self._connect()
        try:
            row = conn.execute(
                """
                SELECT result FROM durable_checkpoints
                WHERE session_id = ? AND idempotency_key = ? AND status = 'COMPLETED'
                ORDER BY updated_at DESC LIMIT 1
                """,
                (self.session_id, idempotency_key),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return (False, None)
        raw = row["result"]
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
