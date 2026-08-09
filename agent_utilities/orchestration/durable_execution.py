"""
Durable Execution Engine (CONCEPT:AU-ECO.messaging.native-backend-abstraction / ORCH-1.36 / OS-5.16)

Provides fault-tolerant, resumable state execution by persisting execution
checkpoints to a durable, crash-safe store — by default the same embedded
SQLite substrate :mod:`agent_utilities.core.sessions` uses for session/turn
recovery. SQLite is chosen over the L1 graph deliberately: the in-memory
epistemic_graph tier is a *cache* (rebuilt on restart), whereas durable
execution must survive process death. With ``state_db_uri`` set
(CONCEPT:AU-OS.state.unified-durable-state-externalization) checkpoints live on the shared Postgres instead, so any
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

import functools
import json
import logging
import os
import sqlite3
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from agent_utilities.core.config import setting
from agent_utilities.core.paths import data_dir
from agent_utilities.orchestration.resilience import (
    DEFAULT_POLICY,
    ResiliencePolicy,
    run_with_resilience,
    run_with_resilience_sync,
)

logger = logging.getLogger(__name__)


class DurableRunVersionMismatch(RuntimeError):
    """Raised on resume when the caller's ``definition_version`` disagrees with
    the version this run was PINNED to when it started (DE5,
    CONCEPT:AU-KG.storage.durable-execution-unit's ``:definitionVersion``).

    Restate hard-pins an in-flight invocation to its starting deployment and
    fails loud on a protocol-incompatible upgrade (`durable-execution-native.md`
    Part 1 row 9, one of only two capabilities restate is genuinely ahead on).
    This is the ``DurableRun`` analog: a code upgrade that changes a step's
    semantics mid-flight must never silently resume under the new code as if
    nothing changed — it fails here instead, with the two versions named, so
    the operator can choose to migrate or discard the stale run explicitly.
    """


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
    """Resolve the durable-execution store path (env-overridable, XDG-friendly).

    Defaults under the agent-utilities data dir (``runtime/``), so it inherits
    the same XDG resolution AND test isolation (``AGENT_UTILITIES_DATA_DIR``) as
    the rest of the harness's durable state; ``DURABLE_EXECUTION_DB`` overrides.
    """
    override = setting("DURABLE_EXECUTION_DB")
    if override:
        return Path(override)
    base = data_dir() / "runtime"
    base.mkdir(parents=True, exist_ok=True)
    return base / "durable_execution.db"


class CheckpointStore(Protocol):
    """Backend seam for durable checkpoints (CONCEPT:AU-OS.state.unified-durable-state-externalization)."""

    def save_checkpoint(
        self,
        session_id: str,
        node_id: str,
        state_json: str,
        status: str,
        idempotency_key: str,
    ) -> None: ...

    def resume_session(self, session_id: str) -> dict[str, Any] | None: ...

    def get_checkpoint(self, session_id: str, node_id: str) -> tuple[str, str] | None:
        """Return ``(state_json, status)`` for one checkpoint, or ``None``."""
        ...

    def mark_completed(
        self, session_id: str, node_id: str, result_json: str
    ) -> None: ...

    def completed_result_raw(
        self, session_id: str, idempotency_key: str
    ) -> tuple[bool, Any]: ...


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

    def get_checkpoint(self, session_id: str, node_id: str) -> tuple[str, str] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT state, status FROM durable_checkpoints
                WHERE session_id = ? AND node_id = ?
                """,
                (session_id, node_id),
            ).fetchone()
        if row is None:
            return None
        return (row["state"], row["status"])

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
    """Shared-Postgres checkpoints (CONCEPT:AU-OS.state.unified-durable-state-externalization) over the ONE state pool."""

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

    def get_checkpoint(self, session_id: str, node_id: str) -> tuple[str, str] | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT state, status FROM durable_checkpoints
                WHERE session_id = %s AND node_id = %s
                """,
                (session_id, node_id),
            ).fetchone()
        if row is None:
            return None
        return (row[0], row[1])

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

    def get_checkpoint(self, node_id: str) -> dict[str, Any] | None:
        """Return ``{"node_id","state","status"}`` for ``node_id`` (or ``None``).

        ``state`` is decoded back to a dict. Used by :class:`DurableRun` to
        recover an in-flight run id after a crash-and-resume.
        """
        row = self._store.get_checkpoint(self.session_id, node_id)
        if row is None:
            return None
        raw_state, status = row
        try:
            state = json.loads(raw_state) if raw_state else {}
        except (TypeError, ValueError):
            state = {}
        if not isinstance(state, dict):
            state = {}
        return {"node_id": node_id, "state": state, "status": status}

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

    async def arun_durable_action(
        self,
        node_id: str,
        action: Callable[[], Awaitable[Any] | Any],
        *,
        idempotency_key: str | None = None,
        policy: ResiliencePolicy | None = None,
        state: dict[str, Any] | None = None,
    ) -> Any:
        """Async twin of :meth:`run_durable_action` for coroutine actions.

        Same exactly-once-effect / at-least-once contract, but ``action`` may be
        a coroutine function (awaited under the async resilience runner). Used by
        live async paths — the ORCH-5.0 goal loop and the queue dispatch worker —
        whose side effects are awaitables and must not be re-applied on a
        crash-and-resume or an at-least-once redelivery.
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
        result = await run_with_resilience(action, policy or DEFAULT_POLICY)
        self.mark_completed(node_id, result=result)
        return result


class DurableRun:
    """A resumable, crash-safe multi-step run over :class:`DurableExecutionManager`.

    The ONE crash-safe checkpoint substrate for the autonomous evolution / SDD
    loop. It resolves (or RESUMES) a run id under a stable ``session_id``, runs
    named steps exactly-once, and finalizes on success. If the process is killed
    — or a step raises — mid-run, the unfinished run stays ``PENDING``;
    re-constructing a ``DurableRun`` with the same ``session_id`` resumes it:
    already-completed steps are skipped and their recorded results replayed, and
    execution continues from the first incomplete step instead of restarting or
    losing the run.

    Wired into ``AgenticEvolutionEngine.run_evolution_cycle`` and
    ``LoopController.run_one_cycle`` (the ``KG_LOOP`` daemon tick), so a crash
    mid-cycle resumes the cycle rather than re-running completed stages. Backend
    selection is inherited from :class:`DurableExecutionManager` (embedded SQLite
    by default; shared Postgres when ``STATE_DB_URI`` is set), so the same run
    can be resumed on another host in the fleet.

    Two DE1/DE5 additions over the base checkpoint substrate, both opt-in via
    constructor args so every existing call site keeps its exact prior
    behavior unchanged:

    * ``engine`` (DE1, CONCEPT:AU-KG.storage.durable-execution-unit) — when
      supplied, every ``step()``/``finish()`` transition mirrors a
      ``:DurableRun`` node via
      :mod:`agent_utilities.knowledge_graph.durable_execution_kg`, the same
      fail-soft, no-op-when-absent posture ``KgAuditSink(engine=None)`` uses.
    * ``definition_version`` (DE5) — when supplied on both the run that
      started and the run that resumes, a mismatch raises
      :class:`DurableRunVersionMismatch` instead of silently resuming under
      different semantics. Omitting it (either side) skips enforcement
      entirely — this is a caller-declared pin, not a default requirement.
    """

    _POINTER = "__run__"

    def __init__(
        self,
        session_id: str,
        *,
        db_path: str | os.PathLike[str] | None = None,
        store: CheckpointStore | None = None,
        engine: Any | None = None,
        definition_version: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.engine = engine
        self.definition_version = definition_version
        self._mgr = DurableExecutionManager(session_id, db_path=db_path, store=store)
        self._auto_seq: dict[str, int] = {}
        pointer = self._mgr.get_checkpoint(self._POINTER)
        if (
            pointer is not None
            and pointer["status"] == "PENDING"
            and pointer["state"].get("run_id")
        ):
            # An earlier run for this session was interrupted — resume it.
            self.run_id = str(pointer["state"]["run_id"])
            self.resumed = True
            pinned_version = pointer["state"].get("definition_version")
            if (
                pinned_version
                and definition_version
                and pinned_version != definition_version
            ):
                raise DurableRunVersionMismatch(
                    f"DurableRun {self.run_id!r} (session {session_id!r}) was "
                    f"pinned to definition_version={pinned_version!r}; resume "
                    f"requested definition_version={definition_version!r}. "
                    "Migrate the run's stored state or discard it explicitly "
                    "before resuming under a different definition version."
                )
        else:
            self.run_id = (
                f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex}"
            )
            self.resumed = False
            self._mgr.save_checkpoint(
                self._POINTER,
                {"run_id": self.run_id, "definition_version": definition_version},
                status="PENDING",
            )
        self._mirror(durable_status="PENDING" if not self.resumed else "RESUMED")

    def _key(self, name: str) -> str:
        return f"{self.run_id}:{name}"

    def _mirror(
        self, *, durable_status: str, checkpoint_ref: str | None = None
    ) -> None:
        """DE1 mirror-on-write — see the class doc's ``engine`` paragraph."""
        from agent_utilities.knowledge_graph.durable_execution_kg import (
            mirror_durable_run,
        )

        mirror_durable_run(
            self.engine,
            session_id=self.session_id,
            run_id=self.run_id,
            durable_status=durable_status,
            checkpoint_ref=checkpoint_ref,
            definition_version=self.definition_version,
        )

    def link_run_trace(self, run_trace_id: str) -> None:
        """Complete DE0's ``:produced``/``:producedBy`` edge to a ``RunTrace``.

        Callers that write a ``RunTrace`` for this run (``observability.
        trace_ontology.trace_id``) can call this once they have its id; a
        no-op when ``engine`` was not supplied to this run.
        """
        from agent_utilities.knowledge_graph.durable_execution_kg import (
            link_durable_run_produced,
        )

        link_durable_run_produced(
            self.engine,
            session_id=self.session_id,
            run_id=self.run_id,
            run_trace_id=run_trace_id,
        )

    def is_done(self, name: str) -> bool:
        """True if ``name`` already completed in this (possibly resumed) run."""
        applied, _ = self._mgr._completed_result_for(self._key(name))
        return applied

    def step(
        self,
        name: str,
        action: Callable[[], Any],
        *,
        state: dict[str, Any] | None = None,
    ) -> Any:
        """Run ``action`` once for this run; on resume, skip it and replay its result.

        Unlike :meth:`DurableExecutionManager.run_durable_action` there is no
        internal retry: an exception — or a hard crash — propagates and leaves
        the step un-completed, so the resumed run re-executes exactly that step
        while every already-completed step stays skipped.
        """
        key = self._key(name)
        applied, prior = self._mgr._completed_result_for(key)
        if applied:
            return prior
        self._mgr.save_checkpoint(
            key, state or {}, status="PENDING", idempotency_key=key
        )
        self._mirror(durable_status="PENDING", checkpoint_ref=key)
        result = action()
        self._mgr.mark_completed(key, result=result)
        self._mirror(durable_status="COMPLETED", checkpoint_ref=key)
        return result

    def _next_auto_seq(self, label: str) -> int:
        seq = self._auto_seq.get(label, 0) + 1
        self._auto_seq[label] = seq
        return seq

    def auto_step(
        self, name: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator (DE3): checkpoint the wrapped call with NO hand-named step.

        Narrows restate's genuine ergonomics lead (Part 1 rows 1/5 — automatic
        any-await-point replay vs. our named-step design) for the shape of
        workload this run actually has: agent-loop boundaries (one tool call,
        one LLM turn) called repeatedly inside a loop. The step name is
        derived from the wrapped function's own ``__qualname__`` (or ``name``
        if given) plus a per-run auto-incrementing call sequence, so calling
        the same decorated function N times in one run produces N distinct,
        independently resumable steps without the caller designing a step
        schema. This does not replicate restate's general-purpose
        bytecode/coroutine replay of ARBITRARY imperative code — it only
        removes the naming burden from :meth:`step`'s existing named-step
        model.

        Caveat: the default (unset ``name``) label is only stable across a
        crash-and-resume if the SAME function object's ``__qualname__``
        resolves identically on restart — true for an ordinary module- or
        class-level function (the common case: the process re-imports the
        same source), but NOT for a locally-defined closure created fresh
        with a different name each attempt. Pass an explicit ``name=`` when
        that stability is not guaranteed.
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            label = name or fn.__qualname__

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                step_name = f"{label}#{self._next_auto_seq(label)}"
                return self.step(step_name, lambda: fn(*args, **kwargs))

            return wrapper

        return decorator

    def sleep_until(self, name: str, wake_at_ms: int) -> bool:
        """Durable timer bound to THIS run's named step (DE6).

        Returns ``True`` once ``wake_at_ms`` has passed (and durably records
        the step as completed, so a later call is a free idempotent replay);
        returns ``False`` — durably checkpointed ``WAITING`` with the deadline
        itself as the checkpoint payload — otherwise. This is the closest
        ``DurableRun``-backend analog to restate's
        ``sleep(ms)``-and-wake-THIS-exact-suspended-continuation
        (`durable-execution-native.md` Part 1 row 7): unlike ``eg-jobs``'s
        cron/interval triggers (dedup'd per tick window, not bound to one
        in-flight execution), this deadline is scoped to exactly one run's
        exactly one named step. It is POLL-based, not push-based — there is
        no wake callback here, only a durably-remembered deadline a caller's
        own retry loop (e.g. the next ``KG_LOOP`` daemon tick) re-checks
        without ever losing WHICH deadline it was waiting on. Narrows, but
        does not fully close, restate's push-based wake semantics.
        """
        key = self._key(name)
        applied, _prior = self._mgr._completed_result_for(key)
        if applied:
            return True
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        if now_ms < int(wake_at_ms):
            self._mgr.save_checkpoint(
                key,
                {"wake_at_ms": int(wake_at_ms)},
                status="PENDING",
                idempotency_key=key,
            )
            self._mirror(durable_status="WAITING", checkpoint_ref=key)
            return False
        self._mgr.save_checkpoint(
            key, {"wake_at_ms": int(wake_at_ms)}, status="PENDING", idempotency_key=key
        )
        self._mgr.mark_completed(key, result=True)
        self._mirror(durable_status="COMPLETED", checkpoint_ref=key)
        return True

    def finish(self) -> None:
        """Mark the run complete so the next run under this session starts fresh."""
        self._mgr.mark_completed(self._POINTER)
        self._mirror(durable_status="COMPLETED")
