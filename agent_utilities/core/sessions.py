#!/usr/bin/python
from __future__ import annotations

"""Durable Session & Autonomous Goal persistence.

CONCEPT:ORCH-5.0 — Durable session and autonomous goal persistence with iterative background goal loops
CONCEPT:ORCH-1.44 — Durable goal registry — goals persist across restarts and stranded runs rehydrate as orphaned instead of silently vanishing

This module houses the schema initialization, memory maps, background runner thread,
and Starlette REST handlers for durable agent sessions and iterative goals.

State backends (CONCEPT:OS-5.16): by default sessions/turns/goals live in the
per-host SQLite file; with ``state_db_uri`` set they live on the shared
Postgres state store, so the gateway is stateless and any host can see the
whole fleet's sessions.

Goal durability (CONCEPT:ORCH-1.44): ``active_goals``/``background_goal_runs``
are an in-memory *cache* over the durable ``goals`` table. Every status change
is persisted; on restart, this host's non-terminal goals are rehydrated as
``orphaned`` (visible + resumable-by-hand) instead of silently vanishing.
"""

import asyncio
import logging
import os
import socket
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse

from agent_utilities.core.config import setting
from agent_utilities.models.goal import GoalIteration, GoalSpec, GoalStatus

logger = logging.getLogger(__name__)

# Resolved paths
DEFAULT_AGENT_DIR = Path(setting("AGENT_WORKSPACE", "workspace"))
DEFAULT_AGENT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache of active runs (durable source of truth is the goals table).
active_goals: dict[str, dict[str, Any]] = {}
background_goal_runs: dict[str, dict[str, Any]] = {}

# Goal statuses that are still "in flight" (rehydration targets on restart).
_NON_TERMINAL_GOAL_STATUSES = ("pending", "running", "validating")

_HOSTNAME = socket.gethostname()


def _owner_token() -> str:
    """Stable owner identity for goal runs: ``hostname:pid``.

    A restart changes the pid, so a goal row carrying this host's name with a
    dead pid is provably orphaned (CONCEPT:ORCH-1.44)."""
    return f"{_HOSTNAME}:{os.getpid()}"


def _identity_metadata() -> dict:
    """Ambient ``{tenant_id, actor_id}`` for stamping into session metadata.

    The tenant-scoped fleet plane (CONCEPT:OS-5.10) aggregates sessions by
    ``metadata.tenant``; stamping the server-minted identity here is what makes
    "show me org X's sessions / which agents client Y spawned" a tenant-scoped
    query (CONCEPT:OS-5.11 + OS-5.14). Best-effort: no actor → ``{}``.
    """
    try:
        from agent_utilities.security.brain_context import current_actor

        actor = current_actor()
        out: dict = {}
        if actor.tenant_id:
            out["tenant"] = actor.tenant_id
            out["tenant_id"] = actor.tenant_id
        if actor.actor_id and actor.actor_id != "system":
            out["actor_id"] = actor.actor_id
        return out
    except Exception:  # noqa: BLE001 — identity stamping is best-effort
        return {}


class StartGoalPayload(BaseModel):
    objective: str
    max_iterations: int = 20
    validation_cmd: str = ""
    constraints: list[str] = []


_SQLITE_DDL = """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT DEFAULT '',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        model TEXT DEFAULT '',
        mode TEXT DEFAULT 'ask',
        workspace TEXT DEFAULT '',
        turn_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active',
        background INTEGER DEFAULT 0,
        needs_input INTEGER DEFAULT 0,
        last_response_preview TEXT DEFAULT '',
        goal_id TEXT DEFAULT '',
        metadata_json TEXT DEFAULT '{}'
    );

    CREATE TABLE IF NOT EXISTS turns (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        turn_number INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT DEFAULT '',
        created_at REAL NOT NULL,
        status TEXT DEFAULT 'completed',
        usage_json TEXT DEFAULT '{}',
        duration_ms INTEGER DEFAULT 0,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS goals (
        goal_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        objective TEXT DEFAULT '',
        owner_host TEXT DEFAULT '',
        total_iterations INTEGER DEFAULT 0,
        total_duration_ms INTEGER DEFAULT 0,
        total_tool_calls INTEGER DEFAULT 0,
        summary TEXT DEFAULT '',
        error TEXT DEFAULT '',
        iterations_json TEXT DEFAULT '[]',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS dispatch_workers (
        worker_id TEXT PRIMARY KEY,
        host TEXT DEFAULT '',
        capacity INTEGER DEFAULT 1,
        active_sessions TEXT DEFAULT '[]',
        queue_backend TEXT DEFAULT '',
        started_at REAL NOT NULL,
        last_heartbeat REAL NOT NULL
    );
"""

# Same logical schema on Postgres (CONCEPT:OS-5.16). REAL epoch timestamps
# become DOUBLE PRECISION; everything else maps 1:1 so the handlers' SQL works
# on both backends through the state-store placeholder adapter.
_PG_DDL = """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT DEFAULT '',
        created_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL,
        model TEXT DEFAULT '',
        mode TEXT DEFAULT 'ask',
        workspace TEXT DEFAULT '',
        turn_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active',
        background INTEGER DEFAULT 0,
        needs_input INTEGER DEFAULT 0,
        last_response_preview TEXT DEFAULT '',
        goal_id TEXT DEFAULT '',
        metadata_json TEXT DEFAULT '{}'
    );
    CREATE TABLE IF NOT EXISTS turns (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        turn_number INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT DEFAULT '',
        created_at DOUBLE PRECISION NOT NULL,
        status TEXT DEFAULT 'completed',
        usage_json TEXT DEFAULT '{}',
        duration_ms INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS goals (
        goal_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        objective TEXT DEFAULT '',
        owner_host TEXT DEFAULT '',
        total_iterations INTEGER DEFAULT 0,
        total_duration_ms INTEGER DEFAULT 0,
        total_tool_calls INTEGER DEFAULT 0,
        summary TEXT DEFAULT '',
        error TEXT DEFAULT '',
        iterations_json TEXT DEFAULT '[]',
        created_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL
    );
    CREATE TABLE IF NOT EXISTS dispatch_workers (
        worker_id TEXT PRIMARY KEY,
        host TEXT DEFAULT '',
        capacity INTEGER DEFAULT 1,
        active_sessions TEXT DEFAULT '[]',
        queue_backend TEXT DEFAULT '',
        started_at DOUBLE PRECISION NOT NULL,
        last_heartbeat DOUBLE PRECISION NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions (updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status);
    CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, turn_number);
    CREATE INDEX IF NOT EXISTS idx_goals_status ON goals (status);
    CREATE INDEX IF NOT EXISTS idx_dispatch_workers_hb
        ON dispatch_workers (last_heartbeat DESC);
"""


def _get_db_path() -> Path:
    """Resolve database path defensively and construct standard sessions schema."""
    # Use standard shared DB resolution
    db_path = (
        Path.home() / ".local" / "share" / "agent-utilities" / "agent_terminal_ui.db"
    )

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize the SQLite schema defensively
    try:
        conn = sqlite3.connect(str(db_path))
        conn.executescript(_SQLITE_DDL)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error defensively initializing SQLite database: {e}")

    return db_path


def _connect_db():
    """Open a connection to the selected sessions backend (CONCEPT:OS-5.16).

    SQLite default → the per-host ``agent_terminal_ui.db`` (path resolved late
    so tests can monkeypatch :func:`_get_db_path`); ``state_db_uri`` set → the
    shared Postgres pool. Same ``?``-placeholder SQL works on both.
    """
    from agent_utilities.core.state_store import open_state_connection

    return open_state_connection("sessions", _get_db_path, _PG_DDL)


# ─────────────────────────────────────────────────────────────────────────
# Durable goal registry (CONCEPT:ORCH-1.44)
# ─────────────────────────────────────────────────────────────────────────


def _status_value(status: Any) -> str:
    return getattr(status, "value", None) or str(status)


def _persist_goal(goal_id: str) -> None:
    """Upsert the in-memory goal entry into the durable ``goals`` table."""
    entry = active_goals.get(goal_id)
    if not entry:
        return
    import json as _json

    try:
        conn = _connect_db()
        cursor = conn.cursor()
        now = time.time()
        cursor.execute(
            """
            INSERT INTO goals (goal_id, session_id, status, objective, owner_host,
                               total_iterations, total_duration_ms, total_tool_calls,
                               summary, error, iterations_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(goal_id) DO UPDATE SET
                status = excluded.status,
                total_iterations = excluded.total_iterations,
                total_duration_ms = excluded.total_duration_ms,
                total_tool_calls = excluded.total_tool_calls,
                summary = excluded.summary,
                error = excluded.error,
                iterations_json = excluded.iterations_json,
                owner_host = excluded.owner_host,
                updated_at = excluded.updated_at
            """,
            (
                goal_id,
                entry.get("session_id", ""),
                _status_value(entry.get("status", "pending")),
                str(entry.get("objective", "")),
                entry.get("owner_host", _owner_token()),
                int(entry.get("total_iterations", 0)),
                int(entry.get("total_duration_ms", 0)),
                int(entry.get("total_tool_calls", 0)),
                str(entry.get("summary", "")),
                str(entry.get("error", "")),
                _json.dumps(make_serializable(entry.get("iterations", []))),
                float(entry.get("created_at", now)),
                now,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error persisting goal {goal_id}: {e}")


def _goal_row_to_entry(row: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a ``goals`` row into the ``active_goals`` entry shape."""
    import json as _json

    try:
        iterations = _json.loads(row.get("iterations_json") or "[]")
    except (TypeError, ValueError):
        iterations = []
    return {
        "goal_id": row.get("goal_id", ""),
        "session_id": row.get("session_id", ""),
        "status": row.get("status", "pending"),
        "objective": row.get("objective", ""),
        "owner_host": row.get("owner_host", ""),
        "iterations": iterations,
        "total_iterations": row.get("total_iterations", 0),
        "total_duration_ms": row.get("total_duration_ms", 0),
        "total_tool_calls": row.get("total_tool_calls", 0),
        "summary": row.get("summary", ""),
        "error": row.get("error", ""),
    }


_rehydrated = False
_rehydrate_lock = threading.Lock()


def rehydrate_goals() -> int:
    """Surface goals stranded by a process restart (CONCEPT:ORCH-1.44).

    Scans the durable ``goals`` table for non-terminal goals that belong to
    this host (same hostname) but have no live run in this process — i.e. a
    previous pid died mid-loop. Those are marked ``orphaned`` (visible and
    explicitly resumable, never silently lost) and loaded into the in-memory
    cache. Goals owned by *other* hostnames (shared Postgres state) are left
    to their owning host's loop. Runs once per process, lazily.
    """
    global _rehydrated
    if _rehydrated:
        return 0
    with _rehydrate_lock:
        if _rehydrated:
            return 0
        _rehydrated = True
        orphaned = 0
        try:
            conn = _connect_db()
            cursor = conn.cursor()
            placeholders = ", ".join("?" for _ in _NON_TERMINAL_GOAL_STATUSES)
            cursor.execute(
                "SELECT goal_id, session_id, status, objective, owner_host, "
                "summary, total_iterations, total_duration_ms, total_tool_calls, "
                f"iterations_json FROM goals WHERE status IN ({placeholders})",  # nosec B608 — placeholders, not values
                _NON_TERMINAL_GOAL_STATUSES,
            )
            rows = [dict(r) for r in cursor.fetchall()]
            me = _owner_token()
            now = time.time()
            for row in rows:
                gid = row.get("goal_id")
                if not gid or gid in background_goal_runs:
                    continue  # live in this process
                owner = str(row.get("owner_host") or "")
                if owner == me:
                    continue
                if owner and owner.split(":", 1)[0] != _HOSTNAME:
                    # Another host owns this goal — leave it to that host.
                    continue
                summary = (
                    "Orphaned by a host restart while "
                    f"'{row.get('status')}' (owner {owner or 'unknown'}); "
                    "resume or cancel explicitly."
                )
                cursor.execute(
                    "UPDATE goals SET status = ?, summary = ?, updated_at = ? "
                    "WHERE goal_id = ?",
                    (GoalStatus.ORPHANED.value, summary, now, gid),
                )
                entry = _goal_row_to_entry(row)
                entry["status"] = GoalStatus.ORPHANED
                entry["summary"] = summary
                active_goals[gid] = entry
                orphaned += 1
                logger.warning(
                    "Rehydrated goal %s as orphaned (session=%s, was %s, owner=%s)",
                    gid,
                    row.get("session_id"),
                    row.get("status"),
                    owner,
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Goal rehydration failed: {e}")
        return orphaned


def _desired_session_action(session_id: str) -> str | None:
    """Read a pending fleet desired-state request for this session (OS-5.18).

    The supervisory plane writes ``pause_requested``/``kill_requested`` into
    the sessions store; the owning host's goal loop honors it here.
    """
    try:
        conn = _connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
    except Exception as e:  # noqa: BLE001 — reconciliation is best-effort
        logger.debug(f"desired-state probe failed for {session_id}: {e}")
        return None
    if not row:
        return None
    status = row[0]
    if status == "pause_requested":
        return "pause"
    if status == "kill_requested":
        return "kill"
    return None


# ─────────────────────────────────────────────────────────────────────────
# Starlette HTTP Route Handlers
# ─────────────────────────────────────────────────────────────────────────


async def get_all_sessions(request: Request) -> JSONResponse:
    """Retrieve durable agent sessions (newest first, paginated)."""
    try:
        params = getattr(request, "query_params", {}) or {}
        limit = max(1, min(int(params.get("limit", 500)), 2000))
        offset = max(0, int(params.get("offset", 0)))
    except (TypeError, ValueError):
        limit, offset = 500, 0
    try:
        conn = _connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = cursor.fetchall()
        res = []
        for row in rows:
            d = dict(row)
            d["background"] = bool(d.get("background", 0))
            d["needs_input"] = bool(d.get("needs_input", 0))
            res.append(d)
        conn.close()
        return JSONResponse(res)
    except Exception as e:
        logger.error(f"Error querying sessions: {e}")
        return JSONResponse([], status_code=500)


async def get_session_details(request: Request) -> JSONResponse:
    """Retrieve details and turn records for a specific session."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse(
            {"error": "session_id path parameter is required"}, status_code=400
        )
    try:
        conn = _connect_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        sess_row = cursor.fetchone()
        if not sess_row:
            conn.close()
            return JSONResponse({"error": "Session not found"}, status_code=404)

        sess_dict = dict(sess_row)
        sess_dict["background"] = bool(sess_dict.get("background", 0))
        sess_dict["needs_input"] = bool(sess_dict.get("needs_input", 0))

        cursor.execute(
            "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_number ASC",
            (session_id,),
        )
        turns = [dict(t) for t in cursor.fetchall()]
        sess_dict["turns"] = turns

        conn.close()
        return JSONResponse(sess_dict)
    except Exception as e:
        logger.error(f"Error retrieving session details: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def delete_session(request: Request) -> JSONResponse:
    """Permanently remove a session and its turns from durable persistence."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse(
            {"error": "session_id path parameter is required"}, status_code=400
        )
    try:
        conn = _connect_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
        return JSONResponse(
            {"status": "success", "message": f"Session {session_id} deleted."}
        )
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def submit_session_reply(request: Request) -> JSONResponse:
    """Submit an interactive user reply turn to a waiting agent session."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse(
            {"error": "session_id path parameter is required"}, status_code=400
        )
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    content = payload.get("content", "").strip()
    if not content:
        return JSONResponse({"error": "Reply content cannot be empty"}, status_code=400)

    try:
        conn = _connect_db()
        cursor = conn.cursor()

        cursor.execute("SELECT turn_count FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return JSONResponse({"error": "Session not found"}, status_code=404)

        turn_num = row[0]
        turn_id = str(uuid.uuid4())

        cursor.execute(
            "INSERT INTO turns (id, session_id, turn_number, role, content, created_at, status, usage_json, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                turn_id,
                session_id,
                turn_num + 1,
                "user",
                content,
                time.time(),
                "completed",
                "{}",
                0,
            ),
        )

        cursor.execute(
            "UPDATE sessions SET turn_count = turn_count + 1, needs_input = 0, updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )

        conn.commit()
        conn.close()

        # Wake up background runner if it is paused waiting for input
        if session_id in background_goal_runs:
            run = background_goal_runs[session_id]
            run["user_reply"] = content
            if run["event"]:
                run["event"].set()

        return JSONResponse(
            {"status": "success", "message": "Reply submitted successfully."}
        )
    except Exception as e:
        logger.error(f"Error submitting session reply: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def cancel_session_run(request: Request) -> JSONResponse:
    """Cancel any active background or goal execution on this session."""
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse(
            {"error": "session_id path parameter is required"}, status_code=400
        )
    cancelled = False
    for goal_id, run in list(background_goal_runs.items()):
        if run["session_id"] == session_id:
            task = run["task"]
            if not task.done():
                task.cancel()
            background_goal_runs.pop(goal_id, None)
            if goal_id in active_goals:
                active_goals[goal_id]["status"] = GoalStatus.CANCELLED
                _persist_goal(goal_id)
            cancelled = True

    try:
        conn = _connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET status = 'cancelled', updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error updating session to cancelled: {e}")

    return JSONResponse({"status": "success", "cancelled": cancelled})


def _append_session_turn(
    session_id: str, iteration_num: int, iteration: GoalIteration, output: str
) -> None:
    """Append an iteration as a console turn on the sessions store (best-effort)."""
    try:
        conn = _connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT turn_count FROM sessions WHERE id = ?", (session_id,))
        tc_row = cursor.fetchone()
        turn_num = tc_row[0] if tc_row else 0
        content_md = (
            f"### Iteration {iteration_num}\n**Action:** {iteration.action}\n"
            f"**Result:** {iteration.result}\n"
        )
        if output:
            content_md += f"\n**Validation Output:**\n```\n{output}\n```"
        cursor.execute(
            "INSERT INTO turns (id, session_id, turn_number, role, content, "
            "created_at, status, usage_json, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                session_id,
                turn_num + 1,
                "assistant",
                content_md,
                time.time(),
                "completed",
                "{}",
                iteration.duration_ms,
            ),
        )
        cursor.execute(
            "UPDATE sessions SET turn_count = turn_count + 1, "
            "last_response_preview = ?, updated_at = ? WHERE id = ?",
            (
                f"Iteration {iteration_num} complete. Success: {iteration.is_complete}",
                time.time(),
                session_id,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error appending turn to sessions store: {e}")


def _set_session_status(session_id: str, status: str, *, guard_desired: bool = False) -> None:
    """Set a session's status (best-effort). ``guard_desired`` won't clobber a
    pending supervisor pause/kill request (OS-5.18)."""
    try:
        conn = _connect_db()
        cursor = conn.cursor()
        if guard_desired:
            cursor.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ? "
                "AND status NOT IN ('pause_requested', 'kill_requested')",
                (status, time.time(), session_id),
            )
        else:
            cursor.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), session_id),
            )
        conn.commit()
        conn.close()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error setting session status: {e}")


async def run_goal_loop(
    session_id: str,
    goal_id: str,
    objective: str,
    validation_cmd: str,
    max_iterations: int,
    constraints: list[str],
):
    """Durable goal execution — a thin adapter onto the unified LoopController.

    The generalized durable run-loop (resume / per-iteration checkpoint / corrigible
    interruption, CONCEPT:OS-5.16 + SAFE-1.5) lives ONCE in
    :meth:`LoopController.run_loop` and is shared by every Loop kind
    (research/develop/skill) — durability is cross-cutting, not goal-specific. This
    adapter registers the goal as a ``develop`` Loop, wires the goal's observability
    (the goals table + the session console turns) and the fleet desired-state signal
    to the controller, and lets it own execution. No separate durable loop remains.
    """
    active_goals[goal_id] = {
        "goal_id": goal_id,
        "session_id": session_id,
        "status": GoalStatus.RUNNING,
        "objective": objective,
        "owner_host": _owner_token(),
        "created_at": time.time(),
        "iterations": [],
        "total_iterations": 0,
        "total_duration_ms": 0,
        "total_tool_calls": 0,
        "summary": "",
        "error": "",
    }
    _persist_goal(goal_id)
    _set_session_status(session_id, "running", guard_desired=True)

    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
        from agent_utilities.knowledge_graph.research.loop_controller import (
            LoopController,
        )
        from agent_utilities.knowledge_graph.research.loops import submit_loop

        engine = IntelligenceGraphEngine.get_active() or IntelligenceGraphEngine()
        # Register the goal as a first-class develop Loop so it is visible to and
        # advanced by the one controller (CONCEPT:KG-2.78).
        submit_loop(
            engine,
            objective,
            kind="develop",
            validation_cmd=validation_cmd,
            loop_id=goal_id,
            max_iterations=max_iterations,
        )
        loop = {
            "id": goal_id,
            "kind": "develop",
            "objective": objective,
            "validation_cmd": validation_cmd,
            "max_iterations": max_iterations,
            "status": "running",
        }

        def _record(iteration_num: int, outcome: dict[str, Any]) -> None:
            cmd_success = outcome.get("status") == "completed"
            output = str(outcome.get("output", ""))
            iteration = GoalIteration(
                iteration=iteration_num,
                action=(
                    f"Executing step {iteration_num} for objective: '{objective}'."
                    + (f" Validation `{validation_cmd}`." if validation_cmd else "")
                ),
                result=f"Iteration step complete. Command success: {cmd_success}",
                validation_output=output,
                is_complete=cmd_success,
                duration_ms=0,
                tool_calls=2 if validation_cmd else 1,
                timestamp=time.time(),
            )
            entry = active_goals.get(goal_id)
            if entry is not None:
                entry["iterations"].append(iteration)
                entry["total_iterations"] = iteration_num
                entry["total_tool_calls"] += iteration.tool_calls
                _persist_goal(goal_id)
            _append_session_turn(session_id, iteration_num, iteration, output)

        # The goal's validation runs in the agent workspace; the controller owns the
        # durable resume/checkpoint and honors the fleet desired-state each iteration.
        controller = LoopController(engine, codebase_root=str(DEFAULT_AGENT_DIR.resolve()))
        result = await controller.run_loop(
            loop,
            max_iterations=max_iterations,
            on_iteration=_record,
            desired_state=lambda: _desired_session_action(session_id),
            sleep_s=2.0,
        )
    except Exception as e:  # noqa: BLE001 — never let a goal crash the worker
        logger.error(f"Goal {goal_id} run_loop failed: {e}")
        result = {"status": "failed", "iterations": 0}

    rstatus = str(result.get("status", "failed"))
    try:
        final = GoalStatus(rstatus)
    except ValueError:
        final = GoalStatus.COMPLETED if rstatus == "completed" else GoalStatus.FAILED
    entry = active_goals.get(goal_id)
    if entry is not None:
        entry["status"] = final
        entry["summary"] = (
            f"Goal finished with status: {final.value}. "
            f"Iterations run: {result.get('iterations', 0)}."
        )
        _persist_goal(goal_id)
    _set_session_status(session_id, final.value)
    background_goal_runs.pop(goal_id, None)


async def create_goal(request: Request) -> JSONResponse:
    """Launch a new backgrounded autonomous goal execution loop."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    objective = body.get("objective", "")
    if not objective:
        return JSONResponse({"error": "objective is required"}, status_code=400)

    session_id = str(uuid.uuid4())
    goal_id = str(uuid.uuid4())

    spec = GoalSpec.parse_goal_input(objective)
    spec.id = goal_id
    spec.session_id = session_id

    max_iter = body.get("max_iterations")
    if max_iter:
        spec.max_iterations = int(max_iter)
    val_cmd = body.get("validation_cmd")
    if val_cmd:
        spec.validation_cmd = val_cmd
    consts = body.get("constraints")
    if consts:
        spec.constraints = consts

    # CONCEPT:ORCH-1.45 — queue-backed goal dispatch: with
    # AGENT_DISPATCH_BACKEND=queue this gateway does NOT run the goal loop
    # in-process. The full spec is persisted into the session's metadata (the
    # queue carries only references), a session-keyed envelope is published,
    # and any host's agent-dispatch-worker claims it, runs the SAME
    # ``run_goal_loop`` body, and writes turns/status back into this store.
    from agent_utilities.orchestration.agent_dispatch import dispatch_queue_enabled

    try:
        queue_mode = dispatch_queue_enabled()
    except Exception as e:  # noqa: BLE001 — a bad flag must not kill goal intake
        logger.error(f"agent_dispatch_backend resolution failed: {e}")
        queue_mode = False

    import json as _json

    # Stamp the originating identity into session metadata so the audit trail
    # and the tenant-scoped fleet plane can attribute this goal to a tenant/actor
    # (CONCEPT:OS-5.14 + OS-5.11). Best-effort: no actor in scope → empty.
    meta: dict = _identity_metadata()
    if queue_mode:
        meta["goal_spec"] = {
            "objective": spec.objective,
            "end_state": spec.end_state,
            "validation_cmd": spec.validation_cmd,
            "max_iterations": spec.max_iterations,
            "constraints": list(spec.constraints or []),
        }
    session_metadata = _json.dumps(meta)

    try:
        conn = _connect_db()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at, model, mode, workspace, turn_count, status, background, needs_input, last_response_preview, goal_id, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                f"Goal: {spec.objective}",
                time.time(),
                time.time(),
                "gpt-4o",
                "ask",
                str(DEFAULT_AGENT_DIR),
                1,
                "queued" if queue_mode else "running",
                1,
                0,
                "Goal queued for dispatch..."
                if queue_mode
                else "Goal loop initialized...",
                goal_id,
                session_metadata,
            ),
        )

        cursor.execute(
            "INSERT INTO turns (id, session_id, turn_number, role, content, created_at, status, usage_json, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                session_id,
                1,
                "user",
                f"/goal {spec.objective}"
                + (f" until {spec.end_state}" if spec.end_state else ""),
                time.time(),
                "completed",
                "{}",
                0,
            ),
        )

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error initializing goal session: {e}")
        return JSONResponse(
            {"error": f"Database initialization failed: {e}"}, status_code=500
        )

    if queue_mode:
        # Durable goal row first (status=pending, unowned), THEN the envelope:
        # a worker that wins the race always finds the record it must claim.
        active_goals[goal_id] = {
            "goal_id": goal_id,
            "session_id": session_id,
            "status": GoalStatus.PENDING,
            "objective": spec.objective,
            "owner_host": "",
            "created_at": time.time(),
            "iterations": [],
            "total_iterations": 0,
            "total_duration_ms": 0,
            "total_tool_calls": 0,
            "summary": "Goal queued for dispatch...",
            "error": "",
        }
        _persist_goal(goal_id)

        from agent_utilities.orchestration.agent_dispatch import (
            KIND_GOAL_LOOP,
            AgentTurnEnvelope,
            enqueue_agent_turn,
        )

        try:
            handle = enqueue_agent_turn(
                AgentTurnEnvelope(
                    session_id=session_id,
                    kind=KIND_GOAL_LOOP,
                    payload_ref=goal_id,
                )
            )
        except Exception as e:  # noqa: BLE001 — surface enqueue failure loudly
            logger.error(f"Goal {goal_id} enqueue failed: {e}")
            active_goals[goal_id]["status"] = GoalStatus.FAILED
            active_goals[goal_id]["error"] = f"dispatch enqueue failed: {e}"
            _persist_goal(goal_id)
            return JSONResponse(
                {"error": f"dispatch enqueue failed: {e}"}, status_code=503
            )

        return JSONResponse(
            {
                "status": "success",
                "goal_id": goal_id,
                "session_id": session_id,
                "objective": spec.objective,
                "validation_cmd": spec.validation_cmd,
                "dispatch": handle,
            }
        )

    task = asyncio.create_task(
        run_goal_loop(
            session_id=session_id,
            goal_id=goal_id,
            objective=spec.objective,
            validation_cmd=spec.validation_cmd,
            max_iterations=spec.max_iterations,
            constraints=spec.constraints,
        )
    )

    background_goal_runs[goal_id] = {
        "task": task,
        "session_id": session_id,
        "user_reply": None,
        "event": asyncio.Event(),
    }

    # Populate in active_goals memory map immediately for lists
    active_goals[goal_id] = {
        "goal_id": goal_id,
        "session_id": session_id,
        "status": GoalStatus.RUNNING,
        "objective": spec.objective,
        "owner_host": _owner_token(),
        "created_at": time.time(),
        "iterations": [],
        "total_iterations": 0,
        "total_duration_ms": 0,
        "total_tool_calls": 0,
        "summary": "Goal loop initialized...",
        "error": "",
    }
    _persist_goal(goal_id)

    return JSONResponse(
        {
            "status": "success",
            "goal_id": goal_id,
            "session_id": session_id,
            "objective": spec.objective,
            "validation_cmd": spec.validation_cmd,
        }
    )


def make_serializable(o: Any) -> Any:
    """Recursively convert Pydantic models and Enums into JSON-serializable types."""
    if hasattr(o, "model_dump"):
        return o.model_dump()
    elif hasattr(o, "dict"):
        return o.dict()
    elif hasattr(o, "value"):  # Enums
        return o.value
    elif isinstance(o, list):
        return [make_serializable(item) for item in o]
    elif isinstance(o, dict):
        return {k: make_serializable(v) for k, v in o.items()}
    return o


async def list_goals(request: Request) -> JSONResponse:
    """Retrieve active + durable goals (in-memory cache overlays the store)."""
    rehydrate_goals()
    merged: dict[str, Any] = {}
    try:
        conn = _connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM goals ORDER BY updated_at DESC LIMIT 200")
        for row in cursor.fetchall():
            entry = _goal_row_to_entry(dict(row))
            merged[entry["goal_id"]] = entry
        conn.close()
    except Exception as e:  # noqa: BLE001 — degrade to the in-memory view
        logger.debug(f"durable goal list unavailable: {e}")
    for gid, entry in active_goals.items():
        merged[gid] = make_serializable(entry)
    return JSONResponse(list(merged.values()))


async def get_goal_iterations(request: Request) -> JSONResponse:
    """Retrieve live-updating iteration steps for a specific goal run."""
    goal_id = request.path_params.get("goal_id")
    if not goal_id:
        return JSONResponse({"error": "Goal run not found"}, status_code=404)
    rehydrate_goals()
    if goal_id in active_goals:
        return JSONResponse(make_serializable(active_goals[goal_id]))
    # Fall back to the durable registry (goal from a previous run / other host).
    try:
        conn = _connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM goals WHERE goal_id = ?", (goal_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return JSONResponse(_goal_row_to_entry(dict(row)))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"durable goal lookup failed: {e}")
    return JSONResponse({"error": "Goal run not found"}, status_code=404)


async def cancel_goal(request: Request) -> JSONResponse:
    """Cancel an active autonomous goal loop."""
    goal_id = request.path_params.get("goal_id")
    if not goal_id or goal_id not in background_goal_runs:
        return JSONResponse({"error": "Active goal run not found"}, status_code=404)

    run = background_goal_runs[goal_id]
    task = run["task"]
    if not task.done():
        task.cancel()

    session_id = run["session_id"]
    background_goal_runs.pop(goal_id, None)

    if goal_id in active_goals:
        active_goals[goal_id]["status"] = GoalStatus.CANCELLED
        active_goals[goal_id]["summary"] = "Goal cancelled by user."
        _persist_goal(goal_id)

    try:
        conn = _connect_db()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET status = 'cancelled', updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error cancelling goal session: {e}")

    return JSONResponse(
        {"status": "success", "message": "Goal cancelled successfully."}
    )
