#!/usr/bin/python
from __future__ import annotations

"""Durable Session & Autonomous Goal persistence.

CONCEPT:AU-ORCH.session.durable-session-autonomous-goal — Durable session and autonomous goal persistence with iterative background goal loops
CONCEPT:AU-ORCH.session.durable-goal-registry-goals — Durable goal registry — goals persist across restarts and expired leases remain visible instead of silently vanishing

This module houses the schema initialization, memory maps, background runner thread,
and Starlette REST handlers for durable agent sessions and iterative goals.

State backends (CONCEPT:AU-OS.state.unified-durable-state-externalization): by default sessions/turns/goals live in the
per-host SQLite file; with ``state_db_uri`` set they live on the shared
Postgres state store, so the gateway is stateless and any host can see the
whole fleet's sessions.

Goal durability (CONCEPT:AU-ORCH.session.durable-goal-registry-goals):
``active_goals``/``background_goal_runs`` are an in-memory UX cache over each
goal's engine-native WorkItem. The Concept node retains definition and
observability only. On restart, non-terminal work retains its exact WorkItem
state without a second lifecycle projection.
"""

import logging
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
from agent_utilities.models.goal import GoalIteration, GoalSpec

logger = logging.getLogger(__name__)

# Resolved paths
DEFAULT_AGENT_DIR = Path(setting("WORKSPACE_PATH", "workspace"))
DEFAULT_AGENT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory UX cache; the deterministic goal-loop WorkItem is authoritative.
active_goals: dict[str, dict[str, Any]] = {}
background_goal_runs: dict[str, dict[str, Any]] = {}

# Goal statuses that are still "in flight" (rehydration targets on restart).
_NON_TERMINAL_GOAL_STATUSES = frozenset(
    {
        "submitted",
        "ready",
        "leased",
        "running",
    }
)

_OWNER_TOKEN = f"goal-owner:{uuid.uuid4().hex}"


def _owner_token() -> str:
    """Return an opaque process-lifetime owner reference.

    Hostnames, user names, and filesystem locations are never persisted.
    WorkItem lease expiry, not host-name comparison, detects stranded work.
    """
    return _OWNER_TOKEN


def _ambient_session_tenant() -> str:
    """The tenant a durable session/goal row is stamped with (AU-P0-5).

    Delegates to the state-store's one ambient-tenant resolver (GraphSession
    first, then the ambient actor) so the sessions store agrees with every
    other Postgres-backed store sharing this pool. ``""`` (unrestricted /
    commons) when nothing is scoped — unchanged from today's behaviour.
    """
    try:
        from agent_utilities.core.state_store import ambient_state_tenant

        return ambient_state_tenant()
    except Exception:  # noqa: BLE001 — tenant stamping must never break intake
        return ""


def _sessions_tenant_predicate() -> tuple[str, tuple]:
    """A ``sessions``-row WHERE fragment + params scoping to the ambient tenant.

    Pushes the tenant check DOWN into the query (AU-P0-5) instead of fetching
    every row and post-filtering in Python: an unscoped/system caller (ambient
    tenant ``""``) gets the fragment ``""`` (unrestricted — today's exact
    behaviour); a tenant-scoped caller gets rows for its own tenant PLUS
    "commons" rows (``tenant_id`` unset/empty), mirroring the RLS convention
    ``PostgreSQLBackend``/the state-store GUC already use.
    """
    tenant = _ambient_session_tenant()
    if not tenant:
        return "", ()
    return " AND (tenant_id = ? OR tenant_id IS NULL OR tenant_id = '')", (tenant,)


def _identity_metadata() -> dict:
    """Ambient ``{tenant_id, actor_id}`` for stamping into session metadata.

    The tenant-scoped fleet plane (CONCEPT:AU-OS.safety.ontological-guardrail) aggregates sessions by
    ``metadata.tenant``; stamping the server-minted identity here is what makes
    "show me org X's sessions / which agents client Y spawned" a tenant-scoped
    query (CONCEPT:AU-OS.observability.run-wide-correlation-id + OS-5.14). Best-effort: no actor → ``{}``.
    """
    try:
        from agent_utilities.security.brain_context import current_actor

        actor = current_actor()
        out: dict = {}
        if actor.tenant_id:
            out["tenant"] = actor.tenant_id
            out["tenant_id"] = actor.tenant_id
        if actor.actor_id and actor.actor_id != "system":
            from agent_utilities.messaging.bus_privacy import bus_reference

            out["actor_id"] = bus_reference(
                "actor", actor.actor_id, tenant=actor.tenant_id or ""
            )
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
        metadata_json TEXT DEFAULT '{}',
        tenant_id TEXT DEFAULT ''
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
# NOTE: goal state is NOT a SQLite table — it lives on the KG Loop node (a develop
# ``Concept``, CONCEPT:AU-KG.research.these-properties-carry). The ``goals`` table was collapsed onto the one Loop
# model so there is a single durable source of truth; see ``_persist_goal`` /
# ``_list_goal_entries`` below.

# Same logical schema on Postgres (CONCEPT:AU-OS.state.unified-durable-state-externalization). REAL epoch timestamps
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
        metadata_json TEXT DEFAULT '{}',
        tenant_id TEXT DEFAULT ''
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
    CREATE TABLE IF NOT EXISTS dispatch_workers (
        worker_id TEXT PRIMARY KEY,
        host TEXT DEFAULT '',
        capacity INTEGER DEFAULT 1,
        active_sessions TEXT DEFAULT '[]',
        queue_backend TEXT DEFAULT '',
        started_at DOUBLE PRECISION NOT NULL,
        last_heartbeat DOUBLE PRECISION NOT NULL
    );
    -- AU-P0-5: idempotent upgrade path for a store created before tenant_id
    -- existed on ``sessions`` — Postgres supports IF NOT EXISTS on ADD COLUMN
    -- so this is a no-op once the column is there (including on brand-new
    -- deployments where CREATE TABLE above already declared it).
    ALTER TABLE sessions ADD COLUMN IF NOT EXISTS tenant_id TEXT DEFAULT '';
    CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions (updated_at DESC);
    CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status);
    CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions (tenant_id);
    CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, turn_number);
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
        # AU-P0-5: upgrade path for a store created before tenant_id existed on
        # ``sessions`` — SQLite's ADD COLUMN has no IF NOT EXISTS, so a
        # duplicate-column error (already-upgraded store) is simply swallowed.
        try:
            conn.execute("ALTER TABLE sessions ADD COLUMN tenant_id TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error defensively initializing SQLite database: {e}")

    return db_path


def _connect_db():
    """Open a connection to the selected sessions backend (CONCEPT:AU-OS.state.unified-durable-state-externalization).

    SQLite default → the per-host ``agent_terminal_ui.db`` (path resolved late
    so tests can monkeypatch :func:`_get_db_path`); ``state_db_uri`` set → the
    shared Postgres pool. Same ``?``-placeholder SQL works on both.
    """
    from agent_utilities.core.state_store import open_state_connection

    return open_state_connection("sessions", _get_db_path, _PG_DDL)


# ─────────────────────────────────────────────────────────────────────────
# Durable goal registry (CONCEPT:AU-ORCH.session.durable-goal-registry-goals)
# ─────────────────────────────────────────────────────────────────────────


# Goal definitions/observability live on a develop ``Concept`` projection. The
# sole writable lifecycle authority is its deterministic WorkItem.
_GOAL_RETURN = (
    "c.id AS goal_id, c.session_id AS session_id, "
    "c.objective AS objective, c.owner_host AS owner_host, c.summary AS summary, "
    "c.error AS error, c.total_iterations AS total_iterations, "
    "c.total_duration_ms AS total_duration_ms, c.total_tool_calls AS total_tool_calls, "
    "c.iterations_json AS iterations_json"
)


def _goal_engine() -> Any:
    """The active KG engine (durable source of truth for goals), or ``None``.

    Best-effort: never constructs one — when no engine is active (e.g. a bare REST
    process) the in-memory ``active_goals`` cache is the live view and goal execution
    (the ``run_goal_loop`` adapter) persists to the KG once its engine is up.
    """
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

        return IntelligenceGraphEngine.get_active()
    except Exception:  # noqa: BLE001
        return None


def _persist_goal(goal_id: str) -> None:
    """Persist the goal entry onto its KG Loop node (CONCEPT:AU-KG.research.these-properties-carry).

    The goal is a develop Loop whose WorkItem is authoritative. This writes
    lifecycle-neutral definition and observability fields only.
    """
    entry = active_goals.get(goal_id)
    if not entry:
        return
    engine = _goal_engine()
    if engine is None:
        return
    import json as _json

    from agent_utilities.messaging.bus_privacy import bus_reference
    from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

    now = time.time()
    privacy = PersistencePrivacyGuard()
    objective, _ = privacy.sanitize_text(str(entry.get("objective", "")))
    summary, _ = privacy.sanitize_text(str(entry.get("summary", "")))
    error, _ = privacy.sanitize_text(str(entry.get("error", "")))
    iterations, _ = privacy.sanitize(make_serializable(entry.get("iterations", [])))
    props = {
        "name": objective or goal_id,
        "objective": objective,
        "loop_kind": "develop",
        "session_id": entry.get("session_id", ""),
        "owner_host": entry.get("owner_host", _owner_token()),
        "summary": summary,
        "error": error,
        "total_iterations": int(entry.get("total_iterations", 0)),
        "total_duration_ms": int(entry.get("total_duration_ms", 0)),
        "total_tool_calls": int(entry.get("total_tool_calls", 0)),
        "iterations_json": _json.dumps(iterations),
        "created_at": float(entry.get("created_at", now)),
        "updated_at": now,
    }
    # CONCEPT:AU-ORCH.session.escalate-breached-goals — goals-as-contracts: carry the SLA + escalation target so
    # the goal_sla maintenance tick can enforce a deadline. Sourced from the goal's
    # constraints (e.g. constraints={"sla_seconds": 3600, "escalate_to": "user"}).
    _constraints = entry.get("constraints") or {}
    _sla = entry.get("sla_seconds") or _constraints.get("sla_seconds")
    if _sla:
        try:
            props["sla_seconds"] = float(_sla)
        except (TypeError, ValueError):
            pass
    _esc = entry.get("escalate_to") or _constraints.get("escalate_to")
    if _esc:
        props["escalate_to"] = bus_reference(
            "goal_escalation", str(_esc), tenant=_ambient_session_tenant()
        )
    try:
        engine.add_node(goal_id, "Concept", properties=props)
    except Exception as e:  # noqa: BLE001 — best-effort persist
        logger.error(f"Error persisting goal {goal_id} to KG: {e}")


def _goal_row_to_entry(row: dict[str, Any]) -> dict[str, Any]:
    """Deserialize a KG goal-node row into the ``active_goals`` entry shape."""
    import json as _json

    try:
        iterations = _json.loads(row.get("iterations_json") or "[]")
    except (TypeError, ValueError):
        iterations = []
    return {
        "goal_id": row.get("goal_id", ""),
        "session_id": row.get("session_id", ""),
        "status": "submitted",
        "objective": row.get("objective", ""),
        "owner_host": row.get("owner_host", ""),
        "iterations": iterations,
        "total_iterations": row.get("total_iterations", 0),
        "total_duration_ms": row.get("total_duration_ms", 0),
        "total_tool_calls": row.get("total_tool_calls", 0),
        "summary": row.get("summary", ""),
        "error": row.get("error", ""),
    }


def _goal_work_item_status(engine: Any, goal_id: str) -> str | None:
    """Return the goal's exact authoritative WorkItem state."""
    from agent_utilities.orchestration import work_item as _wi

    item = _wi.get_work_item(engine, _wi.loop_work_item_id(goal_id))
    if item is None:
        return None
    raw_status = str(item.get("status") or "")
    if raw_status in _wi.WORK_ITEM_STATES:
        return raw_status
    logger.warning("Goal %s has an invalid WorkItem status", goal_id)
    return None


def _load_goal_entry(engine: Any, goal_id: str) -> dict[str, Any] | None:
    """Read one goal's record back from its KG Loop node, or ``None``."""
    try:
        rows = engine.query_cypher(
            f"MATCH (c:Concept) WHERE c.id = $id RETURN {_GOAL_RETURN}",
            {"id": goal_id},
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"goal KG lookup failed: {e}")
        return None
    for r in rows or []:
        if isinstance(r, dict) and r.get("goal_id"):
            entry = _goal_row_to_entry(r)
            entry["status"] = _goal_work_item_status(engine, goal_id) or "submitted"
            return entry
    return None


def _list_goal_entries(engine: Any, *, limit: int = 200) -> list[dict[str, Any]]:
    """List goal records from the KG — develop Loops that carry a ``session_id``."""
    try:
        rows = engine.query_cypher(
            f"MATCH (c:Concept) WHERE c.loop_kind = 'develop' RETURN {_GOAL_RETURN} "
            "LIMIT $limit",
            {"limit": int(limit)},
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"goal KG list failed: {e}")
        return []
    entries: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict) or not row.get("session_id"):
            continue
        entry = _goal_row_to_entry(row)
        status = _goal_work_item_status(engine, str(entry.get("goal_id") or ""))
        if status is None:
            # A definition with no WorkItem has no lifecycle authority.
            continue
        entry["status"] = status
        entries.append(entry)
    return entries


_rehydrated = False
_rehydrate_lock = threading.Lock()


def rehydrate_goals() -> int:
    """Surface goals stranded by a process restart (CONCEPT:AU-ORCH.session.durable-goal-registry-goals).

    Scans authoritative goal WorkItems for non-terminal goals with no live run
    in this process. A current unexpired WorkItem lease is left untouched;
    otherwise the definition is surfaced with its exact WorkItem state and an
    expired-lease summary. Runs once per process, lazily.
    """
    global _rehydrated
    if _rehydrated:
        return 0
    with _rehydrate_lock:
        if _rehydrated:
            return 0
        _rehydrated = True
        stranded = 0
        engine = _goal_engine()
        if engine is None:
            return 0
        try:
            for entry in _list_goal_entries(engine):
                gid = entry.get("goal_id")
                status = str(entry.get("status") or "")
                if not gid or gid in background_goal_runs:
                    continue  # live in this process
                if status not in _NON_TERMINAL_GOAL_STATUSES:
                    continue  # already terminal — nothing to rehydrate
                from agent_utilities.orchestration.work_item import (
                    work_item_view_of_loop,
                )

                work_item = work_item_view_of_loop(engine, str(gid)) or {}
                lease_expires_at = float(work_item.get("lease_expires_at") or 0.0)
                if lease_expires_at > time.time():
                    continue
                summary = (
                    f"Execution lease expired while '{status}'; resume or cancel "
                    "explicitly."
                )
                entry["summary"] = summary
                active_goals[gid] = entry
                _persist_goal(gid)  # refresh definition/observability projection
                stranded += 1
                logger.warning(
                    "Rehydrated goal %s with expired lease (session=%s, status=%s)",
                    gid,
                    entry.get("session_id"),
                    status,
                )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Goal rehydration failed: {e}")
        return stranded


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
    """Retrieve durable agent sessions (newest first, paginated).

    Tenant-scoped at the query level (AU-P0-5, :func:`_sessions_tenant_predicate`)
    rather than fetched-then-filtered: an ambient-tenant caller only ever gets
    its own + commons rows back from the SQL engine itself.
    """
    from agent_utilities.knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:read")
    try:
        params = getattr(request, "query_params", {}) or {}
        limit = max(1, min(int(params.get("limit", 500)), 2000))
        offset = max(0, int(params.get("offset", 0)))
    except (TypeError, ValueError):
        limit, offset = 500, 0
    try:
        conn = _connect_db()
        cursor = conn.cursor()
        tenant_clause, tenant_params = _sessions_tenant_predicate()
        cursor.execute(
            f"SELECT * FROM sessions WHERE 1=1{tenant_clause} "
            "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (*tenant_params, limit, offset),
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
    from agent_utilities.knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:read")
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse(
            {"error": "session_id path parameter is required"}, status_code=400
        )
    try:
        conn = _connect_db()
        cursor = conn.cursor()

        tenant_clause, tenant_params = _sessions_tenant_predicate()
        cursor.execute(
            f"SELECT * FROM sessions WHERE id = ?{tenant_clause}",
            (session_id, *tenant_params),
        )
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
    from agent_utilities.knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:write")
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse(
            {"error": "session_id path parameter is required"}, status_code=400
        )
    try:
        conn = _connect_db()
        cursor = conn.cursor()
        tenant_clause, tenant_params = _sessions_tenant_predicate()
        cursor.execute(
            f"SELECT id FROM sessions WHERE id = ?{tenant_clause}",
            (session_id, *tenant_params),
        )
        if not cursor.fetchone():
            conn.close()
            return JSONResponse({"error": "Session not found"}, status_code=404)
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
    from agent_utilities.knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:write")
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

        tenant_clause, tenant_params = _sessions_tenant_predicate()
        cursor.execute(
            f"SELECT turn_count FROM sessions WHERE id = ?{tenant_clause}",
            (session_id, *tenant_params),
        )
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
    from agent_utilities.knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:write")
    session_id = request.path_params.get("session_id")
    if not session_id:
        return JSONResponse(
            {"error": "session_id path parameter is required"}, status_code=400
        )
    try:
        conn = _connect_db()
        cursor = conn.cursor()
        tenant_clause, tenant_params = _sessions_tenant_predicate()
        cursor.execute(
            f"SELECT id FROM sessions WHERE id = ?{tenant_clause}",
            (session_id, *tenant_params),
        )
        visible = cursor.fetchone()
        conn.close()
    except Exception as e:  # noqa: BLE001
        logger.error("Error authorizing session cancellation: %s", e)
        return JSONResponse({"error": "Session lookup failed"}, status_code=500)
    if not visible:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    cancelled = False
    for goal_id, run in list(background_goal_runs.items()):
        if run["session_id"] == session_id:
            task = run["task"]
            if not task.done():
                task.cancel()
            background_goal_runs.pop(goal_id, None)
            if goal_id in active_goals:
                active_goals[goal_id]["status"] = "cancelled"
                engine = _goal_engine()
                if engine is not None:
                    from agent_utilities.knowledge_graph.research.loops import (
                        mark_loop_status,
                    )

                    mark_loop_status(engine, goal_id, "cancelled", source="user")
                _persist_goal(goal_id)
            cancelled = True

    try:
        conn = _connect_db()
        cursor = conn.cursor()
        tenant_clause, tenant_params = _sessions_tenant_predicate()
        cursor.execute(
            "UPDATE sessions SET status = 'cancelled', updated_at = ? "
            f"WHERE id = ?{tenant_clause}",
            (time.time(), session_id, *tenant_params),
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


def _set_session_status(
    session_id: str, status: str, *, guard_desired: bool = False
) -> None:
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
    interruption, CONCEPT:AU-OS.state.unified-durable-state-externalization + SAFE-1.5) lives ONCE in
    :meth:`LoopController.run_loop` and is shared by every Loop kind
    (research/develop/skill) — durability is cross-cutting, not goal-specific. This
    adapter registers the goal as a ``develop`` Loop, wires the goal's observability
    (the goals table + the session console turns) and the fleet desired-state signal
    to the controller, and lets it own execution. No separate durable loop remains.
    """
    active_goals[goal_id] = {
        "goal_id": goal_id,
        "session_id": session_id,
        "status": "submitted",
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
    _set_session_status(session_id, "running", guard_desired=True)

    engine = None
    try:
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
        from agent_utilities.knowledge_graph.research.loop_controller import (
            LoopController,
        )
        from agent_utilities.knowledge_graph.research.loops import submit_loop

        engine = _goal_engine() or IntelligenceGraphEngine.get_or_create()
        # Register the goal as a first-class develop Loop so it is visible to and
        # advanced by the one controller (CONCEPT:AU-KG.research.these-properties-carry).
        submit_loop(
            engine,
            objective,
            kind="develop",
            validation_cmd=validation_cmd,
            loop_id=goal_id,
            max_iterations=max_iterations,
        )
        _persist_goal(goal_id)
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
        # native WorkItem resume/checkpoint and honors desired state each iteration.
        controller = LoopController(
            engine, codebase_root=str(DEFAULT_AGENT_DIR.resolve())
        )
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

        if engine is not None:
            try:
                from agent_utilities.knowledge_graph.research.loops import (
                    claim_loop,
                    mark_loop_status,
                )
                from agent_utilities.orchestration import work_item as _wi

                item = _wi.get_work_item(engine, _wi.loop_work_item_id(goal_id))
                if (item or {}).get("status") not in _wi.TERMINAL_WORK_ITEM_STATUSES:
                    if _wi.current_work_item_claim(
                        engine, _wi.loop_work_item_id(goal_id)
                    ) or claim_loop(engine, goal_id):
                        mark_loop_status(
                            engine,
                            goal_id,
                            "failed",
                            output="goal loop execution failed",
                            source="goal_runner",
                        )
            except Exception as settle_error:  # noqa: BLE001
                logger.error(
                    "Goal %s WorkItem failure settlement failed: %s",
                    goal_id,
                    settle_error,
                )

    rstatus = str(result.get("status", "failed"))
    if result.get("skipped"):
        # Another fenced owner won the WorkItem. Never project a terminal result
        # from this non-owner; report the actual state and leave the session live.
        actual = _goal_work_item_status(engine, goal_id) if engine else None
        entry = active_goals.get(goal_id)
        if entry is not None:
            entry["status"] = actual or "running"
            entry["summary"] = "Goal execution is owned by another worker."
            _persist_goal(goal_id)
        _set_session_status(session_id, actual or "running", guard_desired=True)
        background_goal_runs.pop(goal_id, None)
        return
    result_statuses = {
        "completed": "succeeded",
        "succeeded": "succeeded",
        "failed": "failed",
        "cancelled": "cancelled",
        "paused": "ready",
        "running": "running",
    }
    final = (
        (_goal_work_item_status(engine, goal_id) if engine else None)
        or result_statuses.get(rstatus)
        or "failed"
    )
    entry = active_goals.get(goal_id)
    if entry is not None:
        entry["status"] = final
        entry["summary"] = (
            f"Goal finished with status: {final}. "
            f"Iterations run: {result.get('iterations', 0)}."
        )
        _persist_goal(goal_id)
    _set_session_status(session_id, final)
    background_goal_runs.pop(goal_id, None)


async def create_goal(request: Request) -> JSONResponse:
    """Launch a new backgrounded autonomous goal execution loop."""
    from agent_utilities.knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:write")
    try:
        body = await request.json()
    except Exception:
        body = {}

    objective = body.get("objective", "")
    if not objective:
        return JSONResponse({"error": "objective is required"}, status_code=400)

    from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

    privacy = PersistencePrivacyGuard()
    objective, _objective_privacy = privacy.sanitize_text(str(objective))

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
        spec.validation_cmd, _validation_privacy = privacy.sanitize_text(str(val_cmd))
    consts = body.get("constraints")
    if consts:
        clean_constraints, _constraints_privacy = privacy.sanitize(consts)
        spec.constraints = (
            list(clean_constraints) if isinstance(clean_constraints, list) else []
        )

    # CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch — the gateway never runs a goal loop
    # in-process. The full spec is persisted into session metadata (the
    # queue carries only references), a session-keyed envelope is published,
    # and any host's agent-dispatch-worker claims it, runs the SAME
    # ``run_goal_loop`` body, and writes turns/status back into this store.
    import json as _json

    # Stamp the originating identity into session metadata so the audit trail
    # and the tenant-scoped fleet plane can attribute this goal to a tenant/actor
    # (CONCEPT:AU-OS.identity.authenticated-identity-enforcement + OS-5.11). Best-effort: no actor in scope → empty.
    meta: dict = _identity_metadata()
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
            "INSERT INTO sessions (id, title, created_at, updated_at, model, mode, workspace, turn_count, status, background, needs_input, last_response_preview, goal_id, metadata_json, tenant_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                f"Goal: {spec.objective}",
                time.time(),
                time.time(),
                "gpt-4o",
                "ask",
                "",
                1,
                "queued",
                1,
                0,
                "Goal queued for dispatch...",
                goal_id,
                session_metadata,
                # AU-P0-5: a queryable tenant column (not just metadata_json) so
                # the row can be RLS-enforced on Postgres and predicate-filtered
                # on any backend — see _sessions_tenant_predicate().
                _ambient_session_tenant(),
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
        logger.error("Error initializing goal session (%s)", type(e).__name__)
        return JSONResponse(
            {"error": "Database initialization failed"}, status_code=500
        )

    # Persist the definition before the envelope. A worker that wins the queue
    # race can always rehydrate the referenced goal.
    active_goals[goal_id] = {
        "goal_id": goal_id,
        "session_id": session_id,
        "status": "submitted",
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
        logger.error("Goal %s enqueue failed: %s", goal_id, e)
        active_goals[goal_id]["status"] = "failed"
        active_goals[goal_id]["error"] = f"dispatch enqueue failed: {e}"
        _persist_goal(goal_id)
        return JSONResponse({"error": "dispatch enqueue failed"}, status_code=503)

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


def make_serializable(o: Any) -> Any:
    """Recursively convert Pydantic models and Enums into JSON-serializable types."""
    if hasattr(o, "model_dump"):
        return o.model_dump()
    elif hasattr(o, "value"):  # Enums
        return o.value
    elif isinstance(o, list):
        return [make_serializable(item) for item in o]
    elif isinstance(o, dict):
        return {k: make_serializable(v) for k, v in o.items()}
    return o


async def list_goals(request: Request) -> JSONResponse:
    """Retrieve active + durable goals (in-memory cache overlays the store)."""
    from agent_utilities.knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:read")
    rehydrate_goals()
    merged: dict[str, Any] = {}
    engine = _goal_engine()
    if engine is not None:
        for entry in _list_goal_entries(engine):
            merged[entry["goal_id"]] = entry
    for gid, entry in active_goals.items():
        merged[gid] = make_serializable(entry)
    return JSONResponse(list(merged.values()))


async def get_goal_iterations(request: Request) -> JSONResponse:
    """Retrieve live-updating iteration steps for a specific goal run."""
    from agent_utilities.knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:read")
    goal_id = request.path_params.get("goal_id")
    if not goal_id:
        return JSONResponse({"error": "Goal run not found"}, status_code=404)
    rehydrate_goals()
    if goal_id in active_goals:
        return JSONResponse(make_serializable(active_goals[goal_id]))
    # Fall back to the KG Loop node (goal from a previous run / other host).
    engine = _goal_engine()
    if engine is not None:
        entry = _load_goal_entry(engine, goal_id)
        if entry is not None:
            return JSONResponse(entry)
    return JSONResponse({"error": "Goal run not found"}, status_code=404)


async def cancel_goal(request: Request) -> JSONResponse:
    """Cancel an active autonomous goal loop."""
    from agent_utilities.knowledge_graph.core.session import resolve_session

    resolve_session(required_scope="kg:write")
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
        active_goals[goal_id]["status"] = "cancelled"
        active_goals[goal_id]["summary"] = "Goal cancelled by user."
        engine = _goal_engine()
        if engine is not None:
            from agent_utilities.knowledge_graph.research.loops import mark_loop_status

            mark_loop_status(engine, goal_id, "cancelled", source="user")
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
