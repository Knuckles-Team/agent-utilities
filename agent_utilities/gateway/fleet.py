"""Native swarm supervisory plane (CONCEPT:OS-5.10).

CONCEPT:OS-5.18 — Fleet supervisory plane at scale — SQL aggregation, paginated and filtered session queries, and desired-state pause and kill reconciliation across hosts

A single pane of glass over the running fleet, exposed through the API gateway —
no separate supervisor service. Everything here surfaces state the ecosystem
*already* maintains:

* **topology / health** — the durable session registry and goal registry in
  :mod:`agent_utilities.core.sessions` (per-host SQLite by default, the shared
  Postgres state store when ``state_db_uri`` is set — CONCEPT:OS-5.16).
  Aggregations run in SQL (``COUNT``/``GROUP BY``) and listings are paginated
  + status-filterable, so the handlers stay O(page), not O(fleet)
  (CONCEPT:OS-5.18).
* **pause / kill** — emergency containment as *desired-state writes*: targets
  local to this gateway are cancelled in-process (fast path) and finalized;
  sessions owned by another host get ``pause_requested``/``kill_requested``,
  which the owning host's goal loop reconciles on its next tick
  (CONCEPT:OS-5.18).
* **approvals** — pending mutation/risk approvals stored as ``Task`` graph nodes,
  read and granted through the parity-covered ``graph_query`` / ``graph_orchestrate``
  tools.

These handlers are plain Starlette callables mounted by the gateway; the
``agent-webui`` Fleet Dashboard consumes them.
"""

from __future__ import annotations

import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from agent_utilities.core import sessions as _sessions

# Statuses that count as "unhealthy" when computing per-domain error rates.
_ERROR_STATUSES = ("failed", "error", "cancelled")
_ACTIVE_STATUSES = ("active", "running")

_MAX_PAGE = 1000


def _domain_sql(dialect: str) -> str:
    """SQL expression deriving a session's enterprise domain from its metadata.

    Mirrors the old per-row Python ``_domain_of`` (domain → team → tenant →
    'default') so aggregation can run in the database on both backends
    (CONCEPT:OS-5.18). Malformed/empty metadata degrades to 'default'.
    """
    if dialect == "postgres":

        def j(key: str) -> str:
            return f"NULLIF(metadata_json::jsonb ->> '{key}', '')"

        valid = "pg_input_is_valid(metadata_json, 'jsonb')"
    else:

        def j(key: str) -> str:
            return f"NULLIF(json_extract(metadata_json, '$.{key}'), '')"

        valid = "json_valid(metadata_json)"
    coalesce = f"COALESCE({j('domain')}, {j('team')}, {j('tenant')}, 'default')"
    return f"CASE WHEN {valid} THEN {coalesce} ELSE 'default' END"


def _page_params(
    request: Request, default_limit: int = 200
) -> tuple[int, int, str | None]:
    """Parse ``limit``/``offset``/``status`` query params with sane bounds."""
    params = getattr(request, "query_params", {}) or {}
    try:
        limit = max(1, min(int(params.get("limit", default_limit)), _MAX_PAGE))
    except (TypeError, ValueError):
        limit = default_limit
    try:
        offset = max(0, int(params.get("offset", 0)))
    except (TypeError, ValueError):
        offset = 0
    status = params.get("status") or None
    return limit, offset, status


def _fetch_sessions(
    status: str | None = None,
    domain: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Read a filtered, paginated page of sessions tagged with their domain."""
    conn = _sessions._connect_db()
    try:
        dom = _domain_sql(conn.dialect)
        where: list[str] = []
        params: list[Any] = []
        if status:
            where.append("status = ?")
            params.append(status)
        if domain:
            where.append(f"{dom} = ?")
            params.append(domain)
        sql = f"SELECT *, {dom} AS domain FROM sessions"  # nosec B608 — expr is a dialect constant
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        cur = conn.cursor()
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _sql_in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def _multi_host_state() -> bool:
    """True when sessions may be owned by other hosts (state externalized)."""
    from agent_utilities.core.state_store import postgres_state_enabled

    return postgres_state_enabled()


async def fleet_health(request: Request) -> JSONResponse:
    """Aggregate swarm-health: counts + per-domain error rates (SQL aggregates)."""
    try:
        _sessions.rehydrate_goals()
    except Exception:  # noqa: BLE001 — health must answer even if rehydrate fails
        pass
    by_status: dict[str, int] = {}
    domains: dict[str, dict[str, float]] = {}
    total = 0
    conn = _sessions._connect_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT status, COUNT(*) FROM sessions GROUP BY status")
        for row in cur.fetchall():
            by_status[str(row[0] or "unknown")] = int(row[1])
        total = sum(by_status.values())

        dom = _domain_sql(conn.dialect)
        cur.execute(
            f"""
            SELECT {dom} AS domain,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status IN {_sql_in(_ACTIVE_STATUSES)} THEN 1 ELSE 0 END) AS active,
                   SUM(CASE WHEN status IN {_sql_in(_ERROR_STATUSES)} THEN 1 ELSE 0 END) AS errored
            FROM sessions GROUP BY 1
            """  # nosec B608 — all interpolations are module constants
        )
        for row in cur.fetchall():
            n_total = int(row[1])
            errored = int(row[3])
            domains[str(row[0])] = {
                "total": n_total,
                "active": int(row[2]),
                "errored": errored,
                "error_rate": round(errored / n_total, 4) if n_total else 0.0,
            }
    finally:
        conn.close()

    active_goals = getattr(_sessions, "active_goals", {})
    return JSONResponse(
        {
            "generated_at": time.time(),
            "sessions": {"total": total, "by_status": by_status},
            "goals": {
                "active": len(getattr(_sessions, "background_goal_runs", {})),
                "tracked": len(active_goals),
            },
            "domains": domains,
        }
    )


async def fleet_topology(request: Request) -> JSONResponse:
    """Live agent/team topology grouped by enterprise domain (paginated)."""
    try:
        _sessions.rehydrate_goals()
    except Exception:  # noqa: BLE001
        pass
    limit, offset, status = _page_params(request)
    rows = _fetch_sessions(status=status, limit=limit, offset=offset)
    domains: dict[str, dict[str, Any]] = {}
    for s in rows:
        dom = domains.setdefault(s["domain"], {"domain": s["domain"], "sessions": []})
        dom["sessions"].append(
            {
                "id": s.get("id"),
                "status": s.get("status"),
                "background": bool(s.get("background", 0)),
                "needs_input": bool(s.get("needs_input", 0)),
                "updated_at": s.get("updated_at"),
            }
        )

    # Totals from SQL aggregates, not the returned page (CONCEPT:OS-5.18).
    conn = _sessions._connect_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sessions")
        row = cur.fetchone()
        total_sessions = int(row[0]) if row else 0
    finally:
        conn.close()

    active_goals = getattr(_sessions, "active_goals", {})
    return JSONResponse(
        {
            "domains": list(domains.values()),
            "goals": _sessions.make_serializable(list(active_goals.values()))
            if hasattr(_sessions, "make_serializable")
            else [],
            "totals": {"domains": len(domains), "sessions": total_sessions},
            "page": {"limit": limit, "offset": offset, "returned": len(rows)},
        }
    )


def _cancel_session_tasks(session_id: str) -> bool:
    """Cancel any in-flight goal-loop task bound to ``session_id`` (in-memory)."""
    cancelled = False
    runs = getattr(_sessions, "background_goal_runs", {})
    for goal_id, run in list(runs.items()):
        if run.get("session_id") == session_id:
            task = run.get("task")
            if task is not None and not task.done():
                task.cancel()
            runs.pop(goal_id, None)
            active = getattr(_sessions, "active_goals", {})
            if goal_id in active:
                goal_status: Any = getattr(
                    _sessions, "GoalStatus", type("G", (), {"CANCELLED": "cancelled"})
                )
                active[goal_id]["status"] = goal_status.CANCELLED
                persist = getattr(_sessions, "_persist_goal", None)
                if callable(persist):
                    persist(goal_id)
            cancelled = True
    return cancelled


async def _set_fleet_status(request: Request, new_status: str) -> JSONResponse:
    """Shared pause/kill: desired-state writes + local fast-path cancel (OS-5.18).

    Body accepts either ``{"session_ids": [...]}`` or ``{"domain": "finance"}``
    (whole-domain containment). Sessions whose goal loop runs in THIS process
    are cancelled immediately and set to the final status. With durable state
    externalized (``state_db_uri``), sessions owned by other hosts instead get
    ``pause_requested``/``kill_requested``, which the owning host's session
    loop reconciles (see ``core.sessions._desired_session_action``). Under the
    single-host SQLite default every session is local, so the final status is
    applied directly — unchanged behavior.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    target_ids: list[str] = list(body.get("session_ids") or [])
    domain = body.get("domain")
    if domain and not target_ids:
        target_ids = [
            s["id"]
            for s in _fetch_sessions(domain=domain, limit=_MAX_PAGE)
            if s.get("id")
        ]

    if not target_ids:
        return JSONResponse(
            {
                "status": "error",
                "message": "Provide session_ids or a domain to target.",
            },
            status_code=400,
        )

    multi_host = _multi_host_state()
    requested_status = "pause_requested" if new_status == "paused" else "kill_requested"

    affected: list[str] = []
    applied: dict[str, str] = {}
    conn = _sessions._connect_db()
    try:
        cur = conn.cursor()
        for sid in target_ids:
            local = _cancel_session_tasks(sid)
            status_to_write = (
                new_status if (local or not multi_host) else requested_status
            )
            cur.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                (status_to_write, time.time(), sid),
            )
            affected.append(sid)
            applied[sid] = status_to_write
        conn.commit()
    finally:
        conn.close()

    return JSONResponse(
        {
            "status": "success",
            "action": new_status,
            "affected": affected,
            "applied": applied,
            "count": len(affected),
        }
    )


async def fleet_pause(request: Request) -> JSONResponse:
    """Pause sessions (by ids or whole domain) — sets status='paused', halts loops."""
    return await _set_fleet_status(request, "paused")


async def fleet_kill(request: Request) -> JSONResponse:
    """Emergency stop: cancel sessions (by ids or whole domain) — blast-radius containment."""
    return await _set_fleet_status(request, "cancelled")


async def fleet_approvals(request: Request) -> JSONResponse:
    """List pending mutation/risk approvals.

    Two sources share this single human queue: ``Task`` graph nodes awaiting a
    decision (orchestrator jobs) and ``ActionApproval`` nodes filed by the
    operational ActionPolicy gate (CONCEPT:OS-5.24 — restart/scale/deploy
    proposals the policy routed to a human; the fleet reconciler executes them
    once granted, CONCEPT:OS-5.25).
    """
    from agent_utilities.mcp.kg_server import _execute_tool, safe_json_load

    cypher = (
        "MATCH (t:Task) WHERE t.status = 'pending' "
        "AND (t.approval_status IS NULL OR t.approval_status = 'pending') "
        "RETURN t LIMIT 200"
    )
    pending: list = []
    note = None
    try:
        res = await _execute_tool("graph_query", action="cypher", cypher=cypher)
        loaded = safe_json_load(res)
        if isinstance(loaded, list):
            pending = loaded
    except Exception as e:
        # Degrade gracefully when the engine/graph is not yet available.
        note = str(e)
    try:
        from agent_utilities.mcp.kg_server import _get_engine

        rows = _get_engine().query_cypher(
            "MATCH (a:ActionApproval {status: 'pending'}) RETURN a LIMIT 200"
        )
        for row in rows or []:
            props = row.get("a") if isinstance(row, dict) else None
            if isinstance(props, dict):
                pending.append(props)
    except Exception as e:
        note = note or str(e)
    payload = {"status": "success", "pending": pending}
    if note:
        payload["note"] = note
    return JSONResponse(payload)


async def fleet_grant_approval(request: Request) -> JSONResponse:
    """Grant or deny a pending approval by job id.

    ``ActionApproval`` ids (``action_approval:...``) resolve in place — the
    decision is stamped on the node and the reconciler's approved-action drain
    executes it on the next tick (CONCEPT:OS-5.24/OS-5.25). Anything else
    falls through to the orchestrator's ``grant_approval``.
    """
    from agent_utilities.mcp.kg_server import _execute_tool, safe_json_load

    try:
        body = await request.json()
    except Exception:
        body = {}
    job_id = body.get("job_id")
    decision = body.get("decision", "approved")
    if not job_id:
        return JSONResponse(
            {"status": "error", "message": "job_id is required"}, status_code=400
        )
    if str(job_id).startswith("action_approval:"):
        status = "approved" if decision in ("approved", "approve", True) else "denied"
        try:
            from agent_utilities.mcp.kg_server import _get_engine

            _get_engine().backend.execute(
                "MATCH (a:ActionApproval {id: $id, status: 'pending'}) "
                "SET a.status = $status, a.decided_unix = $ts",
                {"id": job_id, "status": status, "ts": time.time()},
            )
            return JSONResponse(
                {
                    "status": "success",
                    "result": {"approval_id": job_id, "decision": status},
                }
            )
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
    try:
        res = await _execute_tool(
            "graph_orchestrate",
            action="grant_approval",
            job_id=job_id,
            approval_status=decision,
        )
        return JSONResponse({"status": "success", "result": safe_json_load(res)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


def mount_fleet_routes(app, prefix: str = "") -> None:
    """Mount the supervisory plane onto a Starlette/FastAPI ``app``."""
    # Webhook ingress for monitoring events (CONCEPT:OS-5.15) — sibling module
    # so alert normalization/persistence stays out of the supervisory handlers.
    from agent_utilities.gateway.fleet_events import fleet_events_receive

    def route(path: str, handler, methods: list[str]) -> None:
        app.add_route(prefix + path, handler, methods=methods)

    route("/fleet/health", fleet_health, ["GET"])
    route("/fleet/events", fleet_events_receive, ["POST"])
    route("/fleet/topology", fleet_topology, ["GET"])
    route("/fleet/pause", fleet_pause, ["POST"])
    route("/fleet/kill", fleet_kill, ["POST"])
    route("/fleet/approvals", fleet_approvals, ["GET"])
    route("/fleet/approvals/grant", fleet_grant_approval, ["POST"])
