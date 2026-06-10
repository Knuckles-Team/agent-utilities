"""Native swarm supervisory plane (CONCEPT:OS-5.10).

A single pane of glass over the running fleet, exposed through the API gateway —
no separate supervisor service. Everything here surfaces state the ecosystem
*already* maintains:

* **topology / health** — the durable sqlite session registry and the in-memory
  goal-loop registry in :mod:`agent_utilities.core.sessions`.
* **pause / kill** — emergency containment by reusing the same cancel mechanics
  as :func:`agent_utilities.core.sessions.cancel_session_run` (blast-radius stop
  across a whole domain at once).
* **approvals** — pending mutation/risk approvals stored as ``Task`` graph nodes,
  read and granted through the parity-covered ``graph_query`` / ``graph_orchestrate``
  tools.

These handlers are plain Starlette callables mounted by the gateway; the
``agent-webui`` Fleet Dashboard consumes them.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from agent_utilities.core import sessions as _sessions


def _domain_of(metadata_json: str | None) -> str:
    """Derive a session's enterprise domain from its metadata (default 'default')."""
    if not metadata_json:
        return "default"
    try:
        meta = json.loads(metadata_json)
    except (TypeError, ValueError):
        return "default"
    domain = meta.get("domain") or meta.get("team") or meta.get("tenant")
    return str(domain) if domain else "default"


def _fetch_sessions() -> list[dict[str, Any]]:
    """Read all durable sessions, tagging each with its derived domain."""
    db_path = _sessions._get_db_path()
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        out = []
        for row in cur.fetchall():
            d = dict(row)
            d["domain"] = _domain_of(d.get("metadata_json"))
            out.append(d)
        return out
    finally:
        conn.close()


# Statuses that count as "unhealthy" when computing per-domain error rates.
_ERROR_STATUSES = {"failed", "error", "cancelled"}
_ACTIVE_STATUSES = {"active", "running"}


async def fleet_health(request: Request) -> JSONResponse:
    """Aggregate swarm-health: counts + per-domain error rates."""
    rows = _fetch_sessions()
    by_status: dict[str, int] = {}
    domains: dict[str, dict[str, int]] = {}
    for s in rows:
        status = str(s.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        dom = domains.setdefault(
            s["domain"], {"total": 0, "active": 0, "errored": 0}
        )
        dom["total"] += 1
        if status in _ACTIVE_STATUSES:
            dom["active"] += 1
        if status in _ERROR_STATUSES:
            dom["errored"] += 1

    for dom in domains.values():
        dom["error_rate"] = round(dom["errored"] / dom["total"], 4) if dom["total"] else 0.0

    active_goals = getattr(_sessions, "active_goals", {})
    return JSONResponse(
        {
            "generated_at": time.time(),
            "sessions": {"total": len(rows), "by_status": by_status},
            "goals": {"active": len(getattr(_sessions, "background_goal_runs", {})), "tracked": len(active_goals)},
            "domains": domains,
        }
    )


async def fleet_topology(request: Request) -> JSONResponse:
    """Live agent/team topology grouped by enterprise domain."""
    rows = _fetch_sessions()
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
    active_goals = getattr(_sessions, "active_goals", {})
    return JSONResponse(
        {
            "domains": list(domains.values()),
            "goals": _sessions.make_serializable(list(active_goals.values()))
            if hasattr(_sessions, "make_serializable")
            else [],
            "totals": {"domains": len(domains), "sessions": len(rows)},
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
                active[goal_id]["status"] = getattr(
                    _sessions, "GoalStatus", type("G", (), {"CANCELLED": "cancelled"})
                ).CANCELLED
            cancelled = True
    return cancelled


async def _set_fleet_status(request: Request, new_status: str) -> JSONResponse:
    """Shared pause/kill: set the target sessions' status and cancel their tasks.

    Body accepts either ``{"session_ids": [...]}`` or ``{"domain": "finance"}``
    (whole-domain containment).
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    target_ids: list[str] = list(body.get("session_ids") or [])
    domain = body.get("domain")
    if domain and not target_ids:
        target_ids = [s["id"] for s in _fetch_sessions() if s["domain"] == domain and s.get("id")]

    if not target_ids:
        return JSONResponse(
            {"status": "error", "message": "Provide session_ids or a domain to target."},
            status_code=400,
        )

    affected: list[str] = []
    db_path = _sessions._get_db_path()
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        for sid in target_ids:
            _cancel_session_tasks(sid)
            cur.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE id = ?",
                (new_status, time.time(), sid),
            )
            affected.append(sid)
        conn.commit()
    finally:
        conn.close()

    return JSONResponse(
        {"status": "success", "action": new_status, "affected": affected, "count": len(affected)}
    )


async def fleet_pause(request: Request) -> JSONResponse:
    """Pause sessions (by ids or whole domain) — sets status='paused', halts loops."""
    return await _set_fleet_status(request, "paused")


async def fleet_kill(request: Request) -> JSONResponse:
    """Emergency stop: cancel sessions (by ids or whole domain) — blast-radius containment."""
    return await _set_fleet_status(request, "cancelled")


async def fleet_approvals(request: Request) -> JSONResponse:
    """List pending mutation/risk approvals (Task nodes awaiting a decision)."""
    from agent_utilities.mcp.kg_server import _execute_tool

    cypher = (
        "MATCH (t:Task) WHERE t.status = 'pending' "
        "AND (t.approval_status IS NULL OR t.approval_status = 'pending') "
        "RETURN t LIMIT 200"
    )
    try:
        res = await _execute_tool("graph_query", action="cypher", cypher=cypher)
        from agent_utilities.mcp.kg_server import safe_json_load

        return JSONResponse({"status": "success", "pending": safe_json_load(res)})
    except Exception as e:
        # Degrade gracefully when the engine/graph is not yet available.
        return JSONResponse({"status": "success", "pending": [], "note": str(e)})


async def fleet_grant_approval(request: Request) -> JSONResponse:
    """Grant or deny a pending approval by job id."""
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

    def route(path: str, handler, methods: list[str]) -> None:
        app.add_route(prefix + path, handler, methods=methods)

    route("/fleet/health", fleet_health, ["GET"])
    route("/fleet/topology", fleet_topology, ["GET"])
    route("/fleet/pause", fleet_pause, ["POST"])
    route("/fleet/kill", fleet_kill, ["POST"])
    route("/fleet/approvals", fleet_approvals, ["GET"])
    route("/fleet/approvals/grant", fleet_grant_approval, ["POST"])
