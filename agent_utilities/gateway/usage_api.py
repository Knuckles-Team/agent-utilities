"""Gateway REST surface for usage/cost/observability (CONCEPT:AU-ECO.mcp.usage-cost-observability-surface).

Mounted at ``/api/observability`` next to the dashboard + graph routers, so all
three frontends (agent-webui, agent-terminal-ui, geniusbot) consume one surface.
Mirrors the useful agentsview Huma routes; all SQL is delegated to
``UsageService``. The upload endpoint is the HTTP half of the remote-ingest
transport (CONCEPT:AU-ECO.mcp.client-side-chat-session): clients parse local logs and POST normalized
bundles so a central engine never needs filesystem access to the client.

Mountable by any FastAPI backend::

    from agent_utilities.gateway.usage_api import usage_router
    app.include_router(usage_router, prefix="/api/observability")
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Query

from agent_utilities.security.persistence_privacy import persistence_reference
from agent_utilities.usage.authorization import (
    UsageAuthorizationError,
    require_usage_admin,
    resolve_usage_tenant,
)
from agent_utilities.usage.models import (
    ActivityCell,
    BreakdownEntry,
    ParsedSessionBundle,
    SearchHit,
    SessionDetail,
    SessionRow,
    ToolStat,
    UsageSummary,
)
from agent_utilities.usage.recorder import get_usage_recorder
from agent_utilities.usage.service import get_usage_service

logger = logging.getLogger(__name__)

usage_router = APIRouter(tags=["observability"])


def _tenant_scope(requested: str | None = None) -> str | None:
    """Translate the shared authorization failure into an HTTP response."""

    try:
        return resolve_usage_tenant(requested)
    except UsageAuthorizationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _require_usage_admin() -> None:
    try:
        require_usage_admin()
    except UsageAuthorizationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _filters(
    from_date: str | None,
    to_date: str | None,
    project: str | None,
    agent: str | None,
    model: str | None,
    origin: str | None,
    tenant_id: str | None,
) -> dict:
    return {
        k: v
        for k, v in {
            "from_date": from_date,
            "to_date": to_date,
            "project": project,
            "agent": agent,
            "model": model,
            "origin": origin,
            "tenant_id": tenant_id,
        }.items()
        if v
    }


# Common query params reused across endpoints.
_From = Query(None, alias="from")
_To = Query(None, alias="to")


@usage_router.get("/summary", response_model=UsageSummary)
def summary(
    from_date: str | None = _From,
    to_date: str | None = _To,
    project: str | None = None,
    agent: str | None = None,
    model: str | None = None,
    origin: str | None = None,
    tenant_id: str | None = None,
) -> UsageSummary:
    f = _filters(
        from_date, to_date, project, agent, model, origin, _tenant_scope(tenant_id)
    )
    return get_usage_service().summary(**f)


@usage_router.get("/comparison")
def comparison(
    from_date: str | None = _From,
    to_date: str | None = _To,
    prev_from: str | None = Query(None),
    prev_to: str | None = Query(None),
    project: str | None = None,
    agent: str | None = None,
    tenant_id: str | None = None,
) -> dict:
    svc = get_usage_service()
    tenant = _tenant_scope(tenant_id)
    cur = svc.summary(
        **_filters(from_date, to_date, project, agent, None, None, tenant)
    )
    prev = svc.summary(
        **_filters(prev_from, prev_to, project, agent, None, None, tenant)
    )
    delta = cur.totals.cost_usd - prev.totals.cost_usd
    return {
        "current": cur.model_dump(),
        "previous": prev.model_dump(),
        "cost_delta_usd": round(delta, 6),
    }


@usage_router.get("/by-model", response_model=list[BreakdownEntry])
def by_model(
    from_date: str | None = _From,
    to_date: str | None = _To,
    project: str | None = None,
    agent: str | None = None,
    origin: str | None = None,
    tenant_id: str | None = None,
) -> list[BreakdownEntry]:
    f = _filters(
        from_date, to_date, project, agent, None, origin, _tenant_scope(tenant_id)
    )
    return get_usage_service().by_model(**f)


@usage_router.get("/by-project", response_model=list[BreakdownEntry])
def by_project(
    from_date: str | None = _From,
    to_date: str | None = _To,
    agent: str | None = None,
    origin: str | None = None,
    tenant_id: str | None = None,
) -> list[BreakdownEntry]:
    f = _filters(
        from_date, to_date, None, agent, None, origin, _tenant_scope(tenant_id)
    )
    return get_usage_service().by_project(**f)


@usage_router.get("/by-agent", response_model=list[BreakdownEntry])
def by_agent(
    from_date: str | None = _From,
    to_date: str | None = _To,
    project: str | None = None,
    origin: str | None = None,
    tenant_id: str | None = None,
) -> list[BreakdownEntry]:
    f = _filters(
        from_date, to_date, project, None, None, origin, _tenant_scope(tenant_id)
    )
    return get_usage_service().by_agent(**f)


@usage_router.get("/analytics/tools", response_model=list[ToolStat])
def analytics_tools(
    from_date: str | None = _From,
    to_date: str | None = _To,
    project: str | None = None,
    agent: str | None = None,
    origin: str | None = None,
    tenant_id: str | None = None,
) -> list[ToolStat]:
    f = _filters(
        from_date, to_date, project, agent, None, origin, _tenant_scope(tenant_id)
    )
    return get_usage_service().tools(**f)


@usage_router.get("/analytics/activity", response_model=list[ActivityCell])
@usage_router.get("/analytics/heatmap", response_model=list[ActivityCell])
def analytics_activity(
    from_date: str | None = _From,
    to_date: str | None = _To,
    project: str | None = None,
    agent: str | None = None,
    origin: str | None = None,
    tenant_id: str | None = None,
) -> list[ActivityCell]:
    f = _filters(
        from_date, to_date, project, agent, None, origin, _tenant_scope(tenant_id)
    )
    return get_usage_service().activity(**f)


@usage_router.get("/analytics/session-shape")
def analytics_session_shape(
    from_date: str | None = _From,
    to_date: str | None = _To,
    project: str | None = None,
    agent: str | None = None,
    tenant_id: str | None = None,
) -> dict:
    """Classify sessions into archetypes by message count (agentsview parity)."""
    f = _filters(
        from_date, to_date, project, agent, None, None, _tenant_scope(tenant_id)
    )
    rows = get_usage_service().sessions(limit=10000, **f)
    buckets = {"quick": 0, "standard": 0, "deep": 0, "marathon": 0, "automation": 0}
    for r in rows:
        n = r.message_count
        if n <= 4:
            buckets["quick"] += 1
        elif n <= 20:
            buckets["standard"] += 1
        elif n <= 60:
            buckets["deep"] += 1
        else:
            buckets["marathon"] += 1
    return {"total": len(rows), "shapes": buckets}


@usage_router.get("/top-sessions", response_model=list[SessionRow])
def top_sessions(
    limit: int = 20,
    from_date: str | None = _From,
    to_date: str | None = _To,
    project: str | None = None,
    agent: str | None = None,
    origin: str | None = None,
    tenant_id: str | None = None,
) -> list[SessionRow]:
    f = _filters(
        from_date, to_date, project, agent, None, origin, _tenant_scope(tenant_id)
    )
    return get_usage_service().top_sessions(limit=min(max(limit, 1), 500), **f)


@usage_router.get("/sessions", response_model=list[SessionRow])
def sessions(
    limit: int = 100,
    from_date: str | None = _From,
    to_date: str | None = _To,
    project: str | None = None,
    agent: str | None = None,
    origin: str | None = None,
    tenant_id: str | None = None,
) -> list[SessionRow]:
    f = _filters(
        from_date, to_date, project, agent, None, origin, _tenant_scope(tenant_id)
    )
    return get_usage_service().sessions(limit=min(max(limit, 1), 500), **f)


@usage_router.get("/sessions/{session_id}", response_model=SessionDetail | None)
def session_detail(
    session_id: str, tenant_id: str | None = None
) -> SessionDetail | None:
    tenant = _tenant_scope(tenant_id)
    return get_usage_service().session_detail(session_id, tenant_id=tenant)


@usage_router.get("/search", response_model=list[SearchHit])
def search(q: str, limit: int = 50, tenant_id: str | None = None) -> list[SearchHit]:
    return get_usage_service().search(
        q, limit=min(max(limit, 1), 500), tenant_id=_tenant_scope(tenant_id)
    )


@usage_router.get("/traces")
def traces(
    session_id: str | None = None,
    limit: int = 50,
    tenant_id: str | None = None,
) -> dict:
    """Return privacy-safe Langfuse trace references for the caller's tenant."""

    tenant = _tenant_scope(tenant_id)
    try:
        from agent_utilities.observability.langfuse_exporter import (
            get_langfuse_exporter,
        )

        exporter = get_langfuse_exporter()
        enabled = bool(getattr(exporter, "enabled", False))
    except Exception:  # noqa: BLE001
        enabled = False
    if not enabled:
        return {"enabled": False, "trace_count": 0, "traces": []}
    rows = get_usage_service().sessions(
        limit=min(max(limit, 1), 500), origin="runtime", tenant_id=tenant
    )
    if session_id:
        from agent_utilities.usage.privacy import normalize_run_id

        selected = normalize_run_id(session_id, tenant_id=tenant or "")
        rows = [row for row in rows if row.id == selected]
    return {
        "enabled": True,
        "trace_count": len(rows),
        "traces": [
            {
                "trace_ref": persistence_reference("trace", r.id),
                "project": r.project,
            }
            for r in rows
        ],
    }


@usage_router.post("/sessions/upload")
def upload_sessions(
    bundles: list[ParsedSessionBundle] = Body(...),
    tenant_id: str | None = Query(None),
) -> dict:
    """Ingest pre-parsed session bundles (CONCEPT:AU-ECO.mcp.client-side-chat-session HTTP transport).

    Clients parse their local agent logs and POST normalized bundles here, so a
    central/remote engine never needs to read the client's filesystem.
    """
    authoritative_tenant = _tenant_scope(tenant_id)
    recorder = get_usage_recorder()
    ok = 0
    for bundle in bundles:
        if authoritative_tenant:
            bundle.session.tenant_id = authoritative_tenant
            for call in bundle.tool_calls:
                call.tenant_id = authoritative_tenant
            for event in bundle.usage_events:
                event.tenant_id = authoritative_tenant
        if recorder.record_bundle(bundle):
            ok += 1
    return {"received": len(bundles), "ingested": ok}


@usage_router.post("/sync")
def sync_now() -> dict:
    """Trigger an immediate local-log sync (auto-detect + parse + persist)."""

    _require_usage_admin()
    try:
        from agent_utilities.ingestion.collector import collect_local_sessions

        result = collect_local_sessions()
        return {"status": "ok", **result}
    except Exception as exc:  # noqa: BLE001
        logger.warning("usage_sync_failed error_type=%s", type(exc).__name__)
        return {"status": "error", "detail": "usage sync failed"}
