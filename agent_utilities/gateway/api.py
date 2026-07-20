"""FastAPI router for the service dashboard.

CONCEPT:AU-OS.config.gateway-service-dashboard — Gateway Service Dashboard

Mountable by agent-webui (and any other FastAPI backend)::

    from agent_utilities.gateway.api import dashboard_router
    app.include_router(dashboard_router, prefix="/api/dashboard")
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent_utilities.gateway.aggregator import Aggregator
from agent_utilities.gateway.config import ConfigManager
from agent_utilities.gateway.models import DashboardLayout, WidgetData
from agent_utilities.gateway.registry import get_registry
from agent_utilities.security.error_surface import public_error_payload

logger = logging.getLogger(__name__)
_SERVICE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _dashboard_capabilities(request: Request) -> set[str] | None:
    claims = getattr(request.state, "user_claims", None)
    if not claims or claims.get("auth_type") == "api_key":
        return None
    try:
        from agent_utilities.core.config import config
        from agent_utilities.security.identity import (
            base_capabilities,
            normalize_identity,
        )

        return set(
            base_capabilities(
                normalize_identity(claims), config.identity_group_capability_map
            )
        )
    except Exception:
        raise HTTPException(status_code=403, detail="dashboard capability required") from None


async def _require_dashboard_read(request: Request) -> None:
    if request.url.path.endswith("/health"):
        return
    capabilities = _dashboard_capabilities(request)
    if capabilities is not None and not capabilities.intersection(
        {"gateway:read", "gateway:write", "gateway:admin", "admin"}
    ):
        raise HTTPException(status_code=403, detail="dashboard capability required")


def _require_dashboard_write(request: Request) -> None:
    capabilities = _dashboard_capabilities(request)
    if capabilities is not None and not capabilities.intersection(
        {"gateway:write", "gateway:admin", "admin"}
    ):
        raise HTTPException(status_code=403, detail="dashboard write capability required")


dashboard_router = APIRouter(
    tags=["dashboard"], dependencies=[Depends(_require_dashboard_read)]
)

# Singletons — initialized lazily on first request, PER PROCESS.
#
# Multi-worker audit (CONCEPT:AU-OS.observability.no-op-without-metrics):
# nothing here is mutated post-startup
# except through ``save_layout``, which persists straight to the shared YAML
# file (XDG config dir) — reads always go back to disk, so workers/replicas
# converge on the next request. The only per-process *divergent* state is the
# Aggregator's short read cache (10s TTL) and its thread pool, both of which
# are safe to duplicate per worker. Anything needing strict cross-replica
# consistency belongs to the state-externalization track, not here.
_aggregator: Aggregator | None = None
_config_manager: ConfigManager | None = None


def _get_aggregator() -> Aggregator:
    global _aggregator, _config_manager
    if _aggregator is None:
        _config_manager = ConfigManager()
        _aggregator = Aggregator(config_manager=_config_manager)
    return _aggregator


class DashboardResponse(BaseModel):
    layout: DashboardLayout
    data: dict[str, WidgetData]


class WidgetListItem(BaseModel):
    widget_type: str
    display_name: str
    icon: str
    category: str
    description: str
    supports_websocket: bool


@dashboard_router.get("/layout")
async def get_layout() -> DashboardLayout:
    """Get the current dashboard layout configuration."""
    aggregator = _get_aggregator()
    return aggregator.get_layout()


@dashboard_router.put("/layout")
async def save_layout(layout: DashboardLayout, request: Request) -> dict[str, str]:
    """Save a new dashboard layout configuration."""
    _require_dashboard_write(request)
    aggregator = _get_aggregator()
    aggregator.save_layout(layout)
    return {"status": "saved"}


@dashboard_router.get("/data")
async def get_all_data() -> dict[str, WidgetData]:
    """Fetch data from all active widgets."""
    aggregator = _get_aggregator()
    return await aggregator.fetch_all()


@dashboard_router.get("/data/{service_id}")
async def get_service_data(service_id: str) -> WidgetData:
    """Fetch data for a single service."""
    if not _SERVICE_ID_RE.fullmatch(service_id):
        raise HTTPException(status_code=404, detail="service not found")
    aggregator = _get_aggregator()
    data = await aggregator.fetch_one(service_id)
    if data.status == "error" and "not found" in (data.error or ""):
        raise HTTPException(status_code=404, detail="service not found")
    return data


@dashboard_router.get("/full")
async def get_full_dashboard() -> DashboardResponse:
    """Get layout + data in a single request (initial page load)."""
    aggregator = _get_aggregator()
    layout = aggregator.get_layout()
    data = await aggregator.fetch_all()
    return DashboardResponse(layout=layout, data=data)


@dashboard_router.get("/widgets")
async def list_available_widgets() -> list[WidgetListItem]:
    """List all widget types available for configuration."""
    registry = get_registry()
    registrations = registry.discover_all()
    return [
        WidgetListItem(
            widget_type=r.widget_type,
            display_name=r.display_name,
            icon=r.icon,
            category=r.category.value,
            description=r.description,
            supports_websocket=r.supports_websocket,
        )
        for r in registrations.values()
    ]


@dashboard_router.get("/health")
async def health_check() -> JSONResponse:
    """Return non-fingerprinting liveness for unauthenticated probes."""
    return JSONResponse(
        {"status": "ok"},
        headers={"Cache-Control": "no-store"},
    )


@dashboard_router.get("/discover")
async def discover_services() -> DashboardLayout:
    """Auto-discover services from mcp_config.json and return a layout."""
    config_mgr = ConfigManager()
    return config_mgr._auto_discover()


@dashboard_router.get("/daemon/status")
async def daemon_status() -> dict[str, Any]:
    """Status of the single consolidated KG background daemon (CONCEPT:EG-KG.storage.nonblocking-checkpoint).

    The gateway is the daemon host; this surfaces the one daemon's role, live
    threads, registered maintenance jobs, and queue depth.
    """
    from agent_utilities.gateway.daemon import daemon_status as _status

    return _status()


@dashboard_router.get("/daemon/shards")
async def daemon_shards() -> dict[str, Any]:
    """Engine shard topology + per-shard reachability (CONCEPT:AU-OS.scaling.shard-topology-visibility-per).

    Reports the configured ``GRAPH_SERVICE_ENDPOINTS`` topology (single vs
    sharded), a transport-level reachability probe per shard, and each shard's
    circuit-breaker state. Refreshes the
    ``agent_utilities_engine_shard_up{endpoint}`` gauge as a side effect.
    """
    from agent_utilities.knowledge_graph.core.shard_topology import (
        shard_topology_status,
    )

    return shard_topology_status()


@dashboard_router.post("/daemon/start")
async def daemon_start(request: Request) -> dict[str, Any]:
    """Ensure the single consolidated KG daemon is running in this gateway."""
    _require_dashboard_write(request)
    from agent_utilities.gateway.daemon import daemon_status as _status
    from agent_utilities.gateway.daemon import start_host_daemon

    start_host_daemon()
    return _status()


@dashboard_router.post("/hydrate/{source}")
async def trigger_hydration(source: str, request: Request) -> dict[str, Any]:
    """Manually trigger hydration for a specific external source."""
    import re

    _require_dashboard_write(request)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", source):
        raise HTTPException(status_code=422, detail="invalid hydration source")
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.hydration import HydrationManager

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        raise HTTPException(
            status_code=500, detail="Active Knowledge Graph engine not available"
        )
    try:
        res = HydrationManager().hydrate_source(engine, source)
        return res
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=public_error_payload(exc, logger=logger, code="invalid_request"),
        ) from None
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=public_error_payload(exc, logger=logger)
        ) from None


@dashboard_router.post("/hydrate")
async def trigger_all_hydration(request: Request) -> dict[str, Any]:
    """Manually trigger hydration for all configured/active sources sequentially."""
    _require_dashboard_write(request)
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
    from agent_utilities.knowledge_graph.core.hydration import HydrationManager

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        raise HTTPException(
            status_code=500, detail="Active Knowledge Graph engine not available"
        )
    try:
        res = HydrationManager().hydrate_all(engine)
        return res
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=public_error_payload(exc, logger=logger)
        ) from None


@dashboard_router.get("/hydration-status")
async def get_hydration_status() -> dict[str, Any]:
    """Retrieve configuration status of all hydration sources."""
    from agent_utilities.knowledge_graph.core.hydration import HydrationManager

    return HydrationManager().get_status()
