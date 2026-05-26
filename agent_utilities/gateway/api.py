"""FastAPI router for the service dashboard.

CONCEPT:GW-1.0 — Gateway Service Dashboard

Mountable by agent-webui (and any other FastAPI backend)::

    from agent_utilities.gateway.api import dashboard_router
    app.include_router(dashboard_router, prefix="/api/dashboard")
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_utilities.gateway.aggregator import Aggregator
from agent_utilities.gateway.config import ConfigManager
from agent_utilities.gateway.models import DashboardLayout, WidgetData
from agent_utilities.gateway.registry import get_registry

logger = logging.getLogger(__name__)

dashboard_router = APIRouter(tags=["dashboard"])

# Singletons — initialized on first request
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
async def save_layout(layout: DashboardLayout) -> dict[str, str]:
    """Save a new dashboard layout configuration."""
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
    aggregator = _get_aggregator()
    data = await aggregator.fetch_one(service_id)
    if data.status == "error" and "not found" in (data.error or ""):
        raise HTTPException(status_code=404, detail=data.error)
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
async def health_check() -> dict[str, bool]:
    """Quick health check across all configured services."""
    aggregator = _get_aggregator()
    return await aggregator.health_check()


@dashboard_router.get("/discover")
async def discover_services() -> DashboardLayout:
    """Auto-discover services from mcp_config.json and return a layout."""
    config_mgr = ConfigManager()
    return config_mgr._auto_discover()


@dashboard_router.post("/hydrate/{source}")
async def trigger_hydration(source: str) -> dict[str, Any]:
    """Manually trigger hydration for a specific external source."""
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
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hydration failed: {e}")


@dashboard_router.post("/hydrate")
async def trigger_all_hydration() -> dict[str, Any]:
    """Manually trigger hydration for all configured/active sources sequentially."""
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hydration failed: {e}")


@dashboard_router.get("/hydration-status")
async def get_hydration_status() -> dict[str, Any]:
    """Retrieve configuration status of all hydration sources."""
    from agent_utilities.knowledge_graph.core.hydration import HydrationManager

    return HydrationManager().get_status()
