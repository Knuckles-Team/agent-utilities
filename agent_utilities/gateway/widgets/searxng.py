"""SearXNG widget — metasearch engine status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "searxng"
    display_name = "SearXNG"
    icon = "search"
    category = ServiceCategory.PRODUCTIVITY
    description = "Metasearch engine — privacy-respecting search aggregator"
    env_prefix = "SEARXNG"
    default_url = "https://searxng.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="engines", label="Engines", format="number"),
            WidgetField(key="status", label="Status", format="text", highlight=True),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        import httpx
        url = self._resolve_url(config)
        try:
            resp = httpx.get(f"{url}/config", timeout=5.0, verify=False)
            data = resp.json() if resp.status_code == 200 else {}
            engines = data.get("engines", [])
        except Exception as e:
            logger.debug("SearXNG fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "engines": len(engines) if isinstance(engines, list) else 0,
                "status": "Online",
            },
            status="ok",
        )
