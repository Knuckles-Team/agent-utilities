"""Stirling PDF widget — PDF processing service status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "stirlingpdf"
    display_name = "Stirling PDF"
    icon = "file-text"
    category = ServiceCategory.PRODUCTIVITY
    description = "PDF toolkit — merge, split, convert, and OCR documents"
    env_prefix = "STIRLINGPDF"
    default_url = "https://stirlingpdf.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="status", label="Status", format="text", highlight=True),
            WidgetField(key="tools", label="Tools", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        import httpx
        url = self._resolve_url(config)
        try:
            resp = httpx.get(f"{url}/api/v1/info/status", timeout=5.0, verify=False)
            online = resp.status_code == 200
        except Exception as e:
            logger.debug("Stirling PDF fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={"status": "Online" if online else "Offline", "tools": 40},
            status="ok" if online else "error",
        )
