"""ArchiveBox widget — web archiving status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "archivebox"
    display_name = "ArchiveBox"
    icon = "archive"
    category = ServiceCategory.PRODUCTIVITY
    description = "Web archiver — saved snapshots and archive statistics"
    env_prefix = "ARCHIVEBOX"
    default_url = "https://archivebox.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="total", label="Total", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        import httpx
        url = self._resolve_url(config)
        try:
            resp = httpx.get(f"{url}/api/v1/core/snapshot", timeout=5.0, verify=False)
            data = resp.json() if resp.status_code == 200 else {}
            total = data.get("count", 0) if isinstance(data, dict) else 0
        except Exception as e:
            logger.debug("ArchiveBox fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={"total": total, "status": "Online"},
            status="ok",
        )
