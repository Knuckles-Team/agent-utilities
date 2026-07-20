"""ArchiveBox widget — web archiving status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "archivebox"
    display_name = "ArchiveBox"
    icon = "archive"
    category = ServiceCategory.PRODUCTIVITY
    description = "Web archiver — saved snapshots and archive statistics"
    env_prefix = "ARCHIVEBOX"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="total", label="Total", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        url = self._resolve_url(config)
        try:
            with self._http_client(config, timeout=5.0) as client:
                resp = client.get(f"{url}/api/v1/core/snapshot")
            data = resp.json() if resp.status_code == 200 else {}
            total = data.get("count", 0) if isinstance(data, dict) else 0
        except Exception as e:
            logger.debug("ArchiveBox fetch: %s", type(e).__name__)
            return self._error_data(e)

        return WidgetData(
            fields={"total": total, "status": "Online"},
            status="ok",
        )
