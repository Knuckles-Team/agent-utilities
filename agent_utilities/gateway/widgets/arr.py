"""Arr widget — Sonarr/Radarr/Prowlarr media automation suite."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "arr"
    display_name = "*Arr Suite"
    icon = "tv"
    category = ServiceCategory.MEDIA
    description = "Media automation — Sonarr, Radarr, Prowlarr, and Lidarr"
    env_prefix = "ARR"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="monitored", label="Monitored", format="number"),
            WidgetField(key="missing", label="Missing", format="number", highlight=True),
            WidgetField(key="queued", label="Queued", format="number"),
            WidgetField(key="indexers", label="Indexers", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from arr_mcp.api_client import ArrApi
        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = ArrApi(base_url=url, api_key=token)
        try:
            status = client.get_system_status() or {}
        except Exception as e:
            logger.debug("Arr fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={"monitored": 0, "missing": 0, "queued": 0, "indexers": 0},
            status="ok" if status else "unknown",
        )
