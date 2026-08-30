"""Arr widget — Sonarr/Radarr/Prowlarr media automation suite."""

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
    service_type = "arr"
    display_name = "*Arr Suite"
    icon = "tv"
    category = ServiceCategory.MEDIA
    description = "Media automation — Sonarr, Radarr, Prowlarr, and Lidarr"
    env_prefix = "ARR"

    def get_fields(self) -> list[WidgetField]:
        return self.get_widget_fields("arr")

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        # No `arr_mcp.api_client` module exists. The real clients are the
        # per-service generated `Api` classes under `arr_mcp/api/` — one file
        # per *arr service (Sonarr, Radarr, Prowlarr, Lidarr, Bazarr,
        # Chaptarr, Seerr). This widget has always driven a single generic
        # connection check (all fields below were already hardcoded), so it
        # keeps that behavior and uses Sonarr's client as the representative
        # connection, matching the existing single ARR_URL/ARR_API_KEY config.
        from arr_mcp.api.api_client_sonarr import Api as ArrApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = ArrApi(base_url=url, token=token)
        try:
            status = client.get_system_status() or {}
        except Exception as e:
            logger.debug("Arr fetch: %s", type(e).__name__)
            return self._error_data(e)

        return WidgetData(
            fields={"monitored": 0, "missing": 0, "queued": 0, "indexers": 0},
            status="ok" if status else "unknown",
        )
