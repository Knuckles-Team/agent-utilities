"""Owncast widget — self-hosted live streaming status."""

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
    service_type = "owncast"
    display_name = "Owncast"
    icon = "radio"
    category = ServiceCategory.MEDIA
    description = "Live streaming — broadcast status, viewers, and chat"
    env_prefix = "OWNCAST"
    default_url = "https://owncast.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="live", label="Live", format="text", highlight=True),
            WidgetField(key="viewers", label="Viewers", format="number"),
            WidgetField(key="peak", label="Peak", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from owncast_agent.api_client import OwncastApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = OwncastApi(base_url=url, admin_token=token)

        try:
            status = client.get_status() or {}
            is_live = status.get("online", False)
            viewers = status.get("viewerCount", 0)
            peak = status.get("overallMaxViewerCount", 0)
        except Exception as e:
            logger.debug("Owncast fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "live": "🔴 Live" if is_live else "Offline",
                "viewers": viewers,
                "peak": peak,
            },
            status="ok",
        )
