"""HDHomeRun widget — local network TV tuner device."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets._optional_client import import_client
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "hdhomerun"
    display_name = "HDHomeRun"
    icon = "tv"
    category = ServiceCategory.MEDIA
    description = "Network TV tuner — channel lineup"
    env_prefix = "HDHOMERUN"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="channels", label="Channels", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client(
            "hdhomerun_mcp.api_client", "ApiClientSystem"
        )
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        if not url:
            return WidgetData(status="skipped", error="Missing HDHomeRun url")

        client = client_cls(url=url, verify=self._requests_tls_verify(config))

        try:
            lineup = client.get_lineup() or []
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={
                "channels": len(lineup) if isinstance(lineup, list) else 0,
                "status": "Connected",
            },
            status="ok",
        )
