"""Uptime Kuma widget — service monitoring metrics.

Displays monitor counts by status (up/down/pending) using
the uptime-kuma-agent Python API client.
"""

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
    service_type = "uptime_kuma"
    display_name = "Uptime Kuma"
    icon = "activity"
    category = ServiceCategory.OBSERVABILITY
    description = "Service uptime monitoring — monitor status and response times"
    env_prefix = "UPTIME_KUMA"
    default_url = "https://uptime.local.example.com"
    supports_websocket = True

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="up", label="Up", format="number", highlight=True),
            WidgetField(key="down", label="Down", format="number", highlight=True),
            WidgetField(key="pending", label="Pending", format="number"),
            WidgetField(key="maintenance", label="Maintenance", format="number"),
            WidgetField(key="total", label="Total", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from uptime_kuma_agent.auth import get_client

        client = get_client()

        try:
            monitors = client.get_monitors()
        except Exception as e:
            return WidgetData(status="error", error=str(e))

        up = 0
        down = 0
        pending = 0
        maintenance = 0

        if isinstance(monitors, list):
            for m in monitors:
                status = m.get("active", True)
                monitor_status = m.get("status", 1)  # 1 = up, 0 = down
                if not status:
                    maintenance += 1
                elif monitor_status == 1:
                    up += 1
                elif monitor_status == 0:
                    down += 1
                else:
                    pending += 1

        total = up + down + pending + maintenance

        return WidgetData(
            fields={
                "up": up,
                "down": down,
                "pending": pending,
                "maintenance": maintenance,
                "total": total,
            },
            status="ok" if down == 0 else "error",
        )
