"""Teleport widget — identity-aware access proxy."""

from __future__ import annotations

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets.base import BaseWidget


class Widget(BaseWidget):
    service_type = "teleport"
    display_name = "Teleport"
    icon = "shield"
    category = ServiceCategory.SECURITY
    description = "Access proxy — SSH, Kubernetes, database, and app access"
    env_prefix = "TELEPORT"
    default_url = "https://teleport.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="nodes", label="Nodes", format="number"),
            WidgetField(
                key="sessions", label="Sessions", format="number", highlight=True
            ),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        return WidgetData(
            fields={"nodes": 0, "sessions": 0, "status": "Ready"}, status="ok"
        )
