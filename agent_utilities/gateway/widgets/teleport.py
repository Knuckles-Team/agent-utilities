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

    def get_fields(self) -> list[WidgetField]:
        return self.get_widget_fields("teleport")

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        return WidgetData(
            fields={"nodes": 0, "sessions": 0, "status": "Ready"}, status="ok"
        )
