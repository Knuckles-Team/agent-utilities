"""Systems Manager widget — remote host system health."""

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
    service_type = "systems_manager"
    display_name = "Systems Manager"
    icon = "server"
    category = ServiceCategory.INFRASTRUCTURE
    description = "System health — CPU, memory, disk, and process monitoring"
    env_prefix = "SYSTEMS_MANAGER"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="hosts", label="Hosts", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        return WidgetData(
            fields={"hosts": 0, "status": "Ready"},
            status="ok",
        )
