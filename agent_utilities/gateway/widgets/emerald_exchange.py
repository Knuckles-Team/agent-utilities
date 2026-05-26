"""Emerald Exchange widget — trading platform status."""

from __future__ import annotations

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets.base import BaseWidget


class Widget(BaseWidget):
    service_type = "emerald_exchange"
    display_name = "Emerald Exchange"
    icon = "trending-up"
    category = ServiceCategory.BUSINESS
    description = "Trading platform — portfolio, signals, and market data"
    env_prefix = "EMERALD_EXCHANGE"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="positions", label="Positions", format="number"),
            WidgetField(key="pnl", label="P&L", format="percent"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        return WidgetData(
            fields={"positions": 0, "pnl": 0, "status": "Ready"}, status="ok"
        )
