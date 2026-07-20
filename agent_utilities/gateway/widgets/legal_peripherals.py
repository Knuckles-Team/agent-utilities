"""Legal Peripherals widget — compliance and filing status."""

from __future__ import annotations

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets.base import BaseWidget


class Widget(BaseWidget):
    service_type = "legal_peripherals"
    display_name = "Legal Peripherals"
    icon = "scale"
    category = ServiceCategory.BUSINESS
    description = (
        "Legal compliance — entity filings, EIN status, and state registrations"
    )
    env_prefix = "LEGAL_PERIPHERALS"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="entities", label="Entities", format="number"),
            WidgetField(
                key="pending", label="Pending", format="number", highlight=True
            ),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        return WidgetData(
            fields={"entities": 0, "pending": 0, "status": "Ready"}, status="ok"
        )
