"""ScholarX widget — research paper search and discovery."""

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
    service_type = "scholarx"
    display_name = "ScholarX"
    icon = "book-open"
    category = ServiceCategory.DATA_SCIENCE
    description = "Research — paper search, downloads, and citation analysis"
    env_prefix = "SCHOLARX"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="stored_papers", label="Papers", format="number"),
            WidgetField(key="sources", label="Sources", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        return WidgetData(
            fields={"stored_papers": 0, "sources": 7, "status": "Ready"}, status="ok"
        )
