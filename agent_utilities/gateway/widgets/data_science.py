"""Data Science widget — Jupyter notebooks and analysis environment."""

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
    service_type = "data_science"
    display_name = "Data Science"
    icon = "bar-chart-3"
    category = ServiceCategory.DATA_SCIENCE
    description = "Data science — notebooks, datasets, and analysis pipelines"
    env_prefix = "DATA_SCIENCE"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="status", label="Status", format="text"),
            WidgetField(key="kernels", label="Kernels", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        return WidgetData(fields={"status": "Ready", "kernels": 0}, status="ok")
