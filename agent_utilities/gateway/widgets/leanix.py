"""LeanIX widget — enterprise architecture management."""

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
    service_type = "leanix"
    display_name = "LeanIX"
    icon = "layers"
    category = ServiceCategory.BUSINESS
    description = "EA management — IT landscape, fact sheets, and tech radar"
    env_prefix = "LEANIX"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="fact_sheets", label="Fact Sheets", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from leanix_agent.api_client import LeanIXApi

        token = self._resolve_token(config)
        url = self._resolve_url(config)
        client = LeanIXApi(base_url=url, token=token)
        try:
            fs = client.get_fact_sheets() or []
        except Exception as e:
            logger.debug("LeanIX fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "fact_sheets": len(fs) if isinstance(fs, list) else 0,
                "status": "Connected",
            },
            status="ok",
        )
