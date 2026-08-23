"""PulseLink widget — research/news aggregation service."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets._optional_client import count_items, import_client
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "pulselink"
    display_name = "PulseLink"
    icon = "radio"
    category = ServiceCategory.DATA_SCIENCE
    description = "Research and news aggregation — configured sources"
    env_prefix = "PULSELINK"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="sources", label="Sources", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("pulselink_mcp.api_client", "Api")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        if not url or not token:
            return WidgetData(status="skipped", error="Missing PulseLink url/token")

        try:
            client = client_cls(
                base_url=url,
                token=token,
                tls_profile=self._resolve_tls_profile(config),
            )
            sources = client.sources() or []
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={
                "sources": count_items(sources),
                "status": "Connected",
            },
            status="ok",
        )
