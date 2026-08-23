"""Firefly III widget — personal finance manager."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets._optional_client import import_client
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "firefly_iii"
    display_name = "Firefly III"
    icon = "piggy-bank"
    category = ServiceCategory.LIFESTYLE
    description = "Personal finance manager — accounts and budget summary"
    env_prefix = "FIREFLY_III"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(
                key="summary_entries", label="Summary Entries", format="number"
            ),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client(
            "firefly_iii_mcp.api_client", "ApiClientFireflyIii"
        )
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        if not url or not token:
            return WidgetData(status="skipped", error="Missing Firefly III url/token")

        try:
            client = client_cls(
                base_url=url,
                token=token,
                tls_profile=self._resolve_tls_profile(config),
            )
            summary = client.get_basic_summary() or {}
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={
                "summary_entries": len(summary) if isinstance(summary, dict) else 0,
                "status": "Connected",
            },
            status="ok",
        )
