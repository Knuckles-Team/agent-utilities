"""ARIS widget — Software AG ARIS process-model repository."""

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
    service_type = "aris"
    display_name = "ARIS"
    icon = "workflow"
    category = ServiceCategory.BUSINESS
    description = "Process model repository — models, functions, and events"
    env_prefix = "ARIS"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="models", label="Models", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("aris_mcp.api_client", "ArisApi")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        if not url:
            return WidgetData(status="skipped", error="Missing ARIS url")

        client = client_cls(
            base_url=url,
            token=token,
            verify=self._requests_tls_verify(config),
        )

        try:
            models = client.list_models() or []
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={
                "models": count_items(models),
                "status": "Connected",
            },
            status="ok",
        )
