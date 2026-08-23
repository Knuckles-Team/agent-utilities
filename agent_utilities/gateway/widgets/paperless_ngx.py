"""Paperless-ngx widget — document management system."""

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
    service_type = "paperless_ngx"
    display_name = "Paperless-ngx"
    icon = "file-text"
    category = ServiceCategory.PRODUCTIVITY
    description = "Document management — pending OCR/consumption tasks"
    env_prefix = "PAPERLESS_NGX"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="pending_tasks", label="Pending Tasks", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("paperless_ngx_mcp.api_client", "Api")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        if not url or not token:
            return WidgetData(status="skipped", error="Missing Paperless-ngx url/token")

        try:
            client = client_cls(
                base_url=url,
                token=token,
                tls_profile=self._resolve_tls_profile(config),
            )
            tasks = client.list_tasks(max_pages=1) or []
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={
                "pending_tasks": count_items(tasks),
                "status": "Connected",
            },
            status="ok",
        )
