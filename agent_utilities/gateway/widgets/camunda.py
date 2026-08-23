"""Camunda widget — process orchestration engine (v7 REST API)."""

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
    service_type = "camunda"
    display_name = "Camunda"
    icon = "workflow"
    category = ServiceCategory.BUSINESS
    description = "Process orchestration — deployed definitions and running instances"
    env_prefix = "CAMUNDA"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(
                key="process_definitions", label="Definitions", format="number"
            ),
            WidgetField(
                key="running_instances",
                label="Running",
                format="number",
                highlight=True,
            ),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("camunda_mcp.api_client", "Api")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        if not url:
            return WidgetData(status="skipped", error="Missing Camunda url")

        client = client_cls(
            v7_kwargs={
                "base_url": url,
                "token": token,
                "tls_profile": self._resolve_tls_profile(config),
            }
        )

        try:
            definitions = client.v7.list_process_definitions() or []
            instances = client.v7.list_process_instances() or []
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={
                "process_definitions": len(definitions)
                if isinstance(definitions, list)
                else 0,
                "running_instances": len(instances)
                if isinstance(instances, list)
                else 0,
            },
            status="ok",
        )
