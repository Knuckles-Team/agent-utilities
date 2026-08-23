"""Apache Jena widget — Fuseki triple-store server."""

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
    service_type = "jena"
    display_name = "Apache Jena"
    icon = "database"
    category = ServiceCategory.DATA_SCIENCE
    description = "Fuseki SPARQL triple-store — reachability"
    env_prefix = "JENA"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("jena_mcp.api_client", "Api")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        if not url:
            return WidgetData(status="skipped", error="Missing Jena url")

        client = client_cls(
            base_url=url,
            token=token or None,
            tls_profile=self._resolve_tls_profile(config),
        )

        try:
            client.ping()
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={"status": "Connected"},
            status="ok",
        )
