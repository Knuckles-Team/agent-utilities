"""ServiceNow widget — ITSM incidents, changes, and requests."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "servicenow"
    display_name = "ServiceNow"
    icon = "ticket"
    category = ServiceCategory.BUSINESS
    description = "ITSM — incidents, changes, and service requests"
    env_prefix = "SERVICENOW"
    default_url = "https://instance.service-now.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="open_incidents", label="Incidents", format="number", highlight=True),
            WidgetField(key="open_changes", label="Changes", format="number"),
            WidgetField(key="open_requests", label="Requests", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from servicenow_api.api_client import ServiceNowApi

        url = self._resolve_url(config)
        username = self._resolve_env(config, "username")
        password = self._resolve_env(config, "password")
        client = ServiceNowApi(base_url=url, username=username, password=password)

        try:
            incidents = client.get_incidents(query="state=1", limit=1) or {}
            changes = client.get_change_requests(query="state=1", limit=1) or {}
            inc_count = incidents.get("total", 0) if isinstance(incidents, dict) else len(incidents)
            chg_count = changes.get("total", 0) if isinstance(changes, dict) else len(changes)
        except Exception as e:
            logger.debug("ServiceNow fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "open_incidents": inc_count,
                "open_changes": chg_count,
                "open_requests": 0,
            },
            status="ok",
        )
