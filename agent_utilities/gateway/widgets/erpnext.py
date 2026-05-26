"""ERPNext widget — business ERP status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "erpnext"
    display_name = "ERPNext"
    icon = "building"
    category = ServiceCategory.BUSINESS
    description = "ERP — orders, invoices, inventory, and HR"
    env_prefix = "ERPNEXT"
    default_url = "https://erpnext.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="open_orders", label="Orders", format="number"),
            WidgetField(key="pending_invoices", label="Invoices", format="number"),
            WidgetField(key="employees", label="Employees", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from erpnext_agent.api_client import ERPNextApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = ERPNextApi(base_url=url, api_key=token)

        try:
            info = client.get_info() or {}
            status_text = "Online" if info else "Unknown"
        except Exception as e:
            logger.debug("ERPNext fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "open_orders": 0,
                "pending_invoices": 0,
                "employees": 0,
                "status": status_text,
            },
            status="ok",
        )
