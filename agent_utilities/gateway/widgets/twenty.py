"""Twenty widget — CRM and contact management."""

from __future__ import annotations

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget


class Widget(BaseWidget):
    service_type = "twenty"
    display_name = "Twenty CRM"
    icon = "users"
    category = ServiceCategory.BUSINESS
    description = "CRM — contacts, companies, deals, and pipelines"
    env_prefix = "TWENTY"
    default_url = "https://twenty.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="contacts", label="Contacts", format="number"),
            WidgetField(key="companies", label="Companies", format="number"),
            WidgetField(key="deals", label="Deals", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from twenty_mcp.api_client import TwentyApi
        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = TwentyApi(base_url=url, api_key=token)
        try:
            people = client.list_people() or {}
            companies = client.list_companies() or {}
            p_count = people.get("totalCount", 0) if isinstance(people, dict) else 0
            c_count = companies.get("totalCount", 0) if isinstance(companies, dict) else 0
        except Exception:
            return WidgetData(status="error", error="Connection failed")

        return WidgetData(
            fields={"contacts": p_count, "companies": c_count, "deals": 0},
            status="ok",
        )
