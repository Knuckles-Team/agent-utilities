"""LeanIX widget — enterprise architecture fact sheets (GraphQL API).

The registry has carried a ``leanix`` widget entry since the dashboard's
inception, but no ``leanix.py`` module ever existed — every discovery cycle
silently no-op'd on the missing module. This file closes that gap.
"""

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

_FACT_SHEET_COUNT_QUERY = "{ allFactSheets { totalCount } }"


class Widget(BaseWidget):
    service_type = "leanix"
    display_name = "LeanIX"
    icon = "layout-grid"
    category = ServiceCategory.BUSINESS
    description = "Enterprise architecture — fact sheet inventory"
    env_prefix = "LEANIX"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="fact_sheets", label="Fact Sheets", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("leanix_agent.leanix_gql", "GraphQL")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        if not url or not token:
            return WidgetData(status="skipped", error="Missing LeanIX url/token")

        try:
            client = client_cls(
                url=url,
                token=token,
                tls_profile=self._resolve_tls_profile(config),
            )
            result = client.query(_FACT_SHEET_COUNT_QUERY) or {}
        except Exception as e:
            return self._error_data(e)

        fact_sheets = 0
        if isinstance(result, dict):
            fact_sheets = (
                result.get("data", {}).get("allFactSheets", {}).get("totalCount", 0)
            )

        return WidgetData(
            fields={
                "fact_sheets": fact_sheets,
                "status": "Connected",
            },
            status="ok",
        )
