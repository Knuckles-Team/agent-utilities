"""Microsoft Clarity widget — web analytics data export."""

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
    service_type = "clarity"
    display_name = "Microsoft Clarity"
    icon = "line-chart"
    category = ServiceCategory.DATA_SCIENCE
    description = "Web analytics — session and traffic data export"
    env_prefix = "CLARITY"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="sessions", label="Sessions", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("clarity_api.api_client", "Api")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        if not url or not token:
            return WidgetData(status="skipped", error="Missing Clarity url/token")

        try:
            client = client_cls(
                url=url,
                token=token,
                tls_profile=self._resolve_tls_profile(config),
            )
            response = client.get_data_export(number_of_days=1)
            export = response.json() or []
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={
                "sessions": len(export) if isinstance(export, list) else 0,
                "status": "Connected",
            },
            status="ok",
        )
