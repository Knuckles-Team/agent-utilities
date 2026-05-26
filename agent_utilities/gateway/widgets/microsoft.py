"""Microsoft widget — Microsoft 365 integration status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "microsoft"
    display_name = "Microsoft 365"
    icon = "layout-grid"
    category = ServiceCategory.PRODUCTIVITY
    description = "Microsoft 365 — Teams, Outlook, OneDrive integration"
    env_prefix = "MICROSOFT"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="unread_emails", label="Unread", format="number", highlight=True),
            WidgetField(key="events_today", label="Events", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from microsoft_agent.api_client import MicrosoftApi
        token = self._resolve_token(config)
        client = MicrosoftApi(token=token)
        try:
            mail = client.get_unread_count() or 0
            events = client.get_today_events() or []
        except Exception as e:
            logger.debug("Microsoft fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "unread_emails": mail if isinstance(mail, int) else 0,
                "events_today": len(events) if isinstance(events, list) else 0,
                "status": "Connected",
            },
            status="ok",
        )
