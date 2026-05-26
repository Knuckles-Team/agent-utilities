"""Google Workspace widget — Gmail, Calendar, Drive integration."""

from __future__ import annotations

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets.base import BaseWidget


class Widget(BaseWidget):
    service_type = "google_workspace"
    display_name = "Google Workspace"
    icon = "mail"
    category = ServiceCategory.PRODUCTIVITY
    description = "Google — Gmail, Calendar, Drive, Docs, and Sheets"
    env_prefix = "GOOGLE"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(
                key="unread_emails", label="Unread", format="number", highlight=True
            ),
            WidgetField(key="events_today", label="Events", format="number"),
            WidgetField(key="drive_files", label="Files", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        return WidgetData(
            fields={"unread_emails": 0, "events_today": 0, "drive_files": 0},
            status="ok",
        )
