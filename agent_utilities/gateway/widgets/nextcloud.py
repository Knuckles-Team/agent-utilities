"""Nextcloud widget — cloud storage and collaboration status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "nextcloud"
    display_name = "Nextcloud"
    icon = "cloud"
    category = ServiceCategory.PRODUCTIVITY
    description = "Cloud storage — files, calendars, and collaboration"
    env_prefix = "NEXTCLOUD"
    default_url = "https://nextcloud.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="free_space", label="Free Space", format="bytes"),
            WidgetField(key="files", label="Files", format="number"),
            WidgetField(key="shares", label="Shares", format="number"),
            WidgetField(key="calendars", label="Calendars", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from nextcloud_agent.api_client import NextcloudApi

        url = self._resolve_url(config)
        username = self._resolve_env(config, "username")
        password = self._resolve_env(config, "password")
        client = NextcloudApi(base_url=url, username=username, password=password)

        try:
            files = client.list_files(path="/") or []
            shares = client.list_shares() or []
            calendars = client.list_calendars() or []
            free_space = 0
            try:
                props = client.get_properties(path="/")
                free_space = props.get("free_space", 0) if props else 0
            except Exception:
                pass
        except Exception as e:
            logger.debug("Nextcloud fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "free_space": free_space,
                "files": len(files),
                "shares": len(shares),
                "calendars": len(calendars),
            },
            status="ok",
        )
