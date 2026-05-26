"""Mattermost widget — team communication status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "mattermost"
    display_name = "Mattermost"
    icon = "message-square"
    category = ServiceCategory.COMMUNICATION
    description = "Team chat — channels, users, and message activity"
    env_prefix = "MATTERMOST"
    default_url = "https://mattermost.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="users", label="Users", format="number"),
            WidgetField(key="channels", label="Channels", format="number"),
            WidgetField(key="teams", label="Teams", format="number"),
            WidgetField(key="posts_today", label="Posts Today", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from mattermost_mcp.api_client import MattermostApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = MattermostApi(base_url=url, token=token)

        try:
            users = client.get_users(per_page=1) or {}
            teams = client.get_teams() or []
            total_users = users.get("total_count", 0) if isinstance(users, dict) else len(users)
        except Exception as e:
            logger.debug("Mattermost fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "users": total_users,
                "channels": 0,
                "teams": len(teams) if isinstance(teams, list) else 0,
                "posts_today": 0,
            },
            status="ok",
        )
