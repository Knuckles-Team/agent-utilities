"""Zulip widget — team messaging platform."""

from __future__ import annotations

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets.base import BaseWidget


class Widget(BaseWidget):
    service_type = "zulip"
    display_name = "Zulip"
    icon = "message-circle"
    category = ServiceCategory.COMMUNICATION
    description = "Team messaging — streams, topics, and threaded discussions"
    env_prefix = "ZULIP"
    default_url = "https://zulip.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="streams", label="Streams", format="number"),
            WidgetField(key="unread", label="Unread", format="number", highlight=True),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from zulip_agent.api_client import ZulipApi

        url = self._resolve_url(config)
        email = self._resolve_env(config, "email")
        api_key = self._resolve_token(config)
        client = ZulipApi(base_url=url, email=email, api_key=api_key)
        try:
            streams = client.get_streams() or {}
            stream_list = (
                streams.get("streams", []) if isinstance(streams, dict) else []
            )
        except Exception:
            return WidgetData(status="error", error="Connection failed")

        return WidgetData(
            fields={"streams": len(stream_list), "unread": 0, "status": "Online"},
            status="ok",
        )
