"""Zulip widget — team messaging platform."""

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
    service_type = "zulip"
    display_name = "Zulip"
    icon = "message-circle"
    category = ServiceCategory.COMMUNICATION
    description = "Team messaging — streams, topics, and threaded discussions"
    env_prefix = "ZULIP"

    def get_fields(self) -> list[WidgetField]:
        return self.get_widget_fields("zulip")

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        # zulip_agent does not exist as a distribution — not locally, not on
        # PyPI. Degrade honestly (see ear.py) instead of an unguarded import
        # that would flood the gateway log with dependency_unavailable errors.
        try:
            from zulip_agent.api_client import ZulipApi
        except ImportError:
            return WidgetData(status="skipped", error="zulip-agent not installed")

        url = self._resolve_env(config, "url")
        email = self._resolve_env(config, "email")
        api_key = self._resolve_token(config)

        if not url or not email or not api_key:
            return WidgetData(status="skipped", error="Missing Zulip url/email/key")

        client = ZulipApi(base_url=url, email=email, api_key=api_key)
        try:
            streams = client.get_streams() or {}
            stream_list = (
                streams.get("streams", []) if isinstance(streams, dict) else []
            )
        except Exception as e:
            logger.warning("Zulip fetch failed: %s", e)
            return self._error_data(e)

        return WidgetData(
            fields={"streams": len(stream_list), "unread": 0, "status": "Online"},
            status="ok",
        )
