"""Listmonk widget — newsletter and mailing list management."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "listmonk"
    display_name = "Listmonk"
    icon = "mail"
    category = ServiceCategory.COMMUNICATION
    description = "Newsletter manager — subscribers, lists, and campaigns"
    env_prefix = "LISTMONK"
    default_url = "https://listmonk.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="subscribers", label="Subscribers", format="number"),
            WidgetField(key="lists", label="Lists", format="number"),
            WidgetField(key="campaigns", label="Campaigns", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from listmonk_api.api_client import ListmonkApi
        url = self._resolve_url(config)
        username = self._resolve_env(config, "username", "admin")
        password = self._resolve_env(config, "password")
        client = ListmonkApi(base_url=url, username=username, password=password)
        try:
            subscribers = client.get_subscribers(page=1, per_page=1) or {}
            lists = client.get_lists() or []
            campaigns = client.get_campaigns() or []
            total_subs = subscribers.get("total", 0) if isinstance(subscribers, dict) else 0
        except Exception as e:
            logger.debug("Listmonk fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "subscribers": total_subs,
                "lists": len(lists) if isinstance(lists, list) else 0,
                "campaigns": len(campaigns) if isinstance(campaigns, list) else 0,
            },
            status="ok",
        )
