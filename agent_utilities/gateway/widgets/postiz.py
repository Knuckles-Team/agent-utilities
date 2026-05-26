"""Postiz widget — social media scheduling status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "postiz"
    display_name = "Postiz"
    icon = "share-2"
    category = ServiceCategory.COMMUNICATION
    description = "Social media scheduler — posts, integrations, and analytics"
    env_prefix = "POSTIZ"
    default_url = "https://postiz.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="scheduled", label="Scheduled", format="number"),
            WidgetField(key="published", label="Published", format="number"),
            WidgetField(key="integrations", label="Integrations", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from postiz_agent.api_client import PostizApi
        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = PostizApi(base_url=url, token=token)
        try:
            integrations = client.get_integrations() or []
        except Exception as e:
            logger.debug("Postiz fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "scheduled": 0,
                "published": 0,
                "integrations": len(integrations) if isinstance(integrations, list) else 0,
            },
            status="ok",
        )
