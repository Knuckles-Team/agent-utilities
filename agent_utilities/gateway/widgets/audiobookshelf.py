"""Audiobookshelf widget — audiobook/podcast library server."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets._optional_client import count_items, import_client
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "audiobookshelf"
    display_name = "Audiobookshelf"
    icon = "book-audio"
    category = ServiceCategory.MEDIA
    description = "Audiobook and podcast library — libraries and item counts"
    env_prefix = "AUDIOBOOKSHELF"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="libraries", label="Libraries", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client(
            "audiobookshelf_mcp.api_client", "ApiClientSystem"
        )
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        if not url or not token:
            return WidgetData(
                status="skipped", error="Missing Audiobookshelf url/token"
            )

        client = client_cls(
            base_url=url,
            token=token,
            tls_profile=self._resolve_tls_profile(config),
        )

        try:
            libraries = client.get_libraries() or {}
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={
                "libraries": count_items(libraries, key="libraries"),
                "status": "Connected",
            },
            status="ok",
        )
