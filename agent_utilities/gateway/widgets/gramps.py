"""Gramps widget — genealogy research server."""

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
    service_type = "gramps"
    display_name = "Gramps"
    icon = "trees"
    category = ServiceCategory.LIFESTYLE
    description = "Genealogy research — people and events in the family tree"
    env_prefix = "GRAMPS"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="people", label="People", format="number"),
            WidgetField(key="events", label="Events", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("gramps_mcp.api_client", "Api")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        username = self._resolve_env(config, "username")
        password = self._resolve_env(config, "password")
        if not url or not (token or (username and password)):
            return WidgetData(status="skipped", error="Missing Gramps credentials")

        try:
            client = client_cls(
                url=url,
                token=token or None,
                username=username or None,
                password=password or None,
                tls_profile=self._resolve_tls_profile(config),
            )
            people = client.get_people() or {}
        except Exception as e:
            return self._error_data(e)

        events = 0
        try:
            events = count_items(client.get_events() or {})
        except Exception as e:
            # Best-effort secondary metric: get_people() above already
            # proved the service reachable, so this only degrades one field.
            logger.warning("Gramps events fetch failed: %s", e)

        return WidgetData(
            fields={
                "people": count_items(people),
                "events": events,
            },
            status="ok",
        )
