"""Tunnel Manager widget — SSH tunnel and host inventory status."""

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
    service_type = "tunnel_manager"
    display_name = "Tunnel Manager"
    icon = "network"
    category = ServiceCategory.INFRASTRUCTURE
    description = "SSH tunnels — host inventory, active sessions, and connectivity"
    env_prefix = "TUNNEL_MANAGER"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="hosts", label="Hosts", format="number"),
            WidgetField(
                key="sessions", label="Sessions", format="number", highlight=True
            ),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        # No `tunnel_manager.api_client` module exists. The package's real
        # public client is `HostManager` (tunnel_manager/tunnel_manager.py),
        # which loads the local SSH host inventory on construction — no
        # remote URL/token. It has no session-listing API, so "sessions"
        # stays a placeholder like several other stubbed fields in this file.
        from tunnel_manager import HostManager

        client = HostManager()
        try:
            hosts = client.list_hosts() or {}
        except Exception as e:
            logger.debug("Tunnel Manager fetch: %s", type(e).__name__)
            return self._error_data(e)

        return WidgetData(
            fields={
                "hosts": len(hosts),
                "sessions": 0,
                "status": "Online",
            },
            status="ok",
        )
