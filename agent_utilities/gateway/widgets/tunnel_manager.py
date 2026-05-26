"""Tunnel Manager widget — SSH tunnel and host inventory status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
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
            WidgetField(key="sessions", label="Sessions", format="number", highlight=True),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from tunnel_manager.api_client import TunnelManagerApi
        client = TunnelManagerApi()
        try:
            hosts = client.list_hosts() or []
            sessions = client.list_sessions() or []
        except Exception as e:
            logger.debug("Tunnel Manager fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "hosts": len(hosts) if isinstance(hosts, list) else 0,
                "sessions": len(sessions) if isinstance(sessions, list) else 0,
                "status": "Online",
            },
            status="ok",
        )
