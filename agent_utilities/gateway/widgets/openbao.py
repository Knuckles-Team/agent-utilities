"""OpenBao widget — secrets engine and vault status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "openbao"
    display_name = "OpenBao"
    icon = "lock"
    category = ServiceCategory.SECURITY
    description = "Vault — secrets engines, seal status, and health"
    env_prefix = "OPENBAO"
    default_url = "https://openbao.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="sealed", label="Sealed", format="text", highlight=True),
            WidgetField(key="mounts", label="Mounts", format="number"),
            WidgetField(key="version", label="Version", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from openbao_mcp.api_client import OpenBaoApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = OpenBaoApi(base_url=url, token=token)

        try:
            health = client.get_health() or {}
            mounts = client.get_mounts() or {}
            sealed = health.get("sealed", True)
            version = health.get("version", "unknown")
            mount_count = len(mounts) if isinstance(mounts, dict) else 0
        except Exception as e:
            logger.debug("OpenBao fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "sealed": "Yes" if sealed else "No",
                "mounts": mount_count,
                "version": version,
            },
            status="ok" if not sealed else "error",
        )
