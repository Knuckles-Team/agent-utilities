"""Technitium DNS widget — DNS server metrics and zone counts.

Replaces the deprecated adguard-home-agent with technitium-dns-mcp.
"""

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
    service_type = "technitium"
    display_name = "Technitium DNS"
    icon = "globe"
    category = ServiceCategory.INFRASTRUCTURE
    description = "DNS server — zones, queries, and blocking statistics"
    env_prefix = "TECHNITIUM_DNS"
    default_url = "https://dns.local.example.com"
    supports_websocket = False

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="total_queries", label="Queries", format="number"),
            WidgetField(key="blocked", label="Blocked", format="number", highlight=True),
            WidgetField(key="zones", label="Zones", format="number"),
            WidgetField(key="cached", label="Cached", format="number"),
            WidgetField(key="block_rate", label="Block Rate", format="percent", suffix="%"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from technitium_dns_mcp.api_client import TechnitiumDnsApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)

        client = TechnitiumDnsApi(base_url=url, token=token, verify=False)

        total_queries = 0
        blocked = 0
        zones = 0
        cached = 0
        block_rate = 0.0

        try:
            stats = client.get_stats()
            if isinstance(stats, dict):
                stats_data = stats.get("response", stats)
                total_queries = stats_data.get("totalQueries", 0)
                blocked = stats_data.get("totalBlocked", 0)
                cached = stats_data.get("totalCached", 0)
                if total_queries > 0:
                    block_rate = round((blocked / total_queries) * 100, 1)
        except Exception as e:
            logger.debug("Technitium stats fetch: %s", e)

        try:
            zones_data = client.list_zones()
            if isinstance(zones_data, dict):
                zone_list = zones_data.get("response", {}).get("zones", [])
                zones = len(zone_list) if isinstance(zone_list, list) else 0
            elif isinstance(zones_data, list):
                zones = len(zones_data)
        except Exception as e:
            logger.debug("Technitium zones fetch: %s", e)

        return WidgetData(
            fields={
                "total_queries": total_queries,
                "blocked": blocked,
                "zones": zones,
                "cached": cached,
                "block_rate": block_rate,
            },
            status="ok",
        )
