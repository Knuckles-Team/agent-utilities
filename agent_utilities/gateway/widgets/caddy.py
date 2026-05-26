"""Caddy widget — reverse proxy and TLS certificate status.

Uses caddy-mcp API client for route and upstream health data.
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
    service_type = "caddy"
    display_name = "Caddy"
    icon = "shield-check"
    category = ServiceCategory.INFRASTRUCTURE
    description = "Reverse proxy — routes, TLS certificates, and upstream health"
    env_prefix = "CADDY"
    default_url = "https://caddy.local.example.com:2019"
    supports_websocket = False

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="routes", label="Routes", format="number"),
            WidgetField(
                key="upstreams_healthy",
                label="Healthy",
                format="number",
                highlight=True,
            ),
            WidgetField(
                key="upstreams_unhealthy",
                label="Unhealthy",
                format="number",
                highlight=True,
            ),
            WidgetField(key="tls_certificates", label="TLS Certs", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from caddy_mcp.api_client import CaddyApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)

        client = CaddyApi(base_url=url, token=token, verify=False)

        routes = 0
        upstreams_healthy = 0
        upstreams_unhealthy = 0
        tls_certificates = 0

        try:
            config_data = client.get_config()
            if isinstance(config_data, dict):
                # Count routes from server configs
                apps = config_data.get("apps", {})
                http_app = apps.get("http", {})
                servers = http_app.get("servers", {})
                for server in servers.values():
                    server_routes = server.get("routes", [])
                    routes += len(server_routes)

                # Count TLS certificates
                tls_app = apps.get("tls", {})
                auto_tls = tls_app.get("automation", {})
                policies = auto_tls.get("policies", [])
                for policy in policies:
                    subjects = policy.get("subjects", [])
                    tls_certificates += len(subjects)
        except Exception as e:
            logger.debug("Caddy config fetch: %s", e)

        try:
            upstreams = client.get_reverse_proxy_upstreams()
            if isinstance(upstreams, list):
                for u in upstreams:
                    if u.get("healthy", True):
                        upstreams_healthy += 1
                    else:
                        upstreams_unhealthy += 1
        except Exception as e:
            logger.debug("Caddy upstreams fetch: %s", e)

        return WidgetData(
            fields={
                "routes": routes,
                "upstreams_healthy": upstreams_healthy,
                "upstreams_unhealthy": upstreams_unhealthy,
                "tls_certificates": tls_certificates,
            },
            status="ok" if upstreams_unhealthy == 0 else "error",
        )
