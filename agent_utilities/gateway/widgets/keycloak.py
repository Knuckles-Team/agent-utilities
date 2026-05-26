"""Keycloak widget — identity and access management status."""

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
    service_type = "keycloak"
    display_name = "Keycloak"
    icon = "shield-check"
    category = ServiceCategory.SECURITY
    description = "IAM — realms, users, clients, and SSO sessions"
    env_prefix = "KEYCLOAK"
    default_url = "https://keycloak.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="realms", label="Realms", format="number"),
            WidgetField(key="users", label="Users", format="number"),
            WidgetField(key="clients", label="Clients", format="number"),
            WidgetField(
                key="sessions", label="Sessions", format="number", highlight=True
            ),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from keycloak_agent.api_client import KeycloakApi

        url = self._resolve_url(config)
        username = self._resolve_env(config, "username", "admin")
        password = self._resolve_env(config, "password")
        client = KeycloakApi(base_url=url, username=username, password=password)

        try:
            realms = client.get_realms() or []
            users = client.get_users(realm="master") or []
            clients = client.get_clients(realm="master") or []
            sessions = client.get_sessions(realm="master") or []
        except Exception as e:
            logger.debug("Keycloak fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "realms": len(realms),
                "users": len(users),
                "clients": len(clients),
                "sessions": len(sessions),
            },
            status="ok",
        )
