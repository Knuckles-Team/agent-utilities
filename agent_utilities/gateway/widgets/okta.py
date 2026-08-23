"""Okta widget — identity provider user/group counts."""

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
    service_type = "okta"
    display_name = "Okta"
    icon = "key-round"
    category = ServiceCategory.SECURITY
    description = "Identity provider — managed user count"
    env_prefix = "OKTA"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="users", label="Users", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("okta_agent.api_client", "Api")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        if not url or not token:
            return WidgetData(status="skipped", error="Missing Okta url/token")

        try:
            from okta_agent.api.credentials import SswsToken
        except ImportError:
            return WidgetData(status="skipped", error="okta-agent not installed")

        client = client_cls(
            org_url=url,
            credential=SswsToken(token),
            tls_profile=self._resolve_tls_profile(config),
        )

        try:
            users = client.list_users(limit=200, max_items=200) or {}
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={
                "users": count_items(users, key="users"),
                "status": "Connected",
            },
            status="ok",
        )
