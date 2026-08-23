"""OneTrust widget — privacy/consent platform reachability."""

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
    service_type = "onetrust"
    display_name = "OneTrust"
    icon = "cookie"
    category = ServiceCategory.SECURITY
    description = "Privacy and consent platform — cookie domain inventory"
    env_prefix = "ONETRUST"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="cookie_domains", label="Cookie Domains", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("onetrust_api.api_client", "Api")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        token = self._resolve_token(config)
        client_id = self._resolve_env(config, "client_id")
        client_secret = self._resolve_env(config, "client_secret")
        if not token and not (client_id and client_secret):
            return WidgetData(status="skipped", error="Missing OneTrust credentials")

        try:
            client = client_cls(
                url=self._resolve_url(config) or None,
                token=token or None,
                client_id=client_id or None,
                client_secret=client_secret or None,
                tls_profile=self._resolve_tls_profile(config),
            )
            domains = client.domaindata() or {}
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={
                "cookie_domains": count_items(domains, key="data"),
                "status": "Connected",
            },
            status="ok",
        )
