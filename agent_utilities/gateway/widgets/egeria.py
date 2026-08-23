"""Egeria widget — open metadata and governance platform."""

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
    service_type = "egeria"
    display_name = "Egeria"
    icon = "network"
    category = ServiceCategory.BUSINESS
    description = "Open metadata governance — assets and glossary terms"
    env_prefix = "EGERIA"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="assets", label="Assets", format="number"),
            WidgetField(key="glossary_terms", label="Glossary Terms", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("egeria_mcp.api_client", "EgeriaApi")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        user_id = self._resolve_env(config, "username")
        user_pwd = self._resolve_env(config, "password") or self._resolve_token(config)
        view_server = self._resolve_env(config, "view_server", "view-server")
        if not url or not user_id or not user_pwd:
            return WidgetData(status="skipped", error="Missing Egeria url/credentials")

        try:
            client = client_cls(
                platform_url=url,
                view_server=view_server,
                user_id=user_id,
                user_pwd=user_pwd,
                tls_profile=self._resolve_tls_profile(config),
            )
            assets = client.list_assets() or []
        except Exception as e:
            return self._error_data(e)

        glossary_terms = 0
        try:
            glossary_terms = count_items(client.list_glossary_terms() or [])
        except Exception as e:
            # Best-effort secondary metric: list_assets() above already
            # proved the service reachable, so this only degrades one field.
            logger.warning("Egeria glossary-terms fetch failed: %s", e)

        return WidgetData(
            fields={
                "assets": count_items(assets),
                "glossary_terms": glossary_terms,
            },
            status="ok",
        )
