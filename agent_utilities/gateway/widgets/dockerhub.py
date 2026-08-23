"""Docker Hub widget — image registry repository and tag counts."""

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
    service_type = "dockerhub"
    display_name = "Docker Hub"
    icon = "container"
    category = ServiceCategory.DEVOPS
    description = "Image registry — repositories for a namespace"
    env_prefix = "DOCKERHUB"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="repositories", label="Repositories", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("dockerhub_api.api_client", "Api")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        token = self._resolve_token(config)
        namespace = self._resolve_env(config, "namespace") or self._resolve_env(
            config, "username"
        )
        if not token or not namespace:
            return WidgetData(
                status="skipped", error="Missing Docker Hub token/namespace"
            )

        client = client_cls(
            token=token,
            tls_profile=self._resolve_tls_profile(config),
        )

        try:
            repos = client.get_repositories(namespace) or {}
        except Exception as e:
            return self._error_data(e)

        return WidgetData(
            fields={
                "repositories": count_items(repos),
                "status": "Connected",
            },
            status="ok",
        )
