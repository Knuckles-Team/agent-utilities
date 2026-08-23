"""Sentry widget — error monitoring and performance tracking."""

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
    service_type = "sentry"
    display_name = "Sentry"
    icon = "bug"
    category = ServiceCategory.OBSERVABILITY
    description = "Error tracking — unresolved issues, performance, and releases"
    env_prefix = "SENTRY"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(
                key="unresolved", label="Unresolved", format="number", highlight=True
            ),
            WidgetField(key="projects", label="Projects", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        # sentry_mcp does not exist as a distribution — not locally, not on
        # PyPI. Degrade honestly (see ear.py) instead of an unguarded import
        # that would flood the gateway log with dependency_unavailable errors.
        try:
            from sentry_mcp.api_client import SentryApi
        except ImportError:
            return WidgetData(status="skipped", error="sentry-mcp not installed")

        token = self._resolve_token(config)
        url = self._resolve_env(config, "url")

        if not token or not url:
            return WidgetData(status="skipped", error="Missing Sentry token or url")

        client = SentryApi(base_url=url, token=token)
        try:
            projects = client.list_projects() or []
        except Exception as e:
            logger.warning("Sentry fetch failed: %s", e)
            return self._error_data(e)

        return WidgetData(
            fields={
                "unresolved": 0,
                "projects": len(projects) if isinstance(projects, list) else 0,
                "status": "Connected",
            },
            status="ok",
        )
