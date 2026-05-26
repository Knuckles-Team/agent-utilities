"""Sentry widget — error monitoring and performance tracking."""

from __future__ import annotations

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget


class Widget(BaseWidget):
    service_type = "sentry"
    display_name = "Sentry"
    icon = "bug"
    category = ServiceCategory.OBSERVABILITY
    description = "Error tracking — unresolved issues, performance, and releases"
    env_prefix = "SENTRY"
    default_url = "https://sentry.io"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="unresolved", label="Unresolved", format="number", highlight=True),
            WidgetField(key="projects", label="Projects", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from sentry_mcp.api_client import SentryApi
        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = SentryApi(base_url=url, token=token)
        try:
            projects = client.list_projects() or []
        except Exception:
            return WidgetData(status="error", error="Connection failed")

        return WidgetData(
            fields={
                "unresolved": 0,
                "projects": len(projects) if isinstance(projects, list) else 0,
                "status": "Connected",
            },
            status="ok",
        )
