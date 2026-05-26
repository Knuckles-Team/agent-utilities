"""Atlassian widget — Jira/Confluence integration."""

from __future__ import annotations

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget


class Widget(BaseWidget):
    service_type = "atlassian"
    display_name = "Atlassian"
    icon = "layout-list"
    category = ServiceCategory.PRODUCTIVITY
    description = "Jira & Confluence — issues, sprints, and wiki pages"
    env_prefix = "ATLASSIAN"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="open_issues", label="Open Issues", format="number", highlight=True),
            WidgetField(key="in_progress", label="In Progress", format="number"),
            WidgetField(key="wiki_pages", label="Wiki Pages", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from atlassian_agent.api_client import AtlassianApi
        url = self._resolve_url(config)
        username = self._resolve_env(config, "username")
        token = self._resolve_token(config)
        client = AtlassianApi(base_url=url, username=username, api_token=token)
        try:
            issues = client.search_issues(jql="assignee = currentUser() AND status != Done", max_results=1) or {}
            total = issues.get("total", 0) if isinstance(issues, dict) else 0
        except Exception:
            return WidgetData(status="error", error="Connection failed")

        return WidgetData(
            fields={"open_issues": total, "in_progress": 0, "wiki_pages": 0},
            status="ok",
        )
