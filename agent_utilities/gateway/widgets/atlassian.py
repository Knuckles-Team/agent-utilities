"""Atlassian widget — Jira/Confluence integration."""

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
    service_type = "atlassian"
    display_name = "Atlassian"
    icon = "layout-list"
    category = ServiceCategory.PRODUCTIVITY
    description = "Jira & Confluence — issues, sprints, and wiki pages"
    env_prefix = "ATLASSIAN"

    def get_fields(self) -> list[WidgetField]:
        return self.get_widget_fields("atlassian")

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        # No `atlassian_agent.api_client` module exists. The real clients live
        # under `atlassian_agent/api/api_client_*.py` — one generated class per
        # Atlassian product/deployment (JiraCloudAPI, ConfluenceCloudAPI,
        # AdminCloudAPI, ...), each wrapping a shared `BaseAtlassianClient`
        # (atlassian_agent/api/base.py). This widget reports Jira issue counts,
        # so it drives `JiraCloudAPI`.
        from atlassian_agent.api.api_client_jira_cloud import JiraCloudAPI
        from atlassian_agent.api.base import BaseAtlassianClient

        url = self._resolve_url(config)
        username = self._resolve_env(config, "username")
        token = self._resolve_token(config)
        base_client = BaseAtlassianClient(base_url=url, username=username, token=token)
        client = JiraCloudAPI(base_client)
        try:
            response = client.jira_cloud_search_for_issues_using_jql(
                jql="assignee = currentUser() AND status != Done", max_results=1
            )
            data = response.data if isinstance(response.data, dict) else {}
            total = data.get("total", 0)
        except Exception as e:
            logger.warning("Atlassian fetch failed: %s", e)
            return self._error_data(e)

        return WidgetData(
            fields={"open_issues": total, "in_progress": 0, "wiki_pages": 0},
            status="ok",
        )
