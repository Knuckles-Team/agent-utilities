"""GitHub widget — repository and workflow status."""

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
    service_type = "github"
    display_name = "GitHub"
    icon = "github"
    category = ServiceCategory.DEVOPS
    description = "Repositories — pull requests, issues, and workflow runs"
    env_prefix = "GITHUB"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="repos", label="Repos", format="number"),
            WidgetField(
                key="open_prs", label="Open PRs", format="number", highlight=True
            ),
            WidgetField(key="open_issues", label="Issues", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from github_agent.api_client import GitHubApi

        token = self._resolve_token(config)
        client = GitHubApi(token=token)

        try:
            client.get_authenticated_user() or {}
            repos = client.list_repos() or []
        except Exception as e:
            logger.debug("GitHub fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "repos": len(repos) if isinstance(repos, list) else 0,
                "open_prs": 0,
                "open_issues": 0,
            },
            status="ok",
        )
