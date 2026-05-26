"""GitLab widget — project, pipeline, and merge request metrics.

Uses gitlab-api Python client for data fetching.
"""

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
    service_type = "gitlab"
    display_name = "GitLab"
    icon = "gitlab"
    category = ServiceCategory.DEVOPS
    description = "Source control — projects, pipelines, and merge requests"
    env_prefix = "GITLAB"
    default_url = "https://gitlab.local.example.com"
    supports_websocket = False

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="projects", label="Projects", format="number"),
            WidgetField(
                key="open_mrs", label="Open MRs", format="number", highlight=True
            ),
            WidgetField(key="pipelines_running", label="Running", format="number"),
            WidgetField(
                key="pipelines_failed", label="Failed", format="number", highlight=True
            ),
            WidgetField(key="runners_online", label="Runners", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from gitlab_api.api_client import GitLabApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)

        client = GitLabApi(base_url=url, token=token, verify=False)

        projects = 0
        open_mrs = 0
        pipelines_running = 0
        pipelines_failed = 0
        runners_online = 0

        try:
            project_list = client.get_projects(per_page=1)
            # Use response headers or count for total
            if isinstance(project_list, list):
                # Limited fetch — get count from pagination
                projects = len(project_list)
        except Exception as e:
            logger.debug("GitLab projects fetch: %s", e)

        try:
            mrs = client.get_merge_requests(state="opened", per_page=100)
            if isinstance(mrs, list):
                open_mrs = len(mrs)
        except Exception as e:
            logger.debug("GitLab MRs fetch: %s", e)

        try:
            runners = client.get_runners(status="online")
            if isinstance(runners, list):
                runners_online = len(runners)
        except Exception as e:
            logger.debug("GitLab runners fetch: %s", e)

        return WidgetData(
            fields={
                "projects": projects,
                "open_mrs": open_mrs,
                "pipelines_running": pipelines_running,
                "pipelines_failed": pipelines_failed,
                "runners_online": runners_online,
            },
            status="ok",
        )
