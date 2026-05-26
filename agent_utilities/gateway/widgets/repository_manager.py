"""Repository Manager widget — workspace and project status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "repository_manager"
    display_name = "Repository Manager"
    icon = "git-branch"
    category = ServiceCategory.DEVOPS
    description = "Workspace — projects, validation, and build status"
    env_prefix = "REPOSITORY_MANAGER"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="projects", label="Projects", format="number"),
            WidgetField(key="valid", label="Valid", format="number", highlight=True),
            WidgetField(key="errors", label="Errors", format="number", highlight=True),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from repository_manager.api_client import RepositoryManagerApi
        client = RepositoryManagerApi()
        try:
            repos = client.list_repositories() or []
        except Exception as e:
            logger.debug("Repository Manager fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={"projects": len(repos) if isinstance(repos, list) else 0, "valid": 0, "errors": 0},
            status="ok",
        )
