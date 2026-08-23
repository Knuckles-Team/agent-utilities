"""Repository Manager widget — workspace and project status."""

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
        # No `repository_manager.api_client` module exists. The package's real
        # public client is `Git` (repository_manager/repository_manager.py),
        # which reads the workspace manifest (workspace.yml) rather than
        # calling a remote URL/token. `DEFAULT_WORKSPACE_YML` resolves the
        # same canonical manifest the CLI/MCP surface uses.
        from repository_manager.repository_manager import DEFAULT_WORKSPACE_YML, Git

        client = Git()
        try:
            if not client.load_projects_from_yaml(DEFAULT_WORKSPACE_YML):
                raise RuntimeError("workspace manifest not found")
            repos = client.get_workspace_projects() or []
        except Exception as e:
            logger.debug("Repository Manager fetch: %s", type(e).__name__)
            return self._error_data(e)

        return WidgetData(
            fields={
                "projects": len(repos) if isinstance(repos, list) else 0,
                "valid": 0,
                "errors": 0,
            },
            status="ok",
        )
