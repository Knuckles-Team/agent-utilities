"""Plane widget — project management and issue tracking."""

from __future__ import annotations

import logging
from typing import Any

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "plane"
    display_name = "Plane"
    icon = "layout-kanban"
    category = ServiceCategory.PRODUCTIVITY
    description = "Project management — issues, sprints, and kanban boards"
    env_prefix = "PLANE"
    default_url = "https://plane.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="projects", label="Projects", format="number"),
            WidgetField(
                key="open_issues", label="Open", format="number", highlight=True
            ),
            WidgetField(key="in_progress", label="In Progress", format="number"),
            WidgetField(key="completed", label="Done", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from plane_agent.api_client import PlaneApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = PlaneApi(base_url=url, token=token)

        try:
            workspaces = client.get_workspaces() or []
            projects: list[Any] = []
            if workspaces:
                ws_slug = (
                    workspaces[0].get("slug", "")
                    if isinstance(workspaces[0], dict)
                    else ""
                )
                if ws_slug:
                    projects = client.get_projects(workspace_slug=ws_slug) or []
        except Exception as e:
            logger.debug("Plane fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "projects": len(projects),
                "open_issues": 0,
                "in_progress": 0,
                "completed": 0,
            },
            status="ok",
        )
