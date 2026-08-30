"""Container Manager widget — Docker/Podman status via container-manager-mcp."""

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
    service_type = "container_manager"
    display_name = "Container Manager"
    icon = "container"
    category = ServiceCategory.INFRASTRUCTURE
    description = "Docker/Podman — container, image, volume, and network overview"
    env_prefix = "CONTAINER_MANAGER"

    def get_fields(self) -> list[WidgetField]:
        return self.get_widget_fields("container_manager")

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        # No `container_manager_mcp.api_client` module exists — the package's
        # real public entry point is the `create_manager()` factory
        # (container_manager_mcp/container_manager.py), which auto-detects and
        # connects to the local Docker/Podman runtime and returns a
        # DockerManager/PodmanManager (both ContainerManagerBase). It raises
        # ImportError/ValueError/RuntimeError when no runtime is reachable —
        # all handled by BaseWidget._safe_fetch.
        from container_manager_mcp import create_manager

        client = create_manager()
        try:
            containers = client.list_containers(all=True) or []
            images = client.list_images() or []
            volumes = client.list_volumes() or []
            networks = client.list_networks() or []
            running = sum(
                1 for c in containers if c.get("status", "").lower() == "running"
            )
        except Exception as e:
            logger.debug("Container Manager fetch: %s", type(e).__name__)
            return self._error_data(e)

        return WidgetData(
            fields={
                "containers": len(containers),
                "running": running,
                "images": len(images),
                "volumes": len(volumes) if isinstance(volumes, list) else 0,
                "networks": len(networks),
            },
            status="ok",
        )
