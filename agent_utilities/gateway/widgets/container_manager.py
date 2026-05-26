"""Container Manager widget — Docker/Podman status via container-manager-mcp."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
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
        return [
            WidgetField(key="containers", label="Containers", format="number"),
            WidgetField(key="running", label="Running", format="number", highlight=True),
            WidgetField(key="images", label="Images", format="number"),
            WidgetField(key="volumes", label="Volumes", format="number"),
            WidgetField(key="networks", label="Networks", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from container_manager_mcp.api_client import ContainerManagerApi

        client = ContainerManagerApi()
        try:
            containers = client.list_containers(all_containers=True) or []
            images = client.list_images() or []
            volumes = client.list_volumes() or []
            networks = client.list_networks() or []
            running = sum(1 for c in containers if c.get("State", "").lower() == "running")
        except Exception as e:
            logger.debug("Container Manager fetch: %s", e)
            return WidgetData(status="error", error=str(e))

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
