"""Portainer widget — container management dashboard metrics.

Mirrors Homepage's portainer widget (running/stopped/stacks/volumes) but uses
the portainer-agent Python API client directly instead of HTTP proxy.
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
    service_type = "portainer"
    display_name = "Portainer"
    icon = "container"
    category = ServiceCategory.INFRASTRUCTURE
    description = "Container management — Docker environments, stacks, and services"
    env_prefix = "PORTAINER"
    default_url = "https://portainer.local.example.com"
    supports_websocket = True

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="running", label="Running", format="number", highlight=True),
            WidgetField(key="stopped", label="Stopped", format="number", highlight=True),
            WidgetField(key="stacks", label="Stacks", format="number"),
            WidgetField(key="volumes", label="Volumes", format="number"),
            WidgetField(key="images", label="Images", format="number"),
            WidgetField(key="environments", label="Environments", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from portainer_agent.api_client import PortainerApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)

        client = PortainerApi(base_url=url, token=token, verify=False)

        # Fetch endpoints (environments)
        try:
            endpoints = client.get_endpoints()
            env_count = len(endpoints) if isinstance(endpoints, list) else 0
        except Exception:
            endpoints = []
            env_count = 0

        running = 0
        stopped = 0
        stacks_count = 0
        volumes_count = 0
        images_count = 0

        # Aggregate across all environments
        try:
            docker_info = client.get_docker_dashboard(environment_id=1)
            if isinstance(docker_info, dict):
                containers = docker_info.get("containers", {})
                running = containers.get("running", 0)
                stopped = containers.get("stopped", 0)
                stacks_count = docker_info.get("stacks", 0)
                volumes_count = docker_info.get("volumes", 0)
                images_count = docker_info.get("images", {}).get("total", 0)
        except Exception as e:
            logger.debug("Portainer dashboard fetch: %s", e)
            # Fallback: try listing containers directly
            try:
                containers_list = client.docker_list_containers(
                    environment_id=1, all_containers=True
                )
                if isinstance(containers_list, list):
                    for c in containers_list:
                        state = c.get("State", "").lower()
                        if state == "running":
                            running += 1
                        else:
                            stopped += 1
            except Exception:
                pass

        return WidgetData(
            fields={
                "running": running,
                "stopped": stopped,
                "stacks": stacks_count,
                "volumes": volumes_count,
                "images": images_count,
                "environments": env_count,
            },
            status="ok",
        )
