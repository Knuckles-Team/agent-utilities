"""Ansible Tower widget — AWX/Tower automation status."""

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
    service_type = "ansible_tower"
    display_name = "Ansible Tower"
    icon = "terminal-square"
    category = ServiceCategory.DEVOPS
    description = "Automation — playbooks, job templates, and inventory management"
    env_prefix = "ANSIBLE_TOWER"
    default_url = "https://awx.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="templates", label="Templates", format="number"),
            WidgetField(
                key="running_jobs", label="Running", format="number", highlight=True
            ),
            WidgetField(
                key="failed_jobs", label="Failed", format="number", highlight=True
            ),
            WidgetField(key="hosts", label="Hosts", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from ansible_tower_mcp.api_client import AnsibleTowerApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = AnsibleTowerApi(base_url=url, token=token)
        try:
            templates = client.list_job_templates() or []
            jobs = client.list_jobs(status="running") or []
            failed = client.list_jobs(status="failed") or []
            inventories = client.list_inventories() or []
            hosts = sum(
                i.get("total_hosts", 0) for i in inventories if isinstance(i, dict)
            )
        except Exception as e:
            logger.debug("Ansible Tower fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "templates": len(templates) if isinstance(templates, list) else 0,
                "running_jobs": len(jobs) if isinstance(jobs, list) else 0,
                "failed_jobs": len(failed) if isinstance(failed, list) else 0,
                "hosts": hosts,
            },
            status="ok",
        )
