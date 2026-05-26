"""Config Manager — YAML service configuration with XDG auto-discovery.

CONCEPT:GW-1.0 — Gateway Service Dashboard

Loads dashboard layout from ``~/.config/agent-utilities/services.yaml``
and auto-discovers available services from ``mcp_config.json``.

Uses ``agent_utilities.core.paths`` for all path resolution — no
duplicate XDG logic.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from agent_utilities.core.paths import config_dir, data_dir, mcp_config_path
from agent_utilities.gateway.models import (
    DashboardLayout,
    ServiceCategory,
    ServiceConfig,
    ServiceGroup,
)

logger = logging.getLogger(__name__)

# Map MCP server names to widget types + metadata
_MCP_TO_WIDGET: dict[str, dict[str, Any]] = {
    "portainer-agent": {
        "widget_type": "portainer",
        "name": "Portainer",
        "category": ServiceCategory.INFRASTRUCTURE,
        "icon": "container",
        "env_prefix": "PORTAINER",
    },
    "uptime-kuma-agent": {
        "widget_type": "uptime_kuma",
        "name": "Uptime Kuma",
        "category": ServiceCategory.OBSERVABILITY,
        "icon": "activity",
        "env_prefix": "UPTIME_KUMA",
    },
    "technitium-dns-mcp": {
        "widget_type": "technitium",
        "name": "Technitium DNS",
        "category": ServiceCategory.INFRASTRUCTURE,
        "icon": "globe",
        "env_prefix": "TECHNITIUM_DNS",
    },
    "caddy-mcp": {
        "widget_type": "caddy",
        "name": "Caddy",
        "category": ServiceCategory.INFRASTRUCTURE,
        "icon": "shield-check",
        "env_prefix": "CADDY",
    },
    "gitlab-api": {
        "widget_type": "gitlab",
        "name": "GitLab",
        "category": ServiceCategory.DEVOPS,
        "icon": "gitlab",
        "env_prefix": "GITLAB",
    },
    "jellyfin-mcp": {
        "widget_type": "jellyfin",
        "name": "Jellyfin",
        "category": ServiceCategory.MEDIA,
        "icon": "film",
        "env_prefix": "JELLYFIN",
    },
    "qbittorrent-agent": {
        "widget_type": "qbittorrent",
        "name": "qBittorrent",
        "category": ServiceCategory.MEDIA,
        "icon": "download",
        "env_prefix": "QBITTORRENT",
    },
    "nextcloud-agent": {
        "widget_type": "nextcloud",
        "name": "Nextcloud",
        "category": ServiceCategory.PRODUCTIVITY,
        "icon": "cloud",
        "env_prefix": "NEXTCLOUD",
    },
    "home-assistant-agent": {
        "widget_type": "home_assistant",
        "name": "Home Assistant",
        "category": ServiceCategory.INFRASTRUCTURE,
        "icon": "home",
        "env_prefix": "HOME_ASSISTANT",
    },
    "mealie-mcp": {
        "widget_type": "mealie",
        "name": "Mealie",
        "category": ServiceCategory.LIFESTYLE,
        "icon": "utensils",
        "env_prefix": "MEALIE",
    },
    "container-manager-mcp": {
        "widget_type": "container_manager",
        "name": "Container Manager",
        "category": ServiceCategory.INFRASTRUCTURE,
        "icon": "box",
        "env_prefix": "CONTAINER_MANAGER",
    },
    "mattermost-mcp": {
        "widget_type": "mattermost",
        "name": "Mattermost",
        "category": ServiceCategory.COMMUNICATION,
        "icon": "message-square",
        "env_prefix": "MATTERMOST",
    },
    "keycloak-agent": {
        "widget_type": "keycloak",
        "name": "Keycloak",
        "category": ServiceCategory.SECURITY,
        "icon": "lock",
        "env_prefix": "KEYCLOAK",
    },
    "openbao-mcp": {
        "widget_type": "openbao",
        "name": "OpenBao",
        "category": ServiceCategory.SECURITY,
        "icon": "vault",
        "env_prefix": "BAO",
    },
    "langfuse-agent": {
        "widget_type": "langfuse",
        "name": "Langfuse",
        "category": ServiceCategory.OBSERVABILITY,
        "icon": "line-chart",
        "env_prefix": "LANGFUSE",
    },
    "plane-agent": {
        "widget_type": "plane",
        "name": "Plane",
        "category": ServiceCategory.PRODUCTIVITY,
        "icon": "kanban",
        "env_prefix": "PLANE",
    },
    "servicenow-api": {
        "widget_type": "servicenow",
        "name": "ServiceNow",
        "category": ServiceCategory.BUSINESS,
        "icon": "ticket",
        "env_prefix": "SERVICENOW",
    },
    "erpnext-agent": {
        "widget_type": "erpnext",
        "name": "ERPNext",
        "category": ServiceCategory.BUSINESS,
        "icon": "building-2",
        "env_prefix": "ERPNEXT",
    },
    "wger-agent": {
        "widget_type": "wger",
        "name": "Wger",
        "category": ServiceCategory.LIFESTYLE,
        "icon": "dumbbell",
        "env_prefix": "WGER",
    },
    "owncast-agent": {
        "widget_type": "owncast",
        "name": "Owncast",
        "category": ServiceCategory.MEDIA,
        "icon": "radio",
        "env_prefix": "OWNCAST",
    },
    "legal-peripherals-mcp": {
        "widget_type": "legal_peripherals",
        "name": "Legal Peripherals",
        "category": ServiceCategory.BUSINESS,
        "icon": "scale",
        "env_prefix": "LEGAL",
    },
    "twenty-mcp": {
        "widget_type": "twenty",
        "name": "Twenty CRM",
        "category": ServiceCategory.BUSINESS,
        "icon": "users",
        "env_prefix": "TWENTY",
    },
}


def services_config_path() -> Path:
    """Path to the services configuration YAML file.

    Default: ``~/.config/agent-utilities/services.yaml``
    """
    return config_dir() / "services.yaml"


def dashboard_layout_path() -> Path:
    """Path to the persisted dashboard layout.

    Default: ``~/.local/share/agent-utilities/layout.yaml``
    """
    return data_dir() / "layout.yaml"


class ConfigManager:
    """Manages service dashboard configuration.

    Loads from YAML and can auto-discover services from mcp_config.json.
    Uses ``agent_utilities.core.paths`` for all path resolution.
    """

    def __init__(self, config_path: Path | str | None = None):
        self._config_path = Path(config_path) if config_path else services_config_path()
        self._layout: DashboardLayout | None = None

    def load(self) -> DashboardLayout:
        """Load dashboard layout from YAML config.

        If no YAML config exists, auto-discovers from mcp_config.json.
        """
        if self._config_path.exists():
            return self._load_yaml()
        return self._auto_discover()

    def save(self, layout: DashboardLayout) -> None:
        """Save current layout to YAML."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "settings": {
                "columns": layout.columns,
                "theme": layout.theme,
                "card_size": layout.card_size,
                "show_search": layout.show_search,
                "show_status_indicators": layout.show_status_indicators,
                "auto_refresh": layout.auto_refresh,
                "refresh_interval": layout.refresh_interval,
            },
            "groups": [],
        }

        for group in layout.groups:
            group_data: dict[str, Any] = {
                "name": group.name,
                "order": group.order,
                "collapsed": group.collapsed,
                "icon": group.icon,
                "services": [],
            }
            for svc in group.services:
                svc_data = svc.model_dump(mode="json", exclude_defaults=True)
                group_data["services"].append(svc_data)
            data["groups"].append(group_data)

        with open(self._config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Dashboard config saved to %s", self._config_path)

    def _load_yaml(self) -> DashboardLayout:
        """Load layout from existing YAML file."""
        with open(self._config_path) as f:
            data = yaml.safe_load(f) or {}

        settings = data.get("settings", {})
        groups_data = data.get("groups", [])

        groups = []
        for g in groups_data:
            services = []
            for s in g.get("services", []):
                services.append(ServiceConfig(**s))
            groups.append(
                ServiceGroup(
                    name=g.get("name", ""),
                    services=services,
                    order=g.get("order", 0),
                    collapsed=g.get("collapsed", False),
                    icon=g.get("icon", ""),
                )
            )

        layout = DashboardLayout(
            groups=groups,
            **{k: v for k, v in settings.items() if k in DashboardLayout.model_fields},
        )
        self._layout = layout
        return layout

    def _auto_discover(self) -> DashboardLayout:
        """Auto-discover services from mcp_config.json.

        Reads the MCP config to find configured servers and maps
        them to dashboard widgets.
        """
        mcp_path = mcp_config_path()
        if not mcp_path.exists():
            logger.info("No mcp_config.json found at %s", mcp_path)
            return DashboardLayout()

        with open(mcp_path) as f:
            mcp_config = json.load(f)

        servers = mcp_config.get("mcpServers", mcp_config.get("servers", {}))

        # Group services by category
        category_groups: dict[ServiceCategory, list[ServiceConfig]] = {}

        for server_name, server_config in servers.items():
            mapping = _MCP_TO_WIDGET.get(server_name)
            if not mapping:
                logger.debug("No widget mapping for MCP server: %s", server_name)
                continue

            # Extract URL from server config env vars or args
            env_vars = server_config.get("env", {})
            url = ""
            env_prefix = mapping.get("env_prefix", "")
            if env_prefix:
                url = env_vars.get(f"{env_prefix}_URL", "")
                if not url:
                    url = os.environ.get(f"{env_prefix}_URL", "")

            category = mapping["category"]
            svc = ServiceConfig(
                id=server_name,
                name=mapping["name"],
                widget_type=mapping["widget_type"],
                url=url,
                icon=mapping.get("icon", ""),
                category=category,
                env_prefix=env_prefix,
                href=url,
            )

            if category not in category_groups:
                category_groups[category] = []
            category_groups[category].append(svc)

        groups = []
        for idx, (cat, services) in enumerate(sorted(category_groups.items(), key=lambda x: x[0].value)):
            groups.append(
                ServiceGroup(
                    name=cat.value,
                    services=services,
                    order=idx,
                    icon=services[0].icon if services else "",
                )
            )

        layout = DashboardLayout(groups=groups)
        logger.info(
            "Auto-discovered %d services from mcp_config.json",
            sum(len(g.services) for g in groups),
        )
        return layout

    def get_all_services(self) -> list[ServiceConfig]:
        """Flatten all services from the current layout."""
        layout = self._layout or self.load()
        return [svc for group in layout.groups for svc in group.services]
