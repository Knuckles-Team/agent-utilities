"""Config Manager — YAML service configuration with XDG auto-discovery.

CONCEPT:AU-OS.config.gateway-service-dashboard — Gateway Service Dashboard

Loads dashboard layout from ``~/.config/agent-utilities/services.yaml``
and auto-discovers available services from ``mcp_config.json``.

Uses ``agent_utilities.core.paths`` for all path resolution — no
duplicate XDG logic.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from agent_utilities.core.config import setting
from agent_utilities.core.paths import config_dir, data_dir, mcp_config_path
from agent_utilities.gateway.models import (
    DashboardLayout,
    ServiceCategory,
    ServiceConfig,
    ServiceGroup,
)

logger = logging.getLogger(__name__)

_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_MCP_SERVERS = 512
_INLINE_CREDENTIAL_FIELDS = frozenset(
    {
        "api_key",
        "password",
        "public_key",
        "secret_key",
        "token",
        "username",
    }
)

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
    """Return the XDG-managed services configuration path."""
    return config_dir() / "services.yaml"


def dashboard_layout_path() -> Path:
    """Return the XDG-managed persisted dashboard layout path."""
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
        """Atomically save layout metadata; credential material is never serialized."""
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
                svc_data = svc.model_dump(
                    mode="json",
                    exclude_defaults=True,
                    exclude=_INLINE_CREDENTIAL_FIELDS,
                )
                group_data["services"].append(svc_data)
            data["groups"].append(group_data)

        rendered = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        if len(rendered.encode("utf-8")) > _MAX_CONFIG_BYTES:
            raise ValueError("dashboard configuration exceeds its size boundary")
        fd, temporary_name = tempfile.mkstemp(
            dir=self._config_path.parent,
            prefix=".services-",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            with contextlib.suppress(OSError):
                os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, self._config_path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(temporary_name)

        logger.info("Dashboard config saved")

    def _load_yaml(self) -> DashboardLayout:
        """Load layout from existing YAML file."""
        raw = self._config_path.read_bytes()
        if len(raw) > _MAX_CONFIG_BYTES:
            raise ValueError("dashboard configuration exceeds its size boundary")
        data = yaml.safe_load(raw.decode("utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("dashboard configuration must be an object")

        settings = data.get("settings", {})
        groups_data = data.get("groups", [])

        if not isinstance(settings, dict) or not isinstance(groups_data, list):
            raise ValueError("dashboard configuration has an invalid shape")

        groups = []
        for g in groups_data:
            if not isinstance(g, dict):
                raise ValueError("dashboard group entries must be objects")
            services = []
            raw_services = g.get("services", [])
            if not isinstance(raw_services, list):
                raise ValueError("dashboard services must be a list")
            for s in raw_services:
                if not isinstance(s, dict):
                    raise ValueError("dashboard service entries must be objects")
                if _INLINE_CREDENTIAL_FIELDS.intersection(s):
                    raise ValueError(
                        "persistent inline credentials are forbidden; use credential_refs"
                    )
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
            logger.info("No MCP catalog configured for dashboard discovery")
            return DashboardLayout()

        raw = mcp_path.read_bytes()
        if len(raw) > _MAX_CONFIG_BYTES:
            raise ValueError("MCP catalog exceeds its size boundary")
        mcp_config = json.loads(raw.decode("utf-8"))
        if not isinstance(mcp_config, dict):
            raise ValueError("MCP catalog must be an object")

        servers = mcp_config.get("mcpServers", mcp_config.get("servers", {}))
        if not isinstance(servers, dict) or len(servers) > _MAX_MCP_SERVERS:
            raise ValueError("MCP server catalog has an invalid shape or size")

        # Group services by category
        category_groups: dict[ServiceCategory, list[ServiceConfig]] = {}

        for server_name, server_config in servers.items():
            if not isinstance(server_name, str) or len(server_name) > 128:
                continue
            if not isinstance(server_config, dict):
                continue
            mapping = _MCP_TO_WIDGET.get(server_name)
            if not mapping:
                continue

            # Extract URL from server config env vars or args
            env_vars = server_config.get("env", {})
            if not isinstance(env_vars, dict) or len(env_vars) > 256:
                env_vars = {}
            url = ""
            env_prefix = mapping.get("env_prefix", "")
            if env_prefix:
                candidate = env_vars.get(f"{env_prefix}_URL", "")
                # Persisted catalogs are parsed literally. Runtime templates and
                # secret references stay unresolved here; environment lookup is
                # the only source of concrete discovery values.
                if isinstance(candidate, str) and not candidate.startswith(
                    ("${", "env://", "secret://", "vault://")
                ):
                    url = candidate[:8192]
                if not url:
                    url = setting(f"{env_prefix}_URL", "")

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
        for idx, (cat, services) in enumerate(
            sorted(category_groups.items(), key=lambda x: x[0].value)
        ):
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
        """Flatten all services from the current layout.

        Always re-loads from disk (CONCEPT:AU-OS.observability.no-op-without-metrics): the YAML file is the
        shared source of truth, so a ``save()`` from another gateway
        worker/replica is picked up on the next fetch instead of serving a
        stale in-memory copy forever.
        """
        return [svc for group in self.load().groups for svc in group.services]
