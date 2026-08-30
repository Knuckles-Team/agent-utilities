"""Base widget abstract class — all service widgets inherit from this.

CONCEPT:AU-OS.config.gateway-service-dashboard — Gateway Service Dashboard

Mirrors Homepage's widget.js pattern: each widget defines its fields,
environment variable prefix, and a ``fetch_data()`` method that returns
structured WidgetData.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.security.error_surface import (
    PUBLIC_ERROR_MESSAGES,
    public_error_payload,
)

logger = logging.getLogger(__name__)


_WIDGET_FIELD_SPECS: dict[str, tuple[dict[str, Any], ...]] = {
    "ansible_tower": (
        {"key": "templates", "label": "Templates"},
        {"key": "running_jobs", "label": "Running", "highlight": True},
        {"key": "failed_jobs", "label": "Failed", "highlight": True},
        {"key": "hosts", "label": "Hosts"},
    ),
    "arr": (
        {"key": "monitored", "label": "Monitored"},
        {"key": "missing", "label": "Missing", "highlight": True},
        {"key": "queued", "label": "Queued"},
        {"key": "indexers", "label": "Indexers"},
    ),
    "atlassian": (
        {"key": "open_issues", "label": "Open Issues", "highlight": True},
        {"key": "in_progress", "label": "In Progress"},
        {"key": "wiki_pages", "label": "Wiki Pages"},
    ),
    "container_manager": (
        {"key": "containers", "label": "Containers"},
        {"key": "running", "label": "Running", "highlight": True},
        {"key": "images", "label": "Images"},
        {"key": "volumes", "label": "Volumes"},
        {"key": "networks", "label": "Networks"},
    ),
    "github": (
        {"key": "repos", "label": "Repos"},
        {"key": "open_prs", "label": "Open PRs", "highlight": True},
        {"key": "open_issues", "label": "Issues"},
    ),
    "gitlab": (
        {"key": "projects", "label": "Projects"},
        {"key": "open_mrs", "label": "Open MRs", "highlight": True},
        {"key": "pipelines_running", "label": "Running"},
        {"key": "pipelines_failed", "label": "Failed", "highlight": True},
        {"key": "runners_online", "label": "Runners"},
    ),
    "google_workspace": (
        {"key": "unread_emails", "label": "Unread", "highlight": True},
        {"key": "events_today", "label": "Events"},
        {"key": "drive_files", "label": "Files"},
    ),
    "home_assistant": (
        {"key": "entities", "label": "Entities"},
        {"key": "lights_on", "label": "Lights On", "highlight": True},
        {"key": "automations", "label": "Automations"},
        {"key": "switches_on", "label": "Switches"},
    ),
    "legal_peripherals": (
        {"key": "entities", "label": "Entities"},
        {"key": "pending", "label": "Pending", "highlight": True},
        {"key": "status", "label": "Status", "format": "text"},
    ),
    "lgtm": (
        {"key": "dashboards", "label": "Dashboards"},
        {"key": "alerts_firing", "label": "Firing", "highlight": True},
        {"key": "datasources", "label": "Sources"},
    ),
    "microsoft": (
        {"key": "unread_emails", "label": "Unread", "highlight": True},
        {"key": "events_today", "label": "Events"},
        {"key": "status", "label": "Status", "format": "text"},
    ),
    "ollama": (
        {"key": "models", "label": "Models"},
        {"key": "running", "label": "Running", "highlight": True},
        {"key": "status", "label": "Status", "format": "text"},
    ),
    "openbao": (
        {"key": "sealed", "label": "Sealed", "format": "text", "highlight": True},
        {"key": "mounts", "label": "Mounts"},
        {"key": "version", "label": "Version", "format": "text"},
    ),
    "owncast": (
        {"key": "live", "label": "Live", "format": "text", "highlight": True},
        {"key": "viewers", "label": "Viewers"},
        {"key": "peak", "label": "Peak"},
    ),
    "plane": (
        {"key": "projects", "label": "Projects"},
        {"key": "open_issues", "label": "Open", "highlight": True},
        {"key": "in_progress", "label": "In Progress"},
        {"key": "completed", "label": "Done"},
    ),
    "portainer": (
        {"key": "running", "label": "Running", "highlight": True},
        {"key": "stopped", "label": "Stopped", "highlight": True},
        {"key": "stacks", "label": "Stacks"},
        {"key": "volumes", "label": "Volumes"},
        {"key": "images", "label": "Images"},
        {"key": "environments", "label": "Environments"},
    ),
    "qbittorrent": (
        {"key": "downloading", "label": "Downloading", "highlight": True},
        {"key": "seeding", "label": "Seeding"},
        {"key": "paused", "label": "Paused"},
        {"key": "dl_speed", "label": "↓ Speed", "format": "bytes", "suffix": "/s"},
        {"key": "ul_speed", "label": "↑ Speed", "format": "bytes", "suffix": "/s"},
    ),
    "repository_manager": (
        {"key": "projects", "label": "Projects"},
        {"key": "valid", "label": "Valid", "highlight": True},
        {"key": "errors", "label": "Errors", "highlight": True},
    ),
    "sentry": (
        {"key": "unresolved", "label": "Unresolved", "highlight": True},
        {"key": "projects", "label": "Projects"},
        {"key": "status", "label": "Status", "format": "text"},
    ),
    "servicenow": (
        {"key": "open_incidents", "label": "Incidents", "highlight": True},
        {"key": "open_changes", "label": "Changes"},
        {"key": "open_requests", "label": "Requests"},
    ),
    "technitium": (
        {"key": "total_queries", "label": "Queries"},
        {"key": "blocked", "label": "Blocked", "highlight": True},
        {"key": "zones", "label": "Zones"},
        {"key": "cached", "label": "Cached"},
        {
            "key": "block_rate",
            "label": "Block Rate",
            "format": "percent",
            "suffix": "%",
        },
    ),
    "teleport": (
        {"key": "nodes", "label": "Nodes"},
        {"key": "sessions", "label": "Sessions", "highlight": True},
        {"key": "status", "label": "Status", "format": "text"},
    ),
    "tunnel_manager": (
        {"key": "hosts", "label": "Hosts"},
        {"key": "sessions", "label": "Sessions", "highlight": True},
        {"key": "status", "label": "Status", "format": "text"},
    ),
    "uptime_kuma": (
        {"key": "up", "label": "Up", "highlight": True},
        {"key": "down", "label": "Down", "highlight": True},
        {"key": "pending", "label": "Pending"},
        {"key": "maintenance", "label": "Maintenance"},
        {"key": "total", "label": "Total"},
    ),
    "zulip": (
        {"key": "streams", "label": "Streams"},
        {"key": "unread", "label": "Unread", "highlight": True},
        {"key": "status", "label": "Status", "format": "text"},
    ),
}


class BaseWidget(ABC):
    """Abstract base class for all dashboard service widgets.

    Each subclass must define:
        - service_type: str — unique key (e.g. 'portainer')
        - display_name: str — human-readable name
        - icon: str — Lucide icon name or URL
        - category: ServiceCategory
        - description: str
        - env_prefix: str — for auto-resolving credentials from env vars
        - get_fields() -> list[WidgetField]
        - fetch_data(config) -> WidgetData
    """

    service_type: str = ""
    display_name: str = ""
    icon: str = ""
    category: ServiceCategory = ServiceCategory.CUSTOM
    description: str = ""
    env_prefix: str = ""
    supports_websocket: bool = False

    @staticmethod
    def get_widget_fields(service_type: str) -> list[WidgetField]:
        """Build fresh field metadata for a widget service type."""

        return [
            WidgetField.model_validate(spec)
            for spec in _WIDGET_FIELD_SPECS[service_type]
        ]

    @abstractmethod
    def get_fields(self) -> list[WidgetField]:
        """Return the list of metric fields this widget can display.

        These are used by the frontend to render Block components.
        """
        ...

    @abstractmethod
    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        """Fetch live data from the service.

        Args:
            config: Service configuration including URL, credentials, etc.

        Returns:
            WidgetData with populated fields and status.
        """
        ...

    def _resolve_env(self, config: ServiceConfig, key: str, default: str = "") -> str:
        """Resolve a configuration value from config or environment variables.

        Priority: ephemeral config attribute -> secret reference -> env var -> default

        Args:
            config: ServiceConfig instance
            key: Lowercase key name (e.g. 'url', 'token', 'api_key')
            default: Fallback value
        """
        # Check config object first
        config_val = getattr(config, key, "")
        if config_val:
            return config_val

        # Durable configuration stores references, never credential material.
        # Resolution happens at the last possible runtime boundary so neither
        # API responses nor YAML serialization can expose the resolved value.
        reference = config.credential_refs.get(key.lower())
        if reference:
            from agent_utilities.security.secrets_client import create_secrets_client

            resolved = create_secrets_client().resolve_ref(reference)
            if not resolved:
                raise RuntimeError("configured credential reference did not resolve")
            return resolved

        # Build env var name from prefix: PORTAINER_URL, PORTAINER_TOKEN, etc.
        prefix = config.env_prefix or self.env_prefix
        if prefix:
            env_key = f"{prefix}_{key.upper()}"
            env_val = setting(env_key, "")
            if env_val:
                return env_val

        return default

    def _resolve_url(self, config: ServiceConfig) -> str:
        """Resolve the service URL from config or env vars."""
        url = self._resolve_env(config, "url")
        if not url:
            raise RuntimeError("service URL is not configured")
        return url

    def _resolve_token(self, config: ServiceConfig) -> str:
        """Resolve the API token/key from config or env vars."""
        return (
            self._resolve_env(config, "api_key")
            or self._resolve_env(config, "token")
            or ""
        )

    def _error_data(
        self, exc: BaseException, *, code: str = "dependency_unavailable"
    ) -> WidgetData:
        """Return a correlation-safe widget failure without transport details."""

        payload = public_error_payload(exc, logger=logger, code=code)
        operation_error = payload["error"]
        return WidgetData(
            status="error",
            error=PUBLIC_ERROR_MESSAGES[operation_error["code"]],
            raw=payload,
        )

    def _resolve_tls_profile(self, config: ServiceConfig) -> Any:
        """Resolve this widget's runtime-only trust policy through AgentConfig."""

        from agent_utilities.core.transport_security import (
            resolve_configured_tls_profile,
        )

        profile_ref = config.credential_refs.get(
            "tls_profile_ref"
        ) or config.credential_refs.get("tls_profile")
        return resolve_configured_tls_profile(
            self.env_prefix or self.service_type,
            profile_ref=profile_ref,
        )

    @contextmanager
    def _http_client(
        self,
        config: ServiceConfig,
        *,
        timeout: float = 5.0,
        headers: dict[str, str] | None = None,
    ) -> Iterator[Any]:
        """Yield the canonical HTTPX client with this widget's TLS profile."""

        from agent_utilities.core.http_client import create_http_client

        trust = self._resolve_tls_profile(config)
        try:
            with create_http_client(
                timeout=timeout,
                headers=headers,
                **trust.httpx_kwargs(),
            ) as client:
                yield client
        finally:
            trust.cleanup()

    def _requests_tls_verify(self, config: ServiceConfig) -> bool | str:
        """Return verified Requests-style trust for a fleet connector client.

        Runtime-materialized CA files remain live for the connector client's
        process lifetime and are removed by the transport-security cleanup hook.
        """
        trust = self._resolve_tls_profile(config)
        return trust.requests_kwargs()["verify"]

    def _safe_fetch(self, config: ServiceConfig) -> WidgetData:
        """Wrap fetch_data with error handling."""
        try:
            return self.fetch_data(config)
        except ImportError as exc:
            return self._error_data(exc)
        except Exception as exc:
            return self._error_data(exc)

    def check_health(self, config: ServiceConfig) -> bool:
        """Quick health check — try to reach the service.

        Returns True if the service responds, False otherwise.
        """
        try:
            url = self._resolve_url(config)
            if not url:
                return False
            with self._http_client(config, timeout=5.0) as client:
                resp = client.get(url)
            return resp.status_code < 500
        except Exception:
            return False
