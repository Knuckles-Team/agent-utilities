"""Base widget abstract class — all service widgets inherit from this.

CONCEPT:OS-5.9 — Gateway Service Dashboard

Mirrors Homepage's widget.js pattern: each widget defines its fields,
environment variable prefix, and a ``fetch_data()`` method that returns
structured WidgetData.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from agent_utilities.core.config import setting
from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)

logger = logging.getLogger(__name__)


class BaseWidget(ABC):
    """Abstract base class for all dashboard service widgets.

    Each subclass must define:
        - service_type: str — unique key (e.g. 'portainer')
        - display_name: str — human-readable name
        - icon: str — Lucide icon name or URL
        - category: ServiceCategory
        - description: str
        - env_prefix: str — for auto-resolving credentials from env vars
        - default_url: str — default service URL for auto-discovery
        - get_fields() -> list[WidgetField]
        - fetch_data(config) -> WidgetData
    """

    service_type: str = ""
    display_name: str = ""
    icon: str = ""
    category: ServiceCategory = ServiceCategory.CUSTOM
    description: str = ""
    env_prefix: str = ""
    default_url: str = ""
    supports_websocket: bool = False

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

        Priority: config attribute -> env var -> default

        Args:
            config: ServiceConfig instance
            key: Lowercase key name (e.g. 'url', 'token', 'api_key')
            default: Fallback value
        """
        # Check config object first
        config_val = getattr(config, key, "")
        if config_val:
            return config_val

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
        return self._resolve_env(config, "url", self.default_url)

    def _resolve_token(self, config: ServiceConfig) -> str:
        """Resolve the API token/key from config or env vars."""
        return (
            self._resolve_env(config, "api_key")
            or self._resolve_env(config, "token")
            or ""
        )

    def _safe_fetch(self, config: ServiceConfig) -> WidgetData:
        """Wrap fetch_data with error handling."""
        try:
            return self.fetch_data(config)
        except ImportError as e:
            logger.warning("Widget %s: missing dependency — %s", self.service_type, e)
            return WidgetData(
                status="error",
                error=f"Missing dependency: {e}",
            )
        except Exception as e:
            logger.error(
                "Widget %s: fetch failed — %s", self.service_type, e, exc_info=True
            )
            return WidgetData(
                status="error",
                error=str(e),
            )

    def check_health(self, config: ServiceConfig) -> bool:
        """Quick health check — try to reach the service.

        Returns True if the service responds, False otherwise.
        """
        try:
            import httpx

            url = self._resolve_url(config)
            if not url:
                return False
            resp = httpx.get(url, timeout=5.0, verify=False)  # nosec B501
            return resp.status_code < 500
        except Exception:
            return False
