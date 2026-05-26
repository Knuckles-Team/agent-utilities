"""Widget Registry — auto-discovers and manages service widgets.

CONCEPT:GW-1.0 — Gateway Service Dashboard

Mirrors Homepage's ``widgets.js`` / ``components.js`` registry pattern
but uses Python's import mechanism for discovery. Now lives inside
``agent-utilities`` rather than the former standalone package.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from agent_utilities.gateway.models import WidgetRegistration

if TYPE_CHECKING:
    from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)

# Map of widget_type -> module path within agent_utilities.gateway.widgets
_BUILTIN_WIDGETS: dict[str, str] = {
    "portainer": "agent_utilities.gateway.widgets.portainer",
    "uptime_kuma": "agent_utilities.gateway.widgets.uptime_kuma",
    "technitium": "agent_utilities.gateway.widgets.technitium",
    "caddy": "agent_utilities.gateway.widgets.caddy",
    "gitlab": "agent_utilities.gateway.widgets.gitlab",
    "jellyfin": "agent_utilities.gateway.widgets.jellyfin",
    "qbittorrent": "agent_utilities.gateway.widgets.qbittorrent",
    "nextcloud": "agent_utilities.gateway.widgets.nextcloud",
    "home_assistant": "agent_utilities.gateway.widgets.home_assistant",
    "mealie": "agent_utilities.gateway.widgets.mealie",
    "container_manager": "agent_utilities.gateway.widgets.container_manager",
    "mattermost": "agent_utilities.gateway.widgets.mattermost",
    "keycloak": "agent_utilities.gateway.widgets.keycloak",
    "openbao": "agent_utilities.gateway.widgets.openbao",
    "langfuse": "agent_utilities.gateway.widgets.langfuse",
    "plane": "agent_utilities.gateway.widgets.plane",
    "servicenow": "agent_utilities.gateway.widgets.servicenow",
    "erpnext": "agent_utilities.gateway.widgets.erpnext",
    "wger": "agent_utilities.gateway.widgets.wger",
    "owncast": "agent_utilities.gateway.widgets.owncast",
    "github": "agent_utilities.gateway.widgets.github",
    "searxng": "agent_utilities.gateway.widgets.searxng",
    "media_downloader": "agent_utilities.gateway.widgets.media_downloader",
    "stirlingpdf": "agent_utilities.gateway.widgets.stirlingpdf",
    "microsoft": "agent_utilities.gateway.widgets.microsoft",
    "postiz": "agent_utilities.gateway.widgets.postiz",
    "archivebox": "agent_utilities.gateway.widgets.archivebox",
    "leanix": "agent_utilities.gateway.widgets.leanix",
    "listmonk": "agent_utilities.gateway.widgets.listmonk",
    "tunnel_manager": "agent_utilities.gateway.widgets.tunnel_manager",
    "repository_manager": "agent_utilities.gateway.widgets.repository_manager",
    "systems_manager": "agent_utilities.gateway.widgets.systems_manager",
    "data_science": "agent_utilities.gateway.widgets.data_science",
    "vector_db": "agent_utilities.gateway.widgets.vector_db",
    "documentdb": "agent_utilities.gateway.widgets.documentdb",
    "scholarx": "agent_utilities.gateway.widgets.scholarx",
    "audio_transcriber": "agent_utilities.gateway.widgets.audio_transcriber",
    "arr": "agent_utilities.gateway.widgets.arr",
    "ansible_tower": "agent_utilities.gateway.widgets.ansible_tower",
    "lgtm": "agent_utilities.gateway.widgets.lgtm",
    "emerald_exchange": "agent_utilities.gateway.widgets.emerald_exchange",
    "legal_peripherals": "agent_utilities.gateway.widgets.legal_peripherals",
    "twenty": "agent_utilities.gateway.widgets.twenty",
    "genius_agent": "agent_utilities.gateway.widgets.genius_agent",
    "atlassian": "agent_utilities.gateway.widgets.atlassian",
    "google_workspace": "agent_utilities.gateway.widgets.google_workspace",
    "zulip": "agent_utilities.gateway.widgets.zulip",
    "teleport": "agent_utilities.gateway.widgets.teleport",
    "sentry": "agent_utilities.gateway.widgets.sentry",
    "ollama": "agent_utilities.gateway.widgets.ollama",
}


class Registry:
    """Global registry of available service widgets.

    Provides lazy loading — widgets are only imported when first accessed,
    so frontends don't pay import cost for unused agent-packages.
    """

    def __init__(self) -> None:
        self._widgets: dict[str, type[BaseWidget]] = {}
        self._registrations: dict[str, WidgetRegistration] = {}
        self._loaded: set[str] = set()

    def discover_all(self) -> dict[str, WidgetRegistration]:
        """Attempt to load all builtin widgets and return their registrations.

        Widgets whose agent-package dependencies aren't installed are
        silently skipped (graceful degradation).
        """
        for widget_type, module_path in _BUILTIN_WIDGETS.items():
            if widget_type not in self._loaded:
                self._try_load(widget_type, module_path)
        return dict(self._registrations)

    def get_widget(self, widget_type: str) -> BaseWidget | None:
        """Get an instantiated widget by type key.

        Lazily imports the widget module on first access.
        """
        if widget_type not in self._widgets:
            module_path = _BUILTIN_WIDGETS.get(widget_type)
            if not module_path:
                logger.warning("Unknown widget type: %s", widget_type)
                return None
            self._try_load(widget_type, module_path)

        widget_cls = self._widgets.get(widget_type)
        if widget_cls:
            return widget_cls()
        return None

    def get_registration(self, widget_type: str) -> WidgetRegistration | None:
        """Get widget registration metadata without instantiating."""
        if widget_type not in self._registrations:
            self.get_widget(widget_type)  # Trigger lazy load
        return self._registrations.get(widget_type)

    def list_available(self) -> list[str]:
        """List all widget types that can be loaded (deps installed)."""
        self.discover_all()
        return list(self._registrations.keys())

    def list_all_known(self) -> list[str]:
        """List all known widget types (including those with missing deps)."""
        return list(_BUILTIN_WIDGETS.keys())

    def _try_load(self, widget_type: str, module_path: str) -> None:
        """Attempt to import a widget module. Silently skip on ImportError."""
        self._loaded.add(widget_type)
        try:
            module = importlib.import_module(module_path)
            widget_cls = getattr(module, "Widget", None)
            if widget_cls is None:
                logger.debug(
                    "Widget module %s has no 'Widget' class, skipping", module_path
                )
                return

            self._widgets[widget_type] = widget_cls

            # Build registration from widget class attributes
            instance = widget_cls()
            self._registrations[widget_type] = WidgetRegistration(
                widget_type=instance.service_type,
                display_name=instance.display_name,
                icon=instance.icon,
                category=instance.category,
                description=instance.description,
                available_fields=instance.get_fields(),
                supports_websocket=getattr(instance, "supports_websocket", False),
                env_prefix=getattr(instance, "env_prefix", ""),
                default_url=getattr(instance, "default_url", ""),
            )
            logger.debug("Loaded widget: %s (%s)", widget_type, instance.display_name)

        except ImportError as e:
            logger.debug("Widget %s skipped — missing dependency: %s", widget_type, e)
        except Exception as e:
            logger.warning(
                "Widget %s failed to load: %s", widget_type, e, exc_info=True
            )


# Global singleton
_registry: Registry | None = None


def get_registry() -> Registry:
    """Get or create the global widget registry singleton."""
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry
