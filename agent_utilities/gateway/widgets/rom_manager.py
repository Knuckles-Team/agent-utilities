"""ROM Manager widget — RomM remote game-library server.

Targets the RomM *remote library server* client (``rom_manager.romm.api``),
not the local ROM-conversion facade (which wraps on-disk file operations and
has no reachable service to dashboard).
"""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets._optional_client import import_client
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


def _stat(stats: dict, *names: str) -> int:
    """Case-insensitive lookup across RomM's ``/api/stats`` key spelling."""
    lowered = {str(k).lower(): v for k, v in stats.items()}
    for name in names:
        value = lowered.get(name.lower())
        if isinstance(value, int):
            return value
    return 0


class Widget(BaseWidget):
    service_type = "rom_manager"
    display_name = "ROM Manager"
    icon = "gamepad-2"
    category = ServiceCategory.MEDIA
    description = "RomM game library — platforms and ROM counts"
    env_prefix = "ROM_MANAGER"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="platforms", label="Platforms", format="number"),
            WidgetField(key="roms", label="ROMs", format="number", highlight=True),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("rom_manager.romm.api", "RommApi")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        username = self._resolve_env(config, "username")
        password = self._resolve_env(config, "password")
        if not url or not (token or (username and password)):
            return WidgetData(status="skipped", error="Missing ROM Manager credentials")

        try:
            client = client_cls(
                url=url,
                token=token or None,
                username=username or None,
                password=password or None,
                tls_profile=self._resolve_tls_profile(config),
            )
            stats = client.stats() or {}
        except Exception as e:
            return self._error_data(e)

        if not isinstance(stats, dict):
            stats = {}

        return WidgetData(
            fields={
                "platforms": _stat(stats, "PLATFORMS", "platforms"),
                "roms": _stat(stats, "ROMS", "roms"),
            },
            status="ok",
        )
