"""qBittorrent widget — download client status and transfer stats."""

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
    service_type = "qbittorrent"
    display_name = "qBittorrent"
    icon = "download"
    category = ServiceCategory.MEDIA
    description = "Download client — torrents, speed, and transfer stats"
    env_prefix = "QBITTORRENT"
    default_url = "https://qbittorrent.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(
                key="downloading", label="Downloading", format="number", highlight=True
            ),
            WidgetField(key="seeding", label="Seeding", format="number"),
            WidgetField(key="paused", label="Paused", format="number"),
            WidgetField(key="dl_speed", label="↓ Speed", format="bytes", suffix="/s"),
            WidgetField(key="ul_speed", label="↑ Speed", format="bytes", suffix="/s"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from qbittorrent_agent.api_client import QBittorrentApi

        url = self._resolve_url(config)
        username = self._resolve_env(config, "username", "admin")
        password = self._resolve_env(config, "password")
        client = QBittorrentApi(base_url=url, username=username, password=password)

        try:
            torrents = client.list_torrents() or []
            transfer = client.get_transfer_info() or {}
        except Exception as e:
            logger.debug("qBittorrent fetch: %s", e)
            torrents = []
            transfer = {}

        downloading = sum(
            1 for t in torrents if t.get("state", "").startswith("download")
        )
        seeding = sum(1 for t in torrents if t.get("state", "").startswith("upload"))
        paused = sum(1 for t in torrents if "paused" in t.get("state", "").lower())

        return WidgetData(
            fields={
                "downloading": downloading,
                "seeding": seeding,
                "paused": paused,
                "dl_speed": transfer.get("dl_info_speed", 0),
                "ul_speed": transfer.get("up_info_speed", 0),
            },
            status="ok",
        )
