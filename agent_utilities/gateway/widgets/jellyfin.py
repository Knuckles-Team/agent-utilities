"""Jellyfin widget — media server status and library counts."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "jellyfin"
    display_name = "Jellyfin"
    icon = "play-circle"
    category = ServiceCategory.MEDIA
    description = "Media server — movies, TV shows, music, and live TV"
    env_prefix = "JELLYFIN"
    default_url = "https://jellyfin.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="movies", label="Movies", format="number"),
            WidgetField(key="series", label="Series", format="number"),
            WidgetField(key="episodes", label="Episodes", format="number"),
            WidgetField(key="songs", label="Songs", format="number"),
            WidgetField(key="active_streams", label="Streams", format="number", highlight=True),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from jellyfin_mcp.api_client import JellyfinApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = JellyfinApi(base_url=url, api_key=token)

        try:
            counts = client.get_item_counts()
            sessions = client.get_sessions()
            active = sum(1 for s in (sessions or []) if s.get("NowPlayingItem"))
        except Exception as e:
            logger.debug("Jellyfin fetch: %s", e)
            counts = {}
            active = 0

        return WidgetData(
            fields={
                "movies": counts.get("MovieCount", 0),
                "series": counts.get("SeriesCount", 0),
                "episodes": counts.get("EpisodeCount", 0),
                "songs": counts.get("SongCount", 0),
                "active_streams": active,
            },
            status="ok",
        )
