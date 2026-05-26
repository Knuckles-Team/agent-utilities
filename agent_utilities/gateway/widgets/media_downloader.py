"""Media Downloader widget — yt-dlp download queue status."""

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
    service_type = "media_downloader"
    display_name = "Media Downloader"
    icon = "download-cloud"
    category = ServiceCategory.MEDIA
    description = "Media downloader — yt-dlp queue and completed downloads"
    env_prefix = "MEDIA_DOWNLOADER"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="queued", label="Queued", format="number"),
            WidgetField(key="completed", label="Done", format="number"),
            WidgetField(key="failed", label="Failed", format="number", highlight=True),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from media_downloader.api_client import MediaDownloaderApi

        client = MediaDownloaderApi()
        try:
            status = client.get_status() or {}
        except Exception as e:
            logger.debug("Media Downloader fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "queued": status.get("queued", 0),
                "completed": status.get("completed", 0),
                "failed": status.get("failed", 0),
            },
            status="ok",
        )
