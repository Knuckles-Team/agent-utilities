"""Audio Transcriber widget — Whisper transcription service status."""

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
    service_type = "audio_transcriber"
    display_name = "Audio Transcriber"
    icon = "mic"
    category = ServiceCategory.PRODUCTIVITY
    description = "Whisper transcription — speech-to-text processing"
    env_prefix = "AUDIO_TRANSCRIBER"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="model", label="Model", format="text"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        return WidgetData(
            fields={"model": "whisper-large-v3", "status": "Ready"}, status="ok"
        )
