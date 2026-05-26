"""Ollama widget — local LLM model serving."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "ollama"
    display_name = "Ollama"
    icon = "cpu"
    category = ServiceCategory.DATA_SCIENCE
    description = "Local LLM — models, running processes, and GPU usage"
    env_prefix = "OLLAMA"
    default_url = "http://localhost:11434"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="models", label="Models", format="number"),
            WidgetField(key="running", label="Running", format="number", highlight=True),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        import httpx
        url = self._resolve_url(config)
        try:
            resp = httpx.get(f"{url}/api/tags", timeout=5.0)
            data = resp.json() if resp.status_code == 200 else {}
            models = data.get("models", [])
            ps_resp = httpx.get(f"{url}/api/ps", timeout=5.0)
            ps = ps_resp.json() if ps_resp.status_code == 200 else {}
            running = ps.get("models", [])
        except Exception as e:
            logger.debug("Ollama fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "models": len(models),
                "running": len(running),
                "status": "Online",
            },
            status="ok",
        )
