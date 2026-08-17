"""Ollama widget — local LLM model serving."""

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
    service_type = "ollama"
    display_name = "Ollama"
    icon = "cpu"
    category = ServiceCategory.DATA_SCIENCE
    description = "Local LLM — models, running processes, and GPU usage"
    env_prefix = "OLLAMA"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="models", label="Models", format="number"),
            WidgetField(
                key="running", label="Running", format="number", highlight=True
            ),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        # GOC-87 staged httpx -> httpx2 migration: this widget's two
        # unauthenticated, unpinned, non-streaming diagnostics GETs are the
        # first call family ported to the httpx2-backed adapter (see
        # agent_utilities.httpsupport.transport_factory.MIGRATED_HTTPX2_FAMILIES
        # for why this family qualifies as low-risk).
        from agent_utilities.httpsupport.transport_factory import create_http_client

        url = self._resolve_url(config)
        client = create_http_client(family="gateway-widget-diagnostics", timeout=5.0)
        try:
            resp = client.request("GET", f"{url}/api/tags")
            data = resp.json() if resp.status_code == 200 else {}
            models = data.get("models", [])
            ps_resp = client.request("GET", f"{url}/api/ps")
            ps = ps_resp.json() if ps_resp.status_code == 200 else {}
            running = ps.get("models", [])
        except Exception as e:
            logger.debug("Ollama fetch: %s", type(e).__name__)
            return self._error_data(e)
        finally:
            client.close()

        return WidgetData(
            fields={
                "models": len(models),
                "running": len(running),
                "status": "Online",
            },
            status="ok",
        )
