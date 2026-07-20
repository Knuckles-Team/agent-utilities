"""Langfuse widget — LLM observability and tracing status."""

from __future__ import annotations

import logging

from agent_utilities.core.config import resolve_langfuse_host
from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "langfuse"
    display_name = "Langfuse"
    icon = "eye"
    category = ServiceCategory.OBSERVABILITY
    description = "LLM observability — traces, sessions, and scoring"
    env_prefix = "LANGFUSE"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="traces", label="Traces", format="number"),
            WidgetField(key="sessions", label="Sessions", format="number"),
            WidgetField(key="projects", label="Projects", format="number"),
            WidgetField(key="status", label="Status", format="text", highlight=True),
        ]

    def _resolve_url(self, config: ServiceConfig) -> str:
        """Prefer an explicit widget URL, then the shared Langfuse host policy."""
        resolved = getattr(config, "url", "") or resolve_langfuse_host("")
        if not resolved:
            raise RuntimeError("service URL is not configured")
        return resolved

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        url = self._resolve_url(config)
        public_key = self._resolve_env(config, "public_key")
        secret_key = self._resolve_env(config, "secret_key")

        try:
            from langfuse_agent.api_client import LangfuseApi

            client = LangfuseApi(
                base_url=url, public_key=public_key, secret_key=secret_key
            )
            health = client.health() or {}
            traces = client.get_traces(limit=1) or {}
            total_traces = (
                traces.get("totalItems", 0) if isinstance(traces, dict) else 0
            )
            status_text = (
                health.get("status", "unknown")
                if isinstance(health, dict)
                else "unknown"
            )
        except Exception as e:  # noqa: BLE001 — status widgets must degrade cleanly
            logger.debug("Langfuse fetch failed (%s).", type(e).__name__)
            return WidgetData(status="error", error="Langfuse request failed")

        return WidgetData(
            fields={
                "traces": total_traces,
                "sessions": 0,
                "projects": 0,
                "status": status_text,
            },
            status="ok" if status_text == "OK" else "unknown",
        )
