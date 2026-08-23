"""Kafka widget — Confluent REST Proxy cluster metrics."""

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


class Widget(BaseWidget):
    service_type = "kafka"
    display_name = "Kafka"
    icon = "waypoints"
    category = ServiceCategory.INFRASTRUCTURE
    description = "Streaming platform — topics and consumer groups"
    env_prefix = "KAFKA"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="topics", label="Topics", format="number"),
            WidgetField(
                key="consumer_groups", label="Consumer Groups", format="number"
            ),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        client_cls, missing = import_client("kafka_mcp.api_client", "Api")
        if client_cls is None:
            return WidgetData(status="skipped", error=f"{missing} not installed")

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        if not url:
            return WidgetData(status="skipped", error="Missing Kafka url")

        client = client_cls(base_url=url, token=token or None)

        try:
            topics = client.list_topics() or []
        except Exception as e:
            return self._error_data(e)

        consumer_groups = 0
        try:
            groups = client.list_consumer_groups() or []
            consumer_groups = len(groups) if isinstance(groups, list) else 0
        except Exception as e:
            # Best-effort secondary metric: list_topics() above already
            # proved the service reachable, so this only degrades one field.
            logger.warning("Kafka consumer-group fetch failed: %s", e)

        return WidgetData(
            fields={
                "topics": len(topics) if isinstance(topics, list) else 0,
                "consumer_groups": consumer_groups,
            },
            status="ok",
        )
