"""Vector DB widget — vector database collections and embedding status."""

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
    service_type = "vector_db"
    display_name = "Vector DB"
    icon = "database"
    category = ServiceCategory.DATA_SCIENCE
    description = "Vector database — collections, embeddings, and similarity search"
    env_prefix = "VECTOR"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="collections", label="Collections", format="number"),
            WidgetField(key="points", label="Points", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from vector_mcp.api_client import VectorApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = VectorApi(base_url=url, api_key=token)
        try:
            collections = client.list_collections() or []
            count = len(collections) if isinstance(collections, list) else 0
        except Exception as e:
            logger.debug("Vector DB fetch: %s", type(e).__name__)
            return self._error_data(e)

        return WidgetData(
            fields={"collections": count, "points": 0, "status": "Online"},
            status="ok",
        )
