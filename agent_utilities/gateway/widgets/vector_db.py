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
        # No `vector_mcp.api_client` module exists. The package's real public
        # client is `vector_mcp.vector_api.Api`, and in the PUBLISHED
        # distribution (vector-mcp>=2.1.2, the one the `gateway-widgets` extra
        # installs and the one that ships in the served image) it is a REMOTE
        # REST client: `Api(base_url, token=None, verify=False)` — `base_url`
        # is REQUIRED. An earlier revision of this widget called `Api()` with
        # no arguments, matching the unpublished local sibling checkout whose
        # `Api` is an in-process facade instead; against the published package
        # that raises `TypeError: Api.__init__() missing 1 required positional
        # argument: 'base_url'` on every aggregator poll. Resolve the tile's
        # configured URL/token like every other REST-backed widget here.
        from vector_mcp.vector_api import Api

        client = Api(
            base_url=self._resolve_url(config),
            token=self._resolve_token(config) or None,
        )
        try:
            result = client.list_collections() or {}
            collections = (
                result.get("collections", []) if isinstance(result, dict) else []
            )
            count = len(collections)
        except Exception as e:
            logger.debug("Vector DB fetch: %s", type(e).__name__)
            return self._error_data(e)

        return WidgetData(
            fields={"collections": count, "points": 0, "status": "Online"},
            status="ok",
        )
