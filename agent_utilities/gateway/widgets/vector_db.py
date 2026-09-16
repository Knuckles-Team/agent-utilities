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
        # RF-ADR-009: agent-utilities (workspace phase 4) must not import a
        # sibling connector package (the vector-database connector is phase
        # 7) — no in-process package import here, ever again. Reach it over
        # its MCP server instead, the same fleet transport every mcp_tool source
        # connector already uses for the ~58-server fleet
        # (protocols/source_connectors/connectors/mcp_tool.call_tool_once):
        # a FastMCP client against vector-mcp's `streamable-http` endpoint,
        # never a package import. `VECTOR_URL` keeps its prior meaning (the
        # service's base host:port); the MCP endpoint is that base + FastMCP's
        # default `/mcp` mount path.
        from agent_utilities.protocols.source_connectors.connectors.mcp_package import (
            _run_async,
        )
        from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
            call_tool_once,
        )

        try:
            base_url = self._resolve_url(config).rstrip("/")
            result = _run_async(
                call_tool_once(
                    url=f"{base_url}/mcp",
                    tool="vector_collection_management",
                    action="list_collections",
                    timeout=10.0,
                )
            )
        except Exception as e:
            logger.debug("Vector DB fetch: %s", type(e).__name__)
            return self._error_data(e)

        collections = result.get("collections", []) if isinstance(result, dict) else []
        return WidgetData(
            fields={"collections": len(collections), "points": 0, "status": "Online"},
            status="ok",
        )
