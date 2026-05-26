"""DocumentDB widget — document database status."""

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
    service_type = "documentdb"
    display_name = "DocumentDB"
    icon = "database"
    category = ServiceCategory.DATA_SCIENCE
    description = "Document database — collections and document counts"
    env_prefix = "DOCUMENTDB"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="databases", label="Databases", format="number"),
            WidgetField(key="collections", label="Collections", format="number"),
            WidgetField(key="status", label="Status", format="text"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from documentdb_mcp.api_client import DocumentDBApi

        client = DocumentDBApi()
        try:
            dbs = client.list_databases() or []
        except Exception as e:
            logger.debug("DocumentDB fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "databases": len(dbs) if isinstance(dbs, list) else 0,
                "collections": 0,
                "status": "Online",
            },
            status="ok",
        )
