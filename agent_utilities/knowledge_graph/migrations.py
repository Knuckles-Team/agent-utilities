#!/usr/bin/python
"""Knowledge Graph Migration Framework.

This module handles schema evolution for the Knowledge Graph, ensuring that
node tables and properties are updated as the schema definition grows.
"""

import logging
from typing import Any

from ..models.schema_definition import SCHEMA
from .backends.base import GraphBackend

logger = logging.getLogger(__name__)


class GraphMigrator:
    """Handles schema migrations for the Knowledge Graph."""

    def __init__(self, backend: GraphBackend):
        self.backend = backend

    def run_migrations(self) -> dict[str, Any]:
        """Runs all pending schema migrations."""
        results: dict[str, Any] = {
            "tables_created": 0,
            "columns_added": 0,
            "errors": [],
        }

        # 1. Ensure all node tables exist with all properties
        for node_def in SCHEMA.nodes:
            # Check existing columns to avoid redundant ALTER calls
            existing_cols = set()
            try:
                res = self.backend.execute(
                    f"CALL TABLE_INFO('{node_def.name}') RETURN name"
                )
                existing_cols = {row["name"].lower() for row in res}
            except Exception:
                # Table might not exist yet, backend.create_schema() will handle it
                pass  # nosec B110

            for col_name, col_type in node_def.columns.items():
                if col_name.lower() == "id" or col_name.lower() in existing_cols:
                    continue

                try:
                    # Kùzu (Ladybug) ALTER TABLE ADD property_name property_type
                    clean_type = col_type.split(" PRIMARY KEY")[0]
                    stmt = f"ALTER TABLE {node_def.name} ADD {col_name} {clean_type}"
                    self.backend.execute(stmt)
                    results["columns_added"] += 1
                    logger.info(f"Added column {col_name} to table {node_def.name}")
                except Exception as e:
                    msg = str(e).lower()
                    if (
                        "already has property" in msg
                        or "duplicate" in msg
                        or "already exists" in msg
                    ):
                        continue
                    if "table" in msg and "not found" in msg:
                        continue
                    results["errors"].append(
                        f"Failed to migrate {node_def.name}.{col_name}: {e}"
                    )

        return results


def migrate_graph(backend: GraphBackend):
    """Convenience function to run migrations."""
    migrator = GraphMigrator(backend)
    return migrator.run_migrations()
