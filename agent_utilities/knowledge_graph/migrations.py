#!/usr/bin/python
"""Knowledge Graph Migration Framework.

This module handles schema evolution for the Knowledge Graph, ensuring that
node tables and properties are updated as the schema definition grows.
"""

import logging
import re
from typing import Any

from ..models.schema_definition import SCHEMA
from .backends.base import GraphBackend

logger = logging.getLogger(__name__)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_COLUMN_TYPE = re.compile(
    r"^(?:STRING|BOOLEAN|INT8|INT16|INT32|INT64|UINT8|UINT16|UINT32|UINT64|"
    r"FLOAT|DOUBLE|DATE|TIMESTAMP|INTERVAL|BLOB)(?:\[[1-9][0-9]{0,5}\])?$"
)


def _schema_identifier(value: object) -> str:
    rendered = str(value or "")
    if not _IDENTIFIER.fullmatch(rendered):
        raise ValueError("Schema identifier is invalid")
    return rendered


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
            table_name = _schema_identifier(node_def.name)
            # Check existing columns to avoid redundant ALTER calls
            existing_cols = set()
            try:
                res = self.backend.execute(
                    f"CALL TABLE_INFO('{table_name}') RETURN name"
                )
                existing_cols = {row["name"].lower() for row in res}
            except Exception:
                # Table might not exist yet, backend.create_schema() will handle it
                pass  # nosec B110

            for col_name, col_type in node_def.columns.items():
                column_name = _schema_identifier(col_name)
                if col_name.lower() == "id" or col_name.lower() in existing_cols:
                    continue

                try:
                    # Kùzu (Ladybug) ALTER TABLE ADD property_name property_type
                    clean_type = col_type.split(" PRIMARY KEY")[0]
                    if not _COLUMN_TYPE.fullmatch(clean_type):
                        raise ValueError("Schema column type is invalid")
                    stmt = f"ALTER TABLE {table_name} ADD `{column_name}` {clean_type}"
                    self.backend.execute(stmt)
                    results["columns_added"] += 1
                    logger.info("Added one schema column")
                except Exception as exc:
                    msg = str(exc).lower()
                    if (
                        "already has property" in msg
                        or "duplicate" in msg
                        or "already exists" in msg
                    ):
                        continue
                    if "table" in msg and "not found" in msg:
                        continue
                    results["errors"].append(
                        f"Schema migration failed: error_type={type(exc).__name__}"
                    )

        return results


def migrate_graph(backend: GraphBackend):
    """Convenience function to run migrations."""
    migrator = GraphMigrator(backend)
    return migrator.run_migrations()
