#!/usr/bin/python
"""LadybugDB Graph Backend.

This module provides the LadybugDB implementation of the GraphBackend interface,
supporting strict schema-bound Cypher queries.
"""

import logging
import os
from typing import Any

from .base import GraphBackend

try:
    import ladybug

    LADYBUG_AVAILABLE = True
except ImportError:
    LADYBUG_AVAILABLE = False

from agent_utilities.models.schema_definition import SCHEMA

logger = logging.getLogger(__name__)


class LadybugBackend(GraphBackend):
    """LadybugDB backend implementation."""

    def __init__(self, db_path: str = "knowledge_graph.db"):
        if not LADYBUG_AVAILABLE:
            raise ImportError(
                "ladybug package is not installed. Install with 'pip install ladybug'"
            )
        self.db_path = db_path
        # Use Database and Connection objects as required by newer ladybug versions
        # Add retry logic with jitter for multi-agent startup resilience
        import random
        import time

        max_retries = 5
        last_error: Exception = RuntimeError("Max retries exceeded")
        for attempt in range(max_retries):
            try:
                buffer_size = os.getenv("LADYBUG_MAX_DB_SIZE") or os.getenv(
                    "LADYBUG_BUFFER_SIZE"
                )
                db_params = {}
                if buffer_size:
                    try:
                        db_params["max_db_size"] = int(buffer_size)
                    except ValueError:
                        logger.warning(f"Invalid LADYBUG buffer/db size: {buffer_size}")

                self.db = ladybug.Database(
                    db_path if db_path != ":memory:" else None, **db_params
                )
                self.conn = ladybug.Connection(self.db)
                # Successful connection, perform backup
                self._backup_db()
                return
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                if "lock" in msg or "busy" in msg:
                    wait_time = (2**attempt) + random.random()  # nosec B311
                    logger.warning(
                        f"Graph DB locked, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                elif (
                    "corrupted" in msg
                    or "invalid wal record" in msg
                    or "read out invalid" in msg
                    or "unreachable_code" in msg
                ):
                    logger.error(
                        f"Detected database corruption in {db_path}: {e}. Attempting cleanup and repair..."
                    )
                    self._cleanup_corrupted()
                    # retry immediately after cleanup
                    continue
                else:
                    raise e
        raise last_error

    def close(self) -> None:
        """Close the database connection and database object."""
        if hasattr(self, "conn"):
            del self.conn
        if hasattr(self, "db"):
            del self.db

    def _cleanup_corrupted(self):
        """Removes corrupted SQLite/WAL files."""
        from pathlib import Path

        base_path = Path(self.db_path)
        if base_path.name == ":memory:":
            return

        # Files to remove: main db, wal, shm
        exts = ["", ".wal", "-wal", ".shm", "-shm"]
        for ext in exts:
            p = base_path.parent / (base_path.name + ext)
            if p.exists():
                try:
                    p.unlink()
                    logger.info(f"Cleaned up corrupted file: {p}")
                except Exception as e:
                    logger.error(f"Failed to cleanup {p}: {e}")

    def _backup_db(self):
        """Maintains up to N most recent backups of the database."""
        import datetime
        import shutil
        from pathlib import Path

        from ...config import DEFAULT_KG_BACKUPS

        if DEFAULT_KG_BACKUPS <= 0 or self.db_path == ":memory:":
            return

        base_path = Path(self.db_path)
        if not base_path.exists():
            return

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = base_path.with_name(f"{base_path.name}.{timestamp}.bak")

            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")

            # Prune old backups (keep 3 most recent)
            backups = sorted(
                base_path.parent.glob(f"{base_path.name}.*.bak"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            for old_backup in backups[DEFAULT_KG_BACKUPS:]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
        except Exception as e:
            # Don't crash the server if backup fails, just log it
            logger.warning(f"Database backup failed: {e}")

    def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query on LadybugDB."""
        try:
            res = self.conn.execute(query, params or {})
            if isinstance(res, list):
                if not res:
                    return []
                res = res[0]
            # ladybug returns a QueryResult object. We want a list of dicts.
            return res.rows_as_dict().get_all()
        except Exception as e:
            msg = str(e).lower()
            if (
                "already has property" in msg
                or "duplicate" in msg
                or "already exists" in msg
            ):
                logger.debug(f"LadybugDB expected migration error: {e}")
            elif "table" in msg and "does not exist" in msg:
                logger.warning(f"LadybugDB table not found (check schema): {e}")
            elif "binder exception" in msg:
                logger.error(f"LadybugDB binder issue (invalid property?): {e}")
            else:
                logger.error(f"LadybugDB Cypher execution failed: {e}\nQuery: {query}")
            return []

    def create_schema(self) -> None:
        """Create LadybugDB schema from the unified schema definition.
        Ladybug requires strict DDL for Node and Rel tables.
        """
        # 1. Create Node Tables
        for node in SCHEMA.nodes:
            cols = ", ".join(
                [f"{name} {dtype}" for name, dtype in node.columns.items()]
            )
            stmt = f"CREATE NODE TABLE IF NOT EXISTS {node.name} ({cols});"
            try:
                self.conn.execute(stmt)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Node table creation issue ({node.name}): {e}")

        # 2. Create Rel Tables
        for rel in SCHEMA.edges:
            conns = ", ".join(
                [f"FROM {c['from']} TO {c['to']}" for c in rel.connections]
            )
            stmt = f"CREATE REL TABLE IF NOT EXISTS {rel.type} ({conns});"
            try:
                self.conn.execute(stmt)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Rel table creation issue ({rel.type}): {e}")

        # 3. Create Vector Indices for any FLOAT column named 'embedding'.
        #
        # LadybugDB 0.15.x does not support `CREATE INDEX ... USING HNSW` DDL.
        # Vector indexes are provided by the optional VECTOR extension and must
        # be created via `CALL CREATE_VECTOR_INDEX(table, name, column)`. The
        # extension additionally requires fixed-size arrays (e.g. FLOAT[768]);
        # variable-length FLOAT[] columns raise a binder exception. When the
        # extension is unavailable or the column type is incompatible, we log
        # once at INFO level (not per-node WARNING) to avoid log pollution on
        # every startup.
        embedding_tables = [
            node.name
            for node in SCHEMA.nodes
            if "embedding" in node.columns
            and "FLOAT" in node.columns["embedding"].upper()
        ]
        if embedding_tables:
            try:
                self.conn.execute("INSTALL VECTOR;")
                self.conn.execute("LOAD EXTENSION VECTOR;")
                vector_extension_loaded = True
            except Exception as e:
                logger.info(
                    "LadybugDB VECTOR extension unavailable; skipping vector "
                    "index DDL for %d embedding table(s): %s",
                    len(embedding_tables),
                    e,
                )
                vector_extension_loaded = False

            if vector_extension_loaded:
                skip_reason: str | None = None
                for table in embedding_tables:
                    idx_name = f"idx_{table.lower()}_embedding"
                    stmt = (
                        f"CALL CREATE_VECTOR_INDEX('{table}', "
                        f"'{idx_name}', 'embedding');"
                    )
                    try:
                        self.conn.execute(stmt)
                    except Exception as e:
                        msg = str(e)
                        if "already exists" in msg.lower():
                            continue
                        if "FLOAT/DOUBLE ARRAY" in msg:
                            # Schema uses variable-length FLOAT[]; vector index
                            # needs FLOAT[N]. Record once and skip remaining.
                            skip_reason = msg
                            break
                        logger.warning(f"Vector index creation issue ({idx_name}): {e}")
                if skip_reason is not None:
                    logger.info(
                        "LadybugDB vector indexes skipped for %d table(s): %s. "
                        "Define embedding columns as FLOAT[N] (fixed size) to "
                        "enable HNSW indexing.",
                        len(embedding_tables),
                        skip_reason,
                    )

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Add embedding to an existing message."""
        query = "MATCH (m:Message {id: $id}) SET m.embedding = $emb"
        self.execute(query, {"id": node_id, "emb": embedding})

    def prune(self, criteria: dict[str, Any]) -> None:
        """Prune nodes based on criteria.

        Args:
            criteria: A dictionary defining pruning rules:
                - node_type: (str) Optional filter for specific node labels.
                - age_days: (int) Delete nodes older than this number of days.
                - min_importance: (float) Delete nodes with importance_score below this.
        """
        node_type = criteria.get("node_type", "")
        label = f":{node_type}" if node_type else ""

        where_clauses = []
        params = {}

        if "age_days" in criteria:
            import datetime

            cutoff = (
                datetime.datetime.now() - datetime.timedelta(days=criteria["age_days"])
            ).isoformat()
            where_clauses.append("n.timestamp < $cutoff")
            params["cutoff"] = cutoff

        if "min_importance" in criteria:
            where_clauses.append("n.importance_score < $min_imp")
            params["min_imp"] = criteria["min_importance"]

        if not where_clauses:
            logger.warning("Prune called without any meaningful criteria.")
            return

        where_str = " AND ".join(where_clauses)
        query = f"MATCH (n{label}) WHERE {where_str} DETACH DELETE n"

        logger.info(f"Pruning nodes: {query} with params {params}")
        self.execute(query, params)
