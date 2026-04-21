#!/usr/bin/python
"""LadybugDB Graph Backend.

This module provides the LadybugDB implementation of the GraphBackend interface,
supporting strict schema-bound Cypher queries.
"""

import logging
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
                self.db = ladybug.Database(db_path if db_path != ":memory:" else None)
                self.conn = ladybug.Connection(self.db)
                return
            except Exception as e:
                last_error = e
                if "lock" in str(e).lower() or "busy" in str(e).lower():
                    wait_time = (2**attempt) + random.random()
                    logger.warning(
                        f"Graph DB locked, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                else:
                    raise e
        raise last_error

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
            msg = str(e)
            if "Table" in msg and "does not exist" in msg:
                logger.debug(f"LadybugDB table not found (likely empty DB): {e}")
            elif "Binder exception" in msg:
                logger.debug(f"LadybugDB binder issue (likely empty DB): {e}")
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

        # 3. Create Vector Indices for any FLOAT[] column named 'embedding'
        for node in SCHEMA.nodes:
            if "embedding" in node.columns and "FLOAT[]" in node.columns["embedding"]:
                idx_name = f"idx_{node.name.lower()}_embedding"
                stmt = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {node.name}(embedding) USING HNSW WITH (metric = 'cosine');"
                try:
                    self.conn.execute(stmt)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Vector index creation issue ({idx_name}): {e}")

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
