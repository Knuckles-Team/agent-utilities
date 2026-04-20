#!/usr/bin/python
# coding: utf-8
"""LadybugDB Graph Backend.

This module provides the LadybugDB implementation of the GraphBackend interface,
supporting strict schema-bound Cypher queries.
"""

import logging
from typing import List, Dict, Any, Optional
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
        self.db = ladybug.Database(db_path if db_path != ":memory:" else None)
        self.conn = ladybug.Connection(self.db)

    def execute(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query on LadybugDB."""
        try:
            res = self.conn.execute(query, params or {})
            # ladybug returns a QueryResult object. We want a list of dicts.
            return res.rows_as_dict().get_all()
        except Exception as e:
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
                stmt = f"CREATE VECTOR INDEX IF NOT EXISTS {idx_name} ON {node.name}(embedding) USING HNSW WITH (metric = 'cosine');"
                try:
                    self.conn.execute(stmt)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Vector index creation issue ({idx_name}): {e}")

    def add_embedding(self, node_id: str, embedding: List[float]) -> None:
        """Add embedding to an existing message."""
        query = "MATCH (m:Message {id: $id}) SET m.embedding = $emb"
        self.execute(query, {"id": node_id, "emb": embedding})

    def prune(self, _criteria: Dict[str, Any]) -> None:
        """Prune nodes based on criteria."""
        pass
