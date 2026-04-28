#!/usr/bin/python
"""Graph Database Backends.

Provides the `GraphBackend` ABC and concrete implementations for LadybugDB,
FalkorDB, Neo4j, and Vector-MCP. Use `create_backend()` to instantiate the correct
backend from configuration or environment variables.

Environment Variables:
    GRAPH_BACKEND: Backend type ("ladybug", "falkordb", "neo4j", "vector_mcp"). Default: "ladybug".
    GRAPH_DB_PATH: File path for LadybugDB. Default: "knowledge_graph.db".
    GRAPH_DB_HOST: Host for FalkorDB/Neo4j. Default: "localhost".
    GRAPH_DB_PORT: Port for FalkorDB (6379) or Neo4j (7687).
    GRAPH_DB_URI:  Full URI for Neo4j (e.g., "bolt://localhost:7687").
    GRAPH_DB_USER: Username for Neo4j. Default: "neo4j".
    GRAPH_DB_PASSWORD: Password for Neo4j. Default: "password".
    GRAPH_DB_NAME: Database name for FalkorDB. Default: "agent_graph".
"""

import logging
import os

from .base import GraphBackend
from .falkordb_backend import FalkorDBBackend
from .ladybug_backend import LADYBUG_AVAILABLE, LadybugBackend
from .neo4j_backend import Neo4jBackend
from .vector_mcp_backend import VectorMCPBackend

logger = logging.getLogger(__name__)

_ACTIVE_BACKEND: GraphBackend | None = None

__all__ = [
    "GraphBackend",
    "LadybugBackend",
    "FalkorDBBackend",
    "Neo4jBackend",
    "VectorMCPBackend",
    "LADYBUG_AVAILABLE",
    "create_backend",
    "get_active_backend",
    "set_active_backend",
]


def get_active_backend() -> GraphBackend | None:
    """Retrieve the currently active graph backend instance."""
    return _ACTIVE_BACKEND


def set_active_backend(backend: GraphBackend | None):
    """Explicitly set the active graph backend instance."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def create_backend(
    backend_type: str | None = None,
    db_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
    **kwargs,
) -> GraphBackend | None:
    """Factory function to create the appropriate graph backend.

    Resolves configuration from explicit arguments first, then falls back to
    environment variables, then to sensible defaults. LadybugDB is the default
    backend if nothing is specified.

    Args:
        backend_type: One of "ladybug", "falkordb", "neo4j", "vector_mcp". Falls back to
            ``GRAPH_BACKEND`` env var, then "ladybug".
        db_path: File path for LadybugDB. Falls back to ``GRAPH_DB_PATH``.
        host: Host for FalkorDB/Neo4j. Falls back to ``GRAPH_DB_HOST``.
        port: Port for FalkorDB/Neo4j. Falls back to ``GRAPH_DB_PORT``.
        uri: Full URI for Neo4j. Falls back to ``GRAPH_DB_URI``.
        user: Username for Neo4j. Falls back to ``GRAPH_DB_USER``.
        password: Password for Neo4j. Falls back to ``GRAPH_DB_PASSWORD``.
        db_name: Database name for FalkorDB. Falls back to ``GRAPH_DB_NAME``.
        **kwargs: Additional arguments for Vector-MCP backend (e.g., vector_db_type).

    Returns:
        A configured ``GraphBackend`` instance, or ``None`` if the requested
        backend is not available (e.g., ladybug package not installed).
    """
    global _ACTIVE_BACKEND

    backend_type = (
        (backend_type or os.environ.get("GRAPH_BACKEND") or "ladybug").lower().strip()
    )

    backend: GraphBackend | None = None
    if backend_type == "ladybug":
        if not LADYBUG_AVAILABLE:
            logger.warning(
                "LadybugDB requested but 'ladybug' package is not installed."
            )
            return None
        resolved_path = (
            db_path or os.environ.get("GRAPH_DB_PATH") or "knowledge_graph.db"
        )
        backend = LadybugBackend(resolved_path)

    elif backend_type == "vector_mcp":
        # Vector-MCP backend with fallback to LadybugDB
        vector_db_type = kwargs.get("vector_db_type", "chroma")
        fallback_db = (
            LadybugBackend(db_path or "knowledge_graph.db")
            if LADYBUG_AVAILABLE
            else None
        )
        backend = VectorMCPBackend(
            db_type=vector_db_type,
            fallback_graph_db=fallback_db,
            **{k: v for k, v in kwargs.items() if k != "vector_db_type"},
        )

    elif backend_type == "falkordb":
        resolved_host = host or os.environ.get("GRAPH_DB_HOST") or "localhost"
        resolved_port = port or int(os.environ.get("GRAPH_DB_PORT", "6379"))
        resolved_name = db_name or os.environ.get("GRAPH_DB_NAME") or "agent_graph"
        backend = FalkorDBBackend(
            host=resolved_host, port=resolved_port, db_name=resolved_name
        )

    elif backend_type == "neo4j":
        resolved_uri = uri or os.environ.get("GRAPH_DB_URI") or "bolt://localhost:7687"
        resolved_user = user or os.environ.get("GRAPH_DB_USER") or "neo4j"
        resolved_password = (
            password or os.environ.get("GRAPH_DB_PASSWORD") or "password"
        )
        backend = Neo4jBackend(
            uri=resolved_uri, user=resolved_user, password=resolved_password
        )

    else:
        logger.error(
            f"Unknown graph backend type: '{backend_type}'. Supported: ladybug, falkordb, neo4j, vector_mcp"
        )
        return None

    if backend:
        try:
            backend.create_schema()
            # Run schema migrations to add any missing columns/properties
            if (
                backend_type == "ladybug"
                and os.environ.get("AGENT_UTILITIES_TESTING") != "true"
            ):
                from ..migrations import migrate_graph

                migrate_graph(backend)
        except Exception as e:
            logger.debug(f"Failed to auto-initialize or migrate graph schema: {e}")

    if backend and _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = backend

    return backend
