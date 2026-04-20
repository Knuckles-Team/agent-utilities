#!/usr/bin/python
# coding: utf-8
"""Graph Database Backends.

Provides the `GraphBackend` ABC and concrete implementations for LadybugDB,
FalkorDB, and Neo4j. Use `create_backend()` to instantiate the correct
backend from configuration or environment variables.

Environment Variables:
    GRAPH_BACKEND: Backend type ("ladybug", "falkordb", "neo4j"). Default: "ladybug".
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
from typing import Optional

from .base import GraphBackend
from .ladybug_backend import LadybugBackend, LADYBUG_AVAILABLE
from .falkordb_backend import FalkorDBBackend
from .neo4j_backend import Neo4jBackend

logger = logging.getLogger(__name__)

_ACTIVE_BACKEND: Optional[GraphBackend] = None

__all__ = [
    "GraphBackend",
    "LadybugBackend",
    "FalkorDBBackend",
    "Neo4jBackend",
    "LADYBUG_AVAILABLE",
    "create_backend",
    "get_active_backend",
    "set_active_backend",
]


def get_active_backend() -> Optional[GraphBackend]:
    """Retrieve the currently active graph backend instance."""
    return _ACTIVE_BACKEND


def set_active_backend(backend: GraphBackend):
    """Explicitly set the active graph backend instance."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend


def create_backend(
    backend_type: Optional[str] = None,
    db_path: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    db_name: Optional[str] = None,
) -> Optional[GraphBackend]:
    """Factory function to create the appropriate graph backend.

    Resolves configuration from explicit arguments first, then falls back to
    environment variables, then to sensible defaults. LadybugDB is the default
    backend if nothing is specified.

    Args:
        backend_type: One of "ladybug", "falkordb", "neo4j". Falls back to
            ``GRAPH_BACKEND`` env var, then "ladybug".
        db_path: File path for LadybugDB. Falls back to ``GRAPH_DB_PATH``.
        host: Host for FalkorDB/Neo4j. Falls back to ``GRAPH_DB_HOST``.
        port: Port for FalkorDB/Neo4j. Falls back to ``GRAPH_DB_PORT``.
        uri: Full URI for Neo4j. Falls back to ``GRAPH_DB_URI``.
        user: Username for Neo4j. Falls back to ``GRAPH_DB_USER``.
        password: Password for Neo4j. Falls back to ``GRAPH_DB_PASSWORD``.
        db_name: Database name for FalkorDB. Falls back to ``GRAPH_DB_NAME``.

    Returns:
        A configured ``GraphBackend`` instance, or ``None`` if the requested
        backend is not available (e.g., ladybug package not installed).
    """
    global _ACTIVE_BACKEND

    backend_type = (
        (backend_type or os.environ.get("GRAPH_BACKEND", "ladybug")).lower().strip()
    )

    backend = None
    if backend_type == "ladybug":
        if not LADYBUG_AVAILABLE:
            logger.warning(
                "LadybugDB requested but 'ladybug' package is not installed."
            )
            return None
        resolved_path = db_path or os.environ.get("GRAPH_DB_PATH", "knowledge_graph.db")
        backend = LadybugBackend(resolved_path)

    elif backend_type == "falkordb":
        resolved_host = host or os.environ.get("GRAPH_DB_HOST", "localhost")
        resolved_port = port or int(os.environ.get("GRAPH_DB_PORT", "6379"))
        resolved_name = db_name or os.environ.get("GRAPH_DB_NAME", "agent_graph")
        backend = FalkorDBBackend(
            host=resolved_host, port=resolved_port, db_name=resolved_name
        )

    elif backend_type == "neo4j":
        resolved_uri = uri or os.environ.get("GRAPH_DB_URI", "bolt://localhost:7687")
        resolved_user = user or os.environ.get("GRAPH_DB_USER", "neo4j")
        resolved_password = password or os.environ.get("GRAPH_DB_PASSWORD", "password")
        backend = Neo4jBackend(
            uri=resolved_uri, user=resolved_user, password=resolved_password
        )

    else:
        logger.error(
            f"Unknown graph backend type: '{backend_type}'. Supported: ladybug, falkordb, neo4j"
        )
        return None

    if backend and _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = backend

    return backend
