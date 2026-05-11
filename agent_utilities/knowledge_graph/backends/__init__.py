#!/usr/bin/python
"""Graph Database Backends.

Provides the `GraphBackend` ABC and concrete implementations for Memory,
LadybugDB, FalkorDB, Neo4j, and PostgreSQL. Use `create_backend()` to
instantiate the correct backend from configuration or environment variables.

Architecture (Tiered Graph Engine):
    The engine uses a two-tier architecture:
    - **Tier 1 (Source of Truth)**: A persistent Cypher-capable backend
      (LadybugDB/Neo4j/PostgreSQL) handles all CRUD, schema enforcement,
      vector indexing, and Cypher queries.
    - **Tier 2 (Compute Scratchpad)**: NetworkX is loaded on-demand via
      ``load_subgraph()`` for graph algorithms (PageRank, VF2, spectral
      clustering) that databases cannot perform natively.

    The ``memory`` backend (pure NetworkX) is available for testing/CI
    where no persistence or Cypher support is needed.

Environment Variables:
    GRAPH_BACKEND: Backend type. Default: "ladybug".
        Supported: "memory", "ladybug", "falkordb", "neo4j", "postgresql".
    GRAPH_DB_PATH: File path for LadybugDB. Default: "knowledge_graph.db".
    GRAPH_DB_HOST: Host for FalkorDB/Neo4j. Default: "localhost".
    GRAPH_DB_PORT: Port for FalkorDB (6379) or Neo4j (7687).
    GRAPH_DB_URI:  Full URI for Neo4j or PostgreSQL.
    GRAPH_DB_USER: Username for Neo4j/PostgreSQL. Default: "neo4j".
    GRAPH_DB_PASSWORD: Password for Neo4j/PostgreSQL. Default: "password".
    GRAPH_DB_NAME: Database name for FalkorDB/PostgreSQL. Default: "agent_graph".
"""

import logging
import os

from .base import GraphBackend
from .falkordb_backend import FalkorDBBackend
from .ladybug_backend import LADYBUG_AVAILABLE, LadybugBackend
from .memory_backend import MemoryBackend
from .neo4j_backend import Neo4jBackend
from .postgresql_backend import PostgreSQLBackend

logger = logging.getLogger(__name__)

_ACTIVE_BACKEND: GraphBackend | None = None

__all__ = [
    "GraphBackend",
    "MemoryBackend",
    "LadybugBackend",
    "FalkorDBBackend",
    "Neo4jBackend",
    "PostgreSQLBackend",
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
    environment variables, then to sensible defaults. LadybugDB (self-contained
    SQLite + Cypher) is the default for zero-config startup with full
    persistence. Use ``GRAPH_BACKEND=memory`` for testing/CI only.

    Args:
        backend_type: One of "memory", "ladybug", "falkordb", "neo4j",
            "postgresql". Falls back to ``GRAPH_BACKEND`` env var,
            then "ladybug".
        db_path: File path for LadybugDB. Falls back to ``GRAPH_DB_PATH``.
        host: Host for FalkorDB/Neo4j. Falls back to ``GRAPH_DB_HOST``.
        port: Port for FalkorDB/Neo4j. Falls back to ``GRAPH_DB_PORT``.
        uri: Full URI for Neo4j/PostgreSQL. Falls back to ``GRAPH_DB_URI``.
        user: Username for Neo4j/PostgreSQL. Falls back to ``GRAPH_DB_USER``.
        password: Password for Neo4j/PostgreSQL. Falls back to ``GRAPH_DB_PASSWORD``.
        db_name: Database name for FalkorDB/PostgreSQL. Falls back to ``GRAPH_DB_NAME``.

    Returns:
        A configured ``GraphBackend`` instance, or ``None`` if the requested
        backend is not available (e.g., ladybug package not installed).
    """
    global _ACTIVE_BACKEND

    backend_type = (
        (backend_type or os.environ.get("GRAPH_BACKEND") or "ladybug").lower().strip()
    )

    backend: GraphBackend | None = None

    if backend_type == "memory":
        backend = MemoryBackend()

    elif backend_type == "ladybug":
        if not LADYBUG_AVAILABLE:
            logger.warning(
                "LadybugDB requested but 'ladybug' package is not installed."
            )
            return None
        # Use centralized XDG-aware path resolver
        if db_path:
            resolved_path = db_path
        elif os.environ.get("GRAPH_DB_PATH"):
            resolved_path = os.environ["GRAPH_DB_PATH"]
        else:
            from agent_utilities.core.paths import kg_db_path

            resolved = kg_db_path()
            # Ensure parent directory exists for XDG paths
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved_path = str(resolved)
        backend = LadybugBackend(resolved_path)

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

    elif backend_type == "postgresql":
        resolved_uri = (
            uri
            or os.environ.get("GRAPH_DB_URI")
            or "postgresql://localhost:5432/agent_utilities"
        )
        resolved_name = db_name or os.environ.get("GRAPH_DB_NAME") or "agent_graph"
        backend = PostgreSQLBackend(dsn=resolved_uri, graph_name=resolved_name)

    else:
        logger.error(
            f"Unknown graph backend type: '{backend_type}'. "
            f"Supported: memory, ladybug, falkordb, neo4j, postgresql"
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
