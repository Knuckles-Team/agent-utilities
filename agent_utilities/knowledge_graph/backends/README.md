# Graph Backends

This directory contains the persistence layer implementations for the Knowledge Graph.

## Overview

All backends implement the `GraphBackend` abstract base class defined in `base.py`. This ensures that the engine can interact with any database using a unified interface for Cypher queries, vector search, and functional pruning.

## Supported Backends

### 1. LadybugDB (`ladybug_backend.py`)
- **Type**: Embedded SQLite-based Graph/Vector DB.
- **Status**: Production (Default).
- **Use Case**: Local development, zero-config deployments, and single-agent sessions.

### 2. Neo4j (`neo4j_backend.py`)
- **Type**: Enterprise Distributed Graph DB.
- **Status**: Stub / Experimental.
- **Use Case**: High-concurrency environments, complex multi-user graph analysis.

### 3. FalkorDB (`falkordb_backend.py`)
- **Type**: Redis-based Graph DB.
- **Status**: Stub / Experimental.
- **Use Case**: High-throughput graph workloads requiring Redis speeds.

## Configuration

Backends are selected via environment variables:

- `GRAPH_BACKEND`: `ladybug` (default), `neo4j`, or `falkordb`.
- `GRAPH_DB_PATH`: Local path for LadybugDB (default: `knowledge_graph.db`).
- `GRAPH_DB_URI`: Connection URI for Neo4j.
- `GRAPH_DB_HOST`: Host for FalkorDB.

## Implementing a New Backend

1. Inherit from `GraphBackend` in `base.py`.
2. Implement the required methods: `execute()`, `create_schema()`, `prune()`, and vector-related operations.
3. Register the new backend in the `create_backend()` factory in `__init__.py`.
