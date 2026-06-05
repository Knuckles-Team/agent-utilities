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

### 4. PostgreSQL + pgGraph (`postgresql_backend.py`)
- **Type**: Enterprise PostgreSQL with pgGraph graph extension + pgvector + ParadeDB.
- **Status**: Production.
- **Use Case**: Teams with existing PostgreSQL infrastructure wanting graph, vector, and BM25 capabilities without a separate graph database.
- **Extensions**: pgGraph (CSR graph traversal), pgvector (HNSW embedding search), ParadeDB (BM25 lexical search).
- **Features**: Connection pooling (psycopg_pool), Cypher-to-SQL transpilation, graceful degradation when extensions are absent.

### 5. Memory (`memory_backend.py`)
- **Type**: Pure in-memory NetworkX graph.
- **Status**: Testing / Development.
- **Use Case**: Unit tests, CI pipelines, edge devices, and ephemeral containers.

## Configuration

Backends are selected via environment variables:

- `GRAPH_BACKEND`: `tiered` (default — L1 `epistemic_graph` + L2 `ladybug`, a
  single self-contained binary with no external server), or one of `memory`,
  `file`, `epistemic_graph`, `postgresql`, `ladybug`, `neo4j`, `falkordb`.
- `GRAPH_BACKEND_L1` / `GRAPH_BACKEND_L2`: tiered tier selection. L2 defaults to
  `ladybug`, and auto-switches to `postgresql` when a DSN (`GRAPH_DB_URI` /
  `PGGRAPH_DSN`) is configured. Set `GRAPH_BACKEND_L2=postgresql` to force it.
- `GRAPH_DB_PATH`: Local path for LadybugDB (default: `knowledge_graph.db`).
- `GRAPH_DB_URI`: Connection URI for Neo4j or PostgreSQL.
- `GRAPH_DB_HOST`: Host for FalkorDB.
- `GRAPH_POOL_MIN` / `GRAPH_POOL_MAX`: PostgreSQL connection pool sizing.
- `GRAPH_PGGRAPH_SCHEMA`: Schema for pgGraph table registration (default: `public`).

## Implementing a New Backend

1. Inherit from `GraphBackend` in `base.py`.
2. Implement the required methods: `execute()`, `execute_batch()`, `create_schema()`, `add_embedding()`, `semantic_search()`, `prune()`, and `close()`.
3. Register the new backend in the `create_backend()` factory in `__init__.py`.
4. See `docs/graph_backends_architecture.md` for the full architecture reference.
