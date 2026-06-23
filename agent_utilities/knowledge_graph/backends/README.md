# Graph Backends

This directory contains the `GraphBackend` implementations for the Knowledge Graph.

## Overview

**The epistemic-graph engine is the ONE database — the authority / system of
record** (compute + in-memory cache + semantic/ontology + durable persistence).
It serves all reads and acks all writes. The other backends here are **mirrors**:
durable copies that receive the engine's write stream — losslessly and
asynchronously, via the per-mirror outbox — for interop, BI, external query, and
DR. Every backend implements the `GraphBackend` abstract base class in `base.py`,
so a mutation runs natively on the authority and every mirror through one unified
interface for Cypher queries, vector search, and functional pruning.

## The authority

### epistemic-graph (`epistemic_graph_backend.py`)
- **Type**: Rust-native graph engine, reached out-of-process over MessagePack/UDS.
- **Role**: **THE authority / system of record** — compute + cache + semantic +
  durable persistence in one engine. Default; zero external services.

## Mirror targets

### PostgreSQL + pg-age (`postgresql_backend.py`)
- **Type**: PostgreSQL with the AGE graph extension + pgvector + ParadeDB.
- **Role**: Richest mirror — full openCypher (AGE) + vector + BM25 in a durable,
  BI-friendly store.
- **Extensions**: AGE (native openCypher / CSR graph traversal), pgvector (HNSW
  embedding search), ParadeDB (BM25 lexical search).
- **Features**: Connection pooling (psycopg_pool), Cypher-to-SQL transpilation
  fallback, graceful degradation when extensions are absent.

### Neo4j (`contrib/neo4j_backend.py`)
- **Type**: Distributed graph DB (Bolt).
- **Role**: Native-Cypher mirror for high-concurrency external graph analysis.

### FalkorDB (`contrib/falkordb_backend.py`)
- **Type**: Redis-based graph DB.
- **Role**: Native-Cypher mirror for Redis-speed external workloads.

### LadybugDB (`contrib/ladybug_backend.py`)
- **Type**: Embedded SQLite-based graph/vector DB.
- **Role**: Local zero-infra durable mirror (single file, no server).

### Memory (`memory_backend.py`)
- **Type**: Pure in-memory NetworkX graph.
- **Role**: Testing / development — unit tests, CI, ephemeral containers.

## Configuration

Configured via environment variables:

- `GRAPH_BACKEND`: `epistemic_graph` (default — the engine authority alone, also
  `memory`/`file` snapshot modes) or `fanout` (engine authority + mirrors). The
  `tiered`/`GRAPH_BACKEND_L1`/`GRAPH_BACKEND_L2` scheme is **removed**.
- `GRAPH_AUTHORITY`: read source-of-truth under `fanout` (default
  `epistemic_graph`; may name any durable connection).
- `GRAPH_MIRROR_TARGETS`: JSON/list of mirror connection names (declared in
  `KG_CONNECTIONS`) that receive the fanned-out write stream — replaces the
  removed `GRAPH_BACKEND_L2`.
- `GRAPH_DB_PATH`: Local path for LadybugDB (default: `knowledge_graph.db`).
- `GRAPH_DB_URI` / `PGGRAPH_DSN`: Connection URI for the Neo4j / PostgreSQL mirror.
- `GRAPH_DB_HOST`: Host for FalkorDB.
- `GRAPH_POOL_MIN` / `GRAPH_POOL_MAX`: PostgreSQL connection pool sizing.
- `GRAPH_PGGRAPH_SCHEMA`: Schema for pg-age table registration (default: `public`).

## Implementing a New Backend

1. Inherit from `GraphBackend` in `base.py`.
2. Implement the required methods: `execute()`, `execute_batch()`, `create_schema()`, `add_embedding()`, `semantic_search()`, `prune()`, and `close()`.
3. Register the new backend in the `create_backend()` factory in `__init__.py`.
4. See `docs/graph_backends_architecture.md` for the full architecture reference.
