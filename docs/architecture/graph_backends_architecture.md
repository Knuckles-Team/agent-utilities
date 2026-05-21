# Graph Backend Architecture

The Knowledge Graph engine supports multiple backend implementations through a
unified `GraphBackend` abstract interface. All backends provide the same core
capabilities: Cypher query execution, vector search, and node/edge CRUD.

## Architecture Overview

```mermaid
graph TB
    subgraph "IntelligenceGraphEngine"
        A["KG-2.0: query_cypher()"] --> B["KG-2.0: backend.execute()"]
        C["KG-2.0: add_node()"] --> B
        D["KG-2.0: link_nodes()"] --> B
        E["KG-2.3: search_hybrid()"] --> F["KG-2.3: backend.semantic_search()"]
        G["KG-2.0: load_subgraph()"] --> H["NetworkX\n(Tier 2 Compute)"]
    end

    B --> I{"KG-2.0: Backend Type?"}

    subgraph "KG-2.0: LadybugDB (Default)"
        I -->|ladybug| J["Native Cypher\nSQLite + HNSW"]
    end

    subgraph "KG-2.0: Neo4j"
        I -->|neo4j| K["KG-2.0: Native Cypher\nBolt Protocol"]
    end

    subgraph "FalkorDB"
        I -->|falkordb| L["KG-2.0: Cypher via\nRedis Protocol"]
    end

    subgraph "PostgreSQL + pgGraph"
        I -->|postgresql| M["KG-2.0: Cypher → SQL\nTranspiler"]
        M --> N["PostgreSQL Tables"]
        M --> O["KG-2.0: pgGraph Extension\n(CSR Traversal)"]
        F --> P["KG-2.3: pgvector\n(Cosine Search)"]
        F --> Q["KG-2.3: ParadeDB BM25\n(Lexical Search)"]
    end

    subgraph "Memory (Testing)"
        I -->|memory| R["NetworkX\nIn-Memory"]
    end
```

## Backend Comparison

| Capability | LadybugDB | Neo4j | FalkorDB | PostgreSQL + pgGraph | Memory |
|---|:---:|:---:|:---:|:---:|:---:|
| **Status** | Production | Experimental | Experimental | **Production** | Testing |
| Cypher Support | Native | Native | Native | Transpiled | Basic |
| Vector Search (HNSW) | ✅ | ✅ | ✅ | ✅ pgvector | ✅ NumPy |
| BM25 Lexical Search | — | — | — | ✅ ParadeDB | — |
| Graph Traversal | Cypher | Cypher | Cypher | ✅ pgGraph CSR | NetworkX |
| Connection Pooling | — | ✅ | — | ✅ psycopg_pool | — |
| ACID Transactions | SQLite WAL | ✅ | — | ✅ | — |
| Multi-Agent Concurrent | File Lock | ✅ | ✅ | ✅ | — |
| Persistence | File | Server | Redis | Server | None |
| Zero Config | ✅ | — | — | — | ✅ |

## PostgreSQL Backend Deep Dive

The PostgreSQL backend combines three PostgreSQL extensions into a unified
graph + vector + search layer:

### Three-Layer Architecture

```mermaid
graph LR
    subgraph "Layer 1: Storage"
        A["KG-2.0: Node Tables\n(Agent, Tool, Memory, ...)"] --- B["KG-2.0: kg_edges Table"]
    end

    subgraph "Layer 2: Graph Index"
        C["KG-2.0: pgGraph CSR\n(graph.traverse)\n(graph.shortest_path)\n(graph.search)"]
    end

    subgraph "Layer 3: Search"
        D["KG-2.3: pgvector HNSW\n(Cosine Similarity)"]
        E["KG-2.3: ParadeDB BM25\n(Lexical Ranking)"]
    end

    A --> C
    B --> C
    A --> D
    A --> E
```

### Cypher Transpilation

The engine speaks Cypher; PostgreSQL speaks SQL. The `CypherTranspiler` handles
the translation for all patterns the engine generates:

| Engine Cypher | PostgreSQL SQL |
|---|---|
| `MATCH (n:Agent) WHERE n.id = $id RETURN n` | `SELECT * FROM "Agent" WHERE id = $1` |
| `CREATE (n:Tool {id: $id, name: $name})` | `INSERT INTO "Tool" (id, name) VALUES ($1, $2)` |
| `MATCH (s)-[r:PROVIDES]->(t) MERGE ...` | `INSERT INTO kg_edges ... ON CONFLICT DO UPDATE` |
| `MATCH (n) WHERE toLower(n.name) CONTAINS $q` | `SELECT * FROM ... WHERE LOWER(name) LIKE '%$1%'` |
| Path traversal `(n)-[*1..3]-(t)` | `graph.traverse(seed, max_depth:=3)` |

### Extension Dependencies

| Extension | Required | Purpose |
|---|:---:|---|
| **pgvector** | Recommended | Embedding storage + HNSW cosine search |
| **pgGraph** | Optional | CSR graph traversal, shortest path, component analysis |
| **ParadeDB pg_search** | Optional | BM25 full-text scoring |
| **pg_trgm** | Optional | Trigram similarity for fuzzy text matching |

The backend **gracefully degrades** when extensions are missing — CRUD and basic
search work with plain PostgreSQL; graph traversal requires pgGraph; vector
search requires pgvector.

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GRAPH_BACKEND` | `ladybug` | Backend type: `memory`, `ladybug`, `neo4j`, `falkordb`, `postgresql` |
| `GRAPH_DB_PATH` | `knowledge_graph.db` | File path for LadybugDB |
| `GRAPH_DB_URI` | — | Connection URI for Neo4j or PostgreSQL |
| `GRAPH_DB_HOST` | `localhost` | Host for FalkorDB |
| `GRAPH_DB_PORT` | `6379`/`7687` | Port for FalkorDB/Neo4j |
| `GRAPH_DB_USER` | `neo4j` | Username for Neo4j/PostgreSQL |
| `GRAPH_DB_PASSWORD` | `password` | Password for Neo4j/PostgreSQL |
| `GRAPH_DB_NAME` | `agent_graph` | Database/graph name |
| `GRAPH_POOL_MIN` | `2` | PostgreSQL pool minimum connections |
| `GRAPH_POOL_MAX` | `10` | PostgreSQL pool maximum connections |
| `GRAPH_PGGRAPH_SCHEMA` | `public` | Schema for pgGraph table registration |

### Quick Start: PostgreSQL

```bash
# 1. Start the database
docker compose -f docker/pggraph.compose.yml up -d

# 2. Configure the backend
export GRAPH_BACKEND=postgresql
export GRAPH_DB_URI=postgresql://agent:agent@localhost:5433/agent_kg

# 3. Run the graph-os MCP server
graph-os
```

## Implementing a New Backend

1. Inherit from `GraphBackend` in `backends/base.py`
2. Implement all abstract methods: `execute()`, `execute_batch()`, `create_schema()`,
   `add_embedding()`, `semantic_search()`, `prune()`, `close()`
3. Register in the `create_backend()` factory in `backends/__init__.py`
4. Add optional dependency group to `pyproject.toml`
5. Add integration tests
