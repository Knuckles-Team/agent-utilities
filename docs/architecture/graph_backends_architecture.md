# Graph Backend Architecture

The Knowledge Graph engine supports multiple backend implementations through a
unified `GraphBackend` abstract interface. All backends provide the same core
capabilities: Cypher query execution, vector search, node/edge CRUD, and
optional SPARQL support.

The **default** backend is the zero-dependency Rust-native `EpistemicGraph`
(`GRAPH_BACKEND=memory`/`file`/`epistemic_graph`); the **production** durable
backend is PostgreSQL (`GRAPH_BACKEND=postgresql`), optionally fronted by the
`tiered` write-through store (L1 EpistemicGraph + L3 Postgres). LadybugDB, Neo4j,
and FalkorDB are first-class backends whose drivers install as optional extras
(`backends/contrib/`).

> **Verified parity (KG-2.7).** Node properties (declared / ad-hoc / nested),
> edge existence, **edge properties**, and vector search round-trip on **every**
> backend; the full cross-backend matrix and how to run it live are in
> [backend-parity-and-profile-testing](../guides/backend-parity-and-profile-testing.md).
> SPARQL is served **locally over any backend** via the OWL/RDF layer (see
> [owl_rdf_layer](owl_rdf_layer.md)).

> **PostgreSQL runs Apache AGE (`GRAPH_PG_AGE=1` / `backend_type=age`).** This
> executes **real openCypher** via AGE's `cypher()` function — `count(r)`,
> `RETURN … AS alias`, multi-hop and variable-length traversal all work natively —
> retiring the bounded regex Cypher→SQL transpiler (still the default when AGE is
> off). pgvector continues to back embeddings. Image: `docker/pggraph-age.compose.yml`.

## Architecture Overview

```mermaid
graph TB
    subgraph "IntelligenceGraphEngine"
        A["KG-2.0: query_cypher()"] --> B["KG-2.0: backend.execute()"]
        C["KG-2.0: add_node()"] --> B
        D["KG-2.0: link_nodes()"] --> B
        E["KG-2.3: search_hybrid()"] --> F["KG-2.3: backend.semantic_search()"]
        G["KG-2.0: load_subgraph()"] --> H["Rust GraphComputeEngine\n(Tier 2 Compute)"]
        QR["KG-2.7: QueryRouter"] --> B
        QR --> H
        QR --> F
    end

    B --> I{"KG-2.0: Backend Type?"}

    subgraph "KG-2.7: EpistemicGraph (Default — memory/file)"
        I -->|memory / file / epistemic_graph| EG["Rust-native EpistemicGraph\n(zero-dep working store)"]
    end

    subgraph "KG-2.7: Tiered (L1 EpistemicGraph + L3 Postgres)"
        I -->|tiered| TI["TieredGraphBackend\nwrite-through L1→L3"]
    end

    subgraph "KG-2.0: LadybugDB (opt-in contrib)"
        I -->|ladybug| J["Native Cypher\nSQLite + HNSW"]
    end

    subgraph "KG-2.0: Neo4j (opt-in contrib)"
        I -->|neo4j| K["KG-2.0: Native Cypher\nBolt Protocol"]
    end

    subgraph "FalkorDB (opt-in contrib)"
        I -->|falkordb| L["KG-2.0: Cypher via\nRedis Protocol"]
    end

    subgraph "PostgreSQL (Production durable)"
        I -->|postgresql| M["KG-2.0: Cypher → SQL\nTranspiler (default)"]
        I -->|age / GRAPH_PG_AGE=1| AGE["KG-2.7: Apache AGE\ncypher() — real openCypher"]
        M --> N["PostgreSQL Tables"]
        M --> O["KG-2.0: pgGraph Extension\n(CSR Traversal)"]
        AGE --> AGN["AGE graph (agtype)\n+ kg_embeddings (pgvector)"]
        F --> P["KG-2.3: pgvector\n(Cosine Search)"]
        F --> Q["KG-2.3: ParadeDB BM25\n(Lexical Search)"]
    end

    subgraph "SPARQL Backends"
        I -->|jena_fuseki| S1["Jena Fuseki / EpistemicGraph Compute In-Memory\n(pyjena_fuseki)"]
        I -->|fuseki| S2["Apache Fuseki\n(HTTP SPARQL)"]
    end

    subgraph "Memory (Testing)"
        I -->|memory| R["GraphComputeEngine\nIn-Memory"]
    end
```

## Backend Comparison

| Capability | epistemic_graph (default) | LadybugDB | PostgreSQL (AGE) | Neo4j | FalkorDB |
|---|:---:|:---:|:---:|:---:|:---:|
| **Status** | **Default (Rust-native)** | first-class (extra) | **Production (durable)** | first-class (extra) | first-class (extra) |
| Cypher Support | subset (id-anchored)¹ | Native (Kuzu) | **Native (AGE)** / transpiled | Native | Native |
| Node props (declared/ad-hoc/nested) | ✅ | ✅ (ad-hoc in `metadata`) | ✅ | ✅ | ✅ |
| **Edge properties** | ✅ | ✅ (JSON `r.properties`) | ✅ | ✅ | ✅ |
| Vector Search | ✅ | ✅ | ✅ pgvector | ✅ (`:Embeddable`) | ⚠️ AVX2 host² |
| SPARQL (via OWL/RDF layer) | ✅ local | ✅ local | ✅ local | ✅ local | ✅ local |
| Graph Traversal (multi-hop) | compute/L3¹ | ✅ | ✅ (AGE) | ✅ | ✅ |
| Connection Pooling | UDS client | File Lock | ✅ psycopg_pool | ✅ | — |
| Persistence | optional/in-mem | File | Server | Server | Redis |
| Zero Config | ✅ | ✅ | — | — | — |

¹ epistemic_graph is the in-memory **L1 working store**; `backend.execute` interprets
an operational id-anchored Cypher subset, and multi-hop traversal is served via the
compute layer / tiered L3 — by design. ² FalkorDB vector search is code-correct
(Cypher `CREATE VECTOR INDEX` + `db.idx.vector.queryNodes`) but the `falkordb` image
SIGILLs on 768-dim vector ops on non-AVX2 host CPUs.

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

The engine speaks Cypher; PostgreSQL speaks SQL. The `transpile()` function in
`backends/cypher_transpiler.py` handles the translation for all patterns the
engine generates:

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
| `GRAPH_BACKEND` | `memory` | Backend type: `memory`/`file`/`epistemic_graph` (default, Rust-native), `postgresql` (production durable), `tiered`, `jena_fuseki`, `fuseki`; opt-in contrib: `ladybug`, `neo4j`, `falkordb` |
| `GRAPH_BACKEND_L1` | `epistemic_graph` | L1 working store type when `GRAPH_BACKEND=tiered` |
| `GRAPH_DB_PATH` | `knowledge_graph.db` | File path for EpistemicGraph (`file` mode) / LadybugDB |
| `GRAPH_DB_URI` | — | Connection URI for Neo4j or PostgreSQL |
| `GRAPH_DB_HOST` | `localhost` | Host for FalkorDB |
| `GRAPH_DB_PORT` | `6379`/`7687` | Port for FalkorDB/Neo4j |
| `GRAPH_DB_USER` | `neo4j` | Username for Neo4j/PostgreSQL |
| `GRAPH_DB_PASSWORD` | `password` | Password for Neo4j/PostgreSQL |
| `GRAPH_DB_NAME` | `agent_graph` | Database/graph name |
| `GRAPH_POOL_MIN` | `2` | PostgreSQL pool minimum connections |
| `GRAPH_POOL_MAX` | `10` | PostgreSQL pool maximum connections |
| `GRAPH_PGGRAPH_SCHEMA` | `public` | Schema for pgGraph table registration |
| `GRAPH_FUSEKI_URL` | `http://localhost:3030` | Jena/Apache Fuseki server URL |
| `GRAPH_FUSEKI_DATASET` | `agent_kg` | Fuseki dataset name |
| `GRAPH_FUSEKI_USER` / `GRAPH_FUSEKI_PASSWORD` | — | Optional Fuseki credentials |

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
3. Optionally override `supports_sparql` and `execute_sparql()` for SPARQL support
4. Register in the `create_backend()` factory in `backends/__init__.py`
5. Add optional dependency group to `pyproject.toml`
6. Add integration tests
