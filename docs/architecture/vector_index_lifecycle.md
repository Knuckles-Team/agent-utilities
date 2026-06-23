# HNSW Vector Index Lifecycle

This document describes how the Knowledge Graph manages HNSW (Hierarchical
Navigable Small World) vector indexes for sub-second semantic search across
121K+ nodes.

## Overview

The KG stores embedding vectors (768-dim, `FLOAT[768]`) on 15 node tables.
The exact set is derived dynamically from `schema_definition.py` at runtime (every
node whose `embedding` column is a `FLOAT` array), so it stays in sync as the schema
evolves rather than being hardcoded.
Without HNSW indexes, vector search degrades to **O(N) brute-force** cosine
similarity — scanning all nodes with embeddings. With HNSW indexes, search
is **O(log N)** via LadybugDB's native vector extension.

### The Kuzu Constraint

LadybugDB (Kuzu) **does not support `SET` on properties that are part of a
vector index**. This means:

- ❌ Cannot write new embeddings while HNSW indexes exist on that table
- ✅ Can read/search using HNSW indexes at any time
- ✅ `CREATE_VECTOR_INDEX` skips tables that already have indexes (`IF NOT EXISTS`)

This constraint drives the entire lifecycle design.

## Lifecycle: Drop → Ingest → Build

```
┌───────────┐      ┌──────────────┐      ┌──────────────┐
│  Drop     │      │   Ingest     │      │   Build      │
│  indexes  │─────▶│  (SET emb)   │─────▶│   indexes    │
│  for      │      │   succeeds   │      │   for same   │
│  affected │      │   because    │      │   tables     │
│  tables   │      │   no index   │      │              │
│  ONLY     │      │   blocking   │      │              │
└───────────┘      └──────────────┘      └──────────────┘
```

### Phase 1: Targeted Drop (Pre-Ingestion)

When `submit_task()` is called, it drops HNSW indexes **only for the tables
affected by this task type**:

| Task Type      | Tables Dropped | Tables Left Intact |
|----------------|----------------|--------------------|
| `codebase`     | `Code`         | Article, Concept, Agent, etc. |
| `document`     | `Article`      | Code, Concept, Agent, etc. |
| `conversation` | `Message`      | Code, Article, Concept, etc. |

This means if you're ingesting a PDF, the `Code` table's HNSW index stays
active and searches against code remain O(log N).

The drop is **idempotent** — dropping a non-existent index is silently skipped.
Each table is only dropped once per ingestion batch (tracked via `_dropped_tables` set).

### Phase 2: Ingestion

Standard ingestion proceeds — embeddings are written via `SET n.embedding = $emb`.
This succeeds because the affected table's index was dropped in Phase 1.

### Phase 3: Auto-Build (Post-Ingestion)

After the last worker completes and the task queue is empty:

1. `_maybe_build_vector_indexes()` fires automatically
2. Only rebuilds indexes for tables that were dropped (`_dropped_tables`)
3. Runs in a **background thread** (`KG-IndexBuilder`) — non-blocking
4. Resets all flags so the next ingestion batch re-triggers the cycle

## Embedding Tables (15 total)

These are the node tables with embedding columns, derived from `schema_definition.py`:

| Table | Primary Content | Typical Node Count |
|-------|----------------|-------------------|
| `Code` | Source code (Code/Test/Feature nodes) | ~78K |
| `Symbol` | Code symbols | varies |
| `Article` | Research papers, documents | ~2.6K |
| `Message` | Conversation messages | ~4.6K |
| `Concept` | Extracted concepts | ~79 |
| `Agent` | Registered agents | varies |
| `AgentTemplate` | Agent templates | varies |
| `Tool` | MCP tools | varies |
| `Skill` | Universal skills | varies |
| `DiffEntry` | Git diff entries | varies |
| `CallableResource` | Callable resources | varies |
| `SystemPrompt` | System prompts | varies |
| `KBConcept` | Knowledge base concepts | varies |
| `KBFact` | Knowledge base facts | varies |
| `KnowledgeBaseTopic` | KB topics | varies |

## Search Performance Tiers

| Tier | When Active | Performance | Method |
|------|------------|-------------|--------|
| **HNSW** | After `build_indexes` | O(log N), sub-second | Native `QUERY_VECTOR_INDEX` |
| **Label-scoped fallback** | When HNSW unavailable | ~2.5K comparisons, seconds | Per-table Cypher with LIMIT |
| **Unbounded O(N)** | Never (removed) | 80K+ comparisons, minutes | ~~`MATCH (n) WHERE n.embedding IS NOT NULL`~~ |

The label-scoped fallback queries high-value tables first with conservative limits:
- Article: 500, Concept: 200, KBConcept: 200, KBFact: 200
- Agent: 100, Tool: 200, Skill: 100, Code: 1000

## Manual Operations

### Build Indexes (via MCP)

```
graph_ingest(action="rebuild_indexes")
```

Starts HNSW index building in a **background thread** — returns immediately.
Run this after any bulk ingestion completes or when you want to optimize search.

#### Expected Build Time

| Node Count | Example | Build Time |
|-----------|---------|------------|
| ~80K | Full workspace (78K Code + 2.6K Article) | ~3 minutes |
| ~160K | Double workspace | ~6 minutes |
| ~10K | Small project | ~30 seconds |

Build time scales roughly linearly with total embedded nodes. The `Code` table
(78K nodes) dominates the build time in a typical workspace.

### Check if Search is Using HNSW

If search returns results in sub-second time, HNSW is active. If it takes
2-10 seconds, the label-scoped fallback is being used (still functional,
just not optimal).

## Implementation Files

| File | Component |
|------|-----------|
| `knowledge_graph/backends/contrib/ladybug_backend.py` | `build_vector_indices()`, `drop_vector_indices()` (Ladybug/Kuzu is a demoted `contrib` mirror; the authority is `epistemic_graph_backend.py`, with `postgresql_backend.py` as an optional mirror) |
| `knowledge_graph/retrieval/hybrid_retriever.py` | `_vector_search_native()`, label-scoped fallback |
| `knowledge_graph/core/engine_tasks.py` | `submit_task()` (pre-drop), `_maybe_build_vector_indexes()` (post-build) |
| `mcp/kg_server.py` | `graph_ingest(action="rebuild_indexes")` |

## Key Design Decisions

1. **Targeted drop/build**: Only affects tables written to during ingestion,
   preserving HNSW on unaffected tables for continued fast search.

2. **No startup index build**: Building HNSW on 78K+ Code vectors is CPU/memory
   intensive (~minutes, GBs of RAM). This would block MCP server startup.

3. **Background thread for rebuilds**: Post-ingestion index building runs in a
   daemon thread so worker threads aren't blocked.

4. **Label-scoped fallback as safety net**: Even without HNSW, search is bounded
   to ~2,500 cosine comparisons instead of 80K+, ensuring worst-case is seconds
   rather than minutes.
