# Tiered Graph Engine Architecture

> **CONCEPT:KG-2.0** — Tiered Graph Engine

## Overview

The Knowledge Graph uses a **two-tier architecture** that separates persistent storage from graph computation. This eliminates the dual-write OOM bottleneck and ensures each component is used for what it does best.

```
┌─────────────────────────────────────────────────────────────────┐
│                   IntelligenceGraphEngine                        │
│                                                                 │
│   ┌─────────────────────────┐   ┌─────────────────────────────┐ │
│   │  Tier 1: Source of Truth │   │  Tier 2: Compute Scratchpad │ │
│   │  (Persistent Backend)    │   │  (NetworkX — on-demand)     │ │
│   │                          │   │                             │ │
│   │  • Cypher queries        │   │  • PageRank / centrality    │ │
│   │  • CRUD via MERGE/SET    │   │  • Shortest paths           │ │
│   │  • Schema enforcement    │   │  • Community detection      │ │
│   │  • HNSW vector index     │   │  • VF2 subgraph isomorphism │ │
│   │  • Auto-backup           │   │  • Causal do-calculus       │ │
│   │  • Batch UNWIND          │   │  • Spectral clustering      │ │
│   │  • Cascade DETACH DELETE │   │  • Topological partitioning │ │
│   │                          │   │                             │ │
│   │  PostgreSQL (prod default)│  │  Loaded via load_subgraph() │ │
│   │  epistemic_graph (L1)    │   │  Discarded after use        │ │
│   │  contrib: Ladybug/Neo4j/ │   │                             │ │
│   │  FalkorDB (opt-in)       │   │                             │ │
│   └─────────────────────────┘   └─────────────────────────────┘ │
│                                                                 │
│   Memory-Only Mode (GRAPH_BACKEND=memory):                      │
│   NetworkX serves as BOTH storage and compute — for testing/CI  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Paths

### Write Path (CRUD)

Writes go to the **persistent backend only**. NetworkX is NOT updated on writes, preventing the OOM mirroring bottleneck at enterprise scale.

```python
# Backend available → Tier 1 only
if self.backend and not ephemeral:
    self._upsert_node("Memory", node.id, data)

# No backend (memory-only mode) → NX fallback
else:
    self.graph.add_node(node.id, **node.model_dump())
```

### Read Path (Queries)

Filtered queries use Cypher against the persistent backend. For memory-only mode, the `_query_nx_fallback()` method provides basic pattern matching.

```python
# Cypher query against backend
results = self.backend.execute(
    "MATCH (p:Policy) WHERE p.name CONTAINS $q RETURN p",
    {"q": query}
)
```

### Compute Path (Graph Algorithms)

Graph algorithms use `load_subgraph()` to **selectively load** only the nodes/edges needed for computation — NOT the full graph.

```python
# Load ONLY agent/tool nodes for centrality
subgraph = engine.load_for_centrality(["Agent", "Tool", "Skill"])
scores = nx.pagerank(subgraph)

# Load 3-hop neighborhood for impact analysis
subgraph = engine.load_for_impact_analysis(target_id)
impact = nx.ancestors(subgraph, target_id)
```

## Backend Selection

| Environment | Backend | NetworkX Role | Use Case |
|-------------|---------|---------------|----------|
| **Out-of-box default** | `tiered` (epistemic_graph L1 + **LadybugDB** L2) | Compute scratchpad only | Single self-contained binary, **no external system dependencies** |
| **Production (durable)** | `tiered` with `GRAPH_DB_URI` set → PostgreSQL L2 (pgvector + pg-age) | Compute scratchpad only | Enterprise, high-concurrency multi-agent swarms |
| **Development** | `tiered`+ladybug (default), or `memory` / `file` | Compute scratchpad | Local dev |
| **Testing/CI** | MemoryBackend (`GRAPH_BACKEND=memory`) | Both storage & compute | Unit tests, small graphs |
| **Edge/Embedded** | `tiered`+ladybug or `file` (JSON persistence) | Both + persistence | IoT, offline agents |

> **Note:** The out-of-box default is the zero-infra `tiered` backend: L1
> `epistemic_graph` (always included) + L2 **LadybugDB** (embedded, no server).
> Its L2 auto-switches to PostgreSQL whenever a DSN (`GRAPH_DB_URI` /
> `PGGRAPH_DSN`) is configured, or explicitly via `GRAPH_BACKEND_L2=postgresql`.
> `neo4j` and `falkordb` remain opt-in contrib backends (`backends/contrib/`),
> imported only when requested. The L1 compute tier defaults to
> `epistemic_graph` (`GRAPH_BACKEND_L1`). The production-profile guard rejects a
> ladybug L2 when `APP_PROFILE=production` and no durable DSN is set.

### Configuration

For a detailed walkthrough, compose files, and connection examples, see the [Deploying Graph Databases Guide](graph-db-deployment.md).

```bash
# 1. Out-of-box default — zero-infra tiered (epistemic_graph L1 + LadybugDB L2).
#    Nothing to set; this is what you get when GRAPH_BACKEND is unset.
export GRAPH_BACKEND=tiered            # implicit default
export GRAPH_BACKEND_L1=epistemic_graph
export GRAPH_BACKEND_L2=ladybug        # embedded, no external server

# 2. Production durable — tiered with a PostgreSQL L2 (pgvector + pg-age).
#    Setting a DSN auto-switches L2 to postgres; no other change needed.
export GRAPH_BACKEND=tiered
export GRAPH_DB_URI=postgresql://agent:agent@localhost:5433/agent_kg

# 3. Pure ephemeral in-memory (testing/CI; no persistence)
export GRAPH_BACKEND=memory

# 4. Single durable backend without the L1 compute tier — PostgreSQL only
export GRAPH_BACKEND=postgresql
export GRAPH_DB_URI=postgresql://agent:agent@localhost:5433/agent_kg

# Opt-in contrib backends (imported only when requested):

# export GRAPH_BACKEND=ladybug   # standalone embedded LadybugDB (no L1 tier)

# Neo4j cluster
# export GRAPH_BACKEND=neo4j
# export GRAPH_DB_URI=bolt://localhost:7687
# export GRAPH_DB_USER=neo4j
# export GRAPH_DB_PASSWORD=password

# FalkorDB
# export GRAPH_BACKEND=falkordb
# export GRAPH_DB_HOST=localhost
# export GRAPH_DB_PORT=6380
# export GRAPH_DB_NAME=agent_graph
```

## Why Not a Cypher Parser for NetworkX?

Building a Cypher interpreter for NetworkX would:
1. Take ~3 weeks to implement MATCH, MERGE, SET, DETACH DELETE, WHERE, RETURN
2. Still lack persistence, ACID, vector indexing, and auto-backup
3. Essentially rebuild LadybugDB poorly in Python

The tiered architecture avoids this entirely by using each tool where it excels.

## Key API Methods

| Method | Tier | Purpose |
|--------|------|---------|
| `_upsert_node()` | 1 | Idempotent node write to backend |
| `link_nodes()` | 1 | Relationship creation via Cypher MERGE |
| `query_cypher()` | 1 | Direct Cypher query execution |
| `load_subgraph()` | 1→2 | Gateway: load from backend into NX |
| `load_for_centrality()` | 1→2 | Typed loader for PageRank/centrality |
| `load_for_impact_analysis()` | 1→2 | Typed loader for impact analysis |
| `_is_memory_only` | — | Property: True when no backend exists |

## Migration Notes

The `ephemeral` flag on `add_node()` and `link_nodes()` controls the tier:
- `ephemeral=False` (default): Backend source of truth
- `ephemeral=True`: NX only (temporary compute nodes, council verdicts)

Code that previously relied on dual-writes should use `load_subgraph()` to hydrate NX when graph algorithms are needed.
