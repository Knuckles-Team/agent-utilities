# Tiered Graph Engine Architecture

> **CONCEPT:KG-3.01** — Tiered Graph Engine

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
│   │  LadybugDB (default)     │   │  Loaded via load_subgraph() │ │
│   │  Neo4j / PostgreSQL      │   │  Discarded after use        │ │
│   │  FalkorDB                │   │                             │ │
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
| **Production** | LadybugDB (default) | Compute scratchpad only | Enterprise deployment |
| **Scale-out** | Neo4j / PostgreSQL | Compute scratchpad only | 100K+ employees |
| **Development** | LadybugDB (local SQLite) | Compute scratchpad | Local dev with persistence |
| **Testing/CI** | MemoryBackend (`memory`) | Both storage & compute | Unit tests, small graphs |
| **Edge/Embedded** | MemoryBackend + `save_to_json()` | Both + manual persistence | IoT, offline agents |

### Configuration

```bash
# Default (LadybugDB — zero-config, self-contained SQLite)
export GRAPH_BACKEND=ladybug

# Testing/CI — no persistence needed
export GRAPH_BACKEND=memory

# Enterprise — Neo4j cluster
export GRAPH_BACKEND=neo4j
export GRAPH_DB_URI=bolt://neo4j-cluster:7687
export GRAPH_DB_USER=neo4j
export GRAPH_DB_PASSWORD=secret

# Enterprise — PostgreSQL with Apache AGE
export GRAPH_BACKEND=postgresql
export GRAPH_DB_URI=postgresql://db-host:5432/agent_graph
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
