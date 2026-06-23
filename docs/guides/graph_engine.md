# Graph Engine: One Authority + Mirrors

> **CONCEPT:KG-2.0** — The epistemic-graph engine as the single database.

## Overview

The Knowledge Graph is backed by **one database — the epistemic-graph engine**.
It is the **authority** (the system of record) and does everything in a single
engine: graph **compute**, an in-memory **cache**, **semantic/ontology**
reasoning, AND durable **persistence**. There is no separate "storage tier" and
no separate "compute tier" — it is one engine.

```
┌─────────────────────────────────────────────────────────────────┐
│              epistemic-graph engine (THE database)               │
│                                                                 │
│   • Authority / system of record (durable persistence)          │
│   • In-memory cache for hot reads                               │
│   • Graph compute (PageRank, centrality, shortest paths,        │
│     community detection, VF2 subgraph isomorphism, causal        │
│     do-calculus, spectral clustering, topological partitioning) │
│   • Cypher queries, CRUD via MERGE/SET, schema enforcement      │
│   • HNSW vector index, batch UNWIND, cascade DETACH DELETE      │
│                                                                 │
│   ALL READS are served here. WRITES commit here first.          │
└───────────────────────────────┬─────────────────────────────────┘
                                 │  async, lossless fan-out
                                 │  (durable outbox, replay-on-reconnect)
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│            Optional mirrors — interop / BI / DR only             │
│                                                                 │
│   Postgres / pg-age   ·   Neo4j   ·   FalkorDB   ·   Ladybug     │
│                                                                 │
│   Never on the read path. Never the authority. Populated        │
│   asynchronously for external query, business intelligence,      │
│   and disaster recovery.                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Paths

### Write Path (CRUD)

Writes **commit to the engine** (the authority) first. Once committed, they fan
out **asynchronously and losslessly** to any configured mirrors via a durable
outbox that replays on reconnect — so a mirror being down or slow never blocks a
write and never loses data.

```python
# Commit to the engine authority — this is the durable write
self._upsert_node("Memory", node.id, data)

# The engine asynchronously fans the committed change out to mirrors
# (Postgres/Neo4j/FalkorDB/Ladybug) through a durable, replay-on-reconnect
# outbox. The write returns as soon as the engine has committed.
```

### Read Path (Queries)

**All reads are served by the engine.** Filtered queries run Cypher directly
against the engine; mirrors are never consulted on the read path.

```python
# Cypher query against the engine authority
results = self.engine.execute(
    "MATCH (p:Policy) WHERE p.name CONTAINS $q RETURN p",
    {"q": query}
)
```

### Compute Path (Graph Algorithms)

Graph algorithms run **inside the engine** — PageRank, centrality, impact
analysis, community detection, and subgraph isomorphism are native engine
operations over the authoritative graph, not a separate scratchpad that must be
loaded and discarded.

```python
# Centrality over agent/tool nodes — computed in the engine
scores = engine.pagerank(labels=["Agent", "Tool", "Skill"])

# Impact analysis — 3-hop ancestry over the authoritative graph
impact = engine.impact_analysis(target_id)
```

## Backend Selection

`GRAPH_BACKEND` selects whether mirrors fan out alongside the engine:

| `GRAPH_BACKEND` | What runs | Use case |
|---|---|---|
| `epistemic_graph` **(default)** | The engine only — the tiny, self-contained database | Default everywhere: laptop, edge/offline agents, demos, single-node, and most production. No external system dependencies. |
| `fanout` | The engine + one or more mirrors | When you also need external interop/BI/DR: writes fan out asynchronously to `GRAPH_MIRROR_TARGETS`. |
| `memory` | Pure ephemeral in-memory engine | Unit tests / CI; no persistence. |

> The default is the engine alone — a single self-contained database with no
> external server required. Mirrors are purely optional and only ever receive an
> asynchronous, lossless copy for interop, business intelligence, or disaster
> recovery. They are **never** the authority and **never** on the read path.

### Configuration

For a detailed walkthrough, compose files, and connection examples, see the
[Deploying Graph Databases Guide](graph-db-deployment.md).

```bash
# 1. Default — the engine only (self-contained, zero-infra).
#    Nothing to set; this is what you get when GRAPH_BACKEND is unset.
export GRAPH_BACKEND=epistemic_graph    # implicit default

# 2. Engine + mirrors — fan committed writes out for interop/BI/DR.
export GRAPH_BACKEND=fanout
export GRAPH_MIRROR_TARGETS=postgresql  # comma-separated mirror list
export GRAPH_DB_URI=postgresql://agent:agent@localhost:5433/agent_kg

# 3. Pure ephemeral in-memory (testing/CI; no persistence)
export GRAPH_BACKEND=memory

# Mirror connection examples (only read when a mirror is in GRAPH_MIRROR_TARGETS):

# Postgres / pg-age mirror
# export GRAPH_DB_URI=postgresql://agent:agent@localhost:5433/agent_kg

# Neo4j mirror
# export GRAPH_DB_URI=bolt://localhost:7687
# export GRAPH_DB_USER=neo4j
# export GRAPH_DB_PASSWORD=password

# FalkorDB mirror
# export GRAPH_DB_HOST=localhost
# export GRAPH_DB_PORT=6380
# export GRAPH_DB_NAME=agent_graph

# Ladybug mirror
# export GRAPH_DB_PATH=/data/agent.db
```

## Why the engine does it all

A separate storage layer plus a separate compute layer would mean:
1. Re-implementing persistence, ACID, vector indexing, and backup outside the engine.
2. A dual-write / mirror-on-the-hot-path bottleneck that OOMs at enterprise scale.
3. Read-path dependence on a second system that can be down or stale.

Collapsing everything into one engine — and pushing mirrors strictly
off the hot path, asynchronously — avoids all three.

## Key API Methods

| Method | Purpose |
|--------|---------|
| `_upsert_node()` | Idempotent node write, committed to the engine authority |
| `link_nodes()` | Relationship creation via Cypher MERGE |
| `query_cypher()` | Direct Cypher query against the engine |
| `pagerank()` / `impact_analysis()` | Native in-engine graph compute |

## Deployment shapes

- **Tiny / self-contained** — the embedded engine, one process, no external
  servers. This is `GRAPH_BACKEND=epistemic_graph` (the default).
- **Enterprise** — a shared/remote engine reached over
  `GRAPH_SERVICE_ENDPOINTS` (optionally sharded), with `GRAPH_BACKEND=fanout`
  and `GRAPH_MIRROR_TARGETS` populating Postgres/Neo4j/etc. for interop and DR.

## Migration Notes

The `ephemeral` flag on `add_node()` and `link_nodes()` controls durability:
- `ephemeral=False` (default): committed to the engine authority.
- `ephemeral=True`: kept transient (temporary compute nodes, council verdicts) —
  not persisted and not fanned out to mirrors.
