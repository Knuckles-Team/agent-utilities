# Graph Engine: One Authority + Mirrors

> **CONCEPT:AU-KG.query.object-graph-mapper** — The epistemic-graph engine as the single database.

## Overview

The Knowledge Graph is backed by **one database — the out-of-process
epistemic-graph Rust engine**. It is the authority and combines graph compute,
hot-read caching, semantic/ontology reasoning, and durable persistence. Python
and GraphOS reach it only through the authenticated MessagePack client over a
private UDS or loopback-TCP transport, or an explicitly configured protected
remote endpoint. There is no embedded Python engine and no second read authority.

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
# GraphOS validates the session and commits to the engine authority.
result = await graph_os.call_tool(
    "graph_write",
    {"action": "add_node", "node_id": node.id, "node_type": "Memory", "properties": data},
)
# Mirror delivery proceeds through the durable replayable outbox after commit.
```

### Read Path (Queries)

**All reads are served by the engine.** Filtered queries run Cypher directly
against the engine; mirrors are never consulted on the read path.

```python
# GraphOS applies identity, tenant, scope, and policy before dispatching the
# bounded query to the authoritative engine.
results = await graph_os.call_tool(
    "graph_query",
    {"action": "cypher", "query": approved_query, "params": {"q": query}},
)
```

### Compute Path (Graph Algorithms)

Graph algorithms run **inside the engine** — PageRank, centrality, impact
analysis, community detection, and subgraph isomorphism are native engine
operations over the authoritative graph, not a separate scratchpad that must be
loaded and discarded.

```python
# Bounded personalized PageRank executes inside the authoritative engine.
scores = await graph_os.call_tool(
    "engine_analytics",
    {"action": "personalized_pagerank", "params_json": seed_frontier_json},
)
```

## Authority and projections

Epistemic-graph is always authoritative. Projection declarations determine
whether committed writes also fan out to external stores:

| Declaration | What runs | Use case |
|---|---|---|
| No mirrors **(default)** | The engine only — the tiny, self-contained database | Default everywhere: laptop, edge/offline agents, demos, single-node, and most production. No external system dependencies. |
| `GRAPH_MIRROR_TARGETS` or `role=mirror` | The engine + one or more projections | External interop/BI/DR; writes fan out asynchronously. |

> The default is the engine alone — a single self-contained database with no
> external server required. Mirrors are purely optional and only ever receive an
> asynchronous, lossless copy for interop, business intelligence, or disaster
> recovery. They are **never** the authority and **never** on the read path.

### Configuration

For a detailed walkthrough, compose files, and connection examples, see the
[Deploying Graph Databases Guide](graph-db-deployment.md).

```json
{
  "GRAPH_MIRROR_TARGETS": ["analytics-mirror"],
  "KG_CONNECTIONS": [
    {
      "name": "analytics-mirror",
      "backend": "age",
      "role": "mirror",
      "connection_profile_ref": "secret://graphs/analytics-mirror"
    }
  ]
}
```

Omit both projection settings for the authoritative-engine-only default. The connection
profile is resolved only when the mirror connects and may contain the endpoint,
database selector, identity, verified TLS policy, and credentials. AgentConfig and
documentation retain only the neutral alias and reference. Use
`agent-utilities-doctor` to validate reference resolution, reachability, role, and
TLS before enabling fan-out.

### External sources are not mirrors

`EXTERNAL_GRAPH_CONNECTORS` describes read-only sources from which governed data is
materialized into the engine; it does not place those databases on the operational
read or write path. Neo4j/openCypher, AGE, LadybugDB/Kuzu, remote epistemic-graph,
and GraphQL all use bounded schema discovery, digest-bound mapping approval, drift
checks, and native `ChangeEnvelope` ingestion. Source connection, authentication,
TLS, query, variables, schema, and ontology documents remain behind runtime
references. See [Universal External Graph Connectors](../architecture/universal-external-graph-connectors.md).

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
| GraphOS `graph_write` | Policy-governed node and relationship mutation |
| GraphOS `graph_query` | Identity-, tenant-, scope-, and policy-governed query dispatch |
| GraphOS `engine_analytics` | Bounded native in-engine graph compute |

## Deployment shapes

- **Tiny / self-contained installation** — GraphOS supervises the bundled Rust
  engine as a separate child over a private local transport. No external service
  is managed by the operator. The fixed authority needs no selector.
- **Enterprise** — a shared/remote engine reached over
  `GRAPH_SERVICE_ENDPOINTS` (optionally sharded), with `GRAPH_MIRROR_TARGETS`
  populating Postgres/Neo4j/etc. for interop and DR.
