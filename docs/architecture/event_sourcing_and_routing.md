# Event Sourcing and Query Routing Architecture

## Overview

To enable an Enterprise-grade Knowledge Graph platform, `agent-utilities` employs a sophisticated, Local-First Event Sourcing architecture combined with a Cost-Based Query Router. This allows the system to seamlessly toggle between lightweight, single-process execution (during development) and heavy-duty, multi-node deployment (in production) without any code changes.

## 1. SPARQL Abstraction (`SparqlAdapter`)

The `SparqlAdapter` (found in `agent_utilities.knowledge_graph.backends.sparql.base`) abstracts W3C standard SPARQL operations (`execute_sparql_query`, `execute_sparql_update`, `upload_graph`, `download_graph`).

This enables `agent-utilities` to be entirely vendor-agnostic. Currently, `JenaFusekiBackend` implements this adapter, but it seamlessly paves the way for `StardogBackend` or `NeptuneBackend`.

## 2. Event Sourcing & The EventBus (`EventBackend`)

Because Python's network-bound SPARQL queries are too slow for deep topological algorithms (like PageRank or pathfinding), `agent-utilities` delegates high-performance graph processing to an in-memory Rust layer (`epistemic-graph`).

To prevent Data Drift between the authoritative SPARQL store and the Rust cache, we use an **Event Sourcing Pattern**:

1. When a mutation (`INSERT` / `DELETE`) occurs through the `JenaFusekiBackend`, it intercepts the update.
2. The Backend instantly publishes a `TRIPLE_INSERT` or `TRIPLE_DELETE` event to the `kg.mutations` topic via the `EventBackend`.
3. Consumer systems (like `epistemic-graph`) listen to this topic and dynamically patch their local working set in milliseconds.

### Local-First Paradigm

The `EventBackend` is built on a **Local-First** paradigm:
- **`MemoryEventBackend`:** Uses simple `asyncio.Queue` primitives. It requires zero configuration, no external services, and is perfect for local testing and isolated agents.
- **`RedpandaEventBackend`:** Uses Confluent Redpanda (KRaft mode). When the system scales, it switches to Redpanda by simply providing `REDPANDA_BOOTSTRAP_SERVERS`, enabling cross-container, distributed Pub/Sub.

## 3. Cost-Based Query Router (`QueryRouter`)

Instead of throwing all queries directly at SPARQL (which can result in massive, slow JOINs), the `QueryRouter` intelligently classifies and routes queries based on a "Cost Heuristic."

- **Rust engine (the authority):** the source of truth — extremely fast for `expected_hops >= 2` or `QueryType.TOPOLOGICAL` queries.
- **Working-set cache:** fast for local subset `FILTERED_MATCH` queries.
- **SPARQL mirror:** an optional read mirror used for raw `QueryType.SPARQL` queries; on `requires_freshness=True` the router goes to the engine authority.
- **Vector store:** used for Semantic Similarity (`QueryType.SEMANTIC`).

By analyzing `expected_hops` and `requires_freshness`, the router minimizes latency and offloads work to the most efficient query path automatically.
