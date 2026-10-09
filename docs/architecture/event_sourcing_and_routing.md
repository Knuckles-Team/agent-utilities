# Event Sourcing and Query Routing Architecture

## Overview

To enable an Enterprise-grade Knowledge Graph platform, `agent-utilities` employs a sophisticated, Local-First Event Sourcing architecture combined with a Cost-Based Query Router. This allows the system to smoothly toggle between lightweight, single-process execution (during development) and heavy-duty, multi-node deployment (in production) without any code changes.

## 1. Event Sourcing & The EventBus (`EventBackend`)

Because Python's network-bound queries are too slow for deep topological algorithms (like PageRank or pathfinding), `agent-utilities` delegates high-performance graph processing to an in-memory Rust layer (`epistemic-graph`).

To prevent Data Drift between the engine authority and any working-set cache, this repository use an **Event Sourcing Pattern**:

1. When a mutation (`INSERT` / `DELETE`) commits through the engine authority (`graph_compute.py` / envelope ingestion), it publishes a `TRIPLE_INSERT` or `TRIPLE_DELETE` event to the `kg.mutations` topic via the `EventBackend`.
2. Consumer systems (like `epistemic-graph`) listen to this topic and dynamically patch their local working set in milliseconds.

External SPARQL triplestore federation (a backend exposing `supports_sparql`/`execute_sparql`) is owned by the epistemic-graph engine rather than a backend implementation in this repository.

### Local-First Paradigm

The `EventBackend` is built on a **Local-First** paradigm:
- **`MemoryEventBackend`:** Uses simple `asyncio.Queue` primitives. It requires zero configuration, no external services, and is perfect for local testing and isolated agents.
- **`RedpandaEventBackend`:** Uses Confluent Redpanda (KRaft mode). When the system scales, it switches to Redpanda by simply providing `REDPANDA_BOOTSTRAP_SERVERS`, enabling cross-container, distributed Pub/Sub.

## 2. Cost-Based Query Router (`QueryRouter`)

Instead of throwing all queries directly at one store (which can result in massive, slow JOINs), the `QueryRouter` intelligently classifies and routes queries based on a "Cost Heuristic."

- **Rust engine (the authority):** the source of truth — extremely fast for `expected_hops >= 2` or `QueryType.TOPOLOGICAL` queries.
- **Working-set cache:** fast for local subset `FILTERED_MATCH` queries.
- **SPARQL mirror:** an optional read mirror used for raw `QueryType.SPARQL` queries when a backend declares `supports_sparql`; on `requires_freshness=True` the router goes to the engine authority. No built-in backend currently declares this — it is epistemic-graph's federation surface to populate.
- **Vector store:** used for Semantic Similarity (`QueryType.SEMANTIC`).

By analyzing `expected_hops` and `requires_freshness`, the router minimizes latency and offloads work to the most efficient query path automatically.
