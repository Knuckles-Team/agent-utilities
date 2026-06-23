# Unified Intelligence Graph (UIG)

The Knowledge Graph is the cognitive substrate of the `agent-utilities` ecosystem, providing long-term memory, deep structural codebase awareness, and cross-domain research knowledge.

## Overview

The UIG is built on **one database — the epistemic-graph engine authority** (the Rust-native engine doing compute, in-memory cache, semantic/ontology, and durable persistence), with **NetworkX** retained as an ephemeral in-memory scratchpad for high-performance topological algorithms. Writes fan out to optional durable **mirrors** (Postgres/pg-age, Neo4j, FalkorDB, LadybugDB) for interop and DR. It follows a 12-phase topological pipeline to ingest and analyze workspaces.

## Core Components

- **Graph Engine (`engine.py`)**: The central coordinator for graph operations, linking, and querying.
- **Pipeline (`pipeline/`)**: A 12-phase ingestion pipeline that transforms raw code and metadata into a rich intelligence graph.
- **Backends (`backends/`)**: The epistemic-graph engine authority plus hot-swappable durable mirrors (Postgres/pg-age, Neo4j, FalkorDB, LadybugDB) that receive the engine's fan-out write stream.
- **Knowledge Base (`kb/`)**: An LLM-maintained personal wiki system for domain-specific research and documentation.
- **Maintenance (`maintenance.py`)**: Autonomous routines for vector enrichment, pruning, and summarization.
- **Codemaps (`codemaps.py`)**: Structural codebase analysis and impact prediction.

## Reasoning Views (MAGMA)

The engine supports four orthogonal retrieval views:
- **Semantic**: Conceptual similarity via vector embeddings.
- **Temporal**: Chronological episodic memory with Ebbinghaus decay.
- **Causal**: Reasoning traces and "Why" links between decisions.
- **Entity**: Structural knowledge of People, Organizations, and Code Symbols.

## Usage

The graph is typically accessed via the `knowledge_engine` attribute in `AgentDeps`:

```python
async def search_kg(ctx: RunContext[AgentDeps], query: str):
    results = await ctx.deps.knowledge_engine.search(query)
    return results
```

## Maintenance

- **Persistence**: Ensure `GRAPH_BACKEND` is configured correctly (defaults to `epistemic_graph` — the engine authority; durable by itself).
- **Pruning**: The `GraphMaintainer` runs periodically to remove low-importance nodes (PageRank < 0.05).
- **Schema**: All nodes and edges follow the Pydantic schemas defined in `agent_utilities/models/knowledge_graph.py`.
