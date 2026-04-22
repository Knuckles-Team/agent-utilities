# Unified Intelligence Graph (UIG)

The Knowledge Graph is the cognitive substrate of the `agent-utilities` ecosystem, providing long-term memory, deep structural codebase awareness, and cross-domain research knowledge.

## Overview

The UIG unifies **NetworkX** (for high-performance topological algorithms) and **LadybugDB/Neo4j/FalkorDB** (for persistent Cypher queries and hybrid vector search). It follows a 12-phase topological pipeline to ingest and analyze workspaces.

## Core Components

- **Graph Engine (`engine.py`)**: The central coordinator for graph operations, linking, and querying.
- **Pipeline (`pipeline/`)**: A 12-phase ingestion pipeline that transforms raw code and metadata into a rich intelligence graph.
- **Backends (`backends/`)**: Hot-swappable persistence layers supporting LadybugDB (default), Neo4j, and FalkorDB.
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

- **Persistence**: Ensure `GRAPH_BACKEND` is configured correctly (defaults to `ladybug`).
- **Pruning**: The `GraphMaintainer` runs periodically to remove low-importance nodes (PageRank < 0.05).
- **Schema**: All nodes and edges follow the Pydantic schemas defined in `agent_utilities/models/knowledge_graph.py`.
