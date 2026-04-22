# Unified Intelligence Pipeline

The pipeline is a multi-phase, topological ingestion engine that builds the Knowledge Graph from source code and metadata.

## Overview

The pipeline executes a sequence of specialized **Phases**, each adding a layer of intelligence to the graph. It leverages **NetworkX** for in-memory topological analysis before syncing the final state to the persistent backend.

## The 12 Phases

1. **Memory**: Hydrate existing state from persistence.
2. **Scan**: Perform directory walk and file identification.
3. **Registry**: Ingest prompt frontmatter and MCP definitions.
4. **Parse**: AST parsing via Tree-Sitter to extract symbols.
5. **Resolve**: Map import strings to graph edges.
6. **MRO**: Calculate inheritance hierarchies.
7. **Reference**: Build the call graph.
8. **Communities**: Cluster nodes using Louvain clustering.
9. **Centrality**: Identify critical path objects via PageRank.
10. **Embedding**: Generate semantic vector embeddings.
11. **Registry Int**: Map tools/skills to code structures.
12. **Sync**: Commit the in-memory graph to the Cypher backend.

## Architecture

- **Runner (`runner.py`)**: Executes the pipeline phases in the correct topological order.
- **Phases (`phases/`)**: Individual implementation for each pipeline step.
- **Types (`types.py`)**: Configuration and state models for the pipeline.

## Maintenance

- **Performance**: Large repositories may require phase-specific optimizations.
- **Tree-Sitter**: Adding support for new languages requires updating the `Parse` phase and adding the appropriate Tree-Sitter grammars.
