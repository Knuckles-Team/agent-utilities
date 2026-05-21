# Topological Mincut Partitioning (CONCEPT:KG-2.5)

## Overview
Dynamic Louvain partitioning with Label Propagation fallback to identify emergent topological clusters and communities.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/topological_partition.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Topological Analogy Engine (CONCEPT:KG-2.5)

## Overview
Leverages exact subgraph isomorphism (networkx VF2) and vectorized embeddings (EncPI) to find analogous subgraphs across different domains (cross-domain innovation extraction).

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/analogy_engine.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Topological Graph Visualization (CONCEPT:KG-2.6)

## Overview
Scalable WebGL-based Knowledge Graph visualization engine using Sigma.js and ForceAtlas2 physics for the `agent-webui`. Implements intelligent mass assignment and radial clustering for 100K+ scale.

## Implementation Details
- **Source Code**: ``agent-webui/src/components/knowledge-graph/``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Spectral Cluster Navigator (CONCEPT:KG-2.5)

## Overview
Tuning-free spectral clustering using normalized Laplacian eigengap heuristics for automatic k-selection. OWL-integrated via `skos:Concept` alignment with `broader`/`narrower` edges. Financial regime detection extension. Adapted from contextplus's clustering.ts.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/core/spectral_navigator.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Symbol Blast Radius Analyzer (CONCEPT:KG-2.5)

## Overview
Regex-based symbol usage tracking across Python codebases with definition-line exclusion, low-usage warnings, and KG integration via `BlastRadiusNode`. Impact scoring uses log-scaled usage count × file diversity. Adapted from contextplus's blast-radius.ts.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/core/blast_radius.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Auto-Similarity Memory Graph (CONCEPT:KG-2.3)

## Overview
Auto-creates `SIMILAR_TO` edges between KG memory nodes when cosine similarity ≥ threshold (default 0.72). Exponential decay scoring with stale edge pruning and hub control. Adapted from contextplus's memory-graph.ts.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/memory/auto_similarity.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
