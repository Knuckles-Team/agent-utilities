# Retrieval Quality Gate (CONCEPT:KG-2.6)

## Overview
Systematic retrieval quality measurement with 5-mode failure taxonomy (drift, truncation, staleness, low-relevance, inter-agent), configurable per-SchemaPack relevance thresholds, and temporal freshness scoring. Based on Ambekar (2026) research.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval_quality.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Cross-Agent Context Provenance (CONCEPT:KG-2.6)

## Overview
Tracks retrieval quality scores and failure modes across agent boundaries via `ContextProvenanceRecord`. Detects cascading retrieval degradation in multi-agent pipelines.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval_quality.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Hybrid Search Index (CONCEPT:KG-2.3)

## Overview
Weighted semantic+keyword search scoring (72%/28% default) with CamelCase/snake_case token splitting, phrase boost, and symbol-specific matching. Uses existing `create_embedding_model()` infrastructure. Adapted from contextplus's embedding.ts.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/hybrid_search_scorer.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# RAG-KG Unification (CONCEPT:KG-2.3)

## Overview
Collapses separate RAG vector index into KG-native retrieval using three acceleration layers: similarity-edge shortcuts (O(degree) vs O(N)), spectral cluster scoping (search space reduction), and hybrid semantic+keyword scoring. Drop-in enhancement for HybridRetriever via `retrieve_unified()`.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/unified_rag_kg.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Graph Distillation Migration (CONCEPT:KG-2.6)

## Overview
Migrates standard RAG retrieval to pre-computed SimilarityEdgeNode shortcuts for O(degree) retrieval. Manages distillation index lifecycle: batch creation, incremental updates, stale edge pruning, and coverage health monitoring. Includes `migrate_existing()` for batch migration of legacy nodes.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/graph_distillation.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
