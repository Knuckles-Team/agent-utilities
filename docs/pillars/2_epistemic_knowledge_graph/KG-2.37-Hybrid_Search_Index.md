# Hybrid Search Index (CONCEPT:KG-2.37)

## Overview
Weighted semantic+keyword search scoring (72%/28% default) with CamelCase/snake_case token splitting, phrase boost, and symbol-specific matching. Uses existing `create_embedding_model()` infrastructure. Adapted from contextplus's embedding.ts.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/hybrid_search_scorer.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
