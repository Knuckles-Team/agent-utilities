# RAG-KG Unification (CONCEPT:KG-2.38)

## Overview
Collapses separate RAG vector index into KG-native retrieval using three acceleration layers: similarity-edge shortcuts (O(degree) vs O(N)), spectral cluster scoping (search space reduction), and hybrid semantic+keyword scoring. Drop-in enhancement for HybridRetriever via `retrieve_unified()`.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/unified_rag_kg.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
