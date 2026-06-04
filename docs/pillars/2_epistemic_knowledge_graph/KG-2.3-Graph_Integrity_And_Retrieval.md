# Retrieval Quality Gate (CONCEPT:KG-2.6)

## Overview
Systematic retrieval quality measurement with 5-mode failure taxonomy (drift, truncation, staleness, low-relevance, inter-agent), configurable per-SchemaPack relevance thresholds, and temporal freshness scoring. Based on Ambekar (2026) research.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/retrieval_quality.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Cross-Agent Context Provenance (CONCEPT:KG-2.6)

## Overview
Tracks retrieval quality scores and failure modes across agent boundaries via `ContextProvenanceRecord`. Detects cascading retrieval degradation in multi-agent pipelines.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/retrieval_quality.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Hybrid Search Index (CONCEPT:KG-2.3)

## Overview
Weighted semantic+keyword search scoring (72%/28% default) with CamelCase/snake_case token splitting, phrase boost, and symbol-specific matching. Uses existing `create_embedding_model()` infrastructure. Adapted from contextplus's embedding.ts.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/semantic_retrieval_engine.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# RAG-KG Unification (CONCEPT:KG-2.3)

## Overview
Collapses separate RAG vector index into KG-native retrieval using three acceleration layers: similarity-edge shortcuts (O(degree) vs O(N)), spectral cluster scoping (search space reduction), and hybrid semantic+keyword scoring. Drop-in enhancement for HybridRetriever via `retrieve_unified()`.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/semantic_retrieval_engine.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*
# Graph Distillation Migration (CONCEPT:KG-2.6)

## Overview
Migrates standard RAG retrieval to pre-computed SimilarityEdgeNode shortcuts for O(degree) retrieval. Manages distillation index lifecycle: batch creation, incremental updates, stale edge pruning, and coverage health monitoring. Includes `migrate_existing()` for batch migration of legacy nodes.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/semantic_retrieval_engine.py``
- **Pillar**: KG

## Documentation Coverage
*This is an auto-generated dedicated concept page to ensure 100% documentation coverage across the ecosystem.*

# Evaluation Corpus (CONCEPT:KG-2.3)

## Overview
Fixed corpus evaluation mode for reproducible deep-research benchmarking. Inspired by BrowseComp-Plus (arXiv:2508.06600). Stores named, versioned document sets with optional query-answer pairs. Supports freeze semantics for immutable benchmarks. Integrated into `HybridRetriever` via `corpus_id` parameter for constrained search.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/evaluation_corpus.py``
- **Hot Path**: `HybridRetriever.retrieve_hybrid(corpus_id=...)` → Cypher `WHERE n.id IN $corpus_ids`
- **Python API**: `CorpusManager.create_corpus()`, `CorpusManager.list_corpora()`, `CorpusManager.freeze_corpus()` (in `evaluation_corpus.py`)
- **Pillar**: KG

# Hard Negative Mining (CONCEPT:KG-2.3)

## Overview
Mines challenging distractors from query decomposition to calibrate retriever precision. Uses existing `_decompose_query()` to break complex queries into sub-queries, retrieves per sub-query, and identifies documents that match sub-queries but not the full query. Gated behind `KG_ENABLE_HARD_NEGATIVE_MINING` env var. From BrowseComp-Plus (arXiv:2508.06600).

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/hard_negative_miner.py``
- **Hot Path**: `HybridRetriever.retrieve_hybrid(hard_negatives=...)` → score × 0.5 penalty
- **Pillar**: KG

# nDCG Retrieval Scoring (CONCEPT:KG-2.3)

## Overview
Normalized Discounted Cumulative Gain computation for retrieval quality assessment. Uses binary relevance against gold document sets. Aligns with BrowseComp-Plus evaluation methodology (Section 4.1). Integrated into `RetrievalQualityGate.compute_ndcg()` and consumed by `EvaluationEngine.evaluate_disentangled()`.

## Implementation Details
- **Source Code**: ``agent_utilities/knowledge_graph/retrieval/retrieval_quality.py``
- **Hot Path**: `RetrievalQualityGate.compute_ndcg(results, gold_doc_ids, k=10)`
- **Pillar**: KG
