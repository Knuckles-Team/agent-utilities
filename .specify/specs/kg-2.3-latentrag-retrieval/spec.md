# Spec: Latent-Space Retrieval Optimization (CONCEPT:AU-KG.memory.auto-similarity-memory-graph)

## Pre-Flight Checklist (Mandatory — DSTDD)

- [x] **KG search completed** — `.specify/design/kg-2.3-latentrag-retrieval/design.md` exists
- [x] **Extension point identified** — Extends KG-2.3 (Graph Integrity & Retrieval)
- [x] **C4 diagram created** — showing integration into KG pillar
- [x] **No new CONCEPT: tag** — uses existing KG-2.3
- [ ] **`code-enhancer` audit** — pending implementation
- [ ] **Design validation passes** — pending SDDManager integration

## Design Reference

→ [design.md](../../design/kg-2.3-latentrag-retrieval/design.md)

## Research Sources

| Paper | ArXiv ID | Key Contribution |
|-------|----------|-----------------|
| LatentRAG | 2605.06285v1 | Latent-space retrieval, 90% latency reduction, AutoRefine |
| MINER | 2605.06460v1 | Multi-layer embedding probing, 4.5% nDCG@5 improvement |
| SIRA | 2605.06647v1 | Expected-response sketch, corpus-discriminative retrieval |

## User Stories

### US-1: AutoRefine Post-Retrieval Stage

**As a** KG search consumer, **I want** retrieved results refined against query context before being returned, **so that** low-quality matches are filtered and relevance is improved.

**Acceptance Criteria:**
- [x] `search_hybrid()` gains optional `auto_refine=True` parameter
- [x] AutoRefine re-scores results using query-context dot-product (not raw cosine alone)
- [x] Results below refined threshold are dropped
- [x] Backward compatible: default behavior unchanged

### US-2: Scoring Transparency in API Responses

**As a** developer using KG search, **I want** all search results to include explicit scoring metadata, **so that** I can interpret relevance and debug retrieval quality.

**Acceptance Criteria:**
- [x] All `kg_search` responses include `score` field (cosine similarity, 0.0–1.0)
- [x] Responses include `scoring_method` metadata (`cosine_similarity_hnsw` or `keyword_match`)
- [x] `discover_innovations()` includes `scoring_metadata` in response envelope
- [x] Score interpretation guide in docs: 0.60+ very high, 0.45-0.60 high, 0.35-0.45 moderate, <0.35 low

### US-3: Embedding Quality Diagnostics

**As a** KG operator, **I want** diagnostic tools to assess embedding quality across the corpus, **so that** I can identify when re-embedding or index rebuilds are needed.

**Acceptance Criteria:**
- [x] New `kg_inspect(view='embedding_health')` diagnostic
- [x] Reports: embedding coverage (% nodes with embeddings), avg dimensionality, null rate
- [x] Identifies stale embeddings (nodes updated after last embedding)

## Non-Functional Requirements

- [x] All existing tests continue to pass (zero regression)
- [x] Pre-commit hooks pass cleanly
- [x] Documentation updated in `docs/pillars/2_epistemic_knowledge_graph/KG-2.3-Graph_Integrity_And_Retrieval.md`
- [x] New functionality wired into kg_search and kg_inspect MCP tools
- [x] CONCEPT:AU-KG.memory.auto-similarity-memory-graph tags in all new code and tests
