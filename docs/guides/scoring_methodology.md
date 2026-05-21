# Scoring Methodology & Retrieval Semantics

> **CONCEPT:KG-2.3** — Graph Integrity & Retrieval

This document defines how the Knowledge Graph scores, ranks, and retrieves results across all search operations.

## `top_k` Parameter

### Definition

`top_k` is the **maximum number of results** returned from any search operation. It acts as a hard limit on output volume, not a relevance threshold.

### Semantics

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `top_k=10` | Most tools | Return at most 10 results, ranked by relevance |
| `top_k=5` | Memory recall | Return at most 5 memories |
| `top_k=20` | Innovation discovery | Cast wider net for signal extraction |

### How it Works

1. **HNSW Index Phase**: The LadybugDB HNSW index performs approximate nearest-neighbor search, retrieving `top_k × 3` candidates internally to ensure quality
2. **Quality Gate**: Results below `relevance_threshold` (default 0.2) are filtered
3. **Enrichment**: Innovation signals, metadata, and decay scoring are applied
4. **Truncation**: Final results are truncated to `top_k`

### Best Practices

- Use `top_k=5` for focused, high-precision retrieval (memory recall)
- Use `top_k=10` for balanced search (default for most use cases)
- Use `top_k=20–50` for exploratory discovery (comparative analysis, innovation scanning)
- Never exceed `top_k=100` — diminishing returns and performance degradation

---

## Scoring Methods

### Cosine Similarity (Primary)

All vector-based retrieval uses **cosine similarity** between query embeddings and stored node embeddings.

**Embedding Model**: `text-embedding-3-small` (768 dimensions)

**Formula**:
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

**Interpretation Guide**:

| Score Range | Label | Meaning |
|-------------|-------|---------|
| **0.60+** | Very High | Strong semantic match — nearly identical concepts |
| **0.45–0.60** | High | Related concepts with significant overlap |
| **0.35–0.45** | Moderate | Loosely related; shared domain but different focus |
| **0.20–0.35** | Low | Tangential connection; may be noise |
| **< 0.20** | Noise | Below quality gate — filtered out by default |

### HNSW Index Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `M` | 16 | Max connections per node (balances speed vs accuracy) |
| `ef_construction` | 200 | Index build quality (higher = better recall, slower build) |
| `ef_search` | 100 | Search-time quality (higher = better recall, slower query) |

**Complexity**: O(log N) retrieval with HNSW vs O(N) brute-force scan.

### Keyword Scoring (Hybrid)

For nodes without embeddings, the hybrid retriever falls back to keyword matching:

1. **Exact match**: Query terms appear in node `name`, `description`, or `content`
2. **Scoring**: Binary (0 or 1) — either the keyword appears or it doesn't
3. **Weighting**: Keyword matches contribute to a combined hybrid score

---

## Search Modes

### `hybrid` (Default)

Combines vector similarity + keyword matching. Best for general-purpose queries.

### `concept`

Looks up a specific `CONCEPT:ID` (e.g., `KG-2.1`). Returns exact match + related concepts.

### `analogy`

Finds structurally similar concepts using the graph topology. Used for Extend-Before-Invent checks.

### `memory`

Searches tiered memory nodes with optional Ebbinghaus time-decay scoring (CONCEPT:KG-2.1).

### `discover`

Cross-references query against all ingested content with innovation signal extraction. Returns enriched results with biomimicry/tech signals and concept mapping.

**Signal Taxonomy**:
- **Biomimicry Signals**: Detected via keywords (swarm, colony, pheromone, neural, immune, etc.)
- **Tech Signals**: Detected via keywords (attention, transformer, embedding, RAG, MCP, etc.)
- **Innovation Claims**: Sentences containing signal words (novel, propose, outperform, SOTA, etc.)

---

## Ebbinghaus Time-Decay Scoring

**CONCEPT:KG-2.1** — Research: MEMO Survey (2504.01990v2) §3.2

Applied during memory recall to weight recent/frequently-accessed memories higher.

**Formula**: `relevance = base_score × exp(-λt)` where `λ = ln(2) / half_life`

**Memory Tier Half-Lives**:

| Tier | Half-Life | Use Case |
|------|-----------|----------|
| Working | 5 minutes | Immediate context within a conversation |
| Episodic | 4 hours | Session-level interaction history |
| Semantic | 30 days | Long-term knowledge and patterns |
| Procedural | ∞ (no decay) | Learned skills — never forgotten |

---

## Single-Paper Analysis Workflows

Three methods to analyze a specific ingested research paper against the KG:

### Method 1: Path Query (Most Direct)

```python
# Find all chunks of a specific paper
kg_query(
    cypher="MATCH (a:Article) WHERE a.target_path CONTAINS '2504.01990' RETURN a.id, a.name, a.target_path LIMIT 20"
)
```

### Method 2: Discover Mode (Innovation Extraction)

```python
# Cross-reference a paper's topic against the full KG
kg_search(
    mode="discover",
    query="memory-augmented LLM systems consolidation",
    top_k=20
)
```

### Method 3: Concept Cross-Reference

```python
# Check a paper against a specific concept
kg_search(
    mode="analogy",
    query="KG-2.1 tiered memory context compaction"
)
```

### Per-Paper Relevance Scoring

To score ALL papers against a specific topic:

```python
# Get all unique paper paths
kg_query(
    cypher="MATCH (a:Article) RETURN DISTINCT a.target_path AS path, count(a) AS chunks ORDER BY chunks DESC"
)

# Then for each, run discover mode
kg_search(mode="discover", query="<your concept or feature topic>", top_k=50)
# Filter results by target_path to isolate each paper's score
```

---

## Research Assimilation Tracking

**CONCEPT:KG-2.6** — Research Intelligence

### Assimilation Lifecycle

```
Ingest → Discover → Analyze → Assimilate → Implement
  │          │          │          │           │
  │          │          │          │           └── COMPLETED SDD → auto ASSIMILATED_INTO
  │          │          │          └── kg_write(action='assimilate')
  │          │          └── kg_analyze / comparative-analysis skill
  │          └── kg_search(mode='discover')
  └── kg_ingest(target_path=paper.pdf)
```

### Edge Types

| Edge | Direction | Meaning |
|------|-----------|---------|
| `ASSIMILATED_INTO` | Article → Codebase | Paper reviewed and features extracted |
| `DERIVED_FROM_RESEARCH` | SDDFeature → Article | Feature inspired by paper |
| `IMPLEMENTS_FINDING` | Code → SDDFeature | Code implementing the feature |

### Filtering Already-Implemented Research

```python
# Future comparative analyses skip papers already implemented
kg_search(
    mode="discover",
    query="memory consolidation",
    # Pass exclude_assimilated flag (available in discover_innovations engine method)
)
```
