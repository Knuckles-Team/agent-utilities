# Layered Hybrid Architecture — KG Comparative Analysis Pipeline

> **CONCEPT:AU-KG.query.object-graph-mapper** — Knowledge Graph Comparative Analysis Architecture
>
> This document describes the three-stage **streaming pipeline** for extracting
> actionable features and innovation opportunities from research papers and
> codebases ingested into the agent-utilities Knowledge Graph.

## Architecture Diagram

<div class="admonition architecture" markdown>
<p class="admonition-title">Zero-LLM discovery feeding a bounded streaming synthesis pipeline</p>

Layer 1 (ORCH-1.2, native vector discovery, zero LLM calls): re-ingest with
the v2 schema (types + content + embeddings), cross-reference every concept
against every node, then score and rank matches by cosine similarity. That
ranked output streams into a Pillar Accumulator: a completed pillar
triggers one LLM synthesis call (at most 5, one per pillar), producing
Features; a high-weight paper found along the way triggers one deep
extraction call per paper, producing Blueprints. Both outputs feed the
Results Collector, which writes into the persistent Knowledge Graph with
temporal edges. An `asyncio.Semaphore` (`KG_LLM_CONCURRENCY=4`) bounds both
the synthesis and deep-extraction call sites, capping total concurrent LLM
calls regardless of how many pillars or papers are in flight.
</div>

## Streaming Pipeline Architecture

The pipeline uses a **producer-consumer** pattern to minimize wall-clock time:

1. **Vector discovery** iterates concepts sequentially (fast — vector search only, no LLM).
2. A **pillar accumulator** tracks concept completions per pillar.
3. As soon as all concepts in a pillar finish, **synthesis** fires for that pillar immediately.
4. As soon as any high-weight paper is discovered, **deep extraction** queues immediately.
5. All LLM tasks share a single `asyncio.Semaphore(KG_LLM_CONCURRENCY)`.

This means synthesis for pillar A can run while discovery is still processing pillar B,
and extraction tasks fire as soon as they are discovered.

### Timing Example (KG_LLM_CONCURRENCY=4)

```
Time →  [======= Discovery ========]
         ↓ ORCH done     ↓ KG done     ↓ AHE done
        [SYN:ORCH]      [SYN:KG]      [SYN:AHE]
         ↓ paper X       ↓ paper Y
        [DEEP:X]        [DEEP:Y]

All synthesis and extraction tasks overlap, bounded by 4 concurrent LLM slots.
```

## Layer Descriptions

### Vector Discovery (0 LLM calls)
- Pure cosine similarity between concept embeddings and ingested content
- Cross-references the canonical concepts (see `docs/concepts.yaml`) against all nodes
- Produces ranked match lists with similarity scores
- **Cost:** Zero LLM calls — embedding-only
- **Script:** `concept_cross_reference.py` (shipped in the `comparative-analysis` skill under `universal-skills`)

### LLM Synthesis (1 call per pillar, max 5)
- Triggered when all concepts for a pillar complete discovery
- Top matches per pillar mega-batched into 256K context window
- Extracts: specific techniques, implementation suggestions, agreement/contradiction signals
- Stores enriched edges with `valid_from` temporal metadata
- **Cost:** 1 call per pillar (max 5 for all pillars)

### Deep Extraction (1 call per high-weight paper)
- Triggered immediately when a paper with similarity > 0.80 is found
- Per-paper entity and relationship extraction
- Creates typed edges: `IMPLEMENTS`, `EXTENDS`, `CONTRADICTS`, `PROPOSES_ALTERNATIVE`, `CITES`
- Citation chain tracking and implementation-ready specifications
- **Cost:** ~3-10 LLM calls (depends on number of high-weight papers)
- **Script:** `llm_synthesis.py` (shipped in the `comparative-analysis` skill under `universal-skills`)

## LLM Call Budget

| Layer | Trigger | LLM Calls | Content |
|-------|---------|-----------|------------|
| **Discovery** | All items | 0 | Vector similarity only |
| **Synthesis** | Pillar complete | 1-5 | Top matches mega-batched per pillar |
| **Deep extraction** | similarity > 0.80 | 3-10 | Per-paper deep extraction |
| **Total** | | **4-15** | Down from 500-2000 in naive approach |

## Configuration

| Variable | Default | Description |
|:---------|:--------|:------------|
| `KG_LLM_CONCURRENCY` | `4` | Max concurrent LLM calls. Set to match your inference endpoint capacity. |

**Note**: All LLM routing (endpoints, credential references, model IDs) is
validated through AgentConfig. Durable values live in the XDG `config.json`;
process-scoped environment projection remains an AgentConfig input, not a second
configuration system.

## Temporal Metadata

All edges created by Layers 2 and 3 include Graphiti-inspired temporal metadata:

- **`valid_from`**: ISO timestamp marking when the relationship was established
- **`valid_to`**: Populated only when a relationship is superseded (e.g., new paper contradicts prior finding)
- Enables temporal queries: "What was the state of knowledge about concept X at time T?"

## Related Concepts

- [concepts.yaml](../concepts.yaml) — canonical concept registry (single source of truth) used as cross-reference seeds
- [overview.md](../overview.md) — Architecture overview of agent-utilities
