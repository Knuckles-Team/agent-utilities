# Resilient Retrieval (CONCEPT:AU-KG.retrieval.triviality-gate)

## Overview

Resilient Retrieval makes memory-first retrieval **degrade gracefully** and **skip wasted work**:
a 4-level fallback cascade guarantees a query always returns something even when the vector store
is offline, and a social-closer gate skips retrieval entirely on trivial turns. Assimilated from
memory-os (`scripts/context_enhancer.py`). Extends **KG-2.12** (Memory-First Retrieval).

## How it works

- **4-level fallback cascade.** Beneath the HyDE/two-pass logic in `plan_and_retrieve`:
  hybrid (vector+keyword) → dense-only (inside `retrieve_hybrid`) → **lexical** (backend `CONTAINS`
  keyword scan) → backend/SQLite. If the vector path yields nothing, `_lexical_fallback` runs a
  keyword scan over node content/name for the query's distinctive tokens (low-confidence `_score`,
  `_fallback="lexical"`). Returns `[]` only when no backend exists at all (tier-4 no-op).
- **Social-closer / triviality gate.** `hyde_planner.is_trivial_query` short-circuits the whole
  plan + retrieval on greetings/closers ("ok", "thanks", "👍") or sub-6-char/emoji-only messages,
  saving the planner LLM call and embedding work.

## Key files / API

| Piece | Location |
|---|---|
| Trivial gate | `knowledge_graph/retrieval/hyde_planner.py` (`is_trivial_query`, `SOCIAL_CLOSERS`) |
| Cascade | `knowledge_graph/retrieval/hybrid_retriever.py` (`plan_and_retrieve` gate + `_lexical_fallback`) |

## Wiring (≤3 hops)

`graph_search` → `search_hybrid` → `plan_and_retrieve` (2 hops to the retriever).

## Research provenance

memory-os 4-level cascade + social-closer — `scripts/context_enhancer.py:393-538, 586-589` (verified).
