# Bi-Temporal Memory Layers (CONCEPT:AU-KG.temporal.bi-temporal-memory-layers)

## Overview

Bi-Temporal Memory Layers add a first-class **valid-time axis** to the knowledge graph and a
**procedural memory layer**, so the KG distinguishes *when a fact was stored* from *when the event
actually happened* and can answer "what was true as of date T". Assimilated from Quarq Agent's
Temporal Truth Protocol and three-memory-layer model (`agent-oss/agent.py`) but implemented
**structurally** — as graph properties + edges — rather than as prompt-only date discipline.

Extends **KG-2.1** (Tiered Memory & Context).

## How it works

- **Procedural layer.** `MemoryNode` gains `memory_type ∈ {semantic, episodic, procedural}` and a
  `target_entity`. Procedural rules become first-class nodes scoped to an entity, so selective
  injection is a 1-hop traversal rather than substring matching.
- **Bi-temporal stamping.** Every relationship written through `engine.link_nodes` is stamped by
  `knowledge_graph/core/bitemporal.py:stamp_bitemporal` with `event_time` (when it happened),
  `storage_time` (when saved), `valid_from`, and `valid_to` (open interval = still valid). A
  caller-supplied `event_time` (e.g. a narrative date resolved by the learner) is preserved.
- **As-of queries.** `query_cypher(..., as_of=T)` (surfaced via `graph_query(as_of=...)`)
  post-filters rows to those whose validity interval contains `T` (`valid_from <= T < valid_to`).
- **Contradiction precedence.** `resolve_temporal_contradiction` resolves two conflicting facts by
  later `event_time`, writes a `SUPERSEDES` edge, and closes the loser's `valid_to` — the fact is
  **never deleted**, so history stays queryable.

## Key files / API

| Piece | Location |
|---|---|
| Pure temporal core | `knowledge_graph/core/bitemporal.py` (`stamp_bitemporal`, `is_valid_as_of`, `filter_as_of`, `resolve_precedence`, `supersede`) |
| Procedural fields | `models/knowledge_graph.py` (`MemoryNode.memory_type`, `target_entity`, temporal mirror) |
| Stamping hot path | `knowledge_graph/core/engine.py` (`link_nodes`) |
| As-of + precedence | `knowledge_graph/orchestration/engine_query.py` (`query_cypher(as_of=...)`, `resolve_temporal_contradiction`) |

## Wiring (≤3 hops)

`graph_query(as_of=...)` → `query_cypher` (2 hops); every `graph_write`/`graph_ingest` edge auto-stamps via `link_nodes`.

## Why it's superior to flat-file memory

A flat store only knows storage order; it cannot answer "what was true on date T" or resolve a
contradiction by event time without losing the prior belief. Bi-temporal stamping + as-of queries +
edge-based supersession make all three structural and lossless.

## Research provenance

Quarq Agent Temporal Truth Protocol & memory layers — `agent-oss/agent.py:1058-1466, 2370-2477, 3114-3161`.
