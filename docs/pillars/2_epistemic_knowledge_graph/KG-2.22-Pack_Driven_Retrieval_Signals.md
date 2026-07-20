# KG-2.22 — Pack-Driven Retrieval Signals

**Pillar:** 2 — Epistemic Knowledge Graph · **Status:** live

## What

Three declarative retrieval signals carried on the active Schema Pack and applied in
the hybrid-retrieval hot path:

- **Recency decay** — per node-type temporal boost measured against bi-temporal
  `event_time` (KG-2.11). Exponential or hyperbolic; always `>= 1.0` so unknown
  dates are never penalised.
- **Source trust** — per-source authority multiplier (e.g. `peer_reviewed` 1.3,
  `arxiv` 1.2, `blog` 0.7).
- **Autocut** — trims the long tail at the largest relative score drop ("knee"),
  recall-safe via `autocut_min_results`.

These close the gap to gbrain's `recency-decay.ts`, `source-boost.ts`, and
`autocut.ts` while remaining a strict no-op under the default `core` pack.

## Why

Retrieval quality at scale needs more than vector similarity: fresh material should
surface, trusted sources should outrank noise, and weak tails should be dropped
rather than dilute the context window. Baking these into the pack means a domain can
tune them without code changes.

## How / Wiring

- Declared on `SchemaPack`: `recency_decay`, `source_trust`, `autocut_*`
  (`models/schema_pack.py`), with helpers `recency_spec_for()` / `trust_for()`.
- Applied in `HybridRetriever.retrieve_hybrid` (`knowledge_graph/retrieval/hybrid_retriever.py`):
  `_recency_boost(node, as_of)` and `_source_trust_boost(node)` multiply `_score`
  in the post-fusion boost region; `autocut()` (`retrieval/autocut.py`) trims the
  ranked base set before graph traversal.
- Entry point: `graph_search(..., as_of=...)` → `engine.search_hybrid(as_of=...)` →
  `retrieve_hybrid`. The `as_of` parameter enables "knowledge state as of date D".

## Tests

`tests/knowledge_graph/test_pack_retrieval_signals.py`, `tests/knowledge_graph/test_autocut.py`.
