# Spec: Batched Graph Access for the Assimilation Gap/Rank Stages

> CONCEPT:KG-2.7 (assimilation refinement — no new concept id). Surfaced by the
> first live run of the assimilation engine against the production KG.

## Finding (what we observed)

Running one assimilation cycle against the **live** `epistemic-graph` daemon, the
`rank_features` stage took **117 s for ~91 features** and the full pass over the
2,939 raw `Article` corpus **timed out (>360 s)**. The work itself is tiny — the
cost was almost entirely **network round-trips**.

**Root cause.** The gap/rank stages resolve closure and adjacency with **per-node**
edge lookups: `is_closed()` / `open_features()` call `graph.out_edges(fid)` +
`graph.in_edges(fid)` once **per feature**, and `synergy._adjacency()` calls
`graph.out_edges(nid)` once **per id**. Each call is a UDS round-trip to the Rust
daemon, so cost is `O(features) × latency`. On an in-process test double this is
free; on a live backend it dominates and does not scale.

## User Stories

### US-1: A cycle over the live graph completes in seconds, not minutes

**As** the golden-loop daemon, **I want** the gap/rank stages to read edges in bulk,
**so that** an assimilation cycle on the live KG finishes without timing out.

**Acceptance Criteria:**
- [x] `open_features` / closure detection issues **one** bulk edge traversal, not one per feature.
- [x] `synergy._adjacency` issues **one** bulk edge traversal, not one per id.
- [x] Identical results to the per-node path (status-closed + `SATISFIED_BY`/`DERIVED_FROM_RESEARCH` out, `SUPERSEDES` in).
- [x] Graceful fallback to the per-node path when the graph exposes no bulk edge view (test doubles).

## Non-Functional Requirements

- [x] All existing assimilation tests continue to pass (zero regression).
- [x] Pre-commit / ruff clean.
- [x] No new top-level concept id (refinement of KG-2.7).

## Implementation (done)

- **`dedup.iter_all_edges(graph)`** — returns every `(src, dst, props)` via the
  graph's bulk `edges` view in **one** round-trip; returns `None` when no usable
  bulk view exists (or the view yields no edge data) so callers fall back safely.
- **`gap_analysis._closed_feature_index(engine, feature_types)`** — one node scan +
  one bulk edge scan → `(closed_ids, all_features)`. `open_features` is now a thin
  wrapper over it; per-node `is_closed` remains for single-feature checks + fallback.
- **`synergy._adjacency`** — builds the feature-feature adjacency from the single
  bulk traversal, per-node fallback retained.

Complexity goes from `O(features)` round-trips to **`O(1)`** round-trips per stage.

## Tests

- `test_open_features_uses_bulk_edge_view` — a `SATISFIED_BY` edge closes a feature
  through the bulk path (`_BulkGraph` test double exposing `edges(data=True)`).
- Existing per-node-double tests still pass → fallback path preserved, identical semantics.

## Status

**IMPLEMENTED** — `agent_utilities/knowledge_graph/assimilation/{dedup,gap_analysis,synergy}.py`.
Follow-on (not yet done): push closure/adjacency entirely engine-side via a single
Cypher projection when the L1 interpreter supports edge-property filters, removing
even the one bulk transfer.
