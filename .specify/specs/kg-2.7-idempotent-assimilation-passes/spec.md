# Spec: Idempotent & Isolated Assimilation Passes

> CONCEPT:KG-2.7 (assimilation refinement — no new concept id). Surfaced by
> iterating the assimilation matcher on the live durable KG.

## Finding (what we observed)

While tuning the gap matcher against the live graph, an early **over-liberal** pass
wrote 85 `SATISFIED_BY` edges; a later **precise** pass recognized only 30 — but the
graph then reported **6 open gaps instead of ~61**. The stricter re-run **added** its
30 edges without **removing** the 85 stale ones, so features stayed wrongly "closed".

**Root cause.** `auto_satisfy(write=True)` was **append-only** against a **durable**
graph: each run accretes `SATISFIED_BY` edges, and `is_closed` is true if *any*
closing edge exists. A re-run can therefore never *un-close* a feature it no longer
matches. Compounding it, every experimental pass mutated the **shared production**
graph (the very reason the validated approach is "scratch namespace first").

## User Stories

### US-1: Re-running the matcher REPLACES, never ACCUMULATES

**As** an operator re-running assimilation (e.g. after a matcher fix), **I want** the
pass to be idempotent, **so that** stale matches from an earlier pass don't keep
features wrongly closed.

**Acceptance Criteria:**
- [x] Before writing, `auto_satisfy` drops the prior **auto-written** `SATISFIED_BY` edges of the features it is (re-)evaluating.
- [x] Re-running with a stricter matcher correctly **un-closes** features it no longer matches.
- [x] Only `auto:true` edges are reconciled — human/manual closures are never touched.
- [x] Reconciliation is opt-out (`reconcile=True` default) and a no-op when the engine cannot delete edges (test doubles).

### US-2: Experimental passes don't pollute the shared graph

**As** a developer validating the engine, **I want** to run passes on an isolated
graph, **so that** experiments never leave stale state in the production KG.

**Acceptance Criteria:**
- [x] Documented pattern: validate on a scratch in-process engine / isolated `graph_name` before any live, mutating run (`~/workspace/scratch/assimilation_pilot.py`).
- [ ] *(follow-on)* a first-class `IntelligenceGraphEngine` scratch-namespace helper so live experiments target `graph_name="assim_pilot"` and are dropped after.

## Non-Functional Requirements

- [x] All existing assimilation tests pass (zero regression).
- [x] Pre-commit / ruff clean.
- [x] Reuses the batched bulk-edge scan ([[kg-2.7-batched-graph-access]]) — reconcile costs no extra round-trips.

## Implementation (done)

- **`gap_analysis._clear_auto_satisfied(engine, feature_ids)`** — finds prior
  `SATISFIED_BY` edges with `auto:true` from the target features (via the bulk edge
  view) and removes them through `engine.delete_edge`.
- **`auto_satisfy(..., reconcile=True)`** — calls the reconcile step for its target
  feature set before matching, so each pass yields the *current* truth.

## Tests

- `test_auto_satisfy_reconciles_stale_edges` — match a feature to KG-2.7, repoint its
  declared id to KG-9.9, re-run, and assert the stale `(feature, KG-2.7, SATISFIED_BY)`
  edge was deleted and the feature now matches only the new concept.

## Status

**IMPLEMENTED** (reconcile) — `agent_utilities/knowledge_graph/assimilation/gap_analysis.py`.
**FOLLOW-ON** (isolation helper) — a durable scratch-namespace mode on the engine so
live validation never mutates `__bus__`. Until then, validate on the scratch harness.
