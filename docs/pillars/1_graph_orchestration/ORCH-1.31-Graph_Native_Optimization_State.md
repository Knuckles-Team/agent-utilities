# Graph-Native Optimization State (CONCEPT:ORCH-1.31)

## Overview

Persists the GEPA Pareto frontier (candidates + ancestry) into the durable **epistemic-graph** so
optimization is **resumable and accumulates across sessions** — a synergy unique to our graph-native
architecture (neither predict-rlm nor the GEPA reference impl does this). Extends **ORCH-1.13**
(+KG-2.7 ingestion/persistence).

## How it works

- **Snapshot round-trip** (`ParetoCandidatePool.to_snapshot` / `load_snapshot`) — pure serialization of
  the frontier to/from plain dicts, preserving candidate prompts, scores, and `parent_ids` ancestry.
- **Persist / resume** (`GEPAOptimizer.persist_frontier(run_id)` / `resume_frontier(run_id)`) — writes a
  `GEPAFrontier` graph node (extending the existing `OptimizationTrajectoryNode`/`EvaluatorFeedbackNode`
  writes via `create_or_merge_node`) and reads it back to seed the pool. A killed run resumes from the
  persisted frontier with no loss of the best candidate; prior frontiers are reusable optimization state.
- **Best-effort** — absent a backend, persist/resume degrade quietly (return falsy), never raising.

## Key files / API

| Piece | Location |
|---|---|
| Snapshot + persistence | `rlm/gepa.py` (`ParetoCandidatePool.to_snapshot`/`load_snapshot`, `GEPAOptimizer.persist_frontier`/`resume_frontier`) |

## Wiring (≤3 hops)
`graph_orchestrate(action="rlm_optimize")` → optimizer → `persist_frontier`/`resume_frontier` → graph (≤3 hops).

## Research provenance
Synthesis of GEPA (frontier/ancestry) with agent-utilities' durable epistemic-graph persistence —
verified against `rlm/gepa.py` (`OptimizationTrajectoryNode`/`EvaluatorFeedbackNode`).
