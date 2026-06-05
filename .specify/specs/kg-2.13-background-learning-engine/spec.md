# Spec: Background Learning Engine (KG-2.13)

> References design: `.specify/design/kg-2.13-background-learning-engine/design.md`

## Pre-Flight Checklist
- [x] Design exists; KG-nearest-concepts table completed.
- [x] Extension target identified (KG-2.1, similarity 0.84 ≥ 0.70); reuses KG-2.11, ORCH-1.27, observer seam.
- [x] New CONCEPT:KG-2.13 justified as continuous targeted-edit consolidation.
- [x] Wire-First confirmed: `agent-utilities-memory learn` CLI → `run_learner` → `engine.*_memory_node` = 3 hops.

## User Stories

### US-1 — Relative-date resolution at learn time
**As** the learner, **I want** "yesterday"/"N weeks ago" resolved to absolute dates, **so that** stored `event_time` is a real instant.
- **AC1**: `resolve_relative_dates` converts yesterday/today/tomorrow and "N day|week|month|year(s) ago"; vague recency ("recently") is left untouched.

### US-2 — Defensive edit parsing
**As** the learner, **I want** robust parsing of the LLM's edit list, **so that** malformed output never crashes learning.
- **AC2**: `parse_memory_edits` accepts `{"actions":[...]}` or a bare list embedded in prose, skips invalid rows, returns `[]` on total failure.

### US-3 — Targeted ADD/UPDATE/DELETE as bi-temporal mutations
**As** memory, **I want** the learner to apply targeted edits (not raw dumps), **so that** facts stay consistent.
- **AC3**: ADD creates a `MemoryNode` stamped with event/storage/valid_from + memory_type/target_entity.
- **AC4**: UPDATE rewrites content and re-stamps event/storage time on the existing node.
- **AC5**: DELETE is **soft** — `status="REMOVED"` + `valid_to` set; the node is preserved (history queryable).
- **AC6**: UPDATE/DELETE of a missing id is skipped (counted), never raising.

### US-4 — Concurrency, backoff, sync barrier
**As** the runtime, **I want** bounded concurrency + backoff + a drain barrier, **so that** learning is async but never wedges.
- **AC7**: `with_backoff` retries with exponential delay and succeeds/raises within `max_attempts`.
- **AC8**: `BackgroundLearner` uses `Semaphore(4)`; `schedule` + `await_pending` drain all in-flight tasks.

### US-5 — CLI entry point
- **AC9**: `agent-utilities-memory learn --file|--text [--dry-run] [--now]` runs `run_learner` and prints the edits + counts.

## Non-Functional Requirements
- `tests/unit/knowledge_graph/test_kg_2_13_background_learning.py` (`@pytest.mark.concept(id="KG-2.13")`), ≤60s, no network/LLM.
- `pre-commit` green; concepts.yaml regenerated; 7-artifact mandate satisfied.
