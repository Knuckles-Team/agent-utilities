# Spec: Memory-First Retrieval (KG-2.12)

> References design: `.specify/design/kg-2.12-memory-first-retrieval/design.md`

## Pre-Flight Checklist
- [x] Design exists; KG-nearest-concepts table completed.
- [x] Extension target identified (KG-2.3, similarity 0.88 ≥ 0.70); reuses AHE-3.4, KG-2.6, ORCH-1.27.
- [x] New CONCEPT:KG-2.12 justified as the memory-first retrieval *policy* atop the KG-2.3 retriever.
- [x] Wire-First confirmed: `graph_search → search_hybrid → plan_and_retrieve → retrieve_hybrid` = 3 hops.

## User Stories

### US-1 — HyDE multi-query expansion
**As** a memory recall step, **I want** the planner role to expand a question into multiple vector
formulations + keywords + a search mode, **so that** recall beats single-vector similarity.
- **AC1**: `HydePlan` carries `vector_queries`, `keywords`, `search_mode`; `parse_hyde_plan` tolerates JSON-in-prose and comma-string keywords, and falls back to a single-query plan on garbage.
- **AC2**: `plan_and_retrieve(mode="hyde")` runs each planned query through `retrieve_hybrid` and merges (id-dedup, max-score, sort).

### US-2 — Dual thresholds
**As** retrieval, **I want** standard (0.38) and deep (0.28) thresholds, **so that** aggregations get wide recall and point facts get precision.
- **AC3**: `HYDE_THRESHOLDS` = {standard:0.38, deep:0.28}; `mode="standard"|"deep"` pass the matching threshold to `retrieve_hybrid`.

### US-3 — Self-correcting two-pass
**As** retrieval, **I want** a second pass at the deep threshold **only when the quality gate fails**, **so that** confident-but-thin first passes are repaired without always-on cost.
- **AC4**: with `self_correct=True`, a second pass fires iff `last_quality_report.gate_passed is False`, and merges into the result.

### US-4 — Quantitative-fidelity ledger
**As** a generator, **I want** an ACCEPT/REJECT evidence ledger with extracted numbers, **so that** I aggregate a complete ledger instead of the single most salient row.
- **AC5**: `build_evidence_ledger` marks nodes ≥0.38 ACCEPT, surfaces numeric tokens, and reports accept/reject counts + accepted numbers.

### US-5 — Entry-point wiring
- **AC6**: `search_hybrid` gains `mode`/`self_correct`/`corpus_id`; `graph_search` exposes `hyde`/`deep` modes + `self_correct`; defaults preserve prior behavior.

## Non-Functional Requirements
- `tests/unit/knowledge_graph/test_kg_2_12_memory_first_retrieval.py` (`@pytest.mark.concept(id="KG-2.12")`), ≤60s, no network/LLM.
- `pre-commit` green; concepts.yaml regenerated; 7-artifact mandate satisfied.
