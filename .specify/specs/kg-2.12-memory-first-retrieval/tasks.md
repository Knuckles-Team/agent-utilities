# Tasks: Memory-First Retrieval (KG-2.12)

## T1 — Pure HyDE helpers (US-1,2,4)  [code]
- [x] New `retrieval/hyde_planner.py`: `HydePlan`, `HYDE_THRESHOLDS`/`threshold_for_mode`, `parse_hyde_plan`, `merge_retrievals`, `build_evidence_ledger`.

## T2 — Orchestration method (US-1,2,3,4)  [code]
- [x] `HybridRetriever._generate_hyde_plan` (uses ORCH-1.27 `planner` role) + `plan_and_retrieve` (HyDE/dual-threshold/gated 2nd pass/ledger).

## T3 — Entry-point wiring (US-5)  [code]
- [x] `engine_query.search_hybrid` gains `mode`/`self_correct`/`corpus_id`.
- [x] `graph_search` MCP tool exposes `hyde`/`deep` modes + `self_correct`.

## T4 — Tests (NFR)  [test]
- [x] `tests/unit/knowledge_graph/test_kg_2_12_memory_first_retrieval.py` — AC1–AC6 (10 tests).

## T5 — Artifacts (NFR)  [docs]
- [x] concepts.yaml regen (KG-2.12); CHANGELOG; README KG-2 count; pillar doc note.
