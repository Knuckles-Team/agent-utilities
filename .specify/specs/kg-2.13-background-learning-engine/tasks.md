# Tasks: Background Learning Engine (KG-2.13)

## T1 — Pure helpers (US-1,2)  [code]
- [x] `resolve_relative_dates`, `parse_memory_edits`, `MemoryEdit` in `memory/learning_engine.py`.

## T2 — Targeted edit application (US-3)  [code]
- [x] `BackgroundLearner.apply_edits` — ADD/UPDATE/soft-DELETE as bi-temporal mutations (KG-2.11).

## T3 — Async controls (US-4)  [code]
- [x] `with_backoff` (bounded), `Semaphore(4)`, `schedule` + `await_pending` sync barrier.

## T4 — LLM extraction + CLI entry (US-3,5)  [code]
- [x] `extract_edits` (ORCH-1.27 `learner` role) + `run_learner`; `learn` subcommand in `memory/cli.py`.

## T5 — Tests (NFR)  [test]
- [x] `tests/unit/knowledge_graph/test_kg_2_13_background_learning.py` — AC1–AC8 (9 tests).

## T6 — Artifacts (NFR)  [docs]
- [x] concepts.yaml regen (KG-2.13); CHANGELOG; README EG-KG.compute.backend count; pillar doc note.
