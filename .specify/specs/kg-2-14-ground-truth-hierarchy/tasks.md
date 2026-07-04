# Tasks: Ground-Truth Context Authority (AU-KG.memory.ground-truth-preamble-declaring) — shipped

## T1 — Authority tier  [code] ✅
- [x] `StartupChunk.source_authority` + `_authority_for` + `AUTHORITY_ADVISORY/STANDARD/AUTHORITATIVE` + `AUTHORITY_BOOST` (`memory_engine.py`).

## T2 — Priority boost + preamble  [code] ✅
- [x] `_chunk_priority` boosts authoritative chunks; `_build_authority_preamble` emits the Ground-Truth block, budget-reserved, inserted in `build_payload`.

## T3 — Live-path wiring  [code] ✅
- [x] Preamble emitted on every `build_payload` (CLI `context` / MCP context / `MemoryEngine.build_startup_context`).

## T4 — Tests  [test] ✅
- [x] `tests/unit/knowledge_graph/test_kg_2_14_ground_truth_authority.py` (5 tests).

## T5 — Artifacts  [docs] ✅
- [x] concepts.yaml regen (AU-KG.memory.ground-truth-preamble-declaring); per-concept doc; concept_map/overview rows; CHANGELOG; pillar summary.
