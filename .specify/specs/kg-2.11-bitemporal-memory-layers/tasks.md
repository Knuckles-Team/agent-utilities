# Tasks: Bi-Temporal Memory Layers (KG-2.11)

## T1 — Procedural layer (US-1)  [code]
- [x] Add `memory_type` + `target_entity` (+ node-level temporal mirror) to `MemoryNode` (`models/knowledge_graph.py`).

## T2 — Bi-temporal core (US-2, US-3, US-4)  [code]
- [x] New `knowledge_graph/core/bitemporal.py`: `stamp_bitemporal`, `is_valid_as_of`, `filter_as_of`, `resolve_precedence`, `supersede`.

## T3 — Wire stamping into hot path (US-2)  [code]
- [x] `engine.link_nodes` calls `stamp_bitemporal` for every relationship write (`core/engine.py:415`).

## T4 — As-of queries (US-3)  [code]
- [x] `query_cypher(as_of=...)` post-filters via `filter_as_of` (`orchestration/engine_query.py`).
- [x] `graph_query` MCP tool forwards `as_of` (`mcp/kg_server.py`).

## T5 — Contradiction precedence (US-4)  [code]
- [x] `resolve_temporal_contradiction` writes `SUPERSEDES` + closes loser `valid_to` (`orchestration/engine_query.py`).

## T6 — Tests (NFR)  [test]
- [x] `tests/unit/knowledge_graph/test_kg_2_11_bitemporal.py` — AC1–AC6 (9 tests).

## T7 — Artifacts (NFR)  [docs]
- [x] concepts.yaml regen (KG-2.11); CHANGELOG; README EG-KG.compute.backend range; pillar doc note.
