# Spec: Bi-Temporal Memory Layers (KG-2.11)

> References design: `.specify/design/kg-2.11-bitemporal-memory-layers/design.md`

## Pre-Flight Checklist

- [x] Design document exists; KG-nearest-concepts table completed.
- [x] Extension target identified (KG-2.1, similarity 0.86 ≥ 0.70; +KG-2.3 temporal facet).
- [x] New CONCEPT:KG-2.11 justified as augmentation (valid-time axis + procedural layer).
- [x] Wire-First confirmed: stamping on the `link_nodes` hot path; as-of via `graph_query`.

## User Stories

### US-1 — Procedural memory layer
**As** the memory engine, **I want** `MemoryNode.memory_type ∈ {semantic, episodic, procedural}`
with a `target_entity`, **so that** behavioural rules are first-class and entity-scoped.
- **AC1**: `MemoryNode` accepts `memory_type` + `target_entity`; defaults are backward-compatible (`semantic`, `""`).

### US-2 — First-class bi-temporal stamping
**As** the KG, **I want** every relationship stamped with `event_time`, `storage_time`,
`valid_from`, `valid_to`, **so that** event time is distinguished from storage time.
- **AC2**: `stamp_bitemporal` sets all four; explicit `event_time` is preserved distinct from `storage_time`; re-stamping is idempotent.
- **AC3**: `engine.link_nodes` auto-stamps via `stamp_bitemporal` (verified by reading the hot path).

### US-3 — As-of queries
**As** an agent, **I want** `query_cypher(..., as_of=T)` / `graph_query(as_of=T)` to return only
facts valid at instant T, **so that** "what was true on date T" is answerable.
- **AC4**: `is_valid_as_of` / `filter_as_of` implement `valid_from <= T < valid_to` with open-interval and missing-metadata handling.
- **AC5**: `graph_query` forwards `as_of` to `query_cypher`.

### US-4 — Event-time contradiction precedence
**As** the KG, **I want** contradictions resolved by later `event_time`, **so that** the newer
fact supersedes the older without deleting history.
- **AC6**: `resolve_precedence` picks the later-event_time winner (order-independent); `supersede` closes the loser's `valid_to` at the winner's `event_time` (no delete).
- **AC7**: `resolve_temporal_contradiction` writes a `SUPERSEDES` edge and persists the loser's closed interval.

## Non-Functional Requirements
- `tests/unit/knowledge_graph/test_kg_2_11_bitemporal.py` (`@pytest.mark.concept(id="KG-2.11")`), ≤60s, no network.
- `pre-commit run --all-files` green; concepts.yaml regenerated; 7-artifact mandate satisfied.
