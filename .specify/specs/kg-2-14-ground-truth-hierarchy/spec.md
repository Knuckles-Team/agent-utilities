# Spec: Ground-Truth Context Authority (KG-2.14)

> References design: `.specify/design/kg-2-14-ground-truth-hierarchy/design.md`. Status: **shipped**.

## Pre-Flight Checklist
- [x] Design exists; KG-nearest-concepts table completed.
- [x] Extension target identified (KG-2.1); new CONCEPT:KG-2.14 justified (authority axis).
- [x] Wire-First confirmed: 2 hops from the memory CLI/MCP context / `MemoryEngine.build_startup_context`.
- [x] Success metric defined (redundant re-fetch reduction).

## User Stories

### US-1 — Authoritative injected memory
**As** an agent at startup, **I want** injected memory ranked + declared authoritative, **so that** I
use it directly instead of re-fetching what is already in my prompt.
- **AC1**: `StartupChunk.source_authority ∈ {advisory, standard, authoritative}` (default `standard`).
- **AC2**: durable/curated memory (profile/team/agents_md, identity/preference/active-goal/rule headings)
  classifies as `authoritative` and gets the `AUTHORITY_BOOST` priority bump.
- **AC3**: `build_payload` emits a Ground-Truth Hierarchy preamble naming the authoritative sources and
  instructing "do not re-fetch / injected memory wins", **only** when authoritative sources are present.
- **AC4**: the preamble is budget-reserved (never crowds out content); behavior is backward-compatible.

## Non-Functional Requirements
- `tests/unit/knowledge_graph/test_kg_2_14_ground_truth_authority.py` (`@pytest.mark.concept(id="KG-2.14")`), ≤60s.
- `pre-commit` green; concept registry regenerated; per-concept doc authored.
