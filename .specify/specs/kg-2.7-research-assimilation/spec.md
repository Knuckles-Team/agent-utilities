# Spec: Research Assimilation Tracking (CONCEPT:KG-2.7)

## Pre-Flight Checklist (Mandatory — DSTDD)

- [x] **KG search completed** — `.specify/design/kg-2.7-research-assimilation/design.md` exists
- [x] **Extension point identified** — Extends KG-2.7 (Research Intelligence)
- [x] **C4 diagram created** — showing integration into KG pillar
- [x] **No new CONCEPT: tag** — uses existing KG-2.7
- [ ] **`code-enhancer` audit** — pending implementation
- [ ] **Design validation passes** — pending SDDManager integration

## Design Reference

→ [design.md](../../design/kg-2.7-research-assimilation/design.md)

## User Stories

### US-1: Research-to-Feature Linkage (ASSIMILATED_INTO)

**As a** developer integrating research, **I want** research papers linked to codebases via assimilation edges, **so that** I can track which papers informed which features across any project.

**Acceptance Criteria:**
- [x] `kg_write(action='assimilate')` creates `ASSIMILATED_INTO` edge from Article → Code/Codebase
- [x] Edge properties: `concept_ids[]`, `sdd_feature_ids[]`, `codebase`, `status`, `assimilation_date`
- [x] Status enum: `reviewed` → `assimilated` → `implemented`
- [x] Generalized: `codebase` property disambiguates across multiple projects

### US-2: SDD Feature Nodes with Research Provenance

**As a** SDD author, **I want** feature specifications persisted as KG nodes linked to their source papers, **so that** traceability from feature → paper → innovation claim is maintained.

**Acceptance Criteria:**
- [x] `kg_trace(action='submit_sdd')` creates `SDDFeature` node with `research_sources[]` property
- [x] Auto-creates `DERIVED_FROM_RESEARCH` edges linking SDDFeature → Article nodes
- [x] SDDFeature properties: `id`, `name`, `concept_ids[]`, `research_sources[]`, `status`, `sdd_path`

### US-3: Auto-Detection of Assimilation on SDD Completion

**As an** automated system, **I want** papers automatically marked as assimilated when their derived SDD features reach COMPLETED status, **so that** no manual tracking is needed.

**Acceptance Criteria:**
- [x] When `kg_trace(action='submit_sdd')` receives `status='COMPLETED'`:
  - All linked Articles (via DERIVED_FROM_RESEARCH) get `ASSIMILATED_INTO` edges auto-created
  - Edge status set to `implemented`
- [x] Idempotent: re-submitting COMPLETED status doesn't create duplicate edges

### US-4: Comparative Analysis Exclusion Filter

**As a** comparative analysis user, **I want** future analyses to automatically skip papers whose findings are already implemented, **so that** I focus only on net-new research value.

**Acceptance Criteria:**
- [x] `discover_innovations()` gains `exclude_assimilated: bool = False` parameter
- [x] When True, Article chunks with `ASSIMILATED_INTO` edges (status='implemented') are filtered
- [x] `kg_search(mode='discover')` passes through the flag from MCP callers
- [x] Filter is per-codebase: only excludes papers assimilated into the queried codebase

## Non-Functional Requirements

- [x] All existing tests continue to pass (zero regression)
- [x] Pre-commit hooks pass cleanly
- [x] Documentation updated in `docs/pillars/2_epistemic_knowledge_graph/KG-2.7-Research_Intelligence.md`
- [x] New functionality wired into kg_write, kg_trace, and kg_search MCP tools
- [x] CONCEPT:KG-2.7 tags in all new code and tests
