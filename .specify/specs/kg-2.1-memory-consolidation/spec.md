# Spec: Memory Consolidation Loop (CONCEPT:KG-2.1)

## Pre-Flight Checklist (Mandatory — DSTDD)

- [x] **KG search completed** — `.specify/design/kg-2.1-memory-synthesis/design.md` exists
- [x] **Extension point identified** — Extends KG-2.1 (Tiered Memory & Context)
- [x] **C4 diagram created** — showing integration into KG pillar
- [x] **No new CONCEPT: tag** — uses existing KG-2.1
- [ ] **`code-enhancer` audit** — pending implementation
- [ ] **Design validation passes** — pending SDDManager integration

## Design Reference

→ [design.md](../../design/kg-2.1-memory-synthesis/design.md)

## Research Sources

| Paper | ArXiv ID | Key Contribution |
|-------|----------|-----------------|
| MEMO Survey | 2504.01990v2 | Memory lifecycle, Ebbinghaus decay, CLS theory |
| ParamMem | 2604.27707v1 | Generalization ceiling proof, Trace→Skill→FT pipeline |
| MemReranker | 2605.06132v1 | MemOS, reasoning-aware reranking |

## User Stories

### US-1: Ebbinghaus Time-Decay Scoring

**As a** KG memory consumer, **I want** memory recall results scored with time-decay relevance, **so that** recent and frequently-accessed memories rank higher than stale ones.

**Acceptance Criteria:**
- [x] `recall_memory()` applies exponential decay: `relevance = base_score × exp(-λt)`
- [x] Half-lives configurable per tier: Working=5min, Episodic=4hr, Semantic=30-day
- [x] `kg_memory(action='recall')` MCP tool returns `decay_adjusted_score` field
- [x] Backward compatible: existing recall behavior unchanged when decay disabled

### US-2: Trace→Skill Distillation Rule

**As an** agent system, **I want** successful interaction patterns automatically distilled into reusable SkillNode proposals, **so that** the agent improves with experience.

**Acceptance Criteria:**
- [x] New `TraceToSkillRule` in `SynthesisEngine` detects patterns from ChatTurn/ExecutionTrace nodes
- [x] Minimum evidence: 3 successful traces with shared tool/approach pattern
- [x] Produces `SkillNode` proposals with confidence scoring
- [x] Both timer-driven (periodic) and on-demand (`kg_memory(action='consolidate')`) triggers

### US-3: Memory Poisoning Defense

**As a** security-conscious agent, **I want** memory entries tracked with provenance trust scores, **so that** injected/untrusted content doesn't propagate across sessions.

**Acceptance Criteria:**
- [x] `MemoryNode` gains `trust_score` field (0.0–1.0, default 0.8)
- [x] Trust inherited from source agent's trust level
- [x] Memories with `trust_score < 0.3` quarantined (excluded from default recall)
- [x] Explicit `include_untrusted=True` flag to override quarantine

### US-4: Instruction-Aware Recall Reranking

**As a** memory consumer, **I want** recall results reranked against my current task context, **so that** the most task-relevant memories surface first.

**Acceptance Criteria:**
- [x] Post-retrieval reranker considers task embedding vs memory embedding
- [x] No LLM call required — uses dot-product similarity
- [x] Configurable: can be disabled for raw cosine-only retrieval

## Non-Functional Requirements

- [x] All existing tests continue to pass (zero regression)
- [x] Pre-commit hooks pass cleanly
- [x] Documentation updated in `docs/pillars/2_epistemic_knowledge_graph/KG-2.1-Tiered_Memory_And_Context.md`
- [x] New functionality wired into kg_memory MCP tool
- [x] CONCEPT:KG-2.1 tags in all new code and tests
