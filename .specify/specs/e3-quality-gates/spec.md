# Spec: EPIC 3 — Quality Gate Pipeline (AHE-3.13)

> Design: `.specify/design/ahe-3.13-layered-pre-emit-gate/design.md`. Depends on EPIC 2.

## Pre-Flight Checklist
- [x] Design exists; KG-nearest table (AHE-3.13 max 0.66 vs AHE-3.1) <0.70.
- [x] Extension points: `harness/evaluation_engine.py`, `graph/_router_impl.py` seam, `graph/parallel_engine.py:_synthesize`.
- [x] Wire-First: ≤2 hops from `graph_orchestrate`.
- [ ] Live `kg_search` confirmation.

## User Stories
### US-1 — Discovery-form brief lock (Turn-1)
**As** the orchestrator, **I want** an ambiguous brief to trigger a structured discovery form before dispatch (reusing `/api/human`), **so that** work starts from a locked brief.
- **AC1**: Gate sits between `router_step`→`dispatcher_step`; ambiguous input → form; complete brief → pass-through.
- **AC2**: Gate is config-gated (`warn`|`block`|`off`); default `warn`.

### US-2 — Preflight checklist (P0/P1/P2)
**As** a skill author, **I want** a `references/checklist.md` whose P0 rules block emission, **so that** known failures can't ship.
- **AC3**: `PreflightGate.check()` reads the active skill's checklist; failing P0 blocks (in `block` mode) and is surfaced.
- **AC4**: Per-skill `critique.policy` (E5) toggles enforcement (required/opt-out/opt-in).

### US-3 — 5-dimensional self-critique (pre-emit)
**As** the engine, **I want** outputs scored 1–5 on named dimensions with a fix-and-re-score loop before `_synthesize` returns, **so that** weak output is iterated, not emitted.
- **AC5**: `MultiDimensionalCritique.critique(result)` returns per-dimension scores (coverage, coherence, evidence/citation, safety, specificity); any `<3` triggers one bounded re-derive.
- **AC6**: Critique scores are **persisted to the KG and fed to AHE-3.1/3.2** (self-improving signal).
- **AC7**: A seeded anti-slop antipattern (e.g., invented metric without source) is flagged from the KG-stored registry.

### US-4 — Documented prompt-stack precedence
**As** a developer, **I want** prompt layers composed in documented priority with later/hard/mode-pinned layers overriding, **so that** rule conflicts are auditable.
- **AC8**: A mode-scoped directive provably overrides a conflicting base rule; the composition is inspectable in the run trace.

## Non-Functional Requirements
- `@pytest.mark.concept(id="AHE-3.13")`; ≤60s; no network.
- Default `warn` ⇒ zero regression; `block` opt-in.
- Hot-path overhead budget: critique adds ≤1 extra LLM pass per emit by default.
- Docs: `docs/pillars/3_*/AHE-3.13.md`; concepts.yaml regen.

## Tasks
- [ ] T1 `harness/preflight_gate.py`: checklist parser + `check()`. *(unit)*
- [ ] T2 `harness/evaluation_engine.py`: `MultiDimensionalCritique` + re-score loop. *(unit)*
- [ ] T3 `knowledge_graph/`: anti-slop antipattern registry (nodes + query helper). *(unit)*
- [ ] T4 `graph/_router_impl.py`: discovery gate at the router→dispatcher seam. *(integration)*
- [ ] T5 `graph/parallel_engine.py`: pre-emit critique hook at `_synthesize`. *(integration)*
- [ ] T6 Prompt-composition layer with documented precedence + trace. *(unit)*
- [ ] T7 Feed critique scores to AHE-3.1/3.2. *(integration)*
- [ ] T8 Docs/concepts/wiring-audit/CHANGELOG.
