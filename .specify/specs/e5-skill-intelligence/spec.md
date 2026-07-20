# Spec: EPIC 5 — Skill Intelligence (extends ORCH-1.2 / AU-ECO.mcp.toolkit-live-discovery / AHE-3.1)

> Design: `.specify/design/e5-skill-intelligence/design.md`. Extension-only (no new CONCEPT:ID).
> Depends on EPIC 3 (critique policy + success-rate signal).

## Pre-Flight Checklist
- [x] Design exists; KG-nearest table max 0.74 ≥0.70 → **extend, no new concept**.
- [x] Extension points: `graph/routing/strategies/` (picker), `workflows/skill_compiler.py` (frontmatter), `harness/evaluation_engine.py` (success rate).
- [x] Wire-First: ≤1 hop from routing.

## User Stories
### US-1 — Scenario/category taxonomy
**As** a user, **I want** skills tagged `scenario`/`category`/`source` (from frontmatter or inferred), **so that** discovery groups them semantically.
- **AC1**: `skill_compiler` parses `scenario` ∈ {design,marketing,operation,engineering,finance,hr,sales,personal}, `category`, `source`; untagged skills get an inferred scenario.

### US-2 — Eval-scored self-improving picker
**As** the orchestrator, **I want** `skill_picker.pick(query, context)` to score candidates by keyword overlap + tier fit + **prior success rate from AHE-3.1**, **so that** skills that actually work rank higher.
- **AC2**: A skill with higher historical success rate ranks above a keyword-equal rival (unit).
- **AC3**: A scenario filter narrows candidates before scoring.
- **AC4**: Missing success-rate falls back to a neutral prior (no crash on cold skills).

### US-3 — Per-skill critique policy
**As** a skill author, **I want** `critique.policy` ∈ {required, opt-out, opt-in, null} in frontmatter, **so that** the E3 critique gate is controllable per skill.
- **AC5**: `opt-out` bypasses the critique gate; `required` forces it; default follows the project rollout.

## Non-Functional Requirements
- `@pytest.mark.concept(id="ORCH-1.2")` (extension); ≤60s; no network.
- New routing strategy; existing routing untouched when picker disabled (zero regression).
- Docs: update `docs/pillars/1_graph_orchestration/ORCH-1.2.md` (picker strategy) + ECO-4.6.

## Tasks
- [ ] T1 `workflows/skill_compiler.py`: extend frontmatter (scenario/category/source/critique.policy) + inference. *(unit)*
- [ ] T2 `workflows/skill_picker.py`: multi-factor scorer reading AHE-3.1 win-rates. *(unit)*
- [ ] T3 `graph/routing/strategies/`: register picker strategy; invoke from `_router_impl`. *(integration)*
- [ ] T4 Critique gate (E3) reads `critique.policy`. *(integration)*
- [ ] T5 Docs/wiring-audit/CHANGELOG.
