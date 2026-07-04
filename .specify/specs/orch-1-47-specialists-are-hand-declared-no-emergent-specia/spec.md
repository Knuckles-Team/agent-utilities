# Spec: Emergent Specialist Discovery (ORCH-1.47)

> Status: **proposed**. **Wire-First:** extends `agent_utilities/graph/team_evolution.py`
> (`TeamEvolutionEngine.evaluate_and_evolve`, AHE-3.4) — replace the **hard-coded**
> `"proposed_agent": "error_recovery_specialist"` mutation with a *discovered* specialist
> role — using `agent_utilities/graph/social_system.py` (MASS heterogeneity, ORCH-1.32) as
> the promotion objective. Reuse, do not rebuild: the `MutationProposal` graph path and the
> `CoordinationTrace`/`Episode` reads already exist (`graph/coordination.py`,
> `models/knowledge_graph.py`); `graph/manifest_generators.py` (AU-ORCH.execution.autonomous-department-orchestration) consumes the
> resulting `TeamComposition` role unchanged.

## Pre-Flight Checklist
- [x] Extension target identified: `TeamEvolutionEngine.evaluate_and_evolve`
  (`graph/team_evolution.py`) — currently proposes a single hard-coded label.
- [x] New `CONCEPT:AU-ORCH.execution.swe-agent-system-prompt` justified: AHE-3.4 *proposes a known mutation*; ORCH-1.47 adds
  the missing **discovery** axis — *what* specialization the task stream is under-served by.
  Distinct capability (niche detection), not a tweak to the existing proposer.
- [x] Wire-First confirmed: discovery runs **inside** the existing live
  `evaluate_and_evolve` path; output flows through the existing `MutationProposal` →
  regression-gated golden-loop (AU-AHE.harness.failure-evolution) → `manifest_generators` consumer. No new entry point.
- [x] Success metric: on a synthetic failing/expensive task stream with an under-served niche,
  the discovered specialist's role label/prompt is **derived from the niche cluster** (not a
  constant), and the proposal is gated to only promote when it **raises MASS archetype entropy
  AND projected outcome quality** (no regression-on-homogeneous-stream).

## User Stories

### US-1 — Discover an under-served niche from the task stream
**As** the team-evolution engine, **I want** to cluster failing/expensive
`CoordinationTrace`+`Episode` task embeddings and detect niches where **no existing archetype
has high competence**, **so that** I propose specialists the collective actually lacks.
- **AC1**: a `discover_specialization_niches(team_id)` helper clusters recent
  failing/expensive episodes by embedding and returns niches scored by `(failure_density,
  best_archetype_competence)`; a niche qualifies only when best competence is below a
  module-constant threshold.
- **AC2**: when no qualifying niche exists, `evaluate_and_evolve` proposes **no**
  `add_specialist` mutation (the constant `error_recovery_specialist` is gone — no fallback label).

### US-2 — Instantiate a discovered specialist role through the existing gate
**As** the collective, **I want** the discovered niche turned into a concrete `TeamComposition`
specialist role (label + prompt + tool scope) and pushed through the AHE-3.4 `MutationProposal`
path, **so that** division-of-labor increases only after the regression-gated golden loop accepts it.
- **AC3**: the emitted `MutationProposal` carries a niche-derived `proposed_agent` label,
  a generated role prompt, and a tool scope; it is written via the **existing**
  `PROPOSED_MUTATION` MERGE (no new persistence path).
- **AC4**: a proposal is **promotable** only when it raises MASS archetype entropy
  (`social_system.py`, ORCH-1.32) **and** projected outcome quality vs. the pre-proposal
  baseline; otherwise it is recorded `status='rejected'` with the reason, never auto-applied.

## Non-Functional Requirements
- `tests/unit/graph/test_orch_1_47_emergent_specialist_discovery.py`
  (`@pytest.mark.concept(id="ORCH-1.47")`), ≤60s — covers AC1–AC4 incl. a live-path test that
  calls `evaluate_and_evolve` and asserts the niche-derived proposal (not the old constant).
- `pre-commit run --all-files` green (no new bare `os.environ`; thresholds are module constants).
- Concept registry regenerated (`scripts/build_concepts_yaml.py`; `scripts/check_concepts.py` green).
- Per-concept doc authored (extend `docs/architecture/multi_agent_social_system.md`).
