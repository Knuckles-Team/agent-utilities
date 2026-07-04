# Spec: Program-Synthesis Reasoner with a Low-Complexity (MDL) Prior (KG-2.66)

> Status: **proposed**. **Wire-First:** EXTENDS the KG-2.65 `Reasoner` protocol/registry
> (register a new concrete `Reasoner`, do not rebuild the seam) and the
> `harness/selection_operators.py` operator registry (add an MDL/length objective beside the
> existing verifier-free selectors); searches over the RLM pure-function tool-composition space
> (`rlm/skills.py` `Skill`/`merge_skills`) and validates candidates in the ORCH-1.38 sandbox
> (`rlm/sandboxes/router.py` `SandboxRouter`). Reuse these four seams; build nothing parallel.

## Pre-Flight Checklist
- [x] Extension target identified: KG-2.65 `Reasoner` registry + `harness/selection_operators.py`
  (operator registry) + `rlm/skills.py` (typed composition space) + `rlm/sandboxes/router.py`
  (candidate validation). All four exist and are verified in-repo.
- [x] New CONCEPT:AU-KG.enrichment.multimodal-readers justified: AU's only "synthesis" today is AHE-3.3 (evolving sklearn
  regressors) / AHE-3.2 (evolving prompt/agent text) — search over a fixed parametric or textual
  space, **not** programs/DSLs, and **no** length/MDL/Solomonoff-flavoured prior governs any search
  (zero hits for solomonoff/kolmogorov/MDL/description-length/occam). This adds the missing axis:
  search a structured typed program space biased toward low-complexity solutions.
- [x] Wire-First confirmed: a candidate is `compose skills → sandbox-validate → score = quality −
  λ·description_length` selected through the *existing* selection-operator registry; reachable in
  ≤3 hops from the KG-2.65 reasoner registry (the live `Reasoner`-selection entry point).
- [x] Success metric defined: on a held-out tool-composition task suite, the MDL-prior search finds
  a correct program whose description length is **strictly shorter** than a quality-only (λ=0)
  baseline's at equal pass-rate (Occam preference is *measurable*, not merely configured).
- [x] Dependency noted: KG-2.65 `Reasoner` seam must land first (the report sequences 2.65 → 2.66);
  until then this Reasoner registers behind the same registry the sibling spec introduces.

## User Stories

### US-1 — A program-synthesis Reasoner over a typed DSL
**As** the orchestrator selecting an inference paradigm, **I want** a `ProgramSynthesis` Reasoner
that searches compositions of RLM pure-function tools, **so that** structured tasks are solved by
*searching a program space*, not only by mutating text or floats.
- **AC1**: `ProgramSynthesis` implements the KG-2.65 `Reasoner` protocol (`context+goal →
  action/answer+trace`) and self-registers in the KG-2.65 registry under a capability tag
  (e.g. `program_synthesis`), selectable by task topology like any other `Reasoner`.
- **AC2**: Candidate programs are built by composing `rlm/skills.py` tools via `merge_skills`
  (name-collision = `ValueError`, so composition stays explicit); the search space is the typed
  tool surface, not free text.
- **AC3**: Each candidate is executed/validated in the ORCH-1.38 sandbox through
  `rlm/sandboxes/router.py` `SandboxRouter`; an unsafe or non-terminating candidate is rejected,
  never run in-process.

### US-2 — An explicit low-complexity (MDL) selection objective
**As** the search loop, **I want** a description-length objective in the operator registry, **so
that** among candidates that pass, the *shorter* program wins (a finite stand-in for the universal
prior).
- **AC4**: `harness/selection_operators.py` gains a pure, model-free `description_length` /
  `mdl_score` operator (program size in DSL primitives, no network), beside the existing
  verifier-free selectors — added to the same registry, not a new module.
- **AC5**: The synthesis loop ranks survivors by `quality − λ·description_length` (λ default makes
  the prior active, opt-out via `λ=0` reproducing pure quality selection — backward-compatible per
  the AU-AHE.optimization.telemetry-optimization opt-in rule), and the chosen program is the shortest among top-quality candidates.
- **AC6**: With `λ>0`, the suite demonstrates the success metric: a correct program shorter than the
  λ=0 baseline's at equal pass-rate.

## Non-Functional Requirements
- `tests/unit/knowledge_graph/test_kg_2_66_program_synthesis.py`
  (`@pytest.mark.concept(id="KG-2.66")`), ≤60s, no live engine/LLM — exercise the live path
  (registered `Reasoner` → compose → sandbox-validate → MDL-ranked select) plus a unit test of the
  pure `description_length` operator.
- `pre-commit run --all-files` green (no stubs, no env sprawl, no back-compat shims).
- Concept registry regenerated (`scripts/build_concepts_yaml.py`; `scripts/check_concepts.py` green);
  a `CONCEPT:AU-KG.enrichment.multimodal-readers` marker on the new `Reasoner`.
- Per-concept doc authored (cite the Solomonoff/MDL provenance + arXiv 2606.12683 §4/§5.2 in the
  docstring, name from purpose not paper).
