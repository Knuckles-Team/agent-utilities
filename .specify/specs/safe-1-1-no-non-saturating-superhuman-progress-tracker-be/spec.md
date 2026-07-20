# Spec: Non-Saturating Superhuman Progress Tracker (SAFE-1.1)

> Status: **proposed**.
> **Wire-First:** add a non-saturating eval family as new scorers under `harness/` and a "frontier"
> mode on `harness/benchmark.py` (today fixed-target LongMemEval-S, AHE-3.12) — reusing the existing
> `adversarial_verifier` "Hacker Agent" as the setter in a setter-solver loop, the VariantPool
> tournament (AHE-3.2) for agent-vs-agent duels, and feeding verdicts into the AU-AHE.harness.failure-evolution regression
> gate. Extends the AHE-3.1 reliability-scorer registry; no new harness.

## Pre-Flight Checklist
- [x] **Extension target identified** — every benchmark is fixed-target/saturating: AHE-3.12
  LongMemEval-S (`benchmark.py`, "≥98.2%" vs a human ceiling, frozen gold corpus), AHE-3.1
  `quality_gates.py` (absolute 1-5), `eval_corpus.py` (frozen gold cases), `variant_pool.py`
  (tournament vs a fixed reward corpus). No setter-solver / self-play / Elo / compression /
  saturation-detector primitive exists — AU can't distinguish a capability jump from metric saturation
  (the central §6/§7.3 measurement risk).
- [x] **New CONCEPT:AU-OS.scaling.non-saturating-compression-scorer justified** — a methodology that keeps producing signal **past** the
  human/known-answer ceiling is distinct from the fixed-target scorers (AHE-3.1/3.12); it opens the
  SAFE-1 frictions/measurement pillar's first benchmark family.
- [x] **Wire-First confirmed** — new scorers register in the AHE-3.1 reliability registry; the
  setter reuses `adversarial_verifier`; duels reuse VariantPool (AHE-3.2); results feed AHE-3.18.
- [x] **Success metric defined** — a `benchmark.py` "frontier" run yields a relative score (Elo /
  setter-solver pass-gap / compression ratio) that does **not** clamp at a human ceiling, plus a
  saturation flag when an eval's pass-rate has collapsed to ceiling across recent agent versions.

## User Stories

### US-1 — Signal beyond the human ceiling
**As** the evaluation plane, **I want** non-saturating scores, **so that** I can rank superhuman
agents against each other instead of against a saturated human baseline.
- **AC1**: a setter-solver scorer drives the `adversarial_verifier` "Hacker Agent" to *generate*
  problems at the solver's frontier; the metric is the setter↔solver pass-gap, not an absolute ceiling.
- **AC2**: a self-play/zero-sum **Elo** scorer ranks agent-vs-agent task duels (reuse VariantPool
  pairings); a compression-based `EvalScorer` scores description-length reduction — both relative.
- **AC3**: all three register as AHE-3.1 reliability scorers and run via a `benchmark.py` "frontier" mode.

### US-2 — Saturation detector
**As** an operator, **I want** to be warned when a benchmark has saturated, **so that** I don't read
"no improvement" as "no capability."
- **AC4**: a saturation detector flags any eval whose pass-rate has collapsed to ceiling across the last
  N agent versions and recommends promotion to a frontier/relative scorer.
- **AC5**: frontier scores feed the AU-AHE.harness.failure-evolution regression gate so progress is **tracked even where it
  cannot be forecast** (the §6/§7 unpredictability-floor response).

## Non-Functional Requirements
- `tests/unit/harness/test_safe_1_1_frontier_eval.py` (`@pytest.mark.concept(id="SAFE-1.1")`), ≤60s,
  no live LLM: stub setter/solver + duel outcomes; assert the setter-solver gap and Elo are relative
  (no human-ceiling clamp), the compression scorer scores description-length, and the saturation
  detector flags a ceiling-collapsed eval.
- `pre-commit run --all-files` green; `scripts/build_concepts_yaml.py` re-run so SAFE-1.1 lands in
  `docs/concepts.yaml`; `scripts/check_concepts.py` passes.
- Per-concept doc under `docs/architecture/` (new `safety_measurement.md`, opening the SAFE-1 pillar
  guide), relating SAFE-1.1 to AU-OS.scaling.multi-agent-scaling-law (multi-agent scaling laws) and AU-OS.audit.recursive-improvement-velocity-tracker (RSI velocity) as the
  measurement triad for the unpredictability floor.
