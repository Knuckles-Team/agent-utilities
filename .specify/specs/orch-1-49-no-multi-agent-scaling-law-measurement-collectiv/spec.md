# Spec: Multi-Agent Scaling-Law Eval Harness (AU-ORCH.execution.robust-multi-format-edit)

> Status: **proposed**. Co-satisfies AU-OS.scaling.multi-agent-scaling-law (implement as one harness — see
> `.specify/specs/safe-1-2-multi-agent-scaling-laws-are-not-measured`).
> **Wire-First:** EXTENDS `agent_utilities/graph/parallel_engine.py` (`ParallelEngine`,
> the live collective runner) as the per-config execution seam, and
> `agent_utilities/harness/continuous_evaluation_engine.py` (the existing eval engine
> that has no N axis) as the sweep/scoring host. Reuse `graph/social_system.py` for
> topology variants and `graph/population_drift.py` for collapse gating — **do not
> rebuild** a collective runner, an eval loop, or a drift detector.

## Pre-Flight Checklist
- [x] Extension target identified: `ParallelEngine` (run a collective at a fixed (N, density,
  archetype-mix)) + `continuous_evaluation_engine.py` (sweep + score) — both already live.
- [x] New CONCEPT:AU-ORCH.execution.robust-multi-format-edit justified: AU instruments collective **health** (per-run snapshots)
  but never collective **capability-per-compute** — no harness holds a task fixed and sweeps
  population size / interaction density to fit capability-vs-N. This is the §7.5 "Multi-Agent
  Scaling Laws" quantity; no existing concept measures it (`scaling_signals.py` is load
  autoscaling, `voi_budget_controller.py` is single-agent, `benchmark.py` is single-agent).
- [x] Wire-First confirmed: sweep → `ParallelEngine.run` per config → score fixed task suite →
  fit `capability ~ instances^α` → persist as KG calibration-style nodes (mirror AU-KG.domains.agent-calibration-reputation-tracking),
  ≤ 3 hops from the eval engine entry point.
- [x] Success metric defined: a real small-N sweep produces a fitted exponent α with confidence
  bands and a knee, persisted to the KG and consumable by AHE-3.4 team evolution.

## User Stories

### US-1 — Sweep a fixed task over population size
**As** an evolution operator, **I want** to hold a task suite fixed and sweep
(N, interaction-density, archetype-mix) through the real collective runner, **so that** I can
see whether adding agents raises solved-task quality super- vs sub-linearly.
- **AC1**: A `ScalingLawHarness` accepts a fixed task suite + a grid of `(N, density,
  archetype_mix)` configs and runs each config through `ParallelEngine` (no new collective
  runner), recording collective-quality (reuse the existing per-run quality/judge path) and
  token/compute per config (ECO-4.40 pricing / AU-OS.observability.no-op-without-metrics metrics).
- **AC2**: Topology variants (density / neighborhood) come from `social_system.py`; a config
  whose run trips `population_drift.py` collapse detection is flagged `collapsed` and excluded
  from the fit rather than silently skewing it.

### US-2 — Fit and persist the scaling law
**As** the harness, **I want** to fit `capability ~ instances^α` with confidence bands and
locate the knee, **so that** the empirical curve is a durable, queryable artifact.
- **AC3**: `fit_scaling_law(points)` is **pure** (no engine/LLM): given `(N, quality, compute)`
  points it returns `{alpha, alpha_ci, regime ∈ {sublinear, linear, superlinear}, knee_n}`.
- **AC4**: Each fit is persisted as a KG node mirroring the AU-KG.domains.agent-calibration-reputation-tracking calibration-node shape
  (exponent + confidence band + provenance: task-suite id, config grid, run ids), so a later
  cycle can re-fit and compare.

### US-3 — Feed the empirical knee to team evolution
- **AC5**: The fitted knee / regime is surfaced to AHE-3.4 (`graph/team_evolution.py`) as an
  advisory signal (e.g. "superlinear up to knee_n, sublinear after") — wired as a read on a
  live path, defaulting on, never silent storage.

## Non-Functional Requirements
- `tests/unit/harness/test_orch_1_49_scaling_law_harness.py`
  (`@pytest.mark.concept(id="AU-ORCH.execution.robust-multi-format-edit")`), ≤60s, no live engine/LLM — exercise the pure
  `fit_scaling_law` (super/sub/linear synthetic points) and a `ParallelEngine`-fake sweep
  asserting collapsed configs are excluded and a KG node is written.
- `pre-commit run --all-files` green; `docs/concepts.yaml` regenerated via
  `scripts/build_concepts_yaml.py` (`scripts/check_concepts.py` passes); per-concept doc
  authored for AU-ORCH.execution.robust-multi-format-edit (and the shared AU-OS.scaling.multi-agent-scaling-law note).
- Small-N sweeps run in-repo; genuinely large-N curves need external compute (out of scope) —
  the harness fits AU's *own* local curve as one empirical datapoint.
