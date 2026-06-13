# Spec: Multi-Agent Scaling-Law Harness (SAFE-1.2)

> Status: **proposed**. Co-satisfies ORCH-1.49 — implement as **one** harness.
> **Wire-First**: EXTENDS `agent_utilities/graph/parallel_engine.py` (`ParallelEngine` — the live
> collective executor) and `agent_utilities/harness/continuous_evaluation_engine.py` (the fixed-task
> eval loop that today has no N axis). Reuse `graph/social_system.py` (ORCH-1.32 topology variants) and
> `graph/population_drift.py` (AHE-3.2 collapse gating); do **not** rebuild any of these.

## Pre-Flight Checklist
- [x] Extension target identified: `ParallelEngine` (collective executor) + `continuous_evaluation_engine.py`
      (fixed-task eval); both exist and are live-called. Topology from `social_system.py`, collapse gate
      from `population_drift.py`, cost from `observability` (OS-5.23) / pricing (ECO-4.40).
- [x] New CONCEPT:SAFE-1.2 justified: AU instruments collective **health** (per-run snapshots) but never
      collective **capability-per-compute** — it cannot fit `capability ~ instances^α`. This is a new
      measurement primitive, not a config knob.
- [x] Wire-First confirmed: harness sweeps `(N, density, archetype_mix)` **through the existing**
      `ParallelEngine.run` over a frozen task suite (≤3 hops from a CLI/eval-engine entry point);
      no new collective execution path.
- [x] Success metric defined: for a fixed task suite, the harness produces a fitted scaling curve
      `capability(N)` with α (sub-/super-linear exponent) + confidence bands, persisted as KG nodes
      mirroring KG-2.27 calibration facts.

## User Stories

### US-1 — Sweep population size against a fixed task suite
**As** an evolution operator, **I want** to hold a task suite fixed and sweep collective size/density,
**so that** I can measure whether adding agents raises solved-task quality super- or sub-linearly.
- **AC1**: A `ScalingLawHarness` accepts a task suite + a sweep grid of `(N, density, archetype_mix)` and
  runs each config through the **existing** `ParallelEngine.run` (no new executor), reusing
  `social_system.py` to realize the topology/density for each N.
- **AC2**: Each config records `(N, collective_quality, tokens, compute_cost)` — quality from the fixed
  suite's scorer, tokens/cost from the live OS-5.23 observability / ECO-4.40 pricing inputs (no new meter).
- **AC3**: A config whose `population_drift.py` collapse detector fires is flagged `collapsed=True` and
  excluded from the fit (collapse gating, not silent inclusion).

### US-2 — Fit and persist the capability-per-compute curve
**As** team-evolution (AHE-3.4), **I want** the fitted curve + knee persisted as graph facts,
**so that** team sizing reads an empirical scaling law instead of guessing.
- **AC4**: `fit_scaling_law` is **pure** (no engine/LLM): given the recorded points it fits
  `capability ~ instances^α`, returns `alpha`, per-point residuals, and a confidence band; sub-linear
  (`α<1`), linear (`α≈1`), super-linear (`α>1`) are distinguishable.
- **AC5**: The fitted law + empirical knee persist as KG nodes (mirroring KG-2.27 calibration nodes),
  queryable later; AHE-3.4 can read the knee to bound team size.

### US-3 — Small-N in-repo, large-N opt-in
- **AC6**: Default sweep is small-N (in-repo, no external compute); the grid is caller-supplied so a
  larger sweep is opt-in. The harness never assumes external compute is present.

## Non-Functional Requirements
- `tests/integration/harness/test_scaling_law_harness.py` (`@pytest.mark.concept(id="SAFE-1.2")`), ≤60s,
  no live engine/LLM (stub `ParallelEngine.run` quality returns + pure `fit_scaling_law`).
- `pre-commit run --all-files` green; `docs/concepts.yaml` regenerated via `scripts/build_concepts_yaml.py`
  + `scripts/check_concepts.py` clean.
- Per-concept doc authored (harness + the `capability ~ instances^α` fit, citing §5.4/§7.5).
- Full large-N sweep is on-demand/nightly; CI gates a frozen small-N subset and the pure-fit unit.
