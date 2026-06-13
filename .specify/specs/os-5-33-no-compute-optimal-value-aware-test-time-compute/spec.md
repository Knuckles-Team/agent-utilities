# Spec: Compute-Optimal / Value-Aware Test-Time Compute Governor (OS-5.33)

> Status: **proposed**. Closes part of the AGI→ASI distance (paper "From AGI to ASI" §5.1:
> brute-force search with more compute "fails in virtually all non-toy domains" — gains come
> from search *efficiency* and harnessing "additional returns on test-time compute cost").
>
> **Wire-First:** extends the *live* budget hop `harness/reasoning_effort.py` (`get_budget`,
> `ReasoningBudget`) and the diverse fan-out hop `graph/test_time_diversity.py`
> (`diverse_fan_out_width`, `mean_pairwise_distance`). It **reads existing telemetry**
> (`rlm/telemetry.py` `RunTrace.usage` / `LMUsage`), **existing pricing** (`pricing/catalog.py`
> `PricingCatalog.cost_for`, ECO-4.40), and **writes the existing ledger**
> (`usage/recorder.py` `UsageRecorder.record_run`, OS-5.31). It does NOT add a new sampler,
> a new pricing path, or a new agent-facing knob.

## Pre-Flight Checklist
- [x] Extension target identified: `harness/reasoning_effort.py` (open-loop `get_budget`) +
      `graph/test_time_diversity.py` (`diverse_fan_out_width`, fixed function of effort).
      Verified: effort→`diversity_width=round(1+4·effort)` and `max_search_calls`/subtasks are
      chosen up front and never re-read against observed return.
- [x] New CONCEPT:OS-5.33 justified: it is a *controller* (returns-on-compute estimator +
      adaptive stop), a capability that AHE-3.1 (open-loop budget), AHE-3.16 (open-loop width),
      and ORCH-1.2 (governs *model choice*, not compute *quantity*) each structurally lack.
- [x] Wire-First confirmed: 1–2 hops — `get_budget`/`diverse_fan_out_width` consumers call the
      governor between samples; it consumes `RunTrace.usage` + `mean_pairwise_distance` it
      already computes, prices via `PricingCatalog.cost_for`, and sinks spend via `record_run`.
- [x] Success metric defined: on a fan-out where samples converge early, the governor stops
      before `diversity_width`/`max_search_calls` is exhausted and records the saved spend —
      i.e. fewer samples for equal or better best-of-k quality (Δquality/ΔUSD-driven, not flat).

## User Stories

### US-1 — Adaptive best-of-k that stops on flat marginal return
**As** the test-time fan-out path, **I want** to stop sampling once the next sample's expected
quality gain per dollar drops below a threshold, **so that** compute lands where its marginal
return is highest instead of running a fixed `diversity_width` every time.
- **AC1**: `ComputeGovernor.should_continue(samples, *, effort)` returns `(continue: bool,
  reason: str)`; with ≥2 candidate scores+embeddings it estimates marginal gain as
  `Δbest_quality` blended with the incremental `mean_pairwise_distance` (AHE-3.16), prices the
  *next* sample via `PricingCatalog.cost_for`, and returns `False` when
  `Δquality / ΔUSD < min_return_per_usd`.
- **AC2**: the governor **never exceeds** the open-loop ceiling — it caps at
  `get_budget(effort).diversity_width` / `max_search_calls` and only ever stops *earlier*; with
  fewer than 2 samples or a degenerate (all-equal) score set it returns `True` (no premature
  stop), so behavior is a pure tightening of the existing path.
- **AC3**: a returns-on-compute curve is exposed: `ComputeGovernor.returns_curve()` yields the
  per-sample `(cumulative_usd, best_quality, marginal_gain_per_usd)` points actually observed,
  so the allocation is *measured*, not heuristic.

### US-2 — Spend ledger closes the loop (OS-5.31)
**As** the usage plane, **I want** each governed fan-out's realized vs. ceiling spend recorded,
**so that** test-time scaling is auditable and the threshold is tunable from real data.
- **AC4**: on stop, the governor calls `UsageRecorder.record_run(...)` with the aggregated
  `token_usage` summed from the consumed `RunTrace.usage` (`LMUsage.total`) and a `status`
  distinguishing `converged` (stopped early) from `exhausted` (hit the ceiling).
- **AC5**: the `min_return_per_usd` threshold is a typed `AgentConfig` field with a correct
  universal default (read via `config`, never bare `os.environ` — Configuration discipline); no
  per-feature env flag is added.

## Non-Functional Requirements
- `tests/unit/harness/test_os_5_33_compute_governor.py`, tagged
  `@pytest.mark.concept(id="OS-5.33")`, ≤60s, no live engine/LLM: assert (a) early-convergence
  stops before the ceiling, (b) a still-improving set keeps going to the ceiling, (c)
  `returns_curve()` is monotonic in `cumulative_usd`, (d) a `record_run` spy fires once with the
  summed `LMUsage`. Include a `*_live_path` case driving the existing
  `diverse_fan_out_width` consumer and asserting the governor gates it.
- `pre-commit run --all-files` green (incl. `check_no_env_sprawl.py`, `check_no_stub.py`,
  `check_concepts.py`).
- Concept registry regenerated: `scripts/build_concepts_yaml.py` → `docs/concepts.yaml` carries
  OS-5.33; `scripts/check_concepts.py` passes.
- Per-concept doc authored under `docs/pillars/5_agent_os_infrastructure/` (returns-on-compute
  governor) and cross-linked from `docs/pillars/3_agentic_harness_engineering.md` (AHE-3.1/3.16).
- Related concepts cited in the module docstring: AHE-3.1, AHE-3.16, ORCH-1.29, ORCH-1.2,
  ECO-4.40, OS-5.31 (provenance "From AGI to ASI" §5.1 in docstring, never the identifier).
