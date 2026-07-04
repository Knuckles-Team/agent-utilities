# Tasks: Compute-Optimal Test-Time Compute Governor (AU-OS.scaling.bridge-developer-workspace-mutating)

Wire-first: extend the live budget + diverse-fan-out hops and reuse existing telemetry,
pricing, and ledger. Add the new controller module last, only once it has a live caller.

1. **Confirm the open-loop seams (read, don't rebuild).** Re-read
   `agent_utilities/harness/reasoning_effort.py` (`get_budget` → `ReasoningBudget.diversity_width`,
   `max_search_calls`) and `agent_utilities/graph/test_time_diversity.py`
   (`diverse_fan_out_width`, `mean_pairwise_distance`, `select_diverse`). These give the *ceiling*
   and the *diversity metric* the governor reuses — no new sampler.

2. **Add the `min_return_per_usd` threshold as a typed `AgentConfig` field** in
   `agent_utilities/core/config.py` (default + docstring; read via `config`). No new env flag;
   satisfies `check_no_env_sprawl.py` and AC5.

3. **Build `ComputeGovernor`** in a new `agent_utilities/harness/compute_governor.py`
   (CONCEPT:AU-OS.scaling.bridge-developer-workspace-mutating). Methods: `should_continue(samples, *, effort)` (AC1/AC2), `returns_curve()`
   (AC3). It (a) caps at `get_budget(effort)` ceilings, (b) estimates `Δquality` from candidate
   scores + incremental `mean_pairwise_distance`, (c) prices the next sample via
   `agent_utilities/pricing/catalog.PricingCatalog.cost_for` (ECO-4.40). Cite AHE-3.1/3.16/
   ORCH-1.29/ORCH-1.2/ECO-4.40/AU-OS.observability.persist-this-graph-run + paper §5.1 in the docstring.

4. **Consume `RunTrace.usage`.** Sum `agent_utilities/rlm/telemetry.py` `LMUsage.total`
   (`prompt+completion+sub_lm`) across the consumed samples into the governor's spend accounting
   (ORCH-1.29) — read existing telemetry, add no new field.

5. **Wire the ledger sink (AU-OS.observability.persist-this-graph-run).** On stop, call
   `agent_utilities/usage/recorder.py` `UsageRecorder.record_run(token_usage=…, status=…)` with
   summed usage and `converged`/`exhausted` status (AC4).

6. **Gate the live fan-out (Wire-First).** Make the existing consumer of `diverse_fan_out_width`
   call `ComputeGovernor.should_continue(...)` between samples so the default path is governed ON
   (early-stop within the AU-AHE.harness.width-diverse-best-k ceiling). This is the live invocation, not just an API.

7. **Tests.** Add `tests/unit/harness/test_os_5_33_compute_governor.py`
   (`@pytest.mark.concept(id="AU-OS.scaling.bridge-developer-workspace-mutating")`) covering AC1–AC4 + a `*_live_path` case driving the
   fan-out consumer; assert `record_run` fires once with the summed `LMUsage`.

8. **Regen + docs + gates.** Run `scripts/build_concepts_yaml.py` and `scripts/check_concepts.py`;
   author the per-concept doc under `docs/pillars/5_agent_os_infrastructure/` and cross-link from
   `docs/pillars/3_agentic_harness_engineering.md`; drive `pre-commit run --all-files` fully green.
