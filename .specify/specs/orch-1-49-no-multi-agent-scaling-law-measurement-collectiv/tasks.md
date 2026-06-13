# Tasks: Multi-Agent Scaling-Law Eval Harness (ORCH-1.49)

Wire-first, ordered. Co-satisfies SAFE-1.2 — one harness, one set of tasks.

1. **Pure fitter first (red test).** Add `fit_scaling_law(points)` to a new
   `agent_utilities/harness/scaling_law_harness.py` (`CONCEPT:ORCH-1.49`): given
   `(N, quality, compute)` points return `{alpha, alpha_ci, regime, knee_n}`, pure/LLM-free.
   Write the failing unit test (`tests/unit/harness/test_orch_1_49_scaling_law_harness.py`,
   `@pytest.mark.concept(id="ORCH-1.49")`) with synthetic super/sub/linear point sets.

2. **Sweep over the live collective runner.** In `scaling_law_harness.py` add
   `ScalingLawHarness` that takes a fixed task suite + `(N, density, archetype_mix)` grid and
   runs each config through `agent_utilities/graph/parallel_engine.py` (`ParallelEngine`) — reuse
   its existing quality/judge result path; record token/compute per config via the ECO-4.40
   pricing / OS-5.23 metrics already on that path. **No new collective runner.**

3. **Topology + collapse gating.** Source density/neighborhood variants from
   `agent_utilities/graph/social_system.py`; run each config's result through
   `agent_utilities/graph/population_drift.py` and exclude `collapsed` configs from the fit.

4. **Persist the law as a KG node.** Write the fitted result as a KG node mirroring the KG-2.27
   calibration-node shape (exponent + confidence band + provenance), via the KG facade —
   not a bespoke store.

5. **Wire into the eval engine + team evolution.** Mount the sweep as an entry point on
   `agent_utilities/harness/continuous_evaluation_engine.py` (give it the N axis it lacks), and
   surface the fitted knee/regime to `agent_utilities/graph/team_evolution.py` (AHE-3.4) as a
   read on a live path (default on). Add a `*_live_path` test asserting the side effect.

6. **Gates + docs.** Run `scripts/build_concepts_yaml.py` then `scripts/check_concepts.py`;
   author the per-concept doc for ORCH-1.49 (+ shared SAFE-1.2 note); `pre-commit run
   --all-files` green.
