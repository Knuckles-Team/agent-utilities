# Tasks: Multi-Agent Scaling-Law Harness (SAFE-1.2)

Wire-first, ordered. One harness co-satisfies ORCH-1.49 — do not build twice.

1. **Pure fit primitive (no deps).** Add `fit_scaling_law` (fits `capability ~ instances^α` with α +
   confidence band + residuals) as a pure function in `agent_utilities/harness/` — no engine/LLM imports.
   Unit-test it first (AC4).
2. **Harness over the live executor.** Add `ScalingLawHarness` in `agent_utilities/harness/` that sweeps a
   caller-supplied `(N, density, archetype_mix)` grid through the **existing**
   `agent_utilities/graph/parallel_engine.py::ParallelEngine.run` over a fixed task suite (AC1) — reuse,
   do not add a new collective executor.
3. **Topology variants.** Realize each `(N, density)` config via `agent_utilities/graph/social_system.py`
   (ORCH-1.32 neighborhoods/degree) rather than ad-hoc agent lists (AC1).
4. **Record quality + compute.** Capture `collective_quality` from the fixed suite's scorer and
   `tokens`/`compute_cost` from the live observability/pricing inputs (OS-5.23 / ECO-4.40) — no new meter (AC2).
5. **Collapse gating.** Run `agent_utilities/graph/population_drift.py` (AHE-3.2) per config; flag
   `collapsed=True` and exclude from the fit (AC3).
6. **Wire the N axis into eval.** Extend `agent_utilities/harness/continuous_evaluation_engine.py` to drive
   the harness (the existing fixed-task loop gains the population axis) via a CLI/eval entry point ≤3 hops (AC1, US-3).
7. **Persist the law.** Write the fitted law + knee as KG nodes mirroring KG-2.27 calibration facts;
   expose the knee for AHE-3.4 team sizing (AC5).
8. **Tests.** `tests/integration/harness/test_scaling_law_harness.py`, `@pytest.mark.concept(id="SAFE-1.2")`,
   ≤60s, stubbed `ParallelEngine.run` + pure-fit assertions.
9. **Concept + docs.** Add `CONCEPT:SAFE-1.2` marker at the harness; regenerate `docs/concepts.yaml`
   (`scripts/build_concepts_yaml.py`); run `scripts/check_concepts.py`; author the per-concept doc.
10. **Quality bar.** `pre-commit run --all-files` fully green before merge.
