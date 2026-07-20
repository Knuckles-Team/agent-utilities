# Tasks: Non-Saturating Superhuman Progress Tracker (SAFE-1.1)

Wire-first; extend the AHE-3.1 scorer registry and benchmark harness.

1. **Read the seams.** `harness/reliability_scorers.py` (AHE-3.1 registry), `harness/benchmark.py`
   (AHE-3.12 fixed-target), `harness/variant_pool.py` (AHE-3.2 tournament pairings), the
   `adversarial_verifier` "Hacker Agent", `knowledge_graph/adaptation/failure_analyzer.py` (AU-AHE.harness.failure-evolution gate).
2. **Setter-solver scorer.** Add `harness/frontier_scorers.py::SetterSolverScorer` reusing the Hacker
   Agent as setter; metric = setter↔solver pass-gap (relative, no ceiling).
3. **Elo + compression scorers.** Add a zero-sum Elo scorer over VariantPool duels and a
   compression/description-length `EvalScorer`; register all three in the AHE-3.1 registry.
4. **Frontier mode.** Add a "frontier" mode to `benchmark.py` that runs the relative scorers.
5. **Saturation detector.** Add a detector flagging evals whose pass-rate collapsed to ceiling across the
   last N agent versions; recommend frontier promotion.
6. **Feed the gate.** Route frontier scores into the AU-AHE.harness.failure-evolution regression gate.
7. **Test** `tests/unit/harness/test_safe_1_1_frontier_eval.py` per the spec ACs.
8. **Gates.** `pre-commit run --all-files`; regenerate `docs/concepts.yaml`; `scripts/check_concepts.py`;
   add `docs/architecture/safety_measurement.md`.
