# Tasks: Corpus Collapse / Synthetic-Degeneration Guard (SAFE-1.4)

Wire-first: extend existing modules before adding anything new. Cited files are real and verified.

1. **Extend the existing detector module** `agent_utilities/graph/population_drift.py`.
   Add `CorpusDiversityMonitor` (+ a `CorpusReading` dataclass) **beside** `PopulationDriftMonitor`,
   reusing the existing `wasserstein1` helper. Inputs: per-item embeddings + `provenance` labels;
   outputs `diversity`, `synthetic_fraction`, `drift`, `collapsed`, `low_streak`. Mirror the
   `collapse_threshold`/`patience`/`reset()` contract. Tag the new class docstring `CONCEPT:SAFE-1.4`,
   cite paper §5.5 / §7.4(c),(d) + Shumailov 2024 (provenance in docstring, not the name). (AC1–AC3)

2. **Wire the gate** in `agent_utilities/knowledge_graph/research/auto_merge.py`.
   Add an optional `corpus_reading` arg to `GovernedAutoMerger.evaluate` (line 244) and an opt-in
   `MergePolicy` flag (default = prior behavior); when the reading is `collapsed`, append a
   `corpus diversity`/`synthetic fraction` entry to `MergeEvaluation.failures` alongside the existing
   quality/governance/regression checks. No new path — extend `evaluate()`. (AC4)

3. **Expose corpus provenance/diversity inputs** in `agent_utilities/harness/preference_pairs.py`.
   Extend `reliability_filter` (line 199) — which today only drops `chosen==rejected` — to also surface
   per-pair provenance (human-correction vs synthetic preference node, already distinguished by
   `_from_correction` / `_from_preference_node`) and embeddings the monitor consumes. No bespoke store. (AC5)

4. **Add the test** `tests/unit/graph/test_safe_1_4_corpus_collapse_guard.py`,
   `@pytest.mark.concept(id="SAFE-1.4")`: (a) isolated monitor — diversity-floor streak and
   synthetic-cap both flip `collapsed`; (b) live path — a collapsed reading through
   `GovernedAutoMerger.evaluate` yields non-empty `.failures`, and no reading leaves behavior unchanged.
   ≤60s, no engine/LLM.

5. **Register + document.** Run `scripts/build_concepts_yaml.py` then `scripts/check_concepts.py`;
   author the per-concept doc (note citing §5.5 / §7.4(c),(d), Shumailov 2024, and the AHE-3.2 / KG-2.36
   reuse). Drive `pre-commit run --all-files` fully green before merge.
