# Tasks: Search-Output Distillation Harvester (OS-5.34)

Wire-first; extend existing modules before adding new ones.

1. **Read the seams.** `graph/test_time_diversity.py` (AHE-3.16 candidate set + `mean_pairwise_distance`),
   `harness/verifier.py` (scores), RLM `RunTrace` (ORCH-1.29), `research/preference_pairs.py` (AHE-3.17
   writer shape), `knowledge_graph/distillation/` (KG-2.2 deduplicator), `knowledge_graph/core/engine_tasks.py`
   (`_tick_golden_loop` pattern), `orchestration/action_policy.py` (OS-5.24 gate).
2. **Harvester.** Add `harness/search_distillation.py::SearchDistillationHarvester.harvest(run)` →
   rejection-sample best-of-k → `(prompt, best_trajectory)` SFT rows + `(winner, loser)` preference pairs;
   reuse the `preference_pairs.py` degeneracy guard.
3. **Dedup + persist.** Route rows through the KG-2.2 deduplicator; persist a versioned `SyntheticCorpus`
   node set with provenance (`source_run_id`, `scorer`, `synthetic=true`).
4. **Wire the tick.** Add a `distil-to-data` stage to the golden-loop daemon tick in `engine_tasks.py`,
   gated by the OS-5.24 ActionPolicy exactly like the existing stages; default off.
5. **Trainer enumeration.** Confirm the data-science-mcp trainer (ML-001..007) can enumerate the
   `SyntheticCorpus` node set unchanged; if a reader is missing, add the read-side only.
6. **Test** `tests/unit/knowledge_graph/distillation/test_os_5_34_search_distillation.py` per the spec ACs.
7. **Gates.** `pre-commit run --all-files`; regenerate `docs/concepts.yaml`; `scripts/check_concepts.py`;
   extend `docs/architecture/in_house_training_substrate.md`.
