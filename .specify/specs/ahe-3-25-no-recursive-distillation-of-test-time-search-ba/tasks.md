# Tasks: Recursive Distillation of Test-Time Search (AHE-3.25)

Wire-First order. Build AU-OS.scaling.kg-provenance-panel-data (corpus harvest) FIRST — this spec consumes its corpus and
must not re-harvest. Land AU-OS.safety.model-collapse-guard-self (corpus collapse guard) with it. Gate behind AHE-3.24.

1. **Confirm the AU-OS.scaling.kg-provenance-panel-data corpus seam.** Read the shared corpus API minted by AU-OS.scaling.kg-provenance-panel-data (over
   `agent_utilities/harness/preference_pairs.py` + `agent_utilities/graph/test_time_diversity.py`
   best-of-k + `agent_utilities/harness/verifier.py` scores). Do NOT add a new harvester here.

2. **Add `RecursiveDistiller`** in `agent_utilities/harness/` (new module, named from purpose):
   `snapshot_corpus()` (freeze + version → `DistillationCorpusSnapshot` node, min-size floor),
   `fine_tune(snapshot)` (delegate to data-science-mcp ML-001..007 via the existing MCP source
   path — never train in-process; degrade to `pending` offline), `promote_if_passes(candidate)`
   (call the AU-AHE.evaluation.capability-benchmark-regression-ratchet `CapabilityRatchet`, promote only on monotone pass, else discard+record).

3. **Gate the snapshot with SAFE-1.4.** Before fine-tune, run the AU-OS.safety.model-collapse-guard-self corpus collapse/provenance
   guard (Wasserstein-1 diversity + synthetic-fraction cap) on the snapshot; abort the cycle if it fails.

4. **Register the tick.** Add `_tick_recursive_distillation` in
   `agent_utilities/knowledge_graph/core/engine_tasks.py` mirroring `_tick_golden_loop` (≈ line 1533,
   registered ≈ line 708); opt-in + throttled via `KG_RECURSIVE_DISTILL` (config.setting / AgentConfig —
   NO bare `os.environ`). Strangle the `optimization_engine.py:1745` "out of scope" note by pointing it here.

5. **Persist the cadence-vs-delta ledger.** Link an `EvolutionCycle` node
   `{cadence, capability_delta_vector, compute_cost}` (agenda-4d probe), queryable via graph_query.

6. **Tests** — `tests/unit/harness/test_ahe_3_25_recursive_distillation.py`
   (`@pytest.mark.concept(id="AHE-3.25")`, stubbed corpus/trainer/ratchet) + a `*_live_path` test that the
   registered tick invokes the loop.

7. **Concept + docs + gates.** Add the `CONCEPT:AU-AHE.evaluation.failure-analysis-loop` marker in the `RecursiveDistiller` docstring
   (cite the AlphaZero / agenda-4d arXiv provenance), run `scripts/build_concepts_yaml.py` +
   `scripts/check_concepts.py`, author the per-concept doc section in
   `docs/architecture/in_house_training_substrate.md`, then `pre-commit run --all-files` green.
