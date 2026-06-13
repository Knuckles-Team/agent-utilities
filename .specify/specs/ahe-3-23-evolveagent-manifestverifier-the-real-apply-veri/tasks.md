# Tasks: Verified Apply→Verify→Rollback on the Live Evolution Tick (AHE-3.23)

Wire-first, smallest-diff order. Reuse `ManifestVerifier`/`EvolveAgent`; do not rebuild.

1. **Capture the pre-change baseline.** In
   `agent_utilities/knowledge_graph/adaptation/failure_analyzer.py`, before the
   remediation branch is applied, build the baseline `EvidenceCorpus`
   (targeted evals for the `gaps` + `harness/reliability_scorers.build_reliability_suite`).

2. **Replace the spike gate with the verifier verdict.** Rewrite
   `FailureAnalyzer.make_regression_check(gaps)` (`failure_analyzer.py:459`) to: build the
   post-change `EvidenceCorpus` on the branch, call
   `agent_utilities/harness/verifier.py:ManifestVerifier.verify(baseline, new)`, and
   return pass/fail from `recommendation` (`confirm`=pass; `partial_revert`/`full_revert`
   =fail). Delete the `current > base` occurrence-count path (No-Legacy).

3. **Drive apply/commit through EvolveAgent.** Use
   `agent_utilities/harness/evolve_agent.py:EvolveAgent` for branch+`_git_commit_edit`;
   on `full_revert` auto-abandon the branch (no merge). No edit logic duplicated.

4. **Persist + surface the verdict.** Write `fix_precision`/`fix_recall`/
   `unexpected_regressions`/`recommendation` onto the cycle node emitted by
   `knowledge_graph/research/golden_loop.py` so it is queryable, not log-only.

5. **Confirm the live tick reaches it.** Verify the path
   `engine_tasks.py:_tick_failure_ingest` (1559) → `run_failure_ingest` →
   `make_regression_check` now invokes `ManifestVerifier.verify`; add the degraded-mode
   fallback so the daemon never hard-fails offline.

6. **Tests + gates + docs.** Add
   `tests/integration/harness/test_ahe_3_23_verified_rollback.py`
   (`@pytest.mark.concept(id="AHE-3.23")`, `*_live_path`). Run
   `scripts/build_concepts_yaml.py` + `scripts/check_concepts.py`; correct the AHE-3.0
   entry in `docs/concepts.yaml` to stop overclaiming the verifier as live elsewhere;
   author the per-concept doc; `pre-commit run --all-files` green.
