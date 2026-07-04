# Tasks: Capability-Benchmark Regression Ratchet (AU-AHE.evaluation.capability-benchmark-regression-ratchet)

Wire-first; extend existing modules before adding new ones.

1. **Read the seams.** `research/change_publisher.py:310` (targeted-test gate),
   `knowledge_graph/adaptation/failure_analyzer.py::make_regression_check` (spike monitor it replaces
   as authoritative), `research/promotion_governance.py` (AU-AHE.harness.promotion-governance-validator gate host + `quality_score`),
   `harness/reliability_scorers.py::build_reliability_suite` (AHE-3.1), `harness/benchmark.py` (AHE-3.12).
2. **Ratchet.** Add `harness/capability_ratchet.py::CapabilityRatchet.check(worktree, baseline)` that
   runs the reliability suite (+ LongMemEval-S) in the publisher worktree → per-capability score vector.
3. **Baseline node.** Persist/read a `CapabilityScoreVector` baseline graph node; advance it monotonically
   on pass; bootstrap (establish-without-block) when absent.
4. **Wire governance.** Make `promotion_governance.py` require the ratchet predicate alongside
   `quality_score`/SHACL/constitution; block publish in `change_publisher.py` on a below-baseline verdict;
   record failing capabilities on the AuditLogger node.
5. **Test** `tests/unit/knowledge_graph/research/test_ahe_3_24_capability_ratchet.py` per the spec ACs.
6. **Gates.** `pre-commit run --all-files`; regenerate `docs/concepts.yaml`; `scripts/check_concepts.py`;
   extend `docs/guides/autonomous-evolution.md`.
