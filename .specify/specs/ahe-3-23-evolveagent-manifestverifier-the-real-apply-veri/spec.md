# Spec: Verified Apply→Verify→Rollback on the Live Evolution Tick (AHE-3.23)

> Status: **proposed**.
> **Wire-First:** EXTENDS the already-built but tests-only machinery —
> `agent_utilities/harness/verifier.py` (`ManifestVerifier.verify` → `EvidenceCorpus`
> diff → `partial_revert`/`full_revert`) and `agent_utilities/harness/evolve_agent.py`
> (`EvolveAgent`, applies + git-commits component edits) — by making them the
> **authoritative regression verdict** of the deployed failure-ingest tick
> (`knowledge_graph/core/engine_tasks.py:_tick_failure_ingest` →
> `knowledge_graph/adaptation/failure_analyzer.py:run_failure_ingest` /
> `FailureAnalyzer.make_regression_check`). Reuse both; do not rebuild either.

## Pre-Flight Checklist
- [x] Extension target identified: `failure_analyzer.make_regression_check` is the
      coarse occurrence-count spike monitor (`current > base`, no re-measurement, no
      revert) currently gating the golden loop's auto-merge; it is the seam to replace.
- [x] New CONCEPT:AHE-3.23 justified: this closes the falsifiable
      measure→modify→re-measure→keep-or-revert contract on a **live path** — today
      `ManifestVerifier`/`EvolveAgent` are instantiated only in tests (dead-on-live-path
      per AU's own Wire-First rule); `concepts.yaml` AHE-3.0 listing them as live is the
      overclaim this spec resolves.
- [x] Wire-First confirmed: 1 hop from the deployed daemon tick
      (`_tick_failure_ingest` → `run_failure_ingest` → regression check) to
      `ManifestVerifier.verify`; no new entry point invented.
- [x] Success metric defined: a remediation branch that lowers `fix_recall` /
      introduces `unexpected_regressions` vs the pre-change `EvidenceCorpus` is
      **abandoned** (not merged) on a `full_revert` verdict; the verdict, not the spike
      count, decides keep/revert.

## User Stories

### US-1 — Re-measurement, not a spike count, decides the merge
**As** the failure-driven evolution daemon, **I want** a remediation branch judged by
before/after capability re-measurement, **so that** a "fix" that regresses other
behavior is caught and reverted instead of merged.
- **AC1**: `FailureAnalyzer.make_regression_check(gaps)` returns a check that, for the
  branched change set, builds a **post-change** `EvidenceCorpus` (targeted evals +
  `build_reliability_suite`) and calls `ManifestVerifier.verify(baseline, new)` against
  the **pre-change** baseline corpus captured before the branch.
- **AC2**: the check's pass/fail is the verifier `recommendation`: `confirm` → pass;
  `partial_revert`/`full_revert` → fail. Spike occurrence count is no longer the gate.
- **AC3**: on `full_revert` the branch is **auto-abandoned** (no merge), and the verdict
  (`fix_precision`, `fix_recall`, `unexpected_regressions`, recommendation) is persisted
  on the cycle node so it is queryable, not just logged.

### US-2 — Reuse the existing apply/commit machinery, no second loop
**As** a maintainer, **I want** the existing `EvolveAgent` branch+commit path used as-is,
**so that** there is exactly one apply→verify→rollback engine, not a parallel one.
- **AC4**: the tick drives `EvolveAgent` to apply/commit the change set on its branch and
  the verifier verdict from US-1 governs keep/abandon; no edit-application logic is
  duplicated into `failure_analyzer.py`.
- **AC5**: default-ON for the failure-ingest path it already gates; when the verifier or
  reliability suite cannot run (offline/no engine), the check falls back to the prior
  spike behavior and logs the degraded mode (never hard-fails the daemon).

## Non-Functional Requirements
- Live-path test `tests/integration/harness/test_ahe_3_23_verified_rollback.py`
  (`@pytest.mark.concept(id="AHE-3.23")`), ≤60s, no live engine/LLM: drive the real
  `make_regression_check` seam and assert a regressing branch yields `full_revert` →
  abandon, a clean branch yields `confirm` → keep (name it `*_live_path`).
- `pre-commit run --all-files` green; `scripts/build_concepts_yaml.py` regenerated +
  `scripts/check_concepts.py` passes; per-concept doc authored (AHE-3.0 corrected so the
  verifier/evolve files are listed live only here).
