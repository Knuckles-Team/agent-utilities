# Spec: Capability-Benchmark Regression Ratchet (AU-AHE.evaluation.capability-benchmark-regression-ratchet)

> Status: **proposed**.
> **Wire-First:** insert a `CapabilityRatchet` between the publisher worktree build and merge in
> `knowledge_graph/research/change_publisher.py` (today it runs only proposal-named pytest targets at
> `change_publisher.py:310`) that runs the **existing** `harness/reliability_scorers.py`
> `build_reliability_suite` (AHE-3.1) + LongMemEval-S (`harness/benchmark.py`, AHE-3.12) in that
> worktree and requires post-change scores ≥ a stored baseline before publish. Reuses
> `promotion_governance.py` (AU-AHE.harness.promotion-governance-validator) as the gate host — adds a capability axis to its structural
> `quality_score`, not a new merge path.

## Pre-Flight Checklist
- [x] **Extension target identified** — `change_publisher.py:310` runs only proposal-named targets;
  the live regression gate is `failure_analyzer.make_regression_check` (a coarse occurrence-count
  spike: `current > base`). `reliability_scorers` (AHE-3.1) and LongMemEval-S (AHE-3.12) exist but are
  **not read** by the merge/publish gate — a merged change can pass its own targeted tests and the
  spike monitor while regressing untested capabilities (the paper's "degeneration" mode, §5.3).
- [x] **New CONCEPT:AU-AHE.evaluation.capability-benchmark-regression-ratchet justified** — distinct from AU-AHE.harness.promotion-governance-validator structural governance (SHACL +
  constitution + `quality_score`) and AU-AHE.harness.failure-evolution spike monitoring: this is a *monotone capability*
  ratchet on measured benchmark scores, the verifier-quality safety net agenda 4d demands.
- [x] **Wire-First confirmed** — 1 gate call inserted in the publisher path, reusing the AHE-3.1 suite,
  AHE-3.12 benchmark, and the AU-AHE.harness.promotion-governance-validator governance host; baseline persisted as a graph node.
- [x] **Success metric defined** — a candidate whose post-change per-capability score vector drops
  below the stored baseline (beyond tolerance) on **any** tracked capability is blocked from publish;
  a candidate at-or-above baseline on all tracked capabilities proceeds and updates the baseline.

## User Stories

### US-1 — A change cannot lower overall capability
**As** the deployed evolution loop, **I want** every promoted change scored against a capability
baseline before merge, **so that** a fix cannot silently regress untested capabilities.
- **AC1**: `CapabilityRatchet.check(worktree, baseline)` runs `build_reliability_suite` (+ LongMemEval-S)
  in the publisher worktree and returns a per-capability score vector.
- **AC2**: publish proceeds **iff** every tracked capability scores ≥ `baseline - tolerance`; otherwise
  the branch is blocked and the failing capabilities are recorded on the audit node.
- **AC3**: on success the baseline `CapabilityScoreVector` node is advanced to the new scores (monotone
  ratchet), persisted graph-native with the change-set + concept provenance.

### US-2 — Reuses governance, off-path safe
**As** an operator, **I want** the ratchet to ride the existing governance gate, **so that** no new
publish path or bypass appears.
- **AC4**: the ratchet verdict is consumed by `promotion_governance.py` as an additional required
  predicate alongside `quality_score`/SHACL/constitution — a single governed decision, not a second gate.
- **AC5**: with no baseline node yet, the first run **establishes** the baseline and records it without
  blocking (bootstrap), and the score vector is stored so AU can begin fitting local
  recursive-improvement curves (agenda 4a, feeds AU-AHE.sdd.recursive-improvement-instrumentation-aggregating/AU-OS.audit.recursive-improvement-velocity-tracker).

## Non-Functional Requirements
- `tests/unit/knowledge_graph/research/test_ahe_3_24_capability_ratchet.py`
  (`@pytest.mark.concept(id="AU-AHE.evaluation.capability-benchmark-regression-ratchet")`), ≤60s, no live engine/LLM: stub a score vector; assert a
  below-baseline capability blocks publish, an at/above-baseline set advances the baseline, and the
  bootstrap (no baseline) path establishes without blocking.
- `pre-commit run --all-files` green; `scripts/build_concepts_yaml.py` re-run so AU-AHE.evaluation.capability-benchmark-regression-ratchet lands in
  `docs/concepts.yaml`; `scripts/check_concepts.py` passes.
- Per-concept doc under `docs/guides/` (extend `autonomous-evolution.md`), naming the ratchet as the
  required safety net for AHE-3.22 generated code and AHE-3.25 distilled models.
