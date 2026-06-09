# Spec: Preference-Corpus Reliability (DPO-family: RAPPO + TI-DPO + InSPO)

> **Status: PROPOSED.** Concept: new **AHE-3.x**. Sources: RAPPO (OpenReview LrHfYPFTtg,
> *Keep the Best, Forget the Rest*), TI-DPO (arXiv:2505.19653), InSPO (arXiv:2512.23126),
> DPO (arXiv:2305.18290). Rank 3 in `COMPARATIVE_ANALYSIS.md` — cheap and broadly enabling.

## Pre-Flight Checklist (Mandatory — DSTDD)

- [ ] **KG search completed** — `.specify/design/preference-corpus-reliability/design.md` exists
- [ ] **Extension point identified** — extends `harness/eval_corpus.py` + `trace_distiller` (New Concept: AHE-3.x)
- [ ] **C4 diagram created** — PreferencePair export pipeline into the eval/training substrate
- [ ] **No new CONCEPT: tag** without pillar reference
- [ ] **`code-enhancer` audit** run against proposed changes
- [ ] **Design validation passes** — `SDDManager.validate_design("preference-corpus-reliability")`

## Problem

We already generate preference signal — `training_signals.failure_point()` gives error-attributed
(chosen=corrected / rejected=original) divergence, `trace_distiller`'s `EpisodeToPreferenceRule`
pairs successful vs. failed episodes, and `eval_corpus.py` persists regression cases — but there
is **no first-class, reliability-filtered `PreferencePair` corpus**. DPO-family results converge
on one lesson: corpus *quality* dominates. RAPPO filters ambiguous pairs; TI-DPO weights tokens by
preference contribution; InSPO conditions on an alternative response (self-reflection). All three
layer onto a single clean preference-pair export.

## Existing anchors (reuse, don't reinvent)

- `agent_utilities/harness/eval_corpus.py::EvalCorpus` — `add_case()`/`load_cases()` (graph-backed).
- `agent_utilities/graph/training_signals.py` — `failure_point()`, `difficulty_floor_filter()`.
- `agent_utilities/knowledge_graph/adaptation/trace_distiller.py` — `EpisodeToPreferenceRule`.
- `agent_utilities/knowledge_graph/adaptation/feedback.py::FeedbackService` — `_apply_eval`/`_apply_outcome`.

## User Stories

### US-1: First-class PreferencePair export
**As the** evolution loop, **I want** a `PreferencePair{prompt, chosen, rejected, metadata}`
export consolidated from eval cases + distilled episodes + human corrections, **so that** DPO-style
consumers have one clean source.
**Acceptance Criteria:**
- [ ] A `PreferencePair` model + exporter pulls from `EvalCorpus`, `trace_distiller`, and `FeedbackService`.
- [ ] `*_live_path` test: recording a human correction + a failed/succeeded episode yields a retrievable pair.

### US-2: RAPPO reliability filter
**As the** corpus builder, **I want** ambiguous/low-margin pairs filtered out (order-aware,
"keep the best, forget the rest"), **so that** noisy pairs don't hurt generalization.
**Acceptance Criteria:**
- [ ] A `reliability_filter` drops pairs below a configurable preference-margin/agreement threshold.
- [ ] Dropped counts are logged (no silent truncation per Wire-First).

### US-3 (optional, layered): TI-DPO token weights + InSPO reflective conditioning
**Acceptance Criteria:**
- [ ] Token-importance weights attachable to a pair; an InSPO reflective-conditioning hook optional and off by default.

## Non-Functional Requirements
- [ ] Zero regression; tests < 60s; pre-commit clean; no stubs.
- [ ] Export is incremental/content-addressed (no full-corpus recompute).

## Post-Modification Artifact Mandate (all 7 required before merge)
1. [ ] **/docs** — `docs/pillars/3_agentic_harness_engineering/AHE-3.x-Preference-Corpus.md`
2. [ ] **AGENTS.md** — preference-corpus export capability
3. [ ] **CHANGELOG.md** — Unreleased entry citing OR:LrHfYPFTtg, arXiv:2505.19653, 2512.23126, 2305.18290
4. [ ] **README.md** — AHE pillar concept list/count update
5. [ ] **.specify/** — spec + tasks + design; dual-write to KG
6. [ ] **.specify/reports/** — C4 diagram of the PreferencePair pipeline
7. [ ] **Pytests** — unit (reliability filter, token weights) + `*_live_path` (pair export side effect)
