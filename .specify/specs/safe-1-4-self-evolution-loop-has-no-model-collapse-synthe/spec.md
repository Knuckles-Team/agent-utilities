# Spec: Corpus Collapse / Synthetic-Degeneration Guard (SAFE-1.4)

> Status: **proposed**. Closes part of the AGI→ASI "data wall" (paper §5.5 / §7.4(c),(d)):
> self-generated data is the main data-wall countermeasure but risks model-collapse
> (Shumailov 2024) — recursive distillation can degenerate.
>
> **Wire-First:** extends the existing W1 collapse detector
> `agent_utilities/graph/population_drift.py` (`PopulationDriftMonitor`, AHE-3.2) and
> the merge gate `agent_utilities/knowledge_graph/research/auto_merge.py`
> (`GovernedAutoMerger.evaluate`, line 244). **Reuse, do not rebuild** — `evaluate()`
> already checks quality/governance/regression; add one corpus-diversity check beside them.

## Pre-Flight Checklist
- [x] **Extension target identified.** Detector = `graph/population_drift.py` (`wasserstein1`,
  `PopulationDriftMonitor`); gate hop = `research/auto_merge.py:244` `GovernedAutoMerger.evaluate`.
  Self-generated corpora producers verified: `harness/preference_pairs.py`
  (`reliability_filter` only drops `chosen==rejected`, line 204) and AHE-3.18 `failure_gap` Concepts.
- [x] **New CONCEPT:SAFE-1.4 justified.** AHE-3.2 (`population_drift`) watches the *agent population's
  score* distribution; KG-2.36 notes "collapsed embeddings → degenerate clusters" but only inside
  memory optimization. Nothing watches the self-generated **corpus** (preference pairs / failure_gap
  Concepts / eval additions) for diversity loss or synthetic over-saturation before it re-enters the
  loop. That provenance-aware corpus gate is the new, standalone axis.
- [x] **Wire-First confirmed.** 1 hop: `GovernedAutoMerger.evaluate` gains a `corpus_diversity`
  failure alongside the existing three; the monitor it calls reuses `population_drift.wasserstein1`.
- [x] **Success metric defined.** A synthetic-saturated / diversity-collapsed corpus snapshot makes
  `evaluate().failures` non-empty (proposal stays proposal-only), while a healthy mixed corpus passes.

## User Stories

### US-1 — Provenance-aware diversity reading on a self-generated corpus
**As** the evolution loop, **I want** a per-cycle reading over the corpus that feeds distillation,
**so that** narrowing diversity or synthetic over-saturation is detected before re-ingestion.
- **AC1**: A `CorpusDiversityMonitor` (new, in `graph/population_drift.py` beside `PopulationDriftMonitor`)
  takes per-item embeddings + a `provenance ∈ {human, synthetic}` label and returns a reading with
  `diversity` (mean pairwise embedding distance / spread), `synthetic_fraction`, `drift`
  (W1 vs the previous cycle's diversity distribution, via the existing `wasserstein1`), and `collapsed`.
- **AC2**: `collapsed` is `True` when `diversity ≤ diversity_floor` for `patience` consecutive cycles
  **or** `synthetic_fraction ≥ synthetic_cap` — mirroring the `PopulationDriftMonitor` streak/threshold
  contract (defaults: `diversity_floor=0.05`, `synthetic_cap=0.95`, `patience=2`).
- **AC3**: Pure-Python / embedding-in (no model, no network); empty or single-item corpora return a
  non-collapsed, `drift=None` reading (matches the first-generation case of the existing monitor).

### US-2 — Gate the merge / corpus ingestion on the reading
**As** `GovernedAutoMerger`, **I want** a corpus-collapse failure beside quality/governance/regression,
**so that** a degenerate self-generated corpus blocks promotion instead of silently re-entering.
- **AC4**: `GovernedAutoMerger.evaluate` accepts an optional `corpus_reading` (the US-1 reading); when
  it is `collapsed`, `evaluate()` appends a `"corpus diversity collapsed: …"` / `"synthetic fraction …"`
  failure to `MergeEvaluation.failures` (so `eligible` is false), gated by a `MergePolicy` flag that
  **defaults to the prior behavior** (no reading supplied → unchanged, per the AHE-3.x opt-in rule).
- **AC5**: `harness/preference_pairs.reliability_filter` exposes provenance/diversity inputs the monitor
  can consume from exported pairs (the corpus already distinguishes human-correction vs synthetic
  preference nodes) — no new connector, no bespoke corpus store.

## Non-Functional Requirements
- `tests/unit/graph/test_safe_1_4_corpus_collapse_guard.py` tagged `@pytest.mark.concept(id="SAFE-1.4")`,
  ≤60s, no live engine/LLM (pure monitor + a stub spec through `evaluate`); asserts both the live-path
  gating (a collapsed reading yields non-empty `evaluate().failures`) and the isolated monitor.
- `pre-commit run --all-files` green; `docs/concepts.yaml` regenerated via `scripts/build_concepts_yaml.py`
  and `scripts/check_concepts.py` passes; per-concept doc authored (docstring `CONCEPT:SAFE-1.4` +
  one-paragraph note citing paper §5.5 / §7.4(c),(d) and Shumailov 2024 in `docs/`).
- Opt-in / default-unchanged: with no `corpus_reading`, `GovernedAutoMerger.evaluate` behaves exactly as
  today (Wire-First + No-Legacy: extend the live gate, no parallel path).
