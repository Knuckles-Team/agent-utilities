# Spec: Search-Output Distillation Harvester (OS-5.34)

> Status: **proposed**.
> **Wire-First:** add a harvester that taps the artifacts three existing live paths already
> produce — `graph/test_time_diversity.py` (AHE-3.16 diverse best-of-k candidates),
> `harness/verifier.py` scores, and RLM `RunTrace`s (ORCH-1.29) — and mints a versioned
> synthetic corpus through the **existing** `knowledge_graph/distillation/` dedup pipeline
> (KG-2.2), persisted as graph-native nodes the data-science-mcp trainer (ML-001..007) consumes.
> Run it as one new gated `distil-to-data` stage on the golden-loop daemon tick
> (`knowledge_graph/core/engine_tasks.py`), reusing `research/preference_pairs.py` (AHE-3.17)
> writer shapes. No new training code; no new persistence layer.

## Pre-Flight Checklist
- [x] **Extension target identified** — AHE-3.16 (`test_time_diversity.py`) produces diverse
  best-of-k candidate sets at inference and **discards** winners/losers; RLM `RunTrace`s
  (ORCH-1.29) are kept for GEPA reflection only; `harness/preference_pairs.py` reads only
  eval_case/preference/correction nodes — never best-of-k winners. The artifacts the paper calls
  the synthetic-data goldmine exist and are thrown on the floor.
- [x] **New CONCEPT:OS-5.34 justified** — this is the missing *harvest+curate* edge between
  test-time compute (AHE-3.16) and the trainer (ML-001..007); it is not the trainer (ML-001..007)
  nor the propose-only golden loop (KG-2.7) — it is the data factory between them.
- [x] **Wire-First confirmed** — 1 harvester consuming AHE-3.16 candidate sets + verifier scores +
  RunTraces, writing through the KG-2.2 deduplicator and the AHE-3.17 preference-pair node shape;
  exposed as a `distil-to-data` tick gated by the OS-5.24 ActionPolicy like the other golden-loop stages.
- [x] **Success metric defined** — a run that produced ≥2 scored candidates yields ≥1 deduped SFT row
  `(prompt → best trajectory)` and, when a losing candidate exists, ≥1 `(winner, loser)` preference
  pair, persisted into a versioned `SyntheticCorpus` node set the trainer can enumerate.

## User Stories

### US-1 — Best-of-k winners become trainable rows
**As** the platform, **I want** verified best-of-k outputs harvested into a curated corpus,
**so that** test-time compute is converted into training data instead of discarded.
- **AC1**: a `SearchDistillationHarvester.harvest(run)` reads an AHE-3.16 candidate set + its
  `harness/verifier.py` scores, applies rejection sampling / best-of-k, and emits
  `(prompt, best_trajectory)` SFT rows + `(winner, loser)` preference pairs.
- **AC2**: rows pass through the **existing** KG-2.2 LSH/deduplicator before persistence; near-duplicate
  and trivially-degenerate (`chosen==rejected`) rows are dropped (reuse `preference_pairs.py` guard).
- **AC3**: output is a **versioned** `SyntheticCorpus` node set (provenance: source run id, scorer,
  human-vs-synthetic flag) enumerable by the data-science-mcp trainer (ML-001..007) — no new format.

### US-2 — Gated, off by default, provenance-tagged
**As** an operator, **I want** harvesting gated and provenance-tagged, **so that** synthetic data
cannot silently dominate the training mix.
- **AC4**: harvesting runs only as the `distil-to-data` golden-loop stage behind the OS-5.24
  ActionPolicy gate; with the stage disabled, behavior is unchanged (pure addition).
- **AC5**: every minted row carries `synthetic=true` + source provenance so the SAFE-1.4 corpus
  collapse/diversity guard can measure synthetic-fraction and gate ingestion.

## Non-Functional Requirements
- `tests/unit/knowledge_graph/distillation/test_os_5_34_search_distillation.py`
  (`@pytest.mark.concept(id="OS-5.34")`), ≤60s, no live engine/LLM: stub a candidate set + verifier
  scores; assert best-of-k row + preference pair minted, duplicates dropped, provenance set, and that
  an empty/zero-score run yields no rows.
- `pre-commit run --all-files` green; `scripts/build_concepts_yaml.py` re-run so OS-5.34 lands in
  `docs/concepts.yaml`; `scripts/check_concepts.py` passes.
- Per-concept doc under `docs/architecture/` (extend `in_house_training_substrate.md`), naming the
  SAFE-1.4 collapse guard and the AHE-3.24 capability ratchet as the safety envelope for any model
  later distilled from this corpus (AHE-3.25 closes that loop).
