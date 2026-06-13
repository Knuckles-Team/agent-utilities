# Spec: Recursive Distillation of Test-Time Search (AHE-3.25)

> Status: **proposed**. Closes the AlphaZero-style memetic-RSI loop: search-improved
> outputs → fine-tune the prior → cheaper/stronger next search.
>
> **Wire-First.** This EXTENDS, does not rebuild: it consumes the **shared SearchDistillation
> corpus minted by OS-5.34** (`harness/preference_pairs.py` + the OS-5.34 harvester tapping
> `graph/test_time_diversity.py` best-of-k and `harness/verifier.py` scores) and adds a new
> gated tick alongside `knowledge_graph/research/golden_loop.py` /
> `knowledge_graph/core/engine_tasks.py::_tick_golden_loop`. The fine-tune itself is delegated
> to the existing data-science-mcp trainer (ML-001..007) — not re-implemented here
> (`knowledge_graph/memory/optimization_engine.py:1745` "out of scope" is the seam this closes).
> The new model is gated by the **CapabilityRatchet (AHE-3.24)** before promotion.

## Pre-Flight Checklist
- [x] Extension target identified: shared OS-5.34 corpus + the golden-loop tick registry
  (`engine_tasks.py`); fine-tune delegated to data-science-mcp ML-001..007; gate = AHE-3.24.
- [x] New CONCEPT:AHE-3.25 justified — the **distillation→fine-tune→promote loop** is distinct
  from OS-5.34 (corpus harvest), AHE-3.24 (gate), and AHE-3.17 (`preference_pairs.py`, mining only);
  no existing concept turns the corpus into a promoted generation prior.
- [x] Wire-First confirmed: `engine_tasks.py::_tick_recursive_distillation` (registered like
  `_tick_golden_loop`) → corpus snapshot → trainer job → AHE-3.24 ratchet → promote-or-discard.
- [x] Success metric defined: per cycle, a versioned model node carrying `{cadence, capability_delta_vector,
  compute_cost}`; promotion is **gated** (never lowers a tracked capability) and the loop is propose-only
  by default (`KG_RECURSIVE_DISTILL` opt-in; fine-tune needs external GPU compute).

## User Stories

### US-1 — Harvest the shared corpus into a trainable snapshot
**As** the self-evolution daemon, **I want** to snapshot the OS-5.34 SearchDistillation corpus
(rejection-sampled best-of-k trajectories + winner/loser preference pairs) into a frozen, versioned
training dataset, **so that** a fine-tune run is reproducible and the corpus is not re-mutated mid-run.
- **AC1**: `RecursiveDistiller.snapshot_corpus()` reads the OS-5.34 corpus (NOT a new harvester) and emits
  a frozen `DistillationCorpusSnapshot` node `{snapshot_id, n_sft_rows, n_pref_pairs, source_versions}`.
- **AC2**: When the corpus is below a minimum-size floor, the tick **no-ops** (logged) and promotes nothing.

### US-2 — Trigger an external fine-tune via the existing trainer
**As** the distillation tick, **I want** to dispatch the snapshot to the data-science-mcp trainer
(ML-001..007), **so that** AU produces a candidate generation prior without re-implementing training.
- **AC3**: `RecursiveDistiller.fine_tune(snapshot)` delegates to the ML-001..007 trainer via the existing
  MCP source path and returns a `CandidateModel` ref; it never runs training in-process.
- **AC4**: With no trainer/GPU reachable, the call degrades to a logged `pending` state (propose-only) and
  the loop does not hard-fail offline.

### US-3 — Gate, promote, and log the cadence vs capability delta
**As** the operator, **I want** the candidate gated by the CapabilityRatchet and promoted only on pass,
**so that** recursive distillation cannot silently degenerate (agenda-4d).
- **AC5**: `RecursiveDistiller.promote_if_passes(candidate)` calls the AHE-3.24 `CapabilityRatchet`; promotes
  to "current generation prior" **only** when post-change scores ≥ baseline on every tracked capability,
  else discards the candidate and records the rejection.
- **AC6**: Each cycle persists an `EvolutionCycle`-linked node `{cadence, capability_delta_vector, compute_cost}`
  so distillation cadence vs capability delta is queryable (empirical agenda-4d probe).

## Non-Functional Requirements
- `tests/unit/harness/test_ahe_3_25_recursive_distillation.py` (`@pytest.mark.concept(id="AHE-3.25")`),
  ≤60s, no live engine / GPU / LLM: stub the OS-5.34 corpus + a fake trainer + a fake AHE-3.24 ratchet,
  assert (a) snapshot freezing, (b) trainer delegation, (c) promote-on-pass / discard-on-fail, (d) the
  cadence-vs-delta node is written. Plus a `*_live_path` test asserting the registered tick invokes the loop.
- `pre-commit run --all-files` green; `scripts/build_concepts_yaml.py` regenerated + `scripts/check_concepts.py` passes.
- Per-concept doc authored (`docs/architecture/in_house_training_substrate.md` section or a new doc) describing
  the AlphaZero loop, the OS-5.34/AHE-3.24/SAFE-1.4 coupling, and the `KG_RECURSIVE_DISTILL` opt-in.
- Propose-only + opt-in default; the SAFE-1.4 corpus model-collapse guard MUST gate the snapshot the loop distills.
