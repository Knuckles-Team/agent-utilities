# Comparative Analysis: memory-os → agent-utilities (Innovation Extraction)

**Source:** `ClaudioDrews/memory-os` @ `a4ca094a` (pinned 2026-06-05) — "Hermes Memory OS", a 7-layer
agent memory system.
**Target:** `agent-utilities` (primary, for enhancement).
**Mode:** Lightweight (no live KG). Pipeline: `pin → explore→ledger → verify → score → scaffold-SDD → wiring-audit`.
**Artifacts:** `reports/memory-os-ca/{ledger,verified,scored,wiring}.json`; SDD stubs under `.specify/design|specs/`.

## Method & integrity

11 candidate innovations were extracted by parallel exploration agents (returning structured
Innovation-Ledger rows), then **verified against the actual memory-os source** — all 11 are
`verified` (every claim's evidence tokens were found in the cited files; 0 `claimed-only`, 0
`refuted`). Recommendations were scored by leverage/(effort+risk) and ordered by dependency. A
runnable wiring audit (import-graph ≤3-hop reachability) was run against the proposed target
modules **before** any implementation.

## What memory-os does that's worth assimilating

memory-os is **memory-first like Quarq, but its distinctive idea is making injected memory
authoritative** (Layer 7 "Ground Truth Hierarchy") so the agent stops re-fetching context it was
already given ("memory-zero behavior"). agent-utilities already has strong memory machinery
(KG-2.1 tiered memory, KG-2.11 bi-temporal, KG-2.12 memory-first retrieval, KG-2.13 learner) — the
gaps memory-os fills are around **injection discipline, retrieval resilience, memory hygiene, and
usage-driven trust**.

## Verified innovations → SDD features

Grouped into 6 SDD features extending existing concepts (build order respects dependencies):

| # | SDD feature | Concept | Bundles (verified ledger rows) | Wiring (≤3 hops) |
|---|---|---|---|---|
| **A** | **Ground-Truth Context Authority** | **NEW AU-KG.memory.ground-truth-preamble-declaring** (extends KG-2.1) | ground-truth-hierarchy, multi-source-surgical-injection | ⚠️ `startup_context.py` not statically reachable — needs explicit wiring path |
| **B** | Resilient Retrieval | extends KG-2.12 | four-level-fallback-cascade, social-closer-filter | ✅ `hybrid_retriever.py` (2) |
| **C** | Memory Hygiene | extends KG-2.1 / KG-2.3 | decay-scanner-importance-halflife, semantic-dedup-merge | ✅ `memory_engine.py` (2), `hybrid_retriever.py` (2) |
| **D** | Evidence-Weighted Memory | extends KG-2.6 | trust-scoring-feedback-loop, recall-usage-telemetry, generation-lineage-provenance | ✅ `retrieval_quality.py` (3) |
| **E** | Richer Learning | extends KG-2.13 | typed-session-learning-extraction | ✅ `learning_engine.py` (1) |
| **F** | Self-Curating Wiki | extends KG-2.7 | self-curating-llm-wiki | ⚠️ `physical_distiller.py` (4 hops) — re-target the ingestion engine |

**Highest leverage (score order):** social-closer-filter, ground-truth-hierarchy, generation-lineage-provenance.
**Critical path:** A (ground-truth → multi-source) is foundational; B/C/E are independent and parallelizable; D's telemetry depends on trust scoring.

## Synergy thesis — how each becomes *superior* in agent-utilities

- **A — Ground Truth (flagship, the novel one).** memory-os states authority in a static rulebook.
  agent-utilities can make it *structural*: rank injected memory by provenance + KG-2.11 bi-temporal
  validity + KG-2.6 trust, and emit a startup-context preamble that names the authoritative sources
  with their as-of validity. A graph-grounded authority hierarchy beats a flat prompt rule.
- **B — Resilient retrieval.** The 4-level cascade folds directly into `plan_and_retrieve` (KG-2.12)
  as a degradation ladder beneath the existing HyDE/two-pass logic; the social-closer gate saves the
  HyDE planner call entirely on trivial turns.
- **C — Hygiene.** Decay + semantic-merge become scheduled passes over the durable graph backend
  (bi-temporal aware: archive by `valid_to`, never destroy) — richer than memory-os's flat-store scan.
- **D — Evidence-weighted memory.** Trust feedback + recall/usage telemetry + generation lineage all
  extend the existing `ContextProvenanceRecord`/quality gate (KG-2.6), closing the loop the gate
  currently lacks (it scores retrieval but nothing trains the scores).
- **E — Richer learning.** Typed, outcome-grounded extraction (decision/resolution/note +
  training_value) upgrades the KG-2.13 learner's edit extraction with provenance-grade gating.
- **F — Self-curating wiki.** Structured concept/entity/comparison curation + SHA-256 diff ingest
  extends KG-2.7 research assimilation; re-target the ingestion engine (not the distiller) to stay ≤3 hops.

## Risks / findings from the wiring audit (pre-implementation)
- **A (startup_context.py)** is reachable only via lazy/dynamic import from the tested entry points.
  Confirm the live call path (memory CLI `context` / agent startup hook) and wire the authority
  preamble there explicitly, or the feature risks being a bolt-on.
- **F (physical_distiller.py)** sits 4 import-hops from entry points — exceeds Wire-First. Re-target
  `knowledge_graph/ingestion/engine.py` (reachable from `graph_ingest`) instead.

## Next step
The 11 features are scaffolded as DSTDD stubs under `.specify/design|specs/<id>/`. Promote them per
the build order, resolving the two wiring flags first. Each stub carries verified provenance, the
extension concept, a success metric, and the wiring target — ready to fill and implement.
