# Evidence-Weighted Memory (CONCEPT:AU-KG.retrieval.evidence-weighted-memory)

## Overview

Evidence-Weighted Memory closes the feedback loop the KG-2.6 quality gate lacked: it **trains a
trust score** from whether retrieved memory was actually used, and records a **generation lineage**
linking each answer to the memory ids it was grounded on. Assimilated from memory-os
(`layers/03-fact-store.md`, `icarus/state.py`, `scripts/context_enhancer.py`). Extends **KG-2.6**.

## How it works

- **Bayesian trust.** `bayesian_trust(helpful, total)` = `(helpful + 0.5·w) / (total + w)` — a fact
  retrieved and usually used trends toward 1.0; one retrieved but never used trends toward 0.0; an
  unseen fact stays at the 0.50 prior. Smoothing avoids the overconfident swings of a raw ratio.
- **Recall→usage telemetry.** `UsageTelemetry.record_recall` / `record_usage` log which memories
  were surfaced vs which informed the answer; `usage_rate` and per-node `trust` derive from those
  counts. `flush_to_engine` persists `trust_score` onto memory nodes (the previously-unused
  `store_memory(trust_score=...)` field), so trust survives restarts.
- **Generation lineage.** `build_lineage` records `query → retrieved_ids → used_ids → model +
  context_hash` (order-independent SHA-256), extending `ContextProvenanceRecord` so every answer is
  traceable back to its source memory.

## Key files / API

| Piece | Location |
|---|---|
| Evidence loop | `knowledge_graph/retrieval/retrieval_quality.py` (`bayesian_trust`, `UsageTelemetry`, `LineageRecord`, `build_lineage`) |

## Wiring (≤3 hops)

`graph_search` → retriever → quality gate / telemetry (3 hops). Trust feedback is additive and
default-off, preserving current retrieval behavior.

## Research provenance

memory-os fact trust scoring + telemetry + lineage — `layers/03-fact-store.md`, `icarus/state.py:80-150`,
`scripts/context_enhancer.py:75-104` (verified).
