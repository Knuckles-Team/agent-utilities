# Memory-First Retrieval (CONCEPT:AU-KG.retrieval.memory-first-retrieval)

## Overview

Memory-First Retrieval is the multi-stage recall *policy* layered over the KG-2.3 hybrid retriever:
**HyDE query expansion → dual thresholds → self-correcting two-pass → quantitative-fidelity
ledger**. Assimilated from Quarq Agent (`agent-oss/agent.py`), made graph-native so every retrieved
hit carries backlink-boost + positional encodings. Extends **KG-2.3** (Unified Retrieval).

## How it works

- **HyDE expansion.** The ORCH-1.27 `planner` role emits a structured plan — multiple vector
  formulations (baseline / entity / action / literal-unit) + keywords + a `search_mode` — and each
  sub-query runs through the existing `retrieve_hybrid`; results merge by id-dedup + max-score.
- **Dual thresholds.** `standard` (0.38) for point facts, `deep` (0.28) for aggregations / temporal
  spans (`hyde_planner.HYDE_THRESHOLDS`).
- **Self-correcting two-pass.** If the KG-2.6 quality gate reports `gate_passed=False` after the
  first pass, a second pass re-runs at the deep threshold and merges — an *evidence-based* trigger,
  stronger than Quarq's model-self-report `REQUIRED_DATA`.
- **Quantitative-fidelity ledger.** `build_evidence_ledger` emits an ACCEPT/REJECT table with
  extracted numbers so a generator aggregates a *complete ledger* rather than the single most
  salient row.

## Key files / API

| Piece | Location |
|---|---|
| Pure HyDE helpers | `knowledge_graph/retrieval/hyde_planner.py` (`HydePlan`, `parse_hyde_plan`, `merge_retrievals`, `build_evidence_ledger`, `threshold_for_mode`) |
| Orchestration | `knowledge_graph/retrieval/hybrid_retriever.py` (`plan_and_retrieve`, `_generate_hyde_plan`) |
| Entry surface | `knowledge_graph/orchestration/engine_query.py` (`search_hybrid(mode, self_correct)`); MCP `graph_search(mode="hyde"|"deep", self_correct=True)` |

## Wiring (≤3 hops)

`graph_search` → `search_hybrid` → `plan_and_retrieve` → `retrieve_hybrid` (exactly 3 hops — `plan_and_retrieve` stays a retriever method, not a service).

## Research provenance

Quarq Agent retrieval stack — `agent-oss/agent.py:1817-2825, 2435, 3211`.
