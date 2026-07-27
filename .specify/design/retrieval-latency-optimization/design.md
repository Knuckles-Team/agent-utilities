# Design Document: Retrieval-path latency optimization

> Every feature begins with a design document. This gates creation through
> the Knowledge Graph to enforce the **Extend-Before-Invent** principle.

## KG Analysis (Required)

### Nearest Existing Concepts

| Concept ID | Name | Similarity | Pillar |
|---|---|---|---|
| AU-KG.retrieval.synthesized-cited-answer | code-context retrieval | high | AU-KG |
| AU-KG.compute.* | graph_compute batch primitives | high | AU-KG |
| AU-ORCH.execution.messaging-orchestration-transparency | reply-budget path | med | AU-ORCH |

### Extension Analysis

- **Primary Extension Point**: the existing retrieval/hydration path (`knowledge_graph/retrieval/hybrid_retriever.py`, `knowledge_graph/orchestration/engine_query.py`, `knowledge_graph/retrieval/graph_engineering.py`) and the bounded context compile (`core/contextual_model.py`).
- **Extension Strategy**: augment + specialize.
- **New Concept Required?**: Yes — two named, reusable optimization primitives.

## Problem

A `graph_orchestrate` delegation took ~122s (over the ~55s messaging reply budget), with 4 retrieval degradations. Root cause (measured, engine health fast throughout): retrieval legs still hydrated nodes **one at a time** (`GetNodeProperties` per hit in a loop), and the bounded context-compile kept re-attempting retrieval every LLM round even when the KG held no relevant content (composite score 0.00), burning 4×10s on the degrade-timeout.

## Design

- **`CONCEPT:AU-KG.retrieval.batch-hydrate`** — augment the established batch-hydration pattern (`graph_compute._get_node_properties_batch`, one round-trip) across the remaining per-hit retrieval legs (BFS-fallback neighborhood, tier-3 keyword scan, DCI hop hydration, impact/capability scans, GCE-fallback, entity-neighborhood). Collect the id set from each loop, fetch in ONE round-trip, then dict-lookup in place — identical order/semantics, only the fetch is batched. Fallback paths and edge-property reads are deliberately left untouched.
- **`CONCEPT:AU-KG.retrieval.context-compile-circuit-breaker`** — specialize the existing bounded compile (`_compiled_evidence_and_bundle_bounded`) with a per-process circuit breaker: after 3 consecutive degradations (timeout/exception) OPEN for a 30s cooldown, short-circuiting straight to the degraded `(messages, None)` result (no `to_thread`/`wait_for`), then HALF-OPEN a single probe → CLOSE on success. The fast path (retrieval succeeds) is byte-for-byte unchanged; the breaker only engages when retrieval is already consistently useless.

## Wire-First

Both land on the live delegation hot path (`run_agent` → pydantic-ai model_request → `compile_model_context` → retrieval). Verified by the full `tests/unit/knowledge_graph` sweep (identical pass/fail vs. main — zero regressions) and the standalone breaker state-machine test.
