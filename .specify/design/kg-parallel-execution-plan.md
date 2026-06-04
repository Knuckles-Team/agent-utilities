# KG Enrichment + Enterprise OS — Parallel Execution Plan

> How multiple sessions complete KG‑2.8 (finish) + KG‑2.9 (Enterprise OS) **in
> parallel without stepping on each other**. The work is highly parallelizable
> once the shared chokepoints are removed — most of which are now done.

## Decoupling enablers
1. **Self‑registering extractor registry** (DONE — `enrichment/registry.py`):
   each source = its own module that calls `register_source(...)` at import;
   auto‑discovered. New sources need **zero edits to shared hub files**. Generic
   `write_batch()` persists any `ExtractionBatch` via the one `GraphBackend`.
2. **Tenant isolation** (DONE — KG‑2.8 item C): every session/stream uses its own
   epistemic‑graph tenant → no runtime data collision; durable pggraph is shared
   but label/tenant‑scoped.
3. **Per‑session daemon socket** (RULE): each session runs its own
   `epistemic-graph-server` on a unique `GRAPH_SERVICE_SOCKET` (e.g.
   `/tmp/eg-<stream>.sock`). Then **binary swaps / restarts never disrupt other
   sessions**. (Only ONE stream rebuilds the Rust binary at a time — see below.)
4. **Git worktree / branch per stream** (RULE): isolated working trees, merge via PR.

## Shared chokepoints — coordinate (append‑only, frequent merges)
- `enrichment/__init__.py` exports — prefer the registry; if exporting, append‑only.
- `enrichment/models.py` — additive entity models; low conflict.
- `core/owl_bridge.py` PROMOTABLE_NODE/EDGE sets — append‑only.
- `models/schema_definition.py` — additive node types (durable columns).
- **epistemic‑graph Rust crate + binary** — SERIALIZE: only one stream edits
  `src/` and rebuilds at a time; announce the swap. After KG‑2.8 Phase 2 most
  streams need **no** Rust changes (they consume existing connectors/RPCs).
- `kg_task_queue.db` — scope to your tenant; never clear another stream's queue.

## Streams (parallel unless noted)
| Stream | Scope | Owns (mostly NEW files) | Shared touch | Rust? |
|---|---|---|---|---|
| **S1** KG‑2.8 finish | batched/concurrent LLM cards + embeddings | `enrichment/cards.py`, `semantic.py` | none | no |
| **S2** KG‑2.9.0 infra | inventory.yaml + Docker svc → blast‑radius | `extractors/infra.py`, `ontology_enterprise_os.ttl` | owl_bridge (+edges) | no |
| **S3** KG‑2.9.1 DataConnector | multi‑DB SQL/GraphQL R/W + schema→KG + EG cache | `protocols/data_connector.py`, `connectors/*` | models (DataSource/Table) | no |
| **S4** ServiceNow ITSM | incidents/changes/CI → graph | `extractors/servicenow.py` | none (registry) | no |
| **S5** LeanIX EA | apps/capabilities/ArchiMate | `extractors/leanix.py` | none | no |
| **S6** ERPNext | employees/orders/cost centers | `extractors/erpnext.py` | none | no |
| **S7** Grafana/Prom | dashboards/alerts/metrics → MONITORS | `extractors/grafana.py` | none | no |
| **S8** CrossLinker scale | cross‑category edges over all sources | `crosslink/*` | none | no |

S4–S7 are **fully independent of each other** (the registry is the point). S8
depends on ≥2 sources existing. S2 should land `ontology_enterprise_os.ttl` early
as the shared enterprise base others reference (read‑only for them).

## Sequencing
1. **Foundation (DONE this session):** registry + `ExtractionBatch`/`GraphNode` +
   `write_batch` + tenant isolation → unblocks S2 & S4–S7 immediately.
2. **Wave A (parallel):** S1, S2, S3.
3. **Wave B (parallel):** S4, S5, S6, S7 (each a new extractor file).
4. **Wave C:** S8 cross‑linking + enterprise reasoning over the assembled graph.

## Per‑stream definition‑of‑done
Extractor emits `ExtractionBatch` (typed nodes + edges) for its source; unit test
with a fake client/connector (no live system); registers via `register_source`;
entities embedded + cross‑linkable; ontology terms added to `ontology_*.ttl` +
owl_bridge; no edits to other streams' files. Validate against a live system in an
isolated tenant on the stream's own daemon socket.

## Hand‑off pointer
Each session: read `MEMORY.md` → `kg-enrichment-phase1`, `kg-backend-abstraction`,
this plan, and the `.specify/design/kg-2.8-*` / `kg-2.9-*` strategy docs. Claim a
stream by its row above; work in a worktree; open a PR touching only your owned
files (+ append‑only shared edits).
