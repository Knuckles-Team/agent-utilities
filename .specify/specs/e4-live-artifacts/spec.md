# Spec: EPIC 4 — Live Artifacts (KG-2.24)

> Design: `.specify/design/kg-2.24-live-refreshable-artifact/design.md`. **Lighthouse (with EPIC 1).**
> Branches off after EPIC 1; does not depend on the gates.

## Pre-Flight Checklist
- [x] Design exists; KG-nearest table (KG-2.24 max 0.63 vs KG-2.19) <0.70.
- [x] Extension points: `mcp/kg_server.py` (graph_write), `gateway/api.py` (refresh), `knowledge_graph/` (KG-2.11/2.18/2.19).
- [x] Wire-First: write ≤1 hop (graph_write); refresh ≤2 hops.
- [ ] Live `kg_search` confirmation.

## User Stories
### US-1 — Create a Live Artifact
**As** an agent, **I want** to emit `{template, data, provenance, source_query}` instead of a static blob, **so that** the output can refresh.
- **AC1**: `LiveArtifactStore.create()` persists the artifact node + `source_nodes` edges via `graph_write`.
- **AC2**: Interpolation is injection-safe (`{{data.path}}` + `repeat` directive only; no raw HTML/expressions); JSON bounded (8 levels/100 keys/500 items/16KiB strings/256KiB total) — over-limit rejected.
- **AC3**: `provenance.json` records producing model/CLI (EPIC 1) + cited KG evidence node ids (KG-2.18).

### US-2 — Refresh re-derives from the KG
**As** a consumer, **I want** `POST /api/artifacts/{id}/refresh` to re-run the bound KG query and update `data.json`, **so that** the artifact stays current.
- **AC4**: Mutating a source KG node then refreshing changes `data.json` accordingly (integration).
- **AC5**: A refresh whose new derivation fails validation **preserves the prior render** (bi-temporal valid-time, KG-2.11) and records the failure in `refreshes.jsonl`.

### US-3 — SSRF-safe contract + progress events (descoped renderer/card)
**As** a frontend, **I want** an SSRF-safe artifact contract + postMessage schema and an SSE progress event, **so that** I can render it myself.
- **AC6**: The API emits an artifact contract + bridge schema that validate against a published JSON schema (no in-repo iframe renderer).
- **AC7**: A multi-step run emits ordered progress events with `in_progress`/`completed` transitions over `agent_ui` SSE.

## Non-Functional Requirements
- `@pytest.mark.concept(id="KG-2.24")`; ≤60s; no network.
- New node type + routes only; existing KG writes unaffected (zero regression).
- Docs: `docs/pillars/2_epistemic_knowledge_graph/KG-2.24.md`; concepts.yaml regen.

## Tasks
- [ ] T1 `knowledge_graph/live_artifacts/models.py`: `LiveArtifact` + bounded-JSON validator + safe interpolator. *(unit)*
- [ ] T2 `knowledge_graph/live_artifacts/store.py`: `create()` via graph_write; `source_nodes` edges. *(integration)*
- [ ] T3 `knowledge_graph/live_artifacts/refresh.py`: re-derive from bound KG query; bi-temporal preserve-prior on failure; `refreshes.jsonl`. *(integration: mutate → refresh; forced-fail → prior preserved)*
- [ ] T4 `mcp/kg_server.py`: artifact write helper on `graph_write`. *(integration)*
- [ ] T5 `gateway/api.py`: `/api/artifacts/{id}/refresh`. *(integration)*
- [ ] T6 `server/routers/agent_ui.py`: progress SSE event contract. *(unit)*
- [ ] T7 Publish artifact + bridge JSON schemas; provenance cites model (E1) + evidence (KG-2.18). *(unit)*
- [ ] T8 Docs/concepts/wiring-audit/CHANGELOG.
