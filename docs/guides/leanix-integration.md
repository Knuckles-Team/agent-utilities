# LeanIX ⇄ Knowledge Graph Integration (CONCEPT:KG-2.9)

Mirror SAP LeanIX (your Enterprise Architecture system-of-record) **natively** into
the agent-utilities OWL/RDF knowledge graph, keep it in sync with bite-sized deltas,
and backfeed KG-derived knowledge back into LeanIX.

This is the operator runbook. The end-to-end flow is also driven by the
**`leanix-integration`** agent skill (which references this guide for
troubleshooting). The enterprise self-setup (`agent-utilities-deployment` /
`agent-os-genesis`) delegates the LeanIX data-source step here rather than
duplicating it.

---

## What you get

| Capability | How | Surface |
|---|---|---|
| **Discover the metamodel → OWL** | introspect the live LeanIX data model, generate a faithful `ontology_leanix.ttl` (every fact sheet type → `owl:Class` ArchiMate-aligned, every relation → `owl:ObjectProperty`, every field → `owl:DatatypeProperty`) | `ontology_leanix_sync` MCP / `POST /api/ontology/leanix/sync` |
| **Mirror all fact sheets** | typed extractor + `ingest_external_batch`; every node stamped `externalToolId` + `domain="leanix"` | `source_sync` MCP (`source=leanix mode=full`) / `POST /api/source/sync` / `graph_hydrate source=leanix` |
| **Delta sync** | watermark poll (only fact sheets changed since last run) + webhook narrowing + nightly reconcile for deletions | `source_sync` MCP (`source=leanix mode=delta`/`reconcile`) + `deploy/schedules.yml` |
| **Backfeed to LeanIX** | inferred relationships, enrichment attrs/tags, new fact sheets — fail-closed, dry-run-first | `leanix_writeback` MCP / `POST /api/leanix/writeback` |

---

## 1. Configure

Set these in `~/.config/agent-utilities/config.json` (injected into the env; never
read bare):

```json
{
  "LEANIX_URL": "https://<your-workspace>.leanix.net",
  "LEANIX_TOKEN": "<api-token>",
  "LEANIX_VERIFY_SSL": true,
  "LEANIX_ENABLE_WRITE": false
}
```

- `LEANIX_URL` — your LeanIX base (the token is exchanged for a bearer at
  `/services/mtm/v1/oauth2/token`).
- `LEANIX_TOKEN` (alias `LEANIX_API_TOKEN`) — a LeanIX **technical-user API token**.
- `LEANIX_ENABLE_WRITE` — **fail-closed gate for backfeed**. Leave `false`; live
  write-back refuses unless it is `true`.

Verify resolution:

```python
from agent_utilities.ecosystem.ea_clients import get_leanix_client
print(get_leanix_client())   # None ⇒ not configured
```

## 2. Discover the metamodel → OWL

```bash
# preview the generated ontology without writing
graph-os call ontology_leanix_sync '{"dry_run": true}'
# apply: regenerates ontology_leanix.ttl + registers types for OWL promotion
graph-os call ontology_leanix_sync '{"dry_run": false}'
```

This regenerates `agent_utilities/knowledge_graph/ontology_leanix.ttl` (a
**generated** artifact — do not hand-edit) and registers every fact sheet type via
`register_promotable_node_types` so both reasoning layers see them:

- **DL reasoning** (`owl_reasoning` phase / `owl_closure`) reasons over the TTL —
  loaded through `knowledge_graph/ontology.ttl`'s `owl:imports <…/kg/leanix>`.
- **Structural reasoning** (`owl_bridge`) promotes the LPG nodes via the dynamic
  promotable set.

## 3. Mirror the fact-sheet graph

```bash
graph-os call source_sync '{"source": "leanix", "mode": "full"}'  # full mirror (first run)
# or via hydration:
graph-os call graph_hydrate '{"source": "leanix"}'  # delta by default
```

Confirm:

```cypher
MATCH (n) WHERE n.domain = 'leanix' RETURN n.type, count(*) ORDER BY count(*) DESC
```

Every node carries `externalToolId` (the LeanIX id) and `domain="leanix"` — the
federation key the write-back layer resolves against.

## 4. Keep it in sync (delta)

- **Watermark poll** — `source_sync source=leanix mode=delta` persists the max
  `updatedAt` on a per-source `SourceSyncState` node and, on the next run, fetches
  only newer fact sheets. Runs every 30 min via `deploy/schedules.yml`
  (`leanix-delta-sync`).
- **Reconcile** — `source_sync source=leanix mode=reconcile` compares the live
  fact-sheet id set with the KG and tombstones (`archived=true`) ones deleted in
  LeanIX. Runs nightly (`leanix-reconcile`). Deltas alone never surface deletions —
  keep reconcile on.
- **Webhook (near-real-time, optional)** — point a LeanIX webhook at an endpoint
  that calls `source_sync` with the changed `ids`:

  ```bash
  graph-os call source_sync '{"source": "leanix", "mode": "delta", "ids_json": "[\"<fs-id>\"]"}'
  ```

> **Standardized:** `source_sync` is source-agnostic — `source=camunda`,
> `servicenow`, etc. all sync through the same tool/scheduler. Sources without a
> native delta handler fall back to a full hydrate; LeanIX is the first
> delta-capable source (see `core/source_sync.py` `_DELTA_HANDLERS`).

## 5. Backfeed into LeanIX

**Always dry-run first.** Live writes require `LEANIX_ENABLE_WRITE=true`.

```bash
# preview (default): returns the exact proposed writes, nothing mutates
graph-os call leanix_writeback '{"inferences_json": "[{\"source\":\"app:a1\",\"rel_type\":\"REL_APPLICATION_TO_IT_COMPONENT\",\"target\":\"itcomponent:ic1\"}]", "dry_run": true}'

# apply (needs LEANIX_ENABLE_WRITE=true)
graph-os call leanix_writeback '{"inferences_json": "[...]", "dry_run": false}'
```

- **Inferred relationships** — written between existing fact sheets and tagged
  `agent-utilities:inferred` (reversible). Source: the OWL reasoner's inferences over
  the leanix-domain subgraph.
- **Enrichment** — `enrichments_json`: `[{"node": "...", "patches": [...], "tag": "..."}]`.
- **Auto-create** — `creations_json`: `[{"type": "...", "name": "..."}]` (highest
  risk; only when explicitly supplied).

---

## Architecture (where the code lives)

| Concern | Module |
|---|---|
| LeanIX client (one transport) | `agent_utilities/ecosystem/ea_clients.py` |
| Metamodel → OWL compiler | `agent_utilities/knowledge_graph/ontology/leanix_metamodel.py` |
| Typed extractor (mirror mapping) | `agent_utilities/knowledge_graph/enrichment/extractors/leanix.py` |
| Hydration wiring | `core/hydration.py` (`_hydrate_leanix`) |
| Delta sync / reconcile (source-agnostic) | `core/source_sync.py` |
| Backfeed | `enrichment/leanix_writeback.py` |
| Dynamic OWL promotion | `core/owl_bridge.py` (`DYNAMIC_PROMOTABLE_NODE_TYPES`) |
| Scheduling | `agent_utilities/core/skill_scheduler.py` + `deploy/schedules.yml` |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `get_leanix_client()` returns `None` | `LEANIX_URL`/`LEANIX_TOKEN` unset | Set both in `config.json`; confirm with the snippet in §1. |
| `ontology_leanix_sync` → `{"status":"skipped","reason":"empty LeanIX metamodel..."}` | unreachable host / bad token / SSL | Check `LEANIX_URL` reachability; re-mint the API token; if self-signed, set `LEANIX_VERIFY_SSL=false`. |
| `source_sync source=leanix` returns 0 nodes | token lacks read scope, or empty watermark mismatch | Run `mode=full` once; verify the technical user can read fact sheets in LeanIX. |
| Generated types not reasoned over | `ontology_leanix_sync` never applied (still the 4-class bootstrap), or owl_bridge built before sync | Run `ontology_leanix_sync dry_run=false`; restart the engine/daemon so a fresh `OWLBridge` picks up the dynamic promotable set. |
| Relations missing in the KG | relation fields not in the metamodel, or unusual envelope | Confirm the relation appears in `client.meta_model()`; the extractor walks every `rel*` field tolerantly — file the envelope shape if a custom one is dropped. |
| Delta keeps re-pulling everything | LeanIX `updatedAt` not monotonic, or watermark node not persisted | Check the `LeanixSyncState` node exists and `backend.execute` works; LeanIX has no reliable server-side time filter, so the watermark filters client-side — a missing `updatedAt` field on fact sheets disables it (falls back to full pulls, still correct). |
| Deletions linger in the KG | reconcile not running | Run `source_sync source=leanix mode=reconcile`; confirm `leanix-reconcile` is enabled in `deploy/schedules.yml`. |
| `leanix_writeback` → `{"status":"refused"}` | `LEANIX_ENABLE_WRITE` not set for a live write | Intended fail-closed behavior. Review the dry-run proposals, then set `LEANIX_ENABLE_WRITE=true`. |
| Write-back skips relations | KG nodes lack `externalToolId`, or rel_type has no LeanIX field | Re-mirror so nodes carry the federation key; the rel_type must invert to a LeanIX relation field via the metamodel. |
| Scheduled jobs don't run | daemon scheduler off, or schedule disabled | Ensure the daemon runs the scheduler tick; check `enabled: true` for `leanix-*` in `deploy/schedules.yml`; `/cron calendar` lists due jobs. |

### Diagnostics

```bash
# Are the LeanIX tools registered on both surfaces?
graph-os call multiplexer_status '{}' | grep -i leanix
python scripts/check_surface_parity.py

# Watermark state
# (Cypher) MATCH (n:LeanixSyncState) RETURN n.watermark
```

---

## Safety notes

- LeanIX is the **system-of-record**. Backfeed is fail-closed (`LEANIX_ENABLE_WRITE`)
  and dry-run-by-default; inferred relations carry a provenance tag so they are
  reversible.
- Auto-creating fact sheets is the highest-risk action — supply `creations_json`
  explicitly and review the dry-run first.
- The mirror is read-mostly: ingest/delta never writes to LeanIX.
