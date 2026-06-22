---
name: agent-utilities-source-integration
aliases:
  - source-integration
  - connect-source
  - leanix-integration
  - leanix-sync
  - mirror-leanix
  - ea-integration
  - process-integration
  - camunda-aris-setup
  - connect-camunda
  - connect-aris
  - connect-servicenow
description: >
  One standardized, config-complete path to connect ANY external source to the
  agent-utilities Knowledge Graph — OWL/RDF-natively and bidirectionally. Covers the
  whole source fleet under one model: LeanIX (EA fact sheets), Camunda + ARIS (BPMN/EPC
  processes), ServiceNow/GLPI (ITSM), Egeria (governance), GitLab/GitHub, Twenty (CRM),
  Jira/Plane, and databases. The same flow every time — configure credentials, discover
  (for sources with a queryable metamodel), mirror/ingest, sync with bite-sized deltas,
  and optionally backfeed KG-derived knowledge — via the single `source_sync` entrypoint
  plus per-source ingest/backfeed tools. Use when the user says "connect <X> to the KG",
  "ingest LeanIX/Camunda/ARIS/ServiceNow", "mirror our EA/process tools", "sync
  <source>", or "backfeed to <source>". Assumes a running graph-os. Do NOT use to deploy
  the platform from scratch (use agent-utilities-deployment) or for bare-host bootstrap
  (use agent-os-genesis).
domain: infrastructure
tags:
  - knowledge-graph
  - integration
  - leanix
  - camunda
  - aris
  - servicenow
  - enterprise-architecture
  - ontology
  - sync
requires:
  - graph-os
---

# Connect an External Source to the Knowledge Graph (standardized)

**One model for every source.** Each external system becomes native, reasoned-over KG
data through the same five steps, then (optionally) the KG writes intelligence back. The
mechanism is uniform — the only per-source differences are credentials, which tool
ingests, and whether a metamodel/backfeed exists. Deep per-source runbooks live in
agent-utilities docs (linked in the matrix); this skill is the standard driver.

> **Why not fold this into `agent-os-genesis`?** Genesis is *bare-host → swarm*
> platform bootstrap. Connecting data sources is a **post-platform** step, so genesis
> (and `agent-utilities-deployment`) **delegate here** instead. Run this after the
> platform is up.

## The standard spine

| Step | What | Tool (uniform) |
|---|---|---|
| 1. Configure | per-source creds (see matrix) | `config.json` / `secret-vault-manager` |
| 2. Discover | metamodel → OWL (sources that expose one) | `ontology_leanix_sync` (LeanIX); others use the fixed ArchiMate crosswalk |
| 3. Mirror | full ingest into the KG | `source_sync {"source":"<X>","mode":"full"}` |
| 4. Sync | bite-sized delta + deletion reconcile | `source_sync {"source":"<X>","mode":"delta"\|"reconcile"}` + `deploy/schedules.yml` |
| 5. Backfeed | push KG knowledge back (fail-closed) | `graph_writeback {"target":"<X>", ...}` (unified) — incl. `inventory:true` to create CMDB CIs / ERP items |

`source_sync` is the **single entrypoint for every source**: LeanIX runs an incremental
watermark delta; Camunda/ARIS/Egeria route through the materialize core; the rest hydrate
via the capability registry. Sources without a native delta fall back to a full hydrate
(reported as `delta_capable: false`).

## Step 1 — Configure credentials (per source)

Store via `secret-vault-manager` (OpenBao/Vault) or `config.json` / connector `.env`.

**LeanIX** (agent-utilities config): `LEANIX_URL`, `LEANIX_TOKEN`, `LEANIX_VERIFY_SSL`,
`LEANIX_ENABLE_WRITE` (fail-closed backfeed gate). Verify:
`python -c "from agent_utilities.ecosystem.ea_clients import get_leanix_client; assert get_leanix_client()"`

**Camunda** (`camunda-mcp` env): `CAMUNDA_PLATFORM` (`7`/`8`), `CAMUNDA7_URL` +
`CAMUNDA7_TOKEN` or `CAMUNDA7_USERNAME`/`CAMUNDA7_PASSWORD`; or `CAMUNDA8_*` (Zeebe/
Operate/Tasklist + OAuth2 client-credentials). `CAMUNDA_SSL_VERIFY`.

**ARIS** (`aris-mcp` env): `ARIS_API_BASE`, OAuth2 (`ARIS_OAUTH_URL`/`ARIS_CLIENT_ID`/
`ARIS_CLIENT_SECRET`/`ARIS_TENANT`) or `ARIS_TOKEN` or basic; `ARIS_PATHS_JSON` for
per-tenant path overrides; `ARIS_ENABLE_WRITE` (backfeed gate). Probe `GET {ARIS_API_BASE}/models`.

**ServiceNow / Egeria / GitLab / Twenty / …**: the source's own connector env (read each
connector package's `auth.get_client()`); registered in the hydration `CAPABILITY_REGISTRY`.

## Step 2 — Discover the metamodel (sources that expose one)

LeanIX has a queryable metamodel → mirror it natively as OWL:

```
graph-os call ontology_leanix_sync {"dry_run": true}    # preview generated OWL
graph-os call ontology_leanix_sync {"dry_run": false}   # apply (regenerates ontology_leanix.ttl)
```

Camunda/ARIS/ServiceNow map to the fixed **canonical ArchiMate crosswalk**
(`:BusinessProcess`/`:BusinessTask`/`:flowsTo`, etc.) — no discovery step; skip to Step 3.

## Step 3 — Mirror / ingest

```
graph-os call source_sync {"source": "leanix",  "mode": "full"}
graph-os call source_sync {"source": "camunda", "mode": "full"}   # routes via materialize
graph-os call source_sync {"source": "servicenow", "mode": "full"}
```

Verify: `MATCH (n) WHERE n.domain = '<source>' RETURN n.type, count(*)`. Every node carries
`externalToolId` + `domain=<source>` — the federation key cross-source reasoning and
backfeed resolve against.

## Step 4 — Keep in sync (delta + reconcile)

```
graph-os call source_sync {"source": "<X>", "mode": "delta"}      # only what changed (leanix: watermark)
graph-os call source_sync {"source": "leanix", "mode": "reconcile"} # tombstone deletions (delta-capable sources)
```

Schedule in `agent-utilities/deploy/schedules.yml` — `kind: skill`, `ref: <source>`,
`action: delta|full|reconcile` dispatches generically to `source_sync` (LeanIX ships with
`leanix-delta-sync` every 30 min + nightly `leanix-reconcile`). Webhook: call
`source_sync` with `ids_json` for near-real-time narrowing.

## Step 5 — Backfeed (optional, fail-closed, dry-run-first)

All backfeed flows through one tool — `graph_writeback {"target":"<X>", ...}` —
fail-closed (the target's `*_ENABLE_WRITE` gate) and dry-run by default:
```
# inferred relations / enrichment / new records (preview, then dry_run=false)
graph-os call graph_writeback {"target":"leanix","inferences_json":"[...]","dry_run":true}
graph-os call graph_writeback {"target":"servicenow","enrichments_json":"[...]","dry_run":true}

# create the reconciled inventory as CMDB CIs / ERP items
graph-os call graph_writeback {"target":"servicenow","inventory":true,"dry_run":true}

# Camunda/ARIS process intelligence (target=process)
graph-os call graph_writeback {"target":"process","dry_run":false}   # needs KG_PROCESS_WRITEBACK
```

All backfeed is gated and reversible (inferred links carry a provenance tag). Creating
records upstream and **retire/reconcile** are the highest-risk actions — review the
dry-run first; the inventory-push schedule ships disabled.

## Per-source matrix (depth)

| Source | Discover | Mirror/Sync | Backfeed (`graph_writeback`) | Deep doc |
|---|---|---|---|---|
| **LeanIX** | `ontology_leanix_sync` | `source_sync source=leanix` (watermark delta + reconcile) | `target=leanix` (`LEANIX_ENABLE_WRITE`) | `docs/guides/leanix-integration.md` |
| **ServiceNow** | crosswalk + TRM (cmdb_model/alm_asset) + risk | `source_sync source=servicenow` (materialize) | `target=servicenow` create CIs/enrich/relate/retire + `inventory:true` (`SERVICENOW_ENABLE_WRITE`) | `docs/guides/cmdb-bidirectional-integration.md` |
| **ERPNext** | crosswalk + Asset/Item/Warehouse + stock | `source_sync source=erpnext` (materialize) | `target=erpnext` create Items/Assets/enrich/retire + `inventory:true` (`ERPNEXT_ENABLE_WRITE`) | `docs/guides/cmdb-bidirectional-integration.md` |
| **Camunda** | crosswalk | `source_sync source=camunda` (materialize) | `target=process` (`KG_PROCESS_WRITEBACK`) | `docs/architecture/camunda_aris_kg_integration.md` |
| **ARIS** | crosswalk | `source_sync source=aris` (materialize) | `target=process` (`ARIS_ENABLE_WRITE`) | `docs/architecture/camunda_aris_kg_integration.md` |
| **Egeria** | crosswalk | `source_sync source=egeria` (materialize) | governed-routing federation | `docs/architecture/...` |
| **Twenty / GitLab / Jira / DBs …** | crosswalk | `source_sync source=<X>` (hydration registry) | — | hydration `CAPABILITY_REGISTRY` |

**Technology Reference Model + risk.** ServiceNow (cmdb_model/alm_asset), LeanIX
(ITComponent), and ERPNext (Asset) all map into one vendor-neutral TRM ontology
(`TechnologyProduct`/`AssetInstance`/`TechnologyRisk`), so the portfolio + lifecycle/EOL
risk reason together and the inventory push dedupes across them via `ALIGNED_WITH`.

## Verify end-to-end

- Step 3: `source_sync … mode=full` → `nodes_hydrated`/`nodes` > 0; Cypher count by type.
- Step 4: run `mode=delta` twice → second run only processes changes.
- Backfeed: `*_writeback … dry_run=true` lists proposals, nothing written.
- `python scripts/check_surface_parity.py` (agent-utilities) → 0 unexposed.

## Troubleshooting (quick)

| Symptom | Fix |
|---|---|
| `source_sync` → `skipped: no … client configured` | Set the source's creds (Step 1); for LeanIX verify `get_leanix_client()`. |
| `source_sync source=camunda` → `skipped: no source client` | `camunda-mcp` absent or creds unset — the gateway env must carry the connector vars. |
| `reconcile not supported for '<X>'` | Only delta-capable sources reconcile (LeanIX today); others full-hydrate. |
| `ontology_leanix_sync` → `empty metamodel` | Host unreachable / bad token / SSL — check URL, re-mint token, `LEANIX_VERIFY_SSL=false` for self-signed. |
| Delta re-pulls everything | Missing/non-monotonic `updatedAt`; ensure the `SourceSyncState` node persists. |
| `*_writeback` → `refused` | Intended fail-closed — set the source's write gate after reviewing the dry-run. |

For per-source depth and architecture, follow the matrix's deep-doc links.

## Notes & guardrails

- Connection creds for connector-backed sources (Camunda/ARIS/ServiceNow/…) are
  **connector** config in their own packages; only the agent-utilities-side gates
  (`LEANIX_ENABLE_WRITE`, `KG_PROCESS_WRITEBACK`, `ARIS_ENABLE_WRITE`) are agent-utilities flags.
- Ingest/delta never write upstream; backfeed is always gated + dry-run-first.
- New source? Register it in the hydration `CAPABILITY_REGISTRY` (or add a delta handler
  in `core/source_sync.py` `_DELTA_HANDLERS`) — then it works through this same skill.
