# CMDB / ERP Bidirectional Integration — ServiceNow & ERPNext (CONCEPT:AU-KG.ingest.enterprise-source-extractor)

Mirror ServiceNow and ERPNext into the agent-utilities OWL/RDF knowledge graph —
including the **Technology Reference Model + technology risk** — and backfeed the KG
the other way: enrich records, push inferred relationships, and **create the
reconciled inventory as CMDB CIs / ERPNext Items+Assets**.

This is the operator runbook. The flow is driven by the
`agent-utilities-source-integration` skill; LeanIX has its own guide
(`leanix-integration.md`). Both share one mechanism: inbound `source_sync`, outbound
`graph_writeback`.

## What you get

| Direction | Capability | Surface |
|---|---|---|
| **In** | ServiceNow ITSM + CMDB + TRM (cmdb_model/alm_asset) + risk → typed KG nodes | `source_sync {"source":"servicenow","mode":"full"}` |
| **In** | ERPNext Assets/Items/Warehouses + stock → typed KG nodes | `source_sync {"source":"erpnext","mode":"full"}` |
| **Out** | Enrich existing CIs/Items, push inferred relations | `graph_writeback {"target":"servicenow", ...}` |
| **Out** | Create the reconciled inventory upstream (missing CIs/Items) | `graph_writeback {"target":"servicenow","inventory":true}` |
| **Out** | Retire/decommission upstream | `graph_writeback {"target":"...","retirements_json":"[...]"}` |

Every KG node from these sources carries `externalToolId` + `domain` (the federation
key). Vendor classes map into one **vendor-neutral TRM ontology** (`ontology_trm.ttl`):
`TechnologyProduct` / `TechnologyStandard` / `AssetInstance` / `TechnologyRisk`, so
LeanIX ITComponents, ServiceNow products/CIs, and ERPNext Assets reason together.

## 1. Configure credentials

ServiceNow (`servicenow-api` env): `SERVICENOW_URL`, `SERVICENOW_USER`,
`SERVICENOW_PASSWORD` (or OAuth); **`SERVICENOW_ENABLE_WRITE`** (fail-closed write gate).
ERPNext (`erpnext-agent` env): `ERPNEXT_URL`, `ERPNEXT_TOKEN` (`api_key:api_secret`);
**`ERPNEXT_ENABLE_WRITE`** (fail-closed write gate).

## 2. Mirror in (inbound)

```
graph-os call source_sync {"source": "servicenow", "mode": "full"}
graph-os call source_sync {"source": "erpnext",    "mode": "full"}
```
Verify: `MATCH (n) WHERE n.domain IN ['servicenow','erpnext'] RETURN n.type, count(*)`.
ServiceNow emits `:ConfigurationItem`/`:TechnologyProduct`/`:AssetInstance` (+ EOL/risk
→ `:TechnologyRisk`); ERPNext emits `:AssetInstance`/`:Item`/`:Warehouse`. Scheduled
every 2 h (`servicenow-sync`, `erpnext-sync` in `deploy/schedules.yml`).

## 3. Backfeed out (outbound) — fail-closed, dry-run-first

```
# preview (default) — nothing mutates
graph-os call graph_writeback {"target":"servicenow","enrichments_json":"[{\"node\":\"ci:...\",\"attributes\":{\"comments\":\"...\"}}]","dry_run":true}
# apply: set SERVICENOW_ENABLE_WRITE=true, then dry_run=false
```
- **Inferred relations** (`inferences_json`) — written as CMDB relations, provenance-tagged.
- **Enrichment** (`enrichments_json`) — `patch_cmdb_instance` / ERPNext `update_document`.
- **Retire** (`retirements_json`) — CMDB `install_status=retired` / ERPNext `disabled`.

## 4. Inventory push (the "add inventory as CMDB items" capability)

Create the KG's **reconciled** technology inventory (infra/topology + LeanIX
ITComponents + TRM products/assets, deduplicated via `ALIGNED_WITH` identity) as new
records in the target — skipping anything already present:

```
graph-os call graph_writeback {"target":"servicenow","inventory":true,"dry_run":true}   # preview candidates
# then SERVICENOW_ENABLE_WRITE=true + dry_run=false to create
```
Scheduled `servicenow-inventory-push` ships **disabled** (it mutates the SoR).

## Architecture (where the code lives)

| Concern | Module |
|---|---|
| Unified write-back core + fail-closed gate | `enrichment/writeback/core.py` |
| Per-target sinks | `enrichment/writeback/sinks/{servicenow,erpnext,leanix,capability,process}.py` |
| Inventory push (reconciled set) | `enrichment/writeback/inventory.py` |
| Source adapters (raw API → extractor surface) | `enrichment/source_adapters.py` |
| Extractors (TRM/risk/inventory) | `enrichment/extractors/{servicenow,erpnext}.py` |
| TRM + risk ontology | `ontology_trm.ttl` |
| Inbound sync entrypoint | `core/source_sync.py` |
| Scheduling | `core/skill_scheduler.py` + `deploy/schedules.yml` |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `source_sync source=servicenow` → `skipped: no source client` | `servicenow-api` absent or `SERVICENOW_URL/USER/PASSWORD` unset (the gateway env must carry them). |
| 0 TRM/asset nodes | The instance lacks `cmdb_model`/`alm_asset` plugins, or the technical user can't read them — the extractor probes-and-skips unknown classes. |
| `graph_writeback` → `refused` | Intended fail-closed — set `SERVICENOW_ENABLE_WRITE` / `ERPNEXT_ENABLE_WRITE` after reviewing the dry-run. |
| Inventory push re-creates existing CIs | Reconciliation needs `ALIGNED_WITH` identity — run a reasoning cycle so infra/LeanIX/CMDB nodes are aligned before pushing; candidates already in-`domain` are always skipped. |
| Relations not written to ERPNext | Expected — ERPNext has no generic relation table; relations are skipped (use doctype link-fields). |

## Safety

ServiceNow/ERPNext are systems-of-record. Inbound never writes upstream. All
write-back is fail-closed (`*_ENABLE_WRITE`) + dry-run-by-default; inferred relations
are provenance-tagged; **retire** and **inventory create** are the highest-risk ops
(preview first; inventory-push schedule ships disabled).
