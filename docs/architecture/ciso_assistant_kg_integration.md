# CISO Assistant ↔ Knowledge Graph (bidirectional, OWL/RDF-native)

**CONCEPT:AU-KG.enrichment.ciso-assistant-extraction (extractor) · CONCEPT:AU-KG.enrichment.ciso-2 (writeback sink)**

intuitem **CISO Assistant** is the homelab's open-source GRC system-of-record
(Risk, Compliance & Audit, AppSec, TPRM, BIA, Privacy). This connector federates
its governance data with the one ontology-driven Knowledge Graph hub —
bidirectionally — and, *through that hub*, reconciles it with the **Egeria**
open-metadata catalog and the **Camunda** business-process estate that already
live in the graph.

## The one ontology

CISO Assistant records are mapped to the **canonical** governance classes shared
across the ecosystem (the same classes the Egeria extractor emits, see
`ontology_egeria.ttl` / `ontology_enterprise.ttl`), so they reconcile by GUID /
`qualifiedName` rather than living in a CISO-specific schema:

| CISO Assistant object | KG node (`id` → `type`) |
|---|---|
| policy | `ciso_assistant_policy:{id}` → `Policy` |
| applied / reference control | `ciso_assistant_control:{id}` → `Control` |
| risk scenario | `ciso_assistant_risk:{id}` → `Risk` |
| threat | `ciso_assistant_threat:{id}` → `Threat` |
| risk assessment | `ciso_assistant_risk_assessment:{id}` → `RiskAssessment` |
| compliance assessment | `ciso_assistant_compliance_assessment:{id}` → `ComplianceAssessment` |
| framework | `ciso_assistant_framework:{id}` → `Framework` |
| asset | `ciso_assistant_asset:{id}` → `Asset` |
| incident | `ciso_assistant_incident:{id}` → `Incident` |
| third-party entity | `ciso_assistant_entity:{id}` → `Entity` |

Every node carries `domain="ciso_assistant"`, `externalToolId` (the CISO uuid) and
`qualifiedName` (the CISO `urn`/`ref_id`) — the federation keys.

## Inbound — ingest CISO Assistant INTO the KG

`enrichment/extractors/ciso_assistant.py` (`CATEGORY="ciso_assistant"`,
self-registering) consumes the injected `ciso_assistant_api.Api` facade, calling
its generated list methods (`api_policies_list`, `api_applied_controls_list`,
`api_risk_scenarios_list`, `api_compliance_assessments_list`, …) and emitting:

- **Nodes** — the canonical mapping above.
- **Internal edges** — `Risk —MITIGATED_BY→ Control`, `Risk —PART_OF→
  RiskAssessment`, `ComplianceAssessment —CONFORMS_TO→ Framework`.

The extractor is import-safe (no CISO imports at module top; `client is None`
yields an empty batch) and tolerant of partial client surfaces.

## The crosswalk — CISO ↔ Egeria ↔ Camunda

Reconciliation is the established `ALIGNED_WITH` equivalence pattern. When a CISO
record carries an explicit twin id (populated by an operator mapping field), the
extractor emits an `ALIGNED_WITH` edge to the twin's deterministic node id:

- a CISO control/policy with an Egeria GUID → `ALIGNED_WITH egeria_policy:{guid}`
- a CISO risk/compliance process with a Camunda/BPMN id → `ALIGNED_WITH
  bpmn_process:{id}`

The OWL reasoner (run after every materialize via `OntologyReasoningDriver`)
treats `ALIGNED_WITH` as `sameAs`, so a CISO control aligned to an Egeria policy,
or a CISO compliance process aligned to a Camunda process, collapses into **one
logical concept** — governance intelligence then transits CISO ↔ Egeria ↔ Camunda.
Absent an explicit twin id, the shared `qualifiedName`/`name` lets the reasoner
reconcile by `sameAs`. No edits to the existing Egeria/Camunda connectors are
needed — they already mint the canonical, deterministically-keyed twin nodes.

## Outbound — enrich data back INTO CISO Assistant

`enrichment/writeback/sinks/ciso_assistant.py` (`CisoAssistantSink`,
`domain="ciso_assistant"`) pushes KG-derived governance entities back as CISO
objects via the generated `api_*_create` methods (`Policy`/`Control`/`Asset`/
`Finding`/`SecurityException`/`Entity`). It is **standard tier, fail-closed,
dry-run-first**: live writes require `CISO_ASSISTANT_ENABLE_WRITE`, and every CISO
object must name a `folder` (domain) — never guessed; missing-folder creations are
skipped, not invented.

## Components

| Path | Role |
|---|---|
| `enrichment/extractors/ciso_assistant.py` | inbound extractor (AU-KG.enrichment.ciso-assistant-extraction) |
| `enrichment/writeback/sinks/ciso_assistant.py` | outbound sink (AU-KG.enrichment.ciso-2) |
| `enrichment/materialize.py` | `_CLIENT_MODULES["ciso_assistant"]="ciso_assistant_api"`, `MATERIALIZE_SOURCES` membership |
| `agents/ciso-assistant-api` | the `ciso_assistant_api.Api` client / MCP / agent (CONCEPT:AU-ECO.connector.ciso-assistant) |

## Configuration

The connector resolves its vendor client in-process from the
`ciso_assistant_api` package's `auth.get_client()` — which reads
`CISO_ASSISTANT_URL` + `CISO_ASSISTANT_TOKEN` (or `CISO_ASSISTANT_USERNAME` /
`CISO_ASSISTANT_PASSWORD`) from its own environment. Writeback is gated by
`CISO_ASSISTANT_ENABLE_WRITE` (default off).

## Verification

```bash
# inbound, live (needs a running CISO Assistant backend + the package installed)
source_sync(source="ciso_assistant", mode="delta")   # via served MCP
source_sync(source="ciso_assistant", mode="delta")   # re-run → skipped_unchanged > 0

# outbound writeback dry-run (fail-closed; CISO_ASSISTANT_ENABLE_WRITE unset)
graph_writeback(target="ciso_assistant", dry_run=True)   # proposals only, created == 0

# unit
pytest tests/unit/knowledge_graph/enrichment/test_ciso_assistant_extractor.py \
       tests/unit/knowledge_graph/enrichment/test_ciso_assistant_writeback.py -q
```

See also [Camunda + ARIS ↔ KG](camunda_aris_kg_integration.md) and
[KG Connectors, Ingestors & Enrichers](kg_connectors_and_ingestion.md).
