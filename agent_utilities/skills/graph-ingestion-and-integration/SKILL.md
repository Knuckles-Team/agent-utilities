---
name: graph-ingestion-and-integration
skill_type: skill
description: >-
  Bring documents, repositories, feeds, and external systems into Graph-OS and
  keep them synchronized. Use for source onboarding, connector selection,
  content processing, delta sync, ETL, hydration, feed handling, sharing,
  writeback, ingestion sessions, or freshness and coverage verification.
---

# Graph ingestion and integration

Build an idempotent source-to-graph flow with explicit provenance, freshness,
and verification.

## Choose the route

| Need | Primary operation |
|---|---|
| Ingest one artifact | `graph_ingest` |
| Register or inspect a source | `source_connector` |
| Synchronize changed content | `source_sync` |
| Drain queued source work | `source_drain` |
| Process document content | `document_process` |
| Transform graph data | `graph_etl` |
| Refresh registered sources | `source_sync` with `mode="full"` |
| Manage recurring feeds | `graph_feeds` |
| Export or share selected data | `graph_share` |
| Send approved changes upstream | `graph_writeback` |
| Collect or upload agent session bundles | `ingest_sessions` |

Use one operation directly for a bounded artifact or sync. Delegate through
`graph_workflows` when onboarding a source requires discovery, mapping,
backfill, validation, and a scheduled delta flow.

## Action reference

| Tool | Actions | Notes |
|---|---|---|
| `graph_ingest` | ingests one artifact; `content_type` routes explicitly (`config`, `prompt`, `mcp_server`, `skill`, `document`, `conversation`, `codebase`) or auto-classifies; `action="distill"` exports a KG subgraph to a portable skill-graph, `action="import_pack"` round-trips one back in | delta-skip via a durable content-hash manifest — re-ingesting an unchanged source is a no-op |
| `source_sync` | `source=<connector>` + `mode=full\|delta\|reconcile`; `source="all"` fans out one laned `connector_sync` task per candidate across every registered connector — declarative, computed from the registries, never hand-enumerated | see "Full ingest" below for the one-call fleet-wide sweep |
| `graph_etl` | `action="run"` (pull `source` into the KG and/or load `sink` from the KG — a write-back SoR, a graph store `stardog`/`neo4j`/`age`/`jena_fuseki`, or `sink="table"` for the native engine SQL table), `action="list"` (sources/sinks/backends), `action="lineage"` (recorded runs) | composes ingestion + write-back + graph-store machinery into one source → (ontological transform) → sink flow |
| `graph_ingest` (hydrate) | `graph_ingest(source=<connector>, mode="full")` re-mirrors one external source; `source="all"` fans to the fleet-wide sweep | a thin alias delegating to the same unified `source_sync` core — use `graph_etl`/`source_sync` directly for delta/reconcile modes |
| `graph_feeds` | `list`, `add` (one `url=` or bulk `urls=`), `remove`, `sync` (run the feed sweep now, `mode=delta\|full`) | manages `:FeedSource` nodes (native RSS, FreshRSS, ScholarX arXiv) ingested through one world-model gate |
| `graph_writeback` | `target=leanix\|servicenow\|erpnext\|process\|capability\|…`; ops: `inferences_json`, `enrichments_json`, `creations_json`, `retirements_json`; `action=write\|proposals\|approve` | fail-closed: `dry_run=true` is the default and previews the exact proposed writes; a live write needs the target's own enable flag (e.g. `LEANIX_ENABLE_WRITE`) |
| `graph_share` | `org` (share with the owner's org in place), `commons` (promote a copy into the shared cross-org commons graph), `mark` (attach a mandatory `marking`), `private` (restrict back) | the explicit promotion path for data that is private-to-its-owner by default; actor/owner is the ambient identity, never caller-supplied |

### Full ingest — every source in one fan-out

A full ingest exercises every ingestion family in parallel, each on its own task
lane (`agent_utilities/knowledge_graph/core/task_lanes.py`) so heavy codebase
indexing in the `ingestion` lane can never head-of-line-block connector/feed syncs
in the `connectors`/`worldview` lanes:

```text
# 1) codebase + documents (heavy file-ingestion lane) — workspace + doc + ontology +
#    config + skill paths, resolved via repository-manager
graph_ingest(target_path="<JSON array of paths>")

# 2-4) every connector + both native feed sources, fanned across the connectors/
#      worldview lanes in one declarative call — the candidate set is computed from
#      the registries (_DELTA_HANDLERS, capability registry, PACKAGE_PRESETS,
#      MATERIALIZE_SOURCES) at run time, never hand-enumerated here
source_sync(source="all", mode="full")
```

Use `mode="full"` for a complete (re-)hydrate, `mode="delta"` for an incremental
top-up (the write-layer content-hash delta makes unchanged entities a no-op either
way). Monitor every lane's drain with `graph_jobs(action="list")`.

Every `agents/*` connector also does the complementary **native push**: its own code
writes into the ONE engine as it works (typed OWL nodes + documents + raw blobs, via
the shared `native_ingest` primitive) — so the KG stores the data itself, not just
metadata. Both directions (hub-side pull above, package-side push) are default-on
and engine-guarded (a clean no-op with no reachable engine). The full category→tool
matrix, the connector→OWL-entity reference (20+ connectors), and the per-package
native-push matrix are in
[`references/ingest-connector-reference.md`](references/ingest-connector-reference.md)
(kept in lockstep with the registries — adding a connector needs no change to this
skill or that reference).

## Workflow

### 1. Define the source contract

Record the source kind, stable source identifier, ownership, update cadence,
scope, and expected entities. Use a declarative `mcp_tool` connector preset for
an external system that already has an MCP surface. Reserve native connectors
for governed schemas, zero-infrastructure sources, or engine hot paths.

### 2. Inspect before writing

- Sample a bounded page.
- Confirm identifiers, pagination, timestamps, deletion semantics, and content
  encoding.
- Map source fields to canonical graph types and relationships.
- Reject records that cannot satisfy required identity or policy fields.

For any governed GraphQL hierarchy, use `source_connector` with
`source_type="graphql_document"`. Keep the endpoint, auth, named TLS profile,
queries, hierarchy/document/application/dependency mappings, partial-error
allowlists, ACL, classification, retention, and HMAC key behind the secret
profile reference. No source taxonomy or query belongs in the skill or package.
Use bounded variables and `dry_run`; accept partial data or an optional-field
fallback only when the runtime profile explicitly allowlists the affected field.

For an authoritative hierarchy snapshot, set entity and document scopes
explicitly and run `dry_run` with the prior checkpoint before reconciliation.
Review only the privacy-safe manifest: missing identities, truncation, planned
deletions, and reconciliation eligibility. An incomplete, changed-profile,
truncated, unnormalizable, or unapproved empty snapshot cannot delete prior
knowledge. Dependency records remain reified evidence nodes, and documents use
the native retrieval-index handoff.

For registered Neo4j/openCypher, Apache AGE, LadybugDB/Kuzu, remote
epistemic-graph, or generic GraphQL sources, use
`graph-runtime-and-governance` to run the universal connection sequence:
`graph_configure(action="discover_connection_schema")` →
`graph_configure(action="propose_connection_mapping")` →
`graph_configure(action="approve_connection_mapping")` →
`graph_configure(action="external_graph_doctor")` →
`graph_configure(action="ingest_connection")`, using only a neutral connection
alias and opaque configuration references. Discovery is bounded and read-only.
Standard GraphQL introspection is used only when AgentConfig explicitly permits
it. Without a mapping-policy ref, introspection may generate bounded structural
read/mapping proposals, but exact-digest approval is still mandatory and raw
samples must not persist. Otherwise the profile must supply a bounded read probe. Never treat
Neo4j `elementId()` or AGE `id(n)` as durable identity—require a common stable
property. Run a dry-run before the native ChangeEnvelope import and retain only
opaque identifiers, governance metadata, counts, capability flags, pseudonyms,
and digests. Any schema drift invalidates approval and fails closed.

For a plan, inspection, or dry-run request, return the sampled schema, proposed
mapping, privacy-safe manifest, and verification plan, then stop. Do not approve
a mapping, ingest, reconcile, schedule, or write back unless the corresponding
mutation is explicitly authorized.

### 3. Ingest idempotently

- Prefer `source_sync` for a registered source and `graph_ingest` for a single
  artifact.
- Preserve a neutral source reference, content hash, observed time, and connector version.
- Use deterministic identifiers so retries update rather than duplicate.
- Keep enrichment on the canonical document-processing path.

Example authorized sync and verification plan; use
`graph-query-and-explanation` for the read-back:

```text
source_sync(source="registered-source", mode="delta")
graph_query(cypher="MATCH (n {id: $id}) RETURN n.id LIMIT 1", params='{"id":"synthetic-record"}')
```

### 4. Verify the graph

- Re-run the same delta and expect unchanged content to be skipped.
- Query a representative node and relationship created by the mapping.
- Compare processed, skipped, failed, and deleted counts with the source sample.
- Check freshness and connector coverage before scheduling recurring sync.

### 5. Add writeback only when authorized

- Treat read ingestion and source mutation as separate policy decisions.
- Preview the intended upstream changes.
- Require stable target identifiers, conflict handling, and an audit record.
- Report partial failures without claiming full success.

Use an economy model for extraction, field classification, and batch validation.
Escalate only schema ambiguity, conflict resolution, or final synthesis.

## Guardrails

- Never place credentials or secret values in skill inputs, examples, reports,
  or persisted metadata.
- Do not invent a connector when an existing MCP source preset can express the
  integration.
- Bound pages, batches, retries, and enrichment cost.
- Reject insecure GraphQL transport, mutation/subscription documents, unbounded
  generated roots, repeated cursors, unallowlisted partial errors, and blanket
  optional-field fallbacks.
- Require exact current GraphQL runtime-document formats for connection,
  mapping-policy, and auth refs; missing or unknown versions fail closed.
- Never reconcile a non-empty GraphQL baseline from an empty authoritative result
  unless `allow_empty_snapshot` is part of the exact approved mapping digest.
- Keep TLS verification on by default. Configure platform trust or a complete
  PEM chain through runtime environment/secret projection; never embed a CA
  location or `verify=false` in a source profile.
- Preserve tenant and graph scope across every stage.
- Do not approve a mapping, ingest, reconcile, schedule, or write back without
  explicit authorization for that mutation.
- Do not enable recurring sync or writeback without explicit authorization.
