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
