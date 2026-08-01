# Design Document: External-graph connection is proposal-based — discover, propose, approve, rediscover-and-fail-closed — never direct auto-mapping

> `agent_utilities/knowledge_graph/ingestion/external_graph_schema.py`
> (universal discovery + mapping proposals), `agent_utilities/knowledge_graph/ingestion/graphql_connection.py`
> (the governed GraphOS lifecycle bridging a named connection to the connector).

CONCEPT:AU-KG.ingest.external-graph-mapping-approval ·
CONCEPT:AU-KG.ingest.external-graph-universal-discovery

## Decision — two planes: transient bounded discovery vs. a durable, approved mapping profile

`external_graph_schema.py:1-14`, `graphql_connection.py:1-20`.

**The rejected alternative**: mapping an external property graph's schema
automatically and directly into KG ingestion the moment it's discovered — no
human/agent approval step between "we found this schema shape" and "we're
ingesting through it." That would mean a schema drift (or a semantic
callback's proposal) could silently change what gets ingested with no review
point.

**The design chosen**: the connector is DELIBERATELY split into two planes.

1. **Discovery is transient, bounded, read-only** against a named source —
   no endpoint, credential, local path, raw external identifier, sample
   value, or query result is EVER included in the public discovery result.
   Deterministic mapping runs first; an OPTIONAL semantic callback receives a
   policy-compiled context bundle and may only PROPOSE additional mappings —
   "it can never approve or ingest them" (stated as an explicit invariant,
   not an implementation detail).
2. **The durable object is an encrypted mapping profile plus a pseudonymous
   status** — approval is a SEPARATE, explicit step: discover a bounded
   field graph → validate a secret-backed or structurally-generated policy
   and store a pseudonymous PROPOSAL → approve the exact schema and policy
   digests → REDISCOVER immediately before a dry-run or ingest and FAIL
   CLOSED on drift → only then drain the native connector through the
   authoritative `ChangeEnvelope` path (see
   `.specify/design/kgi-change-envelope-atomic/design.md`).

The "rediscover immediately before ingest and fail closed on drift" step is
the load-bearing safety property: an approval is a snapshot (schema digest +
policy digest), and if the live source's schema has drifted since approval,
ingestion refuses to proceed on stale trust rather than silently ingesting
against a schema that no longer matches what was approved.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ingestion/external_graph_schema.py`,
  `agent_utilities/knowledge_graph/ingestion/graphql_connection.py`,
  `agent_utilities/protocols/source_connectors/connectors/graphql_document.py`.
- **Backward Compatible**: Yes — approval-gated ingestion is the ONLY path;
  there is no legacy auto-map path this displaces within these modules.
- **Breaking Changes**: None (new capability).
- **Known weak point**: the semantic callback's proposals still require a
  human/agent to review and approve — a source with a very large or
  frequently-changing schema could generate proposal churn that makes manual
  approval a bottleneck, and nothing here auto-approves a "trivial" schema
  change (e.g. an added optional field) differently from a substantive one.
