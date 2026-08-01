# Design Document: External graphs are federated by reference, and imported only through a zero-PII, alias-only gate — never bulk-mirrored

> `agent_utilities/knowledge_graph/orchestration/engine_federation.py`
> (`FederationMixin` — reference registration), `agent_utilities/knowledge_graph/ingestion/external_graph.py`
> (the governed, read-only import path), `agent_utilities/knowledge_graph/pipeline/phases/external_graphs.py`
> (the ingestion-pipeline phase that registers SPARQL/LPG endpoint
> references), `agent_utilities/protocols/source_connectors/connectors/graphql_document.py`
> (the GraphQL-specific realization).

CONCEPT:AU-KG.ingest.external-graph-federation

## Decision — reference-only federation, plus a separate governed import path with an alias-only persistence gate

`engine_federation.py:1-4`, `43-47`; `external_graph.py:1-14`.

**The rejected alternative**: bulk-mirroring an external graph's full
content into the local KG the moment it's referenced — the conventional
"integrate an external source" approach, and explicitly NOT what this module
does. `engine_federation.py`'s own module docstring states the scope
directly: "Reference-only federation for external ontologies and graph
sources."

**The design chosen has two deliberately separate halves**:

1. **Reference registration** (`FederationMixin`, `pipeline/phases/external_graphs.py`) —
   external SPARQL/LPG endpoints are registered as lightweight
   `ExternalGraphReferenceNode`s (one per endpoint, deduplicated by a
   `uuid5`-derived node id over the endpoint URL) carrying only
   `endpoint_url`/`graph_type`/name metadata. This is metadata ABOUT the
   external graph's existence, not its content — gated behind
   `ctx.config.enable_external_graphs`, skipped entirely when disabled.
2. **Governed, read-only import** (`external_graph.py`) — the missing piece
   `engine_federation`'s reference registration deliberately does NOT
   provide: actually pulling bounded query ROWS from Neo4j/Apache
   AGE/LadybugDB/other `GraphBackend` implementations through the EXISTING
   named connection registry (not a new connection mechanism), via a
   secret-backed mapping profile that turns bounded query rows into
   canonical `ChangeEnvelope` objects (see
   `.specify/design/kgi-change-envelope-atomic/design.md`), applies a
   ZERO-PII persistence gate, and writes through the SAME lineage/ACL
   /idempotency path as every other source. Only connection and source
   ALIASES are persisted — endpoint URLs, credentials, query text,
   variables, local paths, raw external identifiers, and resolved profile
   content all remain TRANSIENT, never durably stored.

The GraphQL-specific realization (`graphql_document.py`) reuses this SAME
concept id alongside `AU-KG.ingest.universal-data-connector` — see
`.specify/design/kgi-universal-data-connector/design.md` — since GraphQL
document sources are one concrete instance of "external graph federated by
reference, imported through a governed gate," not a separate mechanism.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/orchestration/engine_federation.py`,
  `agent_utilities/knowledge_graph/ingestion/external_graph.py`,
  `agent_utilities/knowledge_graph/pipeline/phases/external_graphs.py`,
  `agent_utilities/protocols/source_connectors/connectors/graphql_document.py`.
- **Backward Compatible**: Yes — `enable_external_graphs` gates the
  reference-registration phase off by default-equivalent behavior when
  unset; the governed import path is opt-in per configured connection.
- **Breaking Changes**: None.
- **Known weak point**: reference registration and governed import are two
  INDEPENDENT code paths that happen to serve the same conceptual goal —
  nothing enforces that an external graph reference registered via
  `engine_federation` is ever actually backed by a governed import
  connection; a reference node can exist with no corresponding import path
  configured, which is a dangling pointer to a source the KG only knows
  about by name.
