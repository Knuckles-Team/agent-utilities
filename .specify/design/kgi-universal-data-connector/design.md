# Design Document: One driver-based, bounded-read connector for SQL and MongoDB — not HTTP, not per-database drivers

> `agent_utilities/protocols/universal_connector.py`; the GraphQL sibling
> that reuses its persistence-reference/id conventions is
> `agent_utilities/protocols/source_connectors/connectors/graphql_document.py`.

CONCEPT:AU-KG.ingest.universal-data-connector

## Decision — a single bounded, read-only, driver-based connector, HTTP kept strictly out

`universal_connector.py:1-27`.

**The rejected alternative, named directly in the module docstring**: routing
database traffic through the same GraphQL/HTTP path as external-graph
sources. Rejected because keeping HTTP OUT of this class is what lets ALL
GraphQL traffic receive the central TLS/SSRF/response-size/schema-approval/
drift controls through one implementation — mixing SQL/Mongo transport into
that path would mean either weakening those controls or duplicating a SECOND,
weaker implementation of them for the database path. `GraphQLSourceAdapter`
(the schema-neutral sibling, `graphql_document.py`) owns HTTP; `UniversalConnector`
owns direct driver connections. Mutations are explicitly out of scope too —
`UniversalConnector` exposes only `read`/`health_check`/`introspect`; writes
go through the governed `ChangeEnvelope`/`MutationBatch` APIs (see
`.specify/design/kgi-change-envelope-atomic/design.md`), never a bespoke
write path bolted onto this class.

**A second load-bearing decision — explicit DSN scheme required, never
silently guessed.** `infer_kind` (`universal_connector.py:103-115`) maps a
DSN's URL scheme (`postgres`, `mysql`, `mongodb`, ...) to a backend kind and
RAISES if the scheme is absent or unrecognized; a bare path (e.g. a SQLite
file path with no scheme) must be paired with an explicit `kind="sqlite"`.
The rejected alternative is a permissive fallback (e.g. "assume Postgres if
unspecified") — rejected because a wrong silent guess against a
credential-bearing connection string is a worse failure mode than an
immediate, loud `ValueError` at connector construction.

**Lazy driver import.** Every backend driver import happens inside
`_driver()`, guarded so a missing driver raises a clear `RuntimeError`, never
a leaking `ImportError` — the module itself is importable with zero database
drivers installed. This lets the connector be declared/tested without paying
for every backend's dependency footprint upfront.

**Irreversible persistence references.** `source_ref` and per-object ids are
derived through `persistence_reference(...)` (`_object_id`,
`universal_connector.py:159-166`) — durable node identifiers never contain a
host, account, endpoint, database path, or credential; the module docstring
states this as an explicit invariant ("Connection values are runtime-only").
The rejected alternative — embedding the DSN or a derivative of it directly
in the node id — would leak connection topology into every downstream
consumer of the KG (query results, exports, logs), which is exactly what the
irreversible reference is designed to prevent.

## Risk Assessment

- **Blast Radius**: `agent_utilities/protocols/universal_connector.py`,
  `agent_utilities/protocols/source_connectors/connectors/database.py`,
  `agent_utilities/protocols/source_connectors/connectors/graphql_document.py`.
- **Backward Compatible**: Yes — adding a new SQL/Mongo backend is a driver
  registration in `_SCHEME_TO_KIND`/`SUPPORTED_KINDS`, not a new connector
  class.
- **Breaking Changes**: None.
- **Known weak point**: `infer_kind`'s scheme→kind mapping is a hardcoded
  dict; a DSN using a nonstandard or aliased scheme for a supported backend
  fails closed with a `ValueError` rather than falling back to any
  heuristic — correct behavior for safety, but it means onboarding a new DSN
  alias requires a code change, not just a config change (unlike
  `mcp_tool`'s fully declarative onboarding — see
  `.specify/design/kgi-mcp-tool-source-connector/design.md`).
