# Design Document: Any fleet MCP server becomes a KG source through config, not code

> `agent_utilities/protocols/source_connectors/connectors/mcp_tool.py`;
> narrative doc `docs/pillars/2_epistemic_knowledge_graph/KG-2.59-MCP_Tool_Source_Connector.md`.

CONCEPT:AU-KG.ingest.mcp-tool-connector

## Decision — one declarative adapter replaces hand-written per-source ingestion drivers

`mcp_tool.py:1-38`.

**The problem**: the fleet has ~58 MCP servers (sql-mcp, objectstore-mcp,
servicenow-api, salesforce, okta, ...), and the naive path to ingesting each
into the KG is a hand-written native driver per database/SaaS — one bespoke
module per source, each re-solving pagination, incremental polling, session
lifecycle, and ACL mapping independently.

**The rejected alternative is exactly that**: per-database/per-SaaS native
ingestion drivers. It is explicitly named as what this connector
"replac[es]" — rejected because it does not scale linearly with the fleet
(~58 servers and growing) and duplicates the same pagination/incremental/ACL
logic dozens of times with dozens of chances to diverge.

**The design chosen**: one declarative adapter (`source_type: "mcp_tool"`)
that turns ANY MCP server's paginated, record-listing tool into a document
source via a config dict, not a code module. It implements the full
ingestion-source contract in one place:

- **Action-routed fleet envelopes** — the fleet convention (`action` +
  `params_json`) is handled via `params_style="json"`; `params_style="args"`
  spreads params as plain tool arguments for non-fleet servers.
- **Pagination** — `cursor` (token or keyset), `page` (offset with exhaustion
  detection), or `none`, with `max_pages`/`max_records` backstops.
- **Session lifecycle** — one MCP client session per `load` run or per `poll`
  batch, reused across every page/detail call, closed cleanly.
- **Incremental poll** — `updated_since_param` binds the prior checkpoint
  watermark into tool params for server-side deltas, with an in-memory
  `updated_field` filter as the belt to that brace.
- **Two-phase list+get** — an optional `detail` call fetches each record's
  body inside the same session, with `{field}` templating from the listed
  record (e.g. objectstore `objects get`, attachment downloads).
- **Permission seam** — `acl_*` field maps project ACL-ish record fields onto
  `ExternalAccess`, feeding the ECO-4.28 permission sync.
- **SQL table sweeps** — a `sql_table` block bootstraps a keyset-paginated
  `SELECT` against sql-mcp, auto-discovering columns via `sql_schema` when
  not given.

Transport resolution is first-match: injected `client` (in-process FastMCP,
tests) → explicit `url` → explicit stdio spec → `server` name resolved
through `mcp_config.json` (the same source the multiplexer uses). No package
import of any fleet repo — runtime MCP calls only, so adding a source is
purely a config change, never a code change or a new dependency.

**A second, narrower rejected alternative — this connector is NOT the
universal answer for every workload.** The KG-2.59 doc names explicit routing
guidance the design deliberately does not override:

- **Bulk hot-path** (millions of rows, in-process): the native `database`
  connector over `UniversalConnector` (Postgres is the proven path) — see
  `.specify/design/kgi-universal-data-connector/design.md`. Routing millions
  of rows through an MCP round-trip per page would be needless overhead when
  a direct DB connection is available and safe.
- **Harvest/poll workloads** (tables, buckets, tickets, SaaS records):
  `mcp_tool` via the fleet server — the per-page MCP round-trip is negligible
  against chunking/embedding/enrichment cost, and every dialect/auth/safety
  concern stays owned by the fleet server (sql-mcp's read-only gate,
  objectstore's size caps) instead of being re-implemented here.
- **No fleet server yet**: fall back to the generic `rest`/`web`/`filesystem`
  connectors rather than blocking on a new MCP server being built first.

## Risk Assessment

- **Blast Radius**: `agent_utilities/protocols/source_connectors/connectors/mcp_tool.py`,
  `agent_utilities/knowledge_graph/ontology/connector_manifest.py`,
  `agent_utilities/knowledge_graph/extraction/second_brain_sync.py`.
- **Backward Compatible**: Yes — adding a new `mcp_tool`-sourced connector is
  a config addition; existing native drivers (e.g. `database`) are
  unaffected.
- **Breaking Changes**: None.
- **Known weak point**: since transport/behavior is entirely config-driven,
  a misconfigured `records_path`/`cursor_param`/pagination mode fails at
  RUNTIME (a bad ingest run) rather than at config-authoring time — there is
  no schema-validation step catching a malformed `mcp_tool` config before it
  is exercised against a live server.
