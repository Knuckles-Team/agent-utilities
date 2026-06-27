# MCP Tool Source Connector (CONCEPT:KG-2.59)

## Overview
One declarative adapter (`source_type: "mcp_tool"`) that turns **any** MCP server's
record-listing tool into a Knowledge-Graph/Company-Brain ingestion source — sql-mcp,
objectstore-mcp, servicenow-api, salesforce, okta, the rest of the ~58-server fleet —
replacing per-database/per-SaaS native ingestion drivers with **config, not code**.
It implements the full ECO-4.25 contract: `load` (full sweep) + `poll` (checkpointed
incremental, ECO-4.26), with action-routed fleet envelopes (`action` + `params_json`),
cursor/page/keyset pagination with exhaustion detection, a tabular
`{columns, rows}` → row-dict zip (sql-mcp), a two-phase list+get `detail` call
(objectstore), ACL field maps feeding the ECO-4.28 permission sync, and one MCP
client session per run reused across every page and detail call.

## Source routing guidance
- **Bulk hot-path** (millions of rows, in-process): the native `database`
  connector over `UniversalConnector` — Postgres is the proven path.
- **Harvest / poll workloads** (tables, buckets, tickets, SaaS records): `mcp_tool`
  via the fleet server. The per-page MCP round-trip is negligible against
  chunking/embedding/enrichment cost, and every dialect/auth/safety concern stays
  in the owning fleet server (sql-mcp's read-only gate, objectstore's size caps).
- **No fleet server yet**: the generic `rest`/`web`/`filesystem` connectors.

## Config shape (all keys overridable per run)
```jsonc
{
  // transport: client (injected) | url | command/args/env | server (mcp_config.json)
  "server": "sql-mcp",
  "tool": "sql_query",                 // the MCP tool to call
  "action": "execute",                  // fleet action routing (action_param: "action")
  "params": {"sql": "...", "params": {"after": 0}, "max_rows": 500},
  "params_style": "json",              // fleet: params JSON-encoded into params_json
  "arguments": {"connection": "warehouse"},  // extra top-level tool args
  "records_path": "result",            // where rows/items live ({columns,rows} auto-zipped)
  "id_field": "id", "title_field": "title", "text_field": "body",
  "updated_field": "updated_at", "doc_type": "record",
  "acl_users_field": "owners", "acl_groups_field": "teams",  // → ExternalAccess
  "pagination": "cursor",              // none | cursor | page
  "cursor_param": "params.after",      // dotted path inside params
  "cursor_record_field": "id",         // keyset: cursor from the last record
  "more_path": "truncated",            // boolean has-more flag in the response
  "updated_since_param": "params.since",  // server-side delta on re-poll
  "batch_size": 200, "max_records": 0, "max_pages": 100
}
```

## Shipped presets (`MCP_TOOL_PRESETS` — named partial configs users extend)
- **`sql-table`** — declarative table sweep against sql-mcp: extend with
  `{"sql_table": {"table": "articles", "key_column": "id", "text_column": "body",
  "updated_column": "updated_at"}}`. Columns are discovered via `sql_schema`
  (action=columns) when not listed; a keyset-paginated `SELECT … WHERE key > :after
  ORDER BY key` is generated with identifiers validated and values bound.
- **`sql-query`** — hand-written SELECT with keyset pagination: supply
  `params.sql` (with `:after`), `cursor_record_field`, and the field map.
- **`objectstore-prefix`** — paginated `objects list` + text-mode size-capped
  `objects get` per key in the same session: extend with
  `{"params": {"bucket": "docs", "prefix": "kb/"}}`.
- **`servicenow-table`** — any Table-API table via `sysparm_offset` paging.
- **`github-repos`** — list repositories from github-mcp as `repository` documents
  (`github_repos` action=list → `data[]`; id/full_name/description/updated_at).
  Extend with the listing scope, e.g. `{"params": {"org": "Knuckles-Team"}}`.
- **`okta-users`** — list identities from okta-mcp as `identity` documents
  (`okta_users` action=list → `data[]`; the client auto-follows Okta's `after`
  cursor, login/email under the nested `profile`). Extend with a filter, e.g.
  `{"params": {"filter": "status eq \"ACTIVE\""}}`.
- **`keycloak-users`** — list realm identities from keycloak-mcp as `identity`
  documents (`keycloak_agent_users` action=list_users → bare user array, so
  `records_path` is ""; flat id/username/email). Realm required, e.g.
  `{"params": {"realm": "master"}}`.

SaaS sources without a shipped preset follow the same pattern — point the config at
the fleet server's listing tool:
- **Salesforce**: `{"server": "salesforce-mcp", "tool": "<soql tool>",
  "params": {"query": "SELECT Id, Subject, Description FROM Case"},
  "records_path": "records", "id_field": "Id", "text_field": "Description",
  "updated_field": "LastModifiedDate", "pagination": "cursor",
  "cursor_path": "nextRecordsUrl"}`.
- **Okta**: `{"server": "okta-mcp", "tool": "<users tool>", "id_field": "id",
  "text_field": "profile.email", "updated_field": "lastUpdated",
  "pagination": "cursor", "cursor_param": "after"}`.

## Wire-First entry points
`source_connector` MCP tool / `POST /connector/run` / `kg.ontology.run_connector`
→ `ContentType.CONNECTOR` ingestion adaptor → KG-2.48 `DocumentProcessor`
(contextual enrichment KG-2.50, permission sync ECO-4.28, checkpoint persistence
KG-2.8). Live-path test:
`tests/integration/protocols/test_mcp_tool_connector_live_path.py` (in-process
FastMCP server with canned sql-mcp / objectstore-mcp envelopes).

## Write-back — the symmetric side (CONCEPT:KG-2.42)

Ingestion *reads* records out of the fleet; the symmetric path *writes* a mutation
back to the source of record. `call_tool_once()` (same module) is the write-side
twin of the connector — one fleet tool call, reusing the identical transport
resolution and `action`+`params_json` assembly — and the governed
**`fleet.write_record`** ontology action (`knowledge_graph/actions/fleet_writeback.py`)
wraps it so every external write runs through the action executor: authorization
(`required_capability="fleet_write"`), approval-gateability, and an audited
`ActionInvocation`. A KG decision can thus update ServiceNow / ERPNext / any fleet
system through the same tools ingestion reads from — turning the KG into a *system
of action*, not just a read model. The caller supplies the exact
server/tool/action/params (no per-vendor write path is hand-coded):

```python
execute_action("fleet.write_record", {
    "server": "servicenow-mcp", "tool": "servicenow_table_api",
    "action": "patch_table_record",
    "params": {"table": "incident", "sys_id": "...", "state": "6"},
})
```

## Connector dual-role — every fleet connector is also an ingestion source
The same `*-mcp` server that exposes a **live MCP tool** is also an **ingestion
source**: any connector whose surface includes a record-listing tool gets a
declarative entry in `MCP_TOOL_PRESETS` (server + tool + action + field-map +
pagination + OWL `doc_type`), so a connector repo is BOTH "call the tool now" AND
"sweep it into the KG" with no new transport code. Beyond the database/SaaS/ops
presets, the catalog covers connector dual-role presets such as `jellyfin-media`,
`rom-manager-roms`, `listmonk-subscribers`/`listmonk-campaigns`, `wger-routines`,
`ansible-tower-inventories`/`-job-templates`/`-projects`, `camunda-tasks`,
`portainer-containers`, `salesforce-sobject` (SOQL-driven), and `erpnext-doctype`
(doctype-driven). Connectors whose surface is pure action/compute/config
(stirlingpdf, vector, caddy, container-manager, systems-manager, clarity) or whose
records carry no text body (kafka topic metadata) have **no preset** by design.

### Per-repo preset contribution (the "2 actions from the same repo" seam)
A connector can ship its OWN ingestion preset **beside its MCP tool** instead of
adding it to the central dict, via the same data-only entry-point pattern the hub
uses for skills/prompts (CONCEPT:OS-5.52):

```toml
# in the connector package's pyproject.toml
[project.entry-points."agent_utilities.source_connector_providers"]
jellyfin-mcp = "jellyfin_mcp.ingestion"
```

The named data subpackage contains a `mcp_source_presets.json` file — a JSON object
of `{preset_name: {server, tool, action, …}}` with the exact `MCP_TOOL_PRESETS`
schema. The hub resolves it via `importlib.resources` (it never imports the
connector's business logic), merges it into the catalog with **contributed presets
taking precedence** over the central dict (they live with the connector and track
its tool surface), and the central dict is the fallback. Discovery is
failure-isolated and cached. Accessors: `get_tool_preset` / `list_tool_presets` /
`all_tool_presets` consult the merged catalog; `reset_contributed_presets_cache()`
clears it after an install.

## Implementation Details
- **Source Code**: `agent_utilities/protocols/source_connectors/connectors/mcp_tool.py`
  (`McpToolSourceConnector`, `McpToolSourceError`, `MCP_TOOL_PRESETS`,
  `SOURCE_PRESET_PROVIDER_GROUP`, `get_tool_preset`/`list_tool_presets`/`all_tool_presets`,
  `call_tool_once`);
  write-back action in `agent_utilities/knowledge_graph/actions/fleet_writeback.py`
  (`fleet.write_record`, registered into `DEFAULT_REGISTRY`)
- **Tests**: `tests/unit/protocols/test_mcp_tool_source_connector.py`,
  `tests/integration/protocols/test_mcp_tool_connector_live_path.py`,
  `tests/unit/knowledge_graph/test_fleet_writeback.py`
- **Pillar**: KG
