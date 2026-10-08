# MCP Tool Surface — the single intent-tool contract

Every MCP server built on agent-utilities serves one tool contract. No mode
switch exists. graph-os and every fleet connector list a small set of
**intent tools**. Each intent tool takes the same condensed input.

| Field | Meaning |
|---|---|
| `action` | The operation id: `"<tool>.<op>"`, `"<tool>"` for a single-operation tool, or `"describe"`. |
| `params` | The operation's arguments as one JSON object. |
| `intent` | Optional natural language. With no `action`, the server routes the intent. |

`action="describe"` lists the operations of an intent tool.
`params={"action": "<operation id>"}` returns the full argument schema of one
operation. Per-operation schemas are served on demand, not in `tools/list`.

The authoritative code is `agent_utilities/mcp/intent_contract.py`.

## graph-os

graph-os lists six intent verbs: `ask`, `find`, `write`, `act`, `manage` and
`why`. It also lists the two MCP Apps launchers that the MCP Apps extension
requires. The granular `graph_*` and `engine_*` tools register on a private
backing server. They dispatch through `_execute_tool`, and no client sees them
in `tools/list`. Fleet access runs through `find`, `act(action="fleet.call")`
and `manage(action="fleet.load" | "fleet.unload")`. Every mutation carries a
preview and a `plan_ref`. Approval-required operations need a session approval
through `manage(action="approve")`.

See `agent_utilities/mcp/graphos_surface.py`.

## Fleet connectors

The shared builder `register_tool_surface` (in
`agent_utilities/mcp/verbose_tools.py`) serves three intent tools on every
connector:

- `find` ranks, lists or describes operations.
- `ask` runs a read-only operation.
- `act` runs any operation, including mutations.

Operation ids reuse the names the connector tools always had. A former call
`servicenow_cmdb(action="get_instance", …)` becomes
`act(action="servicenow_cmdb.get_instance", params={…})`. Each backing tool
keeps its own destructive-operation confirmation.

See `agent_utilities/mcp/connector_surface.py`.

## Backing tools and the typed tier

`register_verbose_tools` introspects the API client class and registers one
backing operation per client method. A normalized operation manifest gives a
method a typed signature. Without a manifest entry, the method takes one
`params_json` argument. Source the manifest from an OpenAPI spec first, then a
crawled documentation site, then a PDF spec. The `api-client-builder` and
`agent-package-builder` skills own acquisition and code generation.

## Surface gates

The tests enforce a small listing. A server lists at most 10 tools. The whole
`tools/list` stays within 4000 tokens, and each tool within 500 tokens. Every
manifest operation stays reachable through the router.

CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse — one MCP tool contract.
CONCEPT:AU-ECO.mcp.tool-mode-standardization — the backing-tool generator (verbose 1:1 operations).
