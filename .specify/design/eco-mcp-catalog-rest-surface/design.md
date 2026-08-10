# Design Document: MCP fleet catalog gets a REST twin

CONCEPT:AU-ECO.mcp.catalog-rest-surface

> `agent_utilities/server/routers/mcp_catalog.py` (GOC-60-W03)

## Decision — REST route that dispatches into the SAME multiplexer methods the MCP meta-tools call

`agent_utilities/mcp/multiplexer.py` already computes the fleet's dispatchable
truth for the `find_tools`/`load_tools`/`list_catalog`/`multiplexer_status` MCP
meta-tools (`list_catalog` at `multiplexer.py:3728`, `multiplexer_status` at
`:4314`). It had no REST route at all — a violation of this repository's own
*Two surfaces by default* rule (`AGENTS.md`: every capability reachable via the
gateway AND MCP). `mcp_catalog.py` is that REST route. It dispatches into the
same `MCPMultiplexer.list_catalog`/`status_snapshot` methods the MCP tools call
(via the process-wide standalone instance in
`agent_utilities.mcp.shared_multiplexer` —
see `CONCEPT:AU-ECO.mcp.shared-multiplexer-singleton`), never re-deriving the
listing logic. Authorization mirrors the MCP-tool-side gate
(`multiplexer._require_fleet_capability("discover")`) so both surfaces require
the same capability, not merely serve the same payload shape. A multiplexer or
catalog failure surfaces as a typed 503 with `status: "DEGRADED"` — never a
silent empty list or generic 500 (GOC-60 lane invariant 1,
`plans/graph-os-completion-program/lanes/GOC-60-acute-user-surface-restoration.md`
§GOC-60-W03).

**The rejected alternative** was a second, REST-native listing implementation
(e.g. reading `mcp_config.json` directly, or a hand-rolled KG query) — that
would drift from the MCP tool payload the moment either surface's logic
changed, and there is a documented parity requirement (GOC-60-W03 acceptance
evidence: "the REST payload equals the MCP tool payload for the same
session") that only holds if both surfaces share one source of truth.

## Risk Assessment

- **Blast Radius**: `agent_utilities/server/routers/mcp_catalog.py`,
  `agent_utilities/mcp/shared_multiplexer.py`, `agent_utilities/mcp/multiplexer.py`
  (read-only from this surface's perspective).
- **Backward Compatible**: Yes — additive route, no existing endpoint changed.
- **Known weak point**: authorization is a hand-mirrored capability check
  (`_require_fleet_capability("discover")`), not a shared decorator — the two
  surfaces could drift apart if one side's gate changes without the other.
