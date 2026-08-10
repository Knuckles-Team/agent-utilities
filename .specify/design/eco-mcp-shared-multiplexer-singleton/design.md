# Design Document: One process-wide standalone MCPMultiplexer for non-serving consumers

CONCEPT:AU-ECO.mcp.shared-multiplexer-singleton

> `agent_utilities/mcp/shared_multiplexer.py` (GOC-60-W03/W04b)

## Decision — a single shared instance, not one per consumer

`attach_fleet_loader` builds a fresh `MCPMultiplexer` for a directly-served
graph-os MCP process (`mcp_server()` in `kg_server.py`) — that process owns the
FastMCP serving loop the multiplexer's live-forwarder bookkeeping is normally
wired against. Two OTHER consumers need the same dispatchable-truth catalog
with no serving loop of their own: the REST twin of `list_catalog`/
`multiplexer_status` (`CONCEPT:AU-ECO.mcp.catalog-rest-surface`, GOC-60-W03),
and the WebUI's governed MCP delegation seam
(`CONCEPT:AU-ECO.mcp.webui-governed-mcp-delegation`,
`agent_utilities/server/webui_mcp_delegation.py`, GOC-60-W04b) plus its
MCP-servers inventory panel, consuming this in-process rather than over HTTP.

`MCPMultiplexer` is explicitly designed to support this — its `_host_mcp`
attribute is documented as optional specifically to "preserve the standalone
probe and unit-test paths, which do not own a FastMCP server or live
forwarders" (`multiplexer.py`). `shared_multiplexer.py` gives those consumers
ONE shared instance rather than several independent ones, so a REST payload,
an MCP payload, and the WebUI's own in-process read all come from the SAME
catalog/probe-cache/session-visibility state instead of drifting against each
other — the whole point of "the REST payload equals the MCP tool payload for
the same session" (GOC-60-W03 acceptance evidence).

**The rejected alternative** was letting each new non-serving consumer build
its own `MCPMultiplexer(...)` instance. That would triple the probe-cache
population cost, and — worse — let each instance's session-visibility state
diverge (a tool mounted in one instance's view but not another's), silently
breaking the parity guarantee GOC-60-W03 exists to establish.
`multiplexer.py` itself is read-only from this lane's perspective (GOC-60
worker instructions): this module only ever consumes its public constructor
and methods, never modifies them.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/shared_multiplexer.py`,
  `agent_utilities/server/routers/mcp_catalog.py`,
  `agent_utilities/server/webui_mcp_delegation.py`.
- **Backward Compatible**: Yes — additive module, the served-process path
  (`attach_fleet_loader`) is untouched.
- **Known weak point**: a process-wide singleton is shared mutable state — a
  bug in one consumer's use of the shared instance (e.g. holding a stale
  reference across a reload) affects every other consumer.
