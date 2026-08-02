# Design Document: The full Knowledge Graph REST API is owned by the persistent API gateway process, not by the `graph-os` MCP server (now slimmed to MCP-tools-only)

CONCEPT:AU-ECO.mcp.knowledge-graph-exposure

> `agent_utilities/gateway/graph_api.py:1-16` (module docstring,
> `register_graph_routes`) and `agent_utilities/mcp/kg_server.py:1-8` (module
> docstring).

## Decision — `agent_utilities/gateway/graph_api.py` (mounted into `server/app.py`) owns every `/graph/*`, `/sessions`, `/goals`, `/tools` HTTP route, importing the SAME route-table/handler implementations `kg_server.py` defines, so the two surfaces never drift; `kg_server.py` itself is slimmed to a thin FastMCP wrapper exposing MCP tools only

Many clients on this host/network need graph access over plain HTTP —
`agent-terminal-ui`, `geniusbot`, subagents, ingestion scripts — each of which used
to be able to open the embedded graph store directly. Funnelling ALL graph HTTP
traffic through one persistent gateway process eliminates the embedded-DB file-lock
contention that arises when many independent clients each hold their own handle to
the store (`graph_api.py:6-9`). The canonical route table and handler
implementations still live in `kg_server.py` — `graph_api.py` imports and mounts
them (`graph_api.py:12-14`) rather than reimplementing them, so REST and MCP are two
thin fronts on one core, and `register_graph_routes` is the single entry point
`server/app.py` calls.

## Rejected alternative — let every client open the embedded graph store directly, or duplicate the route/handler logic between the gateway and the MCP server

Two alternatives are rejected by the decision as stated. First, the status quo it
replaced: each client process (`agent-terminal-ui`, `geniusbot`, subagents,
ingestion scripts) opening the embedded DB file directly — this is named explicitly
as the problem ("eliminates the embedded-DB file-lock contention that arises when
many clients... each open the graph store directly", `graph_api.py:7-9`); an
embedded store has no multi-writer story, so concurrent direct opens is a
correctness bug, not just a performance one. Second, the shape the fix could have
taken instead: give the gateway its OWN independent implementation of graph
query/search/ingest routes, separate from `kg_server.py`'s. That is rejected by the
explicit design constraint — "we import and mount them here so the two never drift"
(`graph_api.py:13-14`) — because two independently maintained copies of the same
route logic (REST and MCP) is exactly the kind of surface-parity drift the fleet's
"two surfaces by default" edict (`scripts/check_surface_parity.py`) exists to catch;
importing one canonical implementation makes drift structurally impossible instead
of relying on a gate to catch it after the fact.

## Risk Assessment

- **Blast Radius**: `agent_utilities/gateway/graph_api.py`,
  `agent_utilities/mcp/kg_server.py`, `agent_utilities/server/app.py`.
- **Backward Compatible**: Yes — route ownership moved, route behaviour and MCP
  tool behaviour are unchanged (same underlying handlers).
- **Known weak point**: the gateway process is now a SINGLE persistent chokepoint
  for all graph HTTP traffic; an outage of that one process removes REST graph
  access for every client that used to be able to open the store directly, trading
  file-lock safety for a single point of failure the gateway's own health/liveness
  machinery (`AU-OS.deployment.liveness-vs-readiness-split`) has to cover.
