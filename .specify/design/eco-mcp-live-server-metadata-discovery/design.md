# Design Document: Discover a configured MCP server's real tool metadata by briefly starting it and calling `list_tools()`, instead of trusting the config's static tool-flag declarations

CONCEPT:AU-ECO.mcp.live-server-metadata-cache

> `agent_utilities/knowledge_graph/core/engine_mcp_discovery.py:216-260`
> (`discover_mcp_tools`).

## Decision — `discover_mcp_tools` starts a declared MCP server (stdio or remote) through the canonical multiplexer probe, calls its real `list_tools()`, and releases the child immediately, applying the SAME `AgentConfig` TLS/auth/egress/environment boundary GraphOS itself uses

Ingesting a fleet MCP server's capabilities into the KG needs real tool metadata —
names, descriptions, input schemas, annotations — not a guess. `discover_mcp_tools`
gets that by making the server actually answer: it normalizes the server config
(`parse_mcp_config`), starts it through `MCPMultiplexer.probe_declaration`
(`engine_mcp_discovery.py:245-249`) — the SAME canonical probe path the
multiplexer itself uses to mount children — and returns the tools it reports,
under a bounded, validated timeout (`_MAX_DISCOVERY_TIMEOUT_SECONDS`). "The canonical
multiplexer probe owns both stdio and remote transports. It applies the same
AgentConfig TLS/auth/egress/environment boundary as GraphOS" (`engine_mcp_discovery.py:222-224`)
is the load-bearing property: discovery is not a second, looser code path that
happens to also connect to MCP servers, it reuses the exact security boundary the
production multiplexer enforces, then releases the child rather than keeping it
resident.

## Rejected alternative — parse the server's declared tool-flag configuration statically, without ever starting it

An MCP server declaration in `mcp_config.json` already carries some static
metadata (command, args, env, declared tool flags). The cheaper alternative is to
trust that declaration and synthesize tool metadata from it without ever actually
running the server. That was rejected because a static declaration can drift from
what the server actually implements — new tools added, old ones removed, schemas
changed — invisibly to anything reading only the config file. Actually starting the
server and calling its real `list_tools()` gets ground truth at the cost of
briefly running the process; reusing the existing probe/boundary infrastructure
(rather than a bespoke lighter-weight "just enough to list tools" client) keeps
that cost from requiring a second, separately-secured connection path.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/engine_mcp_discovery.py`,
  `agent_utilities/mcp/multiplexer.py` (`probe_declaration`).
- **Backward Compatible**: Yes — discovery is invoked explicitly (ingestion time),
  not on every multiplexer mount.
- **Known weak point**: discovery briefly starts every declared server it is asked
  to probe — a server with real side effects on startup (not just tool
  registration) pays that cost on every discovery run, and a slow-starting server
  can consume a meaningful share of the bounded timeout before `list_tools()` is
  even reachable.
