# Design Document: Cross-cutting MCP concerns (auth context, entity linking) are FastMCP `Middleware` classes, not per-tool decorators or inline checks

CONCEPT:AU-ECO.mcp.fastmcp-middleware

> `agent_utilities/mcp/middlewares.py` — `AuthContextMiddleware` (~line 150) and
> `EntityLinkingMiddleware` (`middlewares.py:177`).

## Decision — assimilate FastMCP's `Middleware`/`on_request` interception layer as the ONE place cross-cutting MCP behaviour lives, instead of repeating it inside every tool function

Every MCP tool call needs some behaviour that has nothing to do with the tool's own
logic: minting a `GraphSession` from the caller's claims and binding it to the
request's contextvars (`AuthContextMiddleware`), or, going forward, generic
cross-entity relationship resolution when a tool writes to the KG
(`EntityLinkingMiddleware`). `middlewares.py:177` names this explicitly:
*"Assimilated from FastMCP's 'cross-cutting concern interception layer'"* — i.e. the
module deliberately adopts FastMCP's own `Middleware` base class and
`on_request`/`call_next` chain-of-responsibility shape rather than inventing a
parallel one. `AuthContextMiddleware.on_request` (`middlewares.py:150`) reads the
request's claims once, mints the actor/session, sets them as contextvars for the
duration of the call via `set_actor`/`set_session`, and resets them in a `finally` —
every tool downstream of it, current and future, gets an authenticated actor/session
for free without importing or calling anything itself.

## Rejected alternative — a decorator (or inline call) added to every tool function

The obvious alternative is what most of the codebase's own action-routed tools still
do for their *own* logic: a shared helper function each tool calls at its own top
(the pattern `invoke_client_method`/`_wrap_data_kwargs` use for the concurrency
boundary, `AU-ECO.mcp.standardized-interfaces`). That shape was rejected specifically
for auth-context binding and entity linking because those are concerns about the
**request**, not about any one tool's business logic — a decorator only fires for
tools that remember to apply it, and an MCP server registers tools dynamically from
many modules (condensed tools, verbose auto-wire, per-connector registrars), so
"every tool remembered the decorator" is not a property anything enforces. FastMCP's
middleware chain runs for every request that reaches the server's dispatch path
regardless of which module registered the tool, structurally closing the gap a
per-tool decorator can only close by convention.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/middlewares.py`, every MCP server built via
  `server_factory.create_mcp_server` (all `mcp.add_middleware(...)` call sites).
- **Backward Compatible**: Yes — middleware wraps existing dispatch; tools are
  unmodified.
- **Known weak point**: `EntityLinkingMiddleware.on_request` (`middlewares.py:177`)
  is currently a logged no-op stub ("Example interception") for `kg_write` calls —
  the assimilation of the *mechanism* is real and load-bearing (auth context), but
  the entity-linking *behaviour* it was named for is not implemented yet.
