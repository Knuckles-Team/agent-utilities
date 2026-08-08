# Design Document: Run synchronous MCP tool handlers on a worker thread with a bounded timeout, instead of calling them inline on the gateway's one asyncio loop

CONCEPT:AU-ECO.mcp.gateway-dispatch-isolation

> `agent_utilities/mcp/kg_server.py:206-223` (`_TOOL_CALL_TIMEOUT_S` / the dispatch
> wrapper around `tool_func`).

## Decision — dispatch every synchronous `graph_*`/`engine_*` tool handler through `asyncio.to_thread` under a 320-second timeout, on the ONE gateway event loop that every connected MCP client shares

Most `graph_*`/`engine_*` tools are plain synchronous functions that do blocking
engine I/O. The `graph-os` MCP server, like every FastMCP server here, runs a single
asyncio event loop shared by every connected client. Calling a sync tool inline on
that loop blocks the loop for the duration of the call — an uncompiled engine
surface, a bad action, or a wedged backend does not just fail that one caller, it
**freezes the whole graph-os child and disconnects every other connected MCP
client** at the same time (`kg_server.py:207-208`). The fix wraps dispatch so
synchronous tools run on a worker thread (`to_thread`, which propagates the current
contextvars — actor/session — so auth context survives the hop) and every call is
bounded by `_TOOL_CALL_TIMEOUT_S = 320.0`, deliberately set **longer** than the
delegation wall-clock so a legitimate long-running delegation through
`execute_agent` is never killed by this timeout first. A hung tool now fails loud
with a timeout error and the loop is freed immediately, instead of the whole server
going dark.

## Rejected alternative — leave sync tools inline and rely on each tool's own internal timeout/error handling

Before this, correctness depended on every tool implementation individually being
well-behaved — never blocking indefinitely, never wedging on a dead backend. That
is exactly the failure mode the docstring names as the motivating case ("an
uncompiled engine surface, a bad action, or a wedged backend"): a SINGLE
misbehaving tool call, anywhere in a surface of ~95 tools maintained by many
different modules, degrades every other client's session. Fixing it per-tool would
mean auditing and timeout-wrapping every current and future `graph_*`/`engine_*`
handler individually — the same "N places to remember" shape rejected for the
sibling authority-renewal decision (`AU-ORCH.execution.delegation-hot-path-authority`).
Moving the isolation to the ONE dispatch chokepoint every tool call passes through
makes it structurally impossible for a newly added sync tool to reintroduce the
freeze, without that tool's author having to know this failure mode exists.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/kg_server.py` (the shared `_execute_tool`
  dispatch core every condensed/verbose/intent tool and REST route ultimately calls
  into).
- **Backward Compatible**: Yes — async tools are unaffected (already off the
  blocking path); sync tools behave identically except now bounded and
  thread-isolated.
- **Known weak point**: the 320s ceiling is a single global constant, not
  per-tool-tunable — a legitimately slower bulk operation either has to finish
  under 320s or be restructured as async/streaming; there is no per-tool override.
