# Design Document: Wrap every multiplexer child MCP server in its own bounded-concurrency, pooled, breaker-protected `ChildRuntime`, instead of one shared session per child with no isolation

CONCEPT:AU-ECO.mcp.profile-differences-from-client

> `agent_utilities/mcp/child_resilience.py:1-33` (module docstring, `ChildRuntime`);
> config surface `agent_utilities/core/config.py:3101-3120`
> (`mcp_child_max_concurrency` et al., "per-server override... on the server's
> `mcp_config.json` entry" — a per-child config **profile** that differs from the
> multiplexer's own global default, which the concept id names); always-present
> status tool `agent_utilities/mcp/multiplexer.py:3877`
> (`_register_status_tool`/`multiplexer_status`).

## Decision — each of the multiplexer's ~50 aggregated child MCP servers gets its own `ChildRuntime`: a bounded `asyncio.Semaphore` (with a queue timeout), an optional round-robin session pool, cancellation-safe shielded dispatch, restart-on-crash with backoff, and a per-child circuit breaker — each independently overridable from the global default via a per-server config profile in `mcp_config.json`, with an always-on `multiplexer_status` tool reporting every child's live state

Before this module, "the multiplexer aggregates ~50 child MCP servers behind one
endpoint" meant every child was "a single shared `ClientSession` with no
concurrency control" (`child_resilience.py:5-6`). `ChildRuntime` fixes that at the
per-child boundary: bounded concurrency so one child cannot monopolize the shared
thread/connection budget (`MCP_CHILD_MAX_CONCURRENCY`, overridable per-server);
session pools for remote children that benefit from N round-robin connections;
cancellation-safe dispatch so a caller's timeout/cancel does not corrupt the shared
session's request/response bookkeeping; a supervised restart cycle with exponential
backoff, parking a child as `failed` after too many restarts in a window; and a
circuit breaker (reusing the shared OS-5.23 engine-client breaker state machine,
subclassed for per-child wording) that short-circuits calls to a known-bad child
instead of hammering it. `multiplexer_status` (`multiplexer.py:3877`) is the always
present meta-tool that surfaces every child's state/restart-count/limits/in-flight
count so an operator or agent can see this hardening working (or not) without
guessing.

## Rejected alternative — one shared `ClientSession` per child with no per-child isolation (the pre-existing state this module replaced)

The rejected alternative is not hypothetical — it is stated as the prior, actually
shipped behaviour: "before this module, every child was a single shared
`ClientSession` with no concurrency control: one slow or wedged child head-of-line
blocked every caller, and a crashed child hard-failed all of its tools until the
whole multiplexer was restarted" (`child_resilience.py:5-9`). That shape was
rejected/replaced because a single wedged or crashed child among ~50 should be a
localized failure, not one that degrades or requires restarting the WHOLE
multiplexer — the same "one bad actor takes down everyone sharing the resource"
failure mode `AU-ECO.mcp.gateway-dispatch-isolation` fixes at the dispatch-thread
level, fixed here at the per-child connection/session level instead.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/child_resilience.py`,
  `agent_utilities/mcp/multiplexer.py`, `agent_utilities/core/config.py`
  (`mcp_child_*` settings), `agent_utilities/observability/gateway_metrics.py`.
- **Backward Compatible**: Yes — defaults preserve working behaviour for every
  child; per-server overrides are opt-in via `mcp_config.json`.
- **Known weak point**: the circuit breaker and restart bookkeeping are per-child,
  in-process state — a multiplexer restart resets restart-window/breaker counters,
  so a child that was mid-way through being parked as `failed` gets a fresh count
  after any multiplexer restart, independent of whether the underlying child
  itself actually recovered.
