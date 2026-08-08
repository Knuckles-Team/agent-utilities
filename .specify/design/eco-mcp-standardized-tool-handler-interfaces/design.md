# Design Document: MCP tool handlers dispatch to provider clients through one standardized concurrency + kwarg-normalizing boundary, instead of each connector reimplementing thread-offload and body-param wiring itself

CONCEPT:AU-ECO.mcp.standardized-interfaces

> `agent_utilities/mcp/concurrency.py:1-70` (`invoke_client_method`,
> `_wrap_data_kwargs`); package overview `agent_utilities/mcp/__init__.py:1-12`.

## Decision — every action-routed MCP tool handler calls provider-client methods through `invoke_client_method`, which awaits an async SDK method directly and offloads a synchronous one to a worker thread, and which self-heals the common "LLM passed flat kwargs, the client wants one body dict" mismatch via `_wrap_data_kwargs`

Fleet MCP tools are almost all `async def` (they `await ctx.*` helpers), but the
provider client SDKs they call into are a mix of sync and async across dozens of
connectors. Calling a blocking client method inline on the event loop stalls every
other concurrent request on the worker (`concurrency.py:3-6`) — the same
event-loop-starvation failure mode `AU-ECO.mcp.gateway-dispatch-isolation` fixes at
the top-level dispatch chokepoint, fixed here at the provider-client call boundary
instead. `invoke_client_method` is that ONE boundary every handler is meant to call
through. Layered on top, `_wrap_data_kwargs` fixes a second, unrelated but equally
common failure: an LLM naturally passes a create/update call's fields flat
(`{project_id, name, description}`, mirroring the REST payload shape) while many
client methods take that payload as a single named dict param (`data`/`payload`/`body`
— `concurrency.py:30-31`); when a target method declares exactly one such param and
none was supplied, the extra flat kwargs are folded into it automatically, making the
LLM's natural calling convention just work instead of crashing with `unexpected
keyword argument`.

## Rejected alternative — let each connector call its own client directly, sync or async, and handle its own body-param shape

Nothing in FastMCP forces tool handlers through a shared invocation boundary — each
of the dozens of fleet connectors could call its client SDK directly and, where
needed, write its own `run_in_executor`/thread-offload logic for synchronous
methods, plus its own kwarg→body-dict translation. That is close to what existed
before this module: ad hoc, per-connector handling of the same two problems,
repeated across every connector that happened to hit them. `_wrap_data_kwargs`'s own
design note states the generalization explicitly is safe because it is "a strict
no-op for every method without such a param (the overwhelming majority)"
(`concurrency.py:45-47`) — meaning centralizing it costs nothing for connectors that
never needed it, while removing a whole class of "delegated create/update fails
because the LLM passed the payload flat" bugs from every connector that does, without
each one having to discover and fix the pattern independently.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/concurrency.py`; every action-routed
  fleet MCP tool handler that calls a provider client method.
- **Backward Compatible**: Yes — `invoke_client_method` preserves each method's own
  return value/exception behaviour; `_wrap_data_kwargs` only folds extras when a
  target's single body param is unambiguous and not already supplied.
- **Known weak point**: `_BODY_PARAM_NAMES` (`data`/`payload`/`body`) is a closed,
  hand-picked list of the conventions observed across today's client SDKs
  (`concurrency.py:29-31`) — a future connector whose client uses a different body
  param name silently gets no folding and reproduces the original crash.
