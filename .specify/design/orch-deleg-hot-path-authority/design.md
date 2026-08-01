# Design Document: Authority renewal moves to the convergence point every delegation passes through, not to each entrypoint individually

CONCEPT:AU-ORCH.execution.delegation-hot-path-authority

> `agent_utilities/mcp/kg_server.py:3314-3340`
> (`authority_keepalive_scope`) and its adoption at
> `agent_utilities/orchestration/manager.py:512-530`
> (`Orchestrator.execute_agent`). Introduced by commit `4bdd4877`
> ("renew authority at execute_agent, the real convergence point (D-SNV-5
> follow-up)").

## Decision — open the authority-renewal scope inside `Orchestrator.execute_agent` itself, since every delegation entrypoint converges there

A long-running delegation needs its session authority renewed periodically or
it expires mid-flight with `SessionExpiredError`. The original fix (D-SNV-5)
wired a background keepalive into `_execute_tool`, the MCP dispatch path —
which covers MCP-tool-dispatched delegations, but nothing else. The docstring
names every OTHER entrypoint that reaches the same execution: the
`agent-webui`/REST gateway, the messaging router (Telegram/Mattermost), the
autonomous `agent_dispatch_worker`, `org_runtime`, a governed dynamic
workflow, and the parallel engine. Each of those still expired mid-flight,
because none of them went through `_execute_tool`.

All of those entrypoints converge on exactly one function:
`Orchestrator.execute_agent`. Rather than duplicate the keepalive wiring at
every entrypoint (six-plus call sites, each needing to remember to do it),
`execute_agent` now opens `authority_keepalive_scope` itself
(`manager.py:512`), so the renewal loop runs regardless of which surface the
call came from. The scope is explicitly idempotent/reentrant: a call that
already arrived through the MCP dispatch guard (which opens the same scope)
does not start a second renewal loop — checked by session identity, not by
"am I in the MCP path."

**The rejected alternative is what shipped first and is being corrected
here**: wire the keepalive into each entrypoint individually. That was tried
(the original D-SNV-5 fix at `_execute_tool`) and is the reason this follow-up
exists — a per-entrypoint fix only ever covers the entrypoints someone
remembered to wire, and a new entrypoint (or an existing one nobody thought
of, like `org_runtime`) silently inherits the bug. Moving the fix to the
convergence point makes it structurally impossible to add a new delegation
surface that bypasses it, since anything that doesn't go through
`execute_agent` isn't a delegation in this system at all.

## Risk Assessment

- **Blast Radius**: `agent_utilities/mcp/kg_server.py`,
  `agent_utilities/orchestration/manager.py`.
- **Backward Compatible**: Yes — a caller with a static bearer JWT (no
  server-held `credential_lease`) has nothing to renew; the scope is a no-op
  for that case, checked internally rather than requiring the caller to know
  which authority model it's using.
- **Known weak point**: the fix trusts that `execute_agent` really is the
  single convergence point for all delegation surfaces present and future — a
  new execution path added later that calls into the graph/engine without
  going through `Orchestrator.execute_agent` would silently regress to the
  original per-entrypoint bug with no test currently guarding against that
  specific case.
