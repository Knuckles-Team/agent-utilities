# Design Document: A failed remote MCP child surfaces its real cause, by recursively flattening anyio's ExceptionGroup instead of reporting the group's opaque `str()`

CONCEPT:AU-ORCH.routing.mcp-child-error-unwrap

> Realised by `agent_utilities/orchestration/agent_runner.py:233-266`
> (`_flatten_exception_group`), called from
> `agent_utilities/graph/_router_impl.py:1471-1486` inside
> `expert_executor_step`'s exception handler. Introduced by commit `a79bced3`
> ("Fix agent-stack MCP seams").

## Decision — unwrap at the point where the error becomes user-visible, and de-duplicate the leaves

When an expert step fails by calling a remote MCP tool, the failure arrives
wrapped in anyio's `BaseExceptionGroup`, because the call ran inside a task
group. `BaseExceptionGroup.__str__` produces *"unhandled errors in a TaskGroup
(N sub-exceptions)"* — a message that names neither the failing tool, nor the
error type, nor anything actionable. That string was what the orchestration
layer reported and what a user or an operator saw.

`_flatten_exception_group` walks the group recursively — groups can nest —
and produces de-duplicated `"<ExcType>: <msg>"` leaf strings. The
de-duplication matters because a fan-out to N children that all fail the same
way (one unreachable server, one expired credential) yields N identical leaves,
and repeating the same message N times is barely more informative than the
opaque group string it replaced.

**The rejected alternative is the prior behaviour: report the group as-is.**
The introducing commit states the fix directly — *"BaseExceptionGroup
unwrapping (clear remote-child error instead of 'unhandled errors in a
TaskGroup')."* Reporting the group is what happens by default and costs nothing
to implement; it was rejected because it converts every distinct remote failure
into one indistinguishable message, which makes the entire class of remote-MCP
failures undiagnosable from the trace alone.

## Scope warning — this concept id is over-applied in `agent_runner.py`

This document covers the exception-unwrapping decision only. The same marker
string appears on several unrelated sites in `agent_runner.py` (the module
docstring at `:1`, `run_agent` at `:600`, `_spawn_auth` at `:2354`,
`_execute_single_server` at `:2970`/`:3020`, `_record_execution_trace` at
`:3556`). Those are residue from the OKF-CIS bulk rename, where the old
umbrella id `ORCH-1.21` ("KG-to-LLM Execution Bridge") had been applied loosely
across the whole module. They cover genuinely different features — auth token
refresh, the deterministic single-server bypass (introduced separately by
commit `2463e45d`), and provenance writes — and are **not** part of this
decision. They should be re-pointed at their own concepts rather than read as
evidence about error unwrapping.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/agent_runner.py`,
  `agent_utilities/graph/_router_impl.py`.
- **Backward Compatible**: Yes — this changes the *text* of a reported error,
  not control flow. The exception still propagates.
- **Known weak point**: flattening discards the group's structure. Which child
  raised which leaf, and how the failures nested, are lost — the output is a
  set of leaf strings, so a partial fan-out failure reads the same as a total
  one. Recovering that would need the tool/child identity carried alongside
  each leaf, which this does not do.
