# Design Document: A single-server tool loop is bounded by wall-clock time, not just model round-trips

CONCEPT:AU-ORCH.execution.delegation-wall-clock

> `agent_utilities/orchestration/agent_runner.py:2809-2816` (the
> `_EXECUTE_AGENT_WALL_CLOCK_S` constant) and `agent_runner.py:3082-3090`
> (where it wraps the focused-tools direct tool loop). Introduced by commit
> `227c4487` ("delegation wall-clock timeout (fail-loud on hang)").

## Decision — cap the focused-tools single-server loop at a hard 300s wall clock, independent of `pydantic_ai`'s `UsageLimits`

The single-server direct tool loop already has `UsageLimits` in place
(`request_limit` derived from `max_steps`, `total_tokens_limit` from
`invoker_budget_tokens`) — but those bound *model round-trips and tokens*,
not *elapsed time*. A fleet tool that blocks — the docstring's example: a
systems-manager telemetry call shelling out to a stuck host command — hangs
the entire delegation for however long the underlying MCP client is willing
to wait (observed in production: 1800s), tying up engine connections for the
duration and leaving the caller with no signal other than eventually timing
out at the client layer.

`_EXECUTE_AGENT_WALL_CLOCK_S = 300.0` wraps the tool loop with an explicit
wall-clock timeout well below that 1800s client ceiling, so a blocking tool
fails loud in minutes instead of hanging for the client's full budget. The
value is a named module constant, explicitly NOT exposed as a per-call
knob — "one correct value, auto-behaviour," matching the codebase's
Configuration discipline elsewhere (the same file's `_MAX_BOUND_TOOLS = 20`
constant right above it uses the identical framing).

**The rejected alternative is relying on `UsageLimits` alone.** Round-trip and
token caps bound *how much the model does*, not *how long a single tool call
is allowed to block* — a tool that never returns burns zero round-trips and
zero tokens while it hangs, so neither limit ever fires. The two mechanisms
are complementary, not substitutable: `UsageLimits` stops a chatty loop from
running forever in small steps; the wall clock stops a single blocked step
from running forever in one step. A companion decision recorded at the same
site,`CONCEPT:AU-AHE.harness.runtime-reliability-loop`, treats a completed run
that ate ≥80% of this budget (`_DELEGATION_BUDGET_WARN_FRACTION`) as a
"slow-not-wrong" reliability signal even when it didn't time out — a
run that always finishes but always eats most of its budget would otherwise
never be flagged by the pass/fail reward signal.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/agent_runner.py` (the
  focused-tools/single-server dispatch path only — the toolless multi-agent
  graph path has its own separate timeout handling).
- **Backward Compatible**: Yes — a legitimate multi-step, multi-tool run that
  previously completed in well under 300s is unaffected; only a genuinely
  hung tool call now fails instead of hanging indefinitely.
- **Known weak point**: 300s is a single global constant for every
  single-server delegation regardless of task shape — a legitimately slow but
  healthy tool (e.g. a large batch export) that needs longer than 300s has no
  per-call override and will be killed exactly like a hung one, distinguished
  only after the fact by the caller inspecting why it failed.
