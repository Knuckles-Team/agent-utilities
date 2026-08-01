# Design Document: A named-server delegation that can't reach its real tools fails loud — it never falls through to a toolless graph that could fabricate an answer

CONCEPT:AU-ORCH.execution.no-silent-hallucination ·
CONCEPT:AU-ORCH.execution.task-aware-tool-selection

> `agent_utilities/orchestration/agent_runner.py` (`_fleet_server_failed_result`,
> `_select_relevant_tool_names`). Both introduced by commit `f3bf67a9` ("orch:
> task-aware tool selection + no-silent-hallucination for fleet delegations (F1)").

## Decision — a resolved fleet-server delegation that fails returns a DEGRADED result instead of falling through to the toolless multi-agent graph

`CONCEPT:AU-ORCH.execution.no-silent-hallucination`

`_fleet_server_failed_result` (`agent_runner.py:3725-3737`) exists to be
returned INSTEAD of letting execution fall through to the generic toolless
graph when a named-server delegation's real tools could not be reached. The
degraded result is picked up by `_delegation_degraded` (see
`CONCEPT:AU-ORCH.execution.degraded-no-data-outcome`), producing a truthful
`RunTrace` and negative learning feedback — a caller sees "this delegation
failed to reach its tools," never a plausible-sounding answer synthesized by
a model that had no real tool grounding for the question. This is the
codebase's own name for the general failure class it guards against: a model
asked to answer a tool-shaped question, denied its real tools, that
nonetheless produces a confident, well-formed, WRONG answer because nothing
stopped it from trying.

**The rejected alternative is the fallback this replaces**: when a named
server's tools are unreachable, silently drop back to the toolless
multi-agent graph and let it attempt an answer anyway. That is the
"obvious" graceful-degradation instinct — never leave the user with a hard
error when there's a chance of SOME answer — and it is exactly backwards for
a tool-grounded question: an ungrounded model doesn't know it's ungrounded,
so the resulting text reads as confident and gets recorded as `"completed"`
unless something explicitly catches the substitution. Several other concepts
in this domain (`degraded-no-data-outcome`, `all-tool-calls-errored`,
`focused-tools-fail-closed`) are all independent fixes to different code
paths that were falling into this same trap; `no-silent-hallucination` names
the class of bug all of them close.

### Pointer — `CONCEPT:AU-ORCH.execution.task-aware-tool-selection`

`agent_runner.py:2898-2912` (`_select_relevant_tool_names`). A companion
mechanism, introduced in the same commit, that keeps the single-server
focused-tools path FAST enough to be a viable alternative to the toolless
fallback in the first place: when a fleet server exposes too many tools
(the codebase's own example: `container-manager-mcp` exposes 314), handing
every schema to the model makes the LLM call hang and the run silently fall
through to a hallucinating toolless graph anyway — the exact failure
`no-silent-hallucination` exists to prevent, just reached through a
performance problem instead of a connectivity one. `_select_relevant_tool_names`
returns `None` when the server is small enough to bind wholesale; otherwise a
fast lexical ranker (task-word overlap between the task and each tool's
name+description) picks the top-K relevant tools, hard-capped at
`_MAX_BOUND_TOOLS = 20`. The rejected alternative — binding every tool a
server exposes, unconditionally — is what silently degraded large-server
delegations before this fix; the two concepts are two different failure
modes (connectivity, and schema-count overload) protected by the same
overall guarantee.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/agent_runner.py` (the
  single-server/focused-tools dispatch path).
- **Backward Compatible**: Yes — a server with ≤20 tools is unaffected by
  the selection cap; a small/healthy delegation is unaffected by the
  fail-closed result.
- **Known weak point**: `_select_relevant_tool_names`'s lexical ranker is a
  fast task-word-overlap heuristic, not semantic search — a task phrased
  without vocabulary overlapping any of a large server's tool
  names/descriptions could rank the genuinely-relevant tool outside the
  top-K cap and silently exclude it from the bound set, producing a
  delegation that runs but can't reach the one tool it actually needed.
