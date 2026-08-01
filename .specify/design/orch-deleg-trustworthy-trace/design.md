# Design Document: A RunTrace must be a real, queryable record — not an empty shell

CONCEPT:AU-ORCH.execution.run-trace-status-tool ·
CONCEPT:AU-ORCH.execution.skill-utilization-provenance ·
CONCEPT:AU-ORCH.execution.focused-tools-fail-closed

> `agent_utilities/orchestration/manager.py` (`get_run_trace`, `get_session_runs`),
> `agent_utilities/orchestration/agent_digital_twin.py`, and
> `agent_utilities/orchestration/agent_runner.py` (`_record_execution_trace`,
> the focused-tools dispatch branch). All three landed within days of each other
> (commits `7d739994`, `227c4487`) closing the same family of bug: a delegated
> run's provenance existed in the graph but callers could not see it, or the
> gate protecting it silently let the wrong runs through.

## Decision 1 — `get_run_trace`/`get_session_runs` read the REAL `:RunTrace`/`:ToolCall` graph state, not a dead code path

`CONCEPT:AU-ORCH.execution.run-trace-status-tool`

`Orchestrator.get_run_trace()` (`manager.py:246-258`) documents the bug it fixes
directly: a caller holding the `run_id` an `execute_agent`/`execute_workflow`
MCP call handed back had **no way to query what that run actually did** —
`get_task_status` looked at the `WorkItem` table only, so `status` reported
`"not_found"` for a run that had genuinely executed, with real output and tool
calls already sitting in the graph as a `:RunTrace` node
(`agent_runner._record_execution_trace`, ORCH-1.21) plus `:ToolCall` children
linked by `USED_TOOL` (KG-2.296). `get_run_trace` now reads that node directly
by `run_id` or its canonical trace id, and returns every `ToolCall` in call
order — status/output/duration AND each call's name/args/result/status.
`get_session_runs` (`manager.py:414-424`) is the same fix at the
`:Session`/`HAS_RUN` level, for a multi-step workflow whose steps each write
their own `:RunTrace`.

**The rejected alternative was the status quo**: keep `get_task_status`'s
`WorkItem`-only view and let a caller assume "not_found" means "didn't run."
That is not a neutral default — it actively hides successful, tool-grounded
work from the caller and from any downstream audit.

A second, independent bug fixed at the same time in
`agent_digital_twin.py:480`: the epistemic-graph backend's fast-path Cypher
parser silently **under-matches** (zero rows, no error) a query whose
`:RunTrace` node uses an anonymous inline property-map filter
(`MATCH ({id: $tid})-[:USED_TOOL]->(tc:ToolCall)`), even though the identical
pattern with a bound-but-otherwise-unused variable (`MATCH (t:RunTrace {id:
$tid})-...`) matches correctly. This twin's `tool_calls` were silently empty
for every run until the pattern was rebound — the same class of bug fixed
alongside it in `manager.py`. The lesson recorded at the site: **always bind
a variable to the filtered node**, even when nothing references it
afterward — the parser's fast path depends on it structurally, not just
stylistically.

### Pointer — `CONCEPT:AU-ORCH.execution.skill-utilization-provenance`

`agent_runner.py:983-1000` and `agent_runner.py:3545` (`_record_execution_trace`).
This is the WRITE side of the same story: whether a package **skill** (not a
generic agent) drove a run, and which MCP server's tools it was bound to (the
F7 `skill_bound_server_tools` upgrade below), is captured before the run
starts (`_skill_used`, `_bound_server`, `_skill_id`,
`_skill_instruction_digest`) and stamped onto the `:RunTrace` as opaque
`skill_ref`/`server_ref` properties plus a `USES_SKILL` edge when the trace is
recorded. Without this, "which runs used skill X, and what tools did it
drive" required reconstructing intent from tool-call names after the fact;
with it, it is a single graph traversal. This is the attribute `get_run_trace`
(Decision 1) now makes visible to a caller — write and read halves of one
"RunTrace is a trustworthy record" story.

### Pointer — `CONCEPT:AU-ORCH.execution.focused-tools-fail-closed`

`agent_runner.py:1161-1175`. The third bug in the same family, but on the gate
rather than the record: the focused-tools branch (single-server delegation via
`shape.tool_servers`, resolved independently of `agent_name` by the live-KG
lexical match in `plan_execution_shape`) is entered specifically because a
concrete fleet server was named. The **previous fail-closed check tested
`agent_meta.get("type") == "server"`** — the wrong variable, since
`agent_name` is frequently a generic/passthrough identity (e.g. the messaging
assistant) while the real delegation target is `shape.tool_servers`. A
genuine named-server delegation whose tools could not actually be reached
(unregistered server, auth failure, unreachable) therefore fell through
silently to the toolless multi-agent graph, which could fabricate a
plausible-looking answer stamped `"completed"` — exactly the confident-
hallucination failure `CONCEPT:AU-ORCH.execution.no-silent-hallucination`
exists to catch. The fix tests the right variable so an unreachable named
server fails the run loudly instead of degrading invisibly into a fabricated
"success" that a healthy-looking `:RunTrace` would then also misreport.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/manager.py`,
  `agent_utilities/orchestration/agent_digital_twin.py`,
  `agent_utilities/orchestration/agent_runner.py`.
- **Backward Compatible**: Yes — `get_run_trace`/`get_session_runs` are
  additive read paths; the Cypher rebind and the fail-closed variable fix are
  bug fixes with no schema change.
- **Known weak point**: the anonymous-vs-bound-variable Cypher fast-path
  behavior is a backend implementation quirk, not a documented contract — any
  new query written against the same backend can reintroduce the silent
  zero-row failure unless the convention ("always bind a variable to the
  filtered node") is remembered by the author, not enforced by tooling.
