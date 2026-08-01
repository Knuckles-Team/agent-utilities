# Design Document: A budget is multi-dimensional, enforced at every LLM-call boundary, and exhaustion is always terminal

CONCEPT:AU-ORCH.execution.execution-budget-caps

> Primary definition: `agent_utilities/models/usage.py` (`ExecutionBudget`),
> central enforcement: `agent_utilities/graph/_router_impl.py::dispatcher_step`.
> Reused at every LLM-call boundary in the graph: `agent_utilities/graph/executor.py`,
> `agent_utilities/graph/hierarchical_planner.py`, `agent_utilities/graph/verification.py`,
> `agent_utilities/capabilities/governed_dynamic_workflow.py`,
> `agent_utilities/capabilities/output_repair.py`,
> `agent_utilities/orchestration/agent_runner.py`, `agent_utilities/orchestration/engine.py`.
> The biggest single concept in the `AU-ORCH.execution` domain: 15 source files, 47
> marker sites — sampled across all of them below rather than read from one file.

## The real decision

A graph run's resource ceiling is not one number (a token cap) but a `Pydantic`
model with five independent dimensions, and **every dimension defaults to a real,
finite ceiling — never unbounded**:

```
agent_utilities/models/usage.py:60-73  (ExecutionBudget)
    max_cost_usd: float | None = 10.0
    max_total_tokens: int | None = 500_000
    max_node_transitions: int | None = 50
    max_tool_calls: int | None = 200        # distinct from max_node_transitions:
                                              # one node can invoke several tool calls
    max_duration_seconds: float | None = 600.0
```

The class docstring states the reasoning directly: *"a 5,000-token invoker budget
can still be blown by an oversized tool result or an unattended research loop
without a graph-level backstop"* — so the graph-level `ExecutionBudget` exists
**in addition to**, not instead of, whatever budget an invoker passed down.
Enforcement is centralized in one place, `dispatcher_step`
(`agent_utilities/graph/_router_impl.py:870-889`), which increments
`node_transitions` and checks it against `budget.max_node_transitions` on every
graph transition — the single chokepoint every execution path passes through.

A second, narrower cap rides alongside the five: `per_request_input_tokens_limit`
(`agent_utilities/orchestration/loop_guards.py::DEFAULT_PER_REQUEST_INPUT_TOKENS_LIMIT`).
It is applied independently at **every LLM-call boundary that can receive an
externally-controlled payload** — not just once at the top:

| Boundary | Site |
|---|---|
| Spawned task agent | `graph/executor.py:701` (`spawn_usage_limits`) |
| HTN planner | `graph/hierarchical_planner.py:245` |
| Verifier | `graph/verification.py:319` |
| Governed dynamic-workflow orchestrator | `capabilities/governed_dynamic_workflow.py:975` |
| Delegated/invoked agent run | `orchestration/agent_runner.py:3061` |

Each site's comment gives the same concrete motivating incident:
*"a fleet tool that silently ignored an unknown `limit` argument returned 212 KB
from one call"* (`graph/executor.py:701`, `orchestration/agent_runner.py:3061`) —
a real ServiceNow production failure. `per_request_input_tokens_limit` bounds the
**response** of a single request (checked against the provider-reported
`input_tokens` of the response, so the oversized request is still sent and
billed once — what the cap prevents is *carrying it forward* across further
requests in the same run).

## The rejected alternative — repairing/retrying a budget violation, not treating it as always-terminal

The single most important design choice here is not the shape of the budget —
it is what happens when a dimension is exhausted. Two sites show the rejected
alternative was tried, found broken, and explicitly closed:

**1. `graph/verification.py:885-899` (`error_recovery_step`).** Before this
fix, only the `max node transitions` phrase was in the terminal-keyword list
that stops the planner from retrying. But the dispatcher stamps the **same**
`"Execution budget exceeded: ..."` prefix for all five dimensions (node, tool
calls, tokens, cost, duration) — so a token/cost/duration budget exhaustion
didn't match the keyword list and **was silently retried for up to 2 more
planner rounds, each one burning more of the resource that was already
exhausted**. The fix: treat `result.get("budget_exceeded")` (a boolean the
dispatcher stamps explicitly) as unconditionally terminal, regardless of which
of the five dimensions tripped.

**2. `capabilities/output_repair.py:36-44`.** The output-repair capability
retries a malformed structured output — except when the failure is
`UsageLimitExceeded` (budget exhausted mid-output) or `ContentFilterError`.
The comment states the reasoning explicitly: *"repairing a budget violation
would only spend more of the budget that already tripped"* — the exact
anti-pattern `error_recovery_step` above was hardened against. Both are
**recorded and never retried**, closing with a typed, inspectable error
instead of either an unbounded retry storm or pydantic-ai's own generic
`UnexpectedModelBehavior`.

So the rejected alternative is: treat budget exhaustion as just another
retryable error class, and let the normal recovery/retry machinery run. It
loses because a retry consumes exactly the resource that is already at zero —
the closer a run gets to its cap, the more that mistake compounds, and the
original ServiceNow-class incident (an oversized payload) is precisely the
shape of failure a retry-based recovery would make worse, not better.

## Two other decisions folded into the same footprint

- **Cost governance gates team composition.** `orchestration/engine.py:284-292` —
  when synthesizing a subagent team via KG topology scoping, a supplied
  `delegated_authority` restricts the candidate roster to agents authorised for
  it. Budget enforcement is therefore not purely reactive (stop when exceeded)
  but also proactive (don't grant a team more authority than the budget's
  governance context allows).
- **Termination is a classified outcome, not a generic error.** `orchestration/engine.py:950-963` —
  `run_graph` reports a budget-caused termination as its own outcome dimension
  (`budget_exceeded` + which of the five dimensions tripped), rather than
  folding it into the generic `graph_terminal_error` every other terminal
  failure shares, so telemetry can distinguish "ran out of budget" from "broke."

## Known noise in the marker footprint

Three sites (`base_utilities.py:8,368,394`, `safe_save_model`/`safe_load_model`)
carry this concept id attached to unrelated "Serialization Safety" docstrings —
a copy/paste mistagging, not a fourth instance of this decision. They are not
cited as grounding above and should not be read as evidence for this concept.

## Risk Assessment

- **Blast Radius**: `agent_utilities/models/usage.py`, `agent_utilities/graph/_router_impl.py`,
  `agent_utilities/graph/executor.py`, `agent_utilities/graph/hierarchical_planner.py`,
  `agent_utilities/graph/verification.py`, `agent_utilities/capabilities/governed_dynamic_workflow.py`,
  `agent_utilities/capabilities/output_repair.py`, `agent_utilities/orchestration/agent_runner.py`,
  `agent_utilities/orchestration/engine.py`, `agent_utilities/orchestration/loop_guards.py`.
- **Backward Compatible**: Yes — every dimension has a real default, so an
  unconfigured caller already had governance; `None` opts a single dimension
  out explicitly per-run.
- **Known weak point**: the terminal-keyword matching bug in
  `error_recovery_step` shows this is a pattern that silently regresses if a
  new termination path stamps its own ad hoc error string instead of setting
  `budget_exceeded` — the fix works only because every dimension now routes
  through the same explicit flag.
