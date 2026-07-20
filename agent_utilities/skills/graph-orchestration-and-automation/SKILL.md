---
name: graph-orchestration-and-automation
description: >-
  Plan, delegate, schedule, and verify Graph-OS work across agents and workflows.
  Use for multi-step goals, agent dispatch, workflow execution, loops, schedules,
  sandboxes, run forks or replay, messaging, bus coordination, reachability,
  approvals, or parallel work with dependency-aware synthesis.
---

# Graph orchestration and automation

Convert a goal into bounded work items, execute independent work in parallel,
and verify the synthesized result.

## Direct or delegated

Use a direct tool call when the task is one bounded operation with an obvious
owner and verification. Use `graph_orchestrate` for one named agent,
`graph_agents` for a governed collective, and `graph_workflows` when dependencies
form a reusable DAG.

Do not delegate a trivial lookup. Do not keep a complex task direct merely to
avoid expressing its acceptance criteria.

## Workflow

### 1. Define the goal

Specify the desired state, scope, constraints, evidence required, deadline, and
what must not change. Convert vague completion language into observable checks.

### 2. Build the work graph

- Create one work item per independently verifiable outcome.
- Add dependencies only where data or authorization truly requires ordering.
- Assign domain skills rather than individual one-tool wrappers.
- Mark mutating or externally visible steps for policy review.
- Select an economical model class for deterministic extraction, classification,
  or formatting; reserve stronger reasoning for planning, critique, and synthesis.

### 3. Execute

For a plan or review-only request, return the bounded work graph, dependency
barriers, approval points, and verification criteria, then stop. Do not dispatch,
schedule, persist, load tools, or otherwise execute the plan.

| Need | Primary operation |
|---|---|
| Run one named agent | `graph_orchestrate` |
| Run a swarm or runtime org | `graph_agents` |
| Compile, run, or inspect a workflow | `graph_workflows` |
| Dispatch or inspect a durable job | `graph_jobs` |
| Manage goal state | `graph_goals`, `spec_ticket` |
| Advance a controlled loop | `graph_loops` |
| Create recurring work | `graph_schedules` |
| Isolate execution | `graph_sandbox` |
| Fork, revert, or replay a run | `graph_fork`, `graph_runvcs` |
| Exchange task messages | `graph_message`, `graph_bus`, `graph_broker` |
| Check participants or capability reach | `graph_reach` |

Launch independent work items together, then wait at explicit dependency
barriers. Carry the original goal and acceptance criteria into every delegation.

### 4. Verify and synthesize

- Require each work item to return evidence and a clear status.
- Re-run critical checks from the coordinator rather than trusting summaries.
- Resolve contradictory outputs explicitly.
- Report partial completion and blockers; never convert a timeout into success.

### 5. Close the loop

Persist the final outcome, provenance, approvals, and follow-up items. Remove
temporary schedules or loaded tools that were created only for the run.

## Guardrails

- Keep authorization scope unchanged across delegation.
- Do not send secrets or unrelated context to child agents.
- Bound fan-out, retries, depth, and wall time.
- Require human direction before irreversible or externally visible expansion of
  the requested scope.
