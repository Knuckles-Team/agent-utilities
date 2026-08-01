# Design Document: A model-authored orchestration script gets a sandbox for CODE, never a sandbox for AUTHORITY — and a halted run resumes by replaying, not re-dispatching

CONCEPT:AU-ORCH.execution.dynamic-workflows ·
CONCEPT:AU-ORCH.execution.dynamic-workflow-resume

> `agent_utilities/capabilities/governed_dynamic_workflow.py`,
> `agent_utilities/orchestration/manager.py:56`,
> `agent_utilities/orchestration/engine.py:1736-1856`.

## Decision — `pydantic-ai-harness`'s Monty sandbox evaluates the model-authored SCRIPT only; every catalog function it calls re-enters `Orchestrator.execute_agent`, so authority/tools/credentials never enter the sandbox

`CONCEPT:AU-ORCH.execution.dynamic-workflows`

The module docstring (`governed_dynamic_workflow.py:1-16`) states the
boundary precisely: the optional Harness dependency owns exactly one
thing — evaluating the model-authored orchestration script inside its Monty
sandbox. Every catalog function the script calls re-enters
`Orchestrator.execute_agent`; **the script itself never receives a connector
tool, model client, graph handle, or credential.** Tenant/session authority,
agent and skill resolution, tool contracts, model-class routing, budgets,
cancellation, and RunTrace/ToolCall provenance all stay on the one GraphOS
execution plane — the sandbox only ever gets to say "call catalog function X
with these arguments," never "make this HTTP request" or "use this
credential" directly.

**The rejected alternative is giving the sandboxed script direct access to
the primitives it orchestrates** — connectors, model clients, or a graph
handle — which is the obvious design for a DSL meant to let a model "write
its own orchestration logic." It loses because it would let a
model-generated script bypass every governance surface this codebase
otherwise enforces on delegation (budgets, ActionPolicy, provenance,
authority renewal) — the sandbox would isolate CODE EXECUTION (preventing the
script from crashing or hanging the host process) while doing nothing to
isolate AUTHORITY (the script could still act with the full privilege of
whatever it's allowed to touch). Routing every catalog call back through
`execute_agent` means a dynamically-generated workflow is governed exactly
like a hand-authored one — same budgets, same fail-loud degradation, same
trace.

A second, explicit design choice in the same decision: the Harness dependency
is **optional and loaded lazily**, and callers must EXPLICITLY choose the
ordinary stored-DAG runner (`WorkflowRunner`) as a fallback when Harness is
unavailable — "execution failures never silently change orchestration
engines." The rejected alternative here is a silent engine swap on import
failure, which would make a workflow's actual execution semantics depend on
which optional dependencies happened to be installed, invisibly.

### Pointer — `CONCEPT:AU-ORCH.execution.dynamic-workflow-resume`

`governed_dynamic_workflow.py:106-160, 228, 839-1159`. A halted
`DynamicWorkflow` run (budget exhaustion, cancellation, process restart) does
not restart its script from scratch. `_WorkflowRuntime.resume_cache` is
pre-seeded from prior persisted catalog-call outputs for the SAME
`workflow_run_id`, so a catalog call already completed before the halt
short-circuits to its persisted output instead of re-dispatching through
GraphOS on retry — "the host choke point that makes 'restart produces no
duplicate ToolCalls' hold regardless of what script the model writes on
retry" (the code comment's own framing). Crucially, this resume path carries
an explicit **truthfulness contract**: a resumed run is never reported
identical to a clean single-shot success. `ChildRunEvidence.outcome` has a
dedicated `"replayed"` value (distinct from `ok`/`failed`/`timeout`/
`cancelled`), and `GovernedDynamicWorkflowResult.resumed` /
`.replayed_step_ids` name exactly which steps were reused rather than
re-executed.

**The rejected alternative was tried and rejected for two different
reasons.** Re-running the whole script unconditionally on resume would
duplicate every side-effecting dispatch already completed before the
halt — the exact failure the resume-cache exists to prevent, since a
model-authored script has no reliable way to know on its own which of its
own prior calls already landed. Reporting a resumed run as indistinguishable
from a fresh success (skip the `resumed`/`replayed_step_ids` bookkeeping)
would have been simpler, but it hides from any downstream consumer (an
auditor, the reward/learning loop) that part of the "result" wasn't actually
regenerated this run — a resumed workflow's success is a different claim
than a clean one's, and the schema says so explicitly.

## Risk Assessment

- **Blast Radius**: `agent_utilities/capabilities/governed_dynamic_workflow.py`,
  `agent_utilities/orchestration/manager.py`,
  `agent_utilities/orchestration/engine.py`.
- **Backward Compatible**: Yes — Harness absence falls back to the stored-DAG
  runner explicitly, never silently; resume is additive (a fresh
  `workflow_run_id` has an empty resume cache and behaves exactly like a
  non-resumed run).
- **Known weak point**: the resume cache is keyed on `workflow_run_id`
  identity — a caller that reuses a `workflow_run_id` across semantically
  different script contents (rather than generating a fresh id per logical
  attempt) would incorrectly short-circuit calls that were never actually
  re-issued for THIS script, replaying stale output against new logic.
