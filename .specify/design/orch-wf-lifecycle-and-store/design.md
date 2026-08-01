# Design Document: A stored workflow is a KG subgraph, executed via the SAME `run_agent` executor as every other delegation — with fail-loud guarantees at each seam

CONCEPT:AU-ORCH.execution.workflow-lifecycle-management ·
CONCEPT:AU-ORCH.execution.workflow-persistence-replay ·
CONCEPT:AU-ORCH.execution.no-silent-success-on-empty-workflow ·
CONCEPT:AU-ORCH.execution.workflow-engine-wiring ·
CONCEPT:AU-ORCH.execution.workflow-parallel-bridge ·
CONCEPT:AU-ORCH.execution.best-effort-provenance

> `agent_utilities/workflows/runner.py` (`WorkflowRunner`),
> `agent_utilities/knowledge_graph/workflow_store.py` (`WorkflowStore`),
> `agent_utilities/knowledge_graph/core/owl_bridge.py`. Also documented at
> `docs/pillars/1_graph_orchestration/ORCH-1.9-Workflow_Lifecycle.md`
> (ORCH-1.24) — this doc summarizes and cites that existing architecture doc
> for the head decision, and adds the concept-id text + the four narrower
> pointer decisions that doc doesn't cover.

## Decision — a named workflow is a `WorkflowDefinition` KG subgraph, discovered/persisted like everything else in the graph, executed by wiring the EXISTING `run_agent` step executor rather than building a second execution engine

`CONCEPT:AU-ORCH.execution.workflow-lifecycle-management`

`docs/pillars/.../ORCH-1.9-Workflow_Lifecycle.md` (ORCH-1.24) is explicit that
the `SkillCompiler` path "replaces the former static YAML catalog: any skill
directory becomes a runnable workflow" — that is the rejected alternative
this whole subsystem replaces: a separate, hand-maintained YAML file
enumerating workflows, disconnected from the KG that already indexes
everything else in the system (agents, servers, skills). Instead: a
`SKILL.md`, natural-language spec, or harvested `BusinessProcess` compiles to
a `GraphPlan`, which `WorkflowStore.save_workflow` persists as a
`WorkflowDefinition` KG subgraph (`WorkflowStep` nodes, `TRANSITION_TO`
edges, `REQUIRES_TOOL` edges) — the same graph, the same semantic search, the
same provenance model as everything else, with zero separate
sync/consistency surface to maintain.

Execution (`WorkflowRunner._execute_plan_via_agents`,
`CONCEPT:AU-ORCH.execution.workflow-engine-wiring`, `runner.py:953-965`)
wires the SAME `run_agent` executor every other delegation in the codebase
uses: steps with satisfied dependencies run concurrently as a wave, each via
`run_agent(step.id, step.task, engine=...)` against its resolved MCP
toolset, with upstream step outputs threaded into dependent steps' context.
Because `run_agent` already records its own `RunTrace`/`:ToolCall` nodes,
workflow execution is fully visible over graph-os "with zero extra
plumbing" — the alternative (a bespoke workflow-step executor with its own
provenance recording) would have duplicated that machinery and risked
drifting from it.

A ready step whose `kind` is `"gate"`/`"approval"` is NOT run by an agent at
all: `gate_checker` (default: a `:satisfiedBy` out-edge check,
`_default_gate_checker`) decides approved/rejected, and rejection marks the
step plus its on-success downstream as skipped while still running any
`on_reject` target — human/governance checkpoints are a first-class step
kind, not bolted on separately (ORCH §7.1 delta 3).

### Pointer — `CONCEPT:AU-ORCH.execution.workflow-persistence-replay`

`agent_utilities/knowledge_graph/workflow_store.py:3-50, 247-268, 464+`. The
persistence half of the same decision, grounded in its own module: a
`WorkflowDefinition` round-trips `GraphPlan ⇄ KG subgraph` via
`save_workflow`/`load_workflow`, with a Cypher-backend path and an
in-memory-NetworkX fallback (`_load_workflow_nx`) kept behaviorally
equivalent for engines without a live graph backend. `save_from_execution`
auto-caches a successful ad-hoc `RunTrace`+`GraphResponse` as a reusable
`WorkflowDefinition` under a collision-resistant privacy-safe auto-name
(`_automatic_workflow_name`), so a workflow doesn't have to be authored up
front — a good run can become a reusable template retroactively, the
opposite of requiring every workflow to start as a hand-written definition.

### Pointer — `CONCEPT:AU-ORCH.execution.no-silent-success-on-empty-workflow`

`agent_utilities/workflows/runner.py:107-121` (`WorkflowHasNoStepsError`,
commit `d51ff6de`, D-FSR-1). A stored `WorkflowDefinition` MAY legitimately
have zero steps (a placeholder or an unauthored template) — that's not the
bug. The bug being fixed: *executing* a zero-step definition previously
reported success having done nothing, and the caller had no way to
distinguish "ran fine" from "ran nothing." The fix raises
`WorkflowHasNoStepsError` from `_execute_plan_via_agents` — the single choke
point `execute_by_name`/`resume`/`resume_localized` all route through — so
every caller is covered, including four call sites that invoke
`Orchestrator.execute_workflow`/`WorkflowRunner` directly and never pass
through the `graph_workflows` MCP tool's own SHACL/ACL pre-dispatch gate
(`ticket_playbooks._dispatch_workflow`, `loop_controller._default_skill_runner`,
`weights_distillation._dispatch_train_workflow`, `schedule_engine`'s
`kind in ("workflow", "agent")` dispatch). All four already wrap the call in
`try/except Exception` and degrade to failed/skipped, so raising at the one
choke point turns a fake success into a correctly-surfaced failure
everywhere at once, not just at the MCP boundary.

### Pointer — `CONCEPT:AU-ORCH.execution.workflow-engine-wiring` (bug-fix facet)

`agent_utilities/workflows/runner.py:459-475`. The second half of the same
concept id, distinct from the wiring decision above: `AgentExecutionResult`
(the `ParallelEngine` wave result type) carries no `task` field — it lives in
`.metadata`. Reading `r.task` directly raised `AttributeError` and crashed
every wired `execute_workflow` run **after the steps had already executed**
— the worst place for a crash, since the work was done but the result never
got reported. The fix falls back through `getattr(r, "task", None) or
(r.metadata or {}).get("task") or r.agent_id` — three tiers, never `None`.

### Pointer — `CONCEPT:AU-ORCH.execution.workflow-parallel-bridge`

`agent_utilities/workflows/runner.py:400-410`
(`execute_via_parallel_engine`). Converts a `GraphPlan` to an
`ExecutionManifest` and delegates wave dispatch to `ParallelEngine.execute()`
rather than `WorkflowRunner` reimplementing concurrent-wave execution itself
— "ensuring a single execution path" (the docstring's own words) for
anything that needs coordination-history reads, Global Workspace broadcast,
or execution-hierarchy bookkeeping, all of which already live in
`ParallelEngine`. The rejected alternative — a second, workflow-specific wave
scheduler — would have needed to reimplement or duplicate every one of those
`ParallelEngine` behaviors to stay consistent with non-workflow parallel
execution.

### Pointer — `CONCEPT:AU-ORCH.execution.best-effort-provenance`

`agent_utilities/workflows/runner.py:37-70, 382-392`,
`agent_utilities/knowledge_graph/core/owl_bridge.py:511-513`. When an
executed workflow was compiled from a harvested `BusinessProcess` (a
`(:WorkflowDefinition)-[:REALIZES]->(:BusinessProcess)` edge, ORCH-1.41),
completion closes the provenance loop with an `EXECUTED_PROCESS` edge from
the run's `RunTrace` to the `BusinessProcess`. `WorkflowRunner(lineage_sink=
...)` additionally accepts an OPTIONAL callable invoked once per close-out
with a normalized lineage record (process/workflow/run ids, status, step
counts, duration) — explicitly so a deployment can wire an external lineage
system-of-record (the docstring names egeria-mcp's `assert_lineage`)
**without agent-utilities taking a hard dependency on it**. The rejected
alternative is a hard dependency on one specific lineage backend; the
best-effort optional-sink seam keeps the KG-native `EXECUTED_PROCESS` edge as
the source of truth and treats any external system as a downstream, swappable
consumer.

## Risk Assessment

- **Blast Radius**: `agent_utilities/workflows/runner.py`,
  `agent_utilities/knowledge_graph/workflow_store.py`,
  `agent_utilities/knowledge_graph/core/owl_bridge.py`.
- **Backward Compatible**: Yes for all six — the zero-step guard and the
  `task`-field fallback are bug fixes that only change previously-broken
  behavior; persistence, parallel-bridge, and lineage-sink are additive.
- **Known weak point**: `WorkflowHasNoStepsError` is enforced at the single
  Python choke point `_execute_plan_via_agents`, not at the KG schema level —
  a future dispatch path that constructs and runs a `GraphPlan` without going
  through `WorkflowRunner` at all would silently regress to the pre-D-FSR-1
  fake-success behavior.
