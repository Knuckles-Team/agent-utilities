# Design Document: An ingested skill/workflow must run on the local LLM against real tools, not stay search-corpus-only

CONCEPT:AU-ORCH.execution.execution-seam-closure

> Already fully documented outside `.specify/design/`: `docs/architecture/orchestration-execution-seam.md`
> (the canonical write-up) and `docs/architecture/north-star-architecture.md:131-135`
> (the one-paragraph summary, Pillar 5). This document transplants the concept
> id into `.specify/design/` per the concept-lineage rule
> (`CONCEPT:AU-OS.governance.concept-lineage-parent-doc`) rather than
> restating the existing architecture doc; read `orchestration-execution-seam.md`
> for the full flow diagram and governance detail. Code sites:
> `agent_utilities/orchestration/manager.py:605-609`, `:973-981`.

## The real decision

Before this seam closed, the substrate already had every individual part —
the step executor (`WorkflowRunner`), the tool-binding loop (`run_agent`,
ORCH-1.21), the model router, the SHACL/ACL gate, and the ingested DAGs
(KG-2.97) — **but they were not connected**
(`orchestration-execution-seam.md:9-11`). Three separate gaps existed
simultaneously, and closing all three is what "the seam" refers to:

| Gap | Before | After |
|---|---|---|
| `execute_workflow` ignored the stored DAG | `Orchestrator.execute_workflow` → `AgentOrchestrationEngine` ran one generic `dynamic_worker`, regardless of what workflow was requested | routes to `WorkflowRunner.execute_by_name`, which loads the stored `WorkflowDefinition`/`WorkflowStep` DAG and runs **each step via `run_agent`** on the local LLM, in dependency-wave order |
| Ingested skills weren't executable | a `:Skill` (or cold `AGENT_SKILL`) node was search corpus only | `_resolve_agent_from_kg` hydrates the skill's instruction body as the system prompt + `USES_TOOL` tools, binding it into a runnable `CallableResource (AGENT_SKILL)` |
| Tool calls weren't visible | only a run-level `RunTrace` was written | every tool call is persisted as a privacy-guarded `:ToolCall` linked `RunTrace -[:USED_TOOL]-> :ToolCall` |

The manager-level code (`orchestration/manager.py:973-981`) states the
"before" state as directly as the architecture doc does:

> *"This previously constructed a generic `AgentOrchestrationEngine` whose
> no-completion-state path ran ONE `dynamic_worker` agent and never loaded the
> ingested `WorkflowDefinition`/`WorkflowStep` DAG — so a stored/ingested
> workflow (the KG-2.97 `WorkflowStore` shape) was dispatchable but never
> executed."*

The fix routes through `WorkflowRunner` (ORCH-1.24): `load_workflow(name)` →
build dependency waves → run each step on the local LLM. The SHACL+ACL
ontology gate (ORCH-1.42) was moved to run at **this** chokepoint —
`_execute_plan_via_agents` — rather than only in the `graph_workflows` MCP
handler, because (per the same comment) **four separate production callers**
bypassed the MCP handler entirely: `knowledge_graph.adaptation.ticket_playbooks._dispatch_workflow`,
`knowledge_graph.research.loop_controller._default_skill_runner`, and others
— an instance of "enforce at the chokepoint, not one entrypoint" applied to
this exact seam.

`resolve_capability` (`orchestration/manager.py:605-609`) closes the
resolution half of the same seam: it resolves a task against the KG's hybrid
index before local-vLLM execution, but an **explicit agent name remains
authoritative** — an unresolved task routes to the KG-bound
`agent-utilities-expert` rather than exposing raw skill bodies or fleet tool
schemas to the calling harness.

## The rejected alternative

The rejected alternative is the literal prior behaviour, not a hypothetical:
`execute_workflow` running a generic `dynamic_worker` regardless of which
workflow was requested, and an ingested `:Skill` node existing purely as
search corpus for retrieval without ever becoming something a local LLM could
actually run. Under that design, "delegation" was structurally incapable of
truly handing off a stored capability — the KG could describe what should
happen, but nothing turned that description into an executed run with real
MCP tools bound. The seam is explicitly framed as the platform's *delegation
keystone* (`orchestration-execution-seam.md:"Why this is the delegation
keystone"`) precisely because the delegate-and-verify operating model is only
safe if a delegated run is (a) actually executed against real tools, not a
stub, and (b) fully visible and steerable afterward via `RunTrace`/`:ToolCall`
provenance — neither was true before this seam closed.

### Pointer — related ids at the same seam (not in this batch)

`AU-ORCH.dispatch.dispatch-half-skill-ingestion` (the skill-hydration half),
`AU-ORCH.execution.rich-result-wrapper` (the `run_id`/trace_ref handle
returned to the caller), and `KG-2.296` (the `:ToolCall` provenance write) are
three co-located ids on the same seam, each with its own row in the gap table
above and its own marker sites in `docs/architecture/orchestration-execution-seam.md`.
They are named here for context but are **not** claimed as covered by this
document — they are outside this batch's 13 assigned concept ids and should
be verified/documented on their own terms if not already covered elsewhere.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/manager.py`,
  `agent_utilities/workflows/runner.py`,
  `agent_utilities/knowledge_graph.adaptation.ticket_playbooks`,
  `agent_utilities/knowledge_graph.research.loop_controller`.
- **Backward Compatible**: Yes — additive routing; the generic
  `dynamic_worker` path still exists for non-workflow requests.
- **Known weak point**: the SHACL+ACL gate had to be moved to the chokepoint
  because four production callers were already bypassing the MCP handler —
  a fifth new caller introduced without going through
  `_execute_plan_via_agents` would reopen the same governance gap by a
  different route.
