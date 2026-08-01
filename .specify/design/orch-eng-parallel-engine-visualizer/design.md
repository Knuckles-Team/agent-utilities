# Design Document: One engine, one manifest contract, from 1 agent to 300+

CONCEPT:AU-ORCH.execution.parallel-engine-visualizer

> `agent_utilities/graph/parallel_engine.py` (`ParallelEngine`, primary),
> `agent_utilities/models/execution_manifest.py` (`ExecutionManifest`, the input
> contract), `agent_utilities/graph/manifest_generators.py` (conversion layer,
> pointer below), `agent_utilities/workflows/visualizer.py` (Mermaid rendering,
> folded in — see "Why the visualizer is part of this decision, not a separate
> one"), `agent_utilities/core/execution/protocol.py` (the `ExecutionEngine`
> Protocol every engine now conforms to). Second-biggest concept in the domain:
> 9 source files, 63 marker sites — sampled across all of them below.

## The real decision

`ParallelEngine` (`graph/parallel_engine.py:188-`) is the **single** engine
for every scale of agentic execution — the module docstring states it plainly:

> *"The single engine that handles every execution from a trivial 1-agent LLM
> call to a 300-agent enterprise swarm. The **same code path** runs for all
> scales."*

Its own docstring lists what it **replaces**, by name:

```
graph/parallel_engine.py:7-13
    - DynamicSubgraphOrchestrator (team execution)
    - HeavyThinkingOrchestrator (parallel reasoning + deliberation)
    - RLMEnvironment.run_parallel_sub_calls() (parallel sub-calls)
    - SubagentPatternRouter (pattern selection)
    - CoordinationLayer (protocol selection — now a subcomponent)
    - WorkflowRunner (wave-based batch execution)
```

That list of six fragmented systems **is** the rejected alternative, made
concrete: a per-scale, per-pattern engine (a "team" engine, a "swarm" engine, a
"heavy thinking" engine, a "workflow" engine, each with its own scheduling and
coordination code) versus one engine whose scale-dependent behaviour is
entirely a property of its input, never its code path:

```
graph/parallel_engine.py:191-197
    - 1 agent (trivial query → inline execution)
    - 3-5 agents (team of specialists → standard parallel)
    - 10-50 agents (department-scale → wave batching)
    - 50-300+ agents (enterprise swarm → hierarchical synthesis)
```

The **input contract** that makes this possible is `ExecutionManifest`
(`models/execution_manifest.py:1-15`): *"the single universal input to the
`ParallelEngine`. Every execution ... is expressed as a manifest."* An
`AgentSpec` (`execution_manifest.py:32`) describes one logical agent
invocation, with fan-out expressed declaratively via `partitions` rather than
via a separate fan-out code path.

`execute()` is documented as **the only entry point**
(`parallel_engine.py:240-242`) and its flow is uniform regardless of scale:
resolve auto-configuration → build a DAG and schedule dependency waves
(`_schedule_waves`, using `rx.topological_generations`, falling back to fully
sequential on a detected cycle, `parallel_engine.py:588-596`) → execute each
wave under `asyncio.Semaphore` backpressure, with a per-agent-type circuit
breaker (`AgentTypeCircuitBreaker`, `parallel_engine.py:100-116`) → synthesize.

**The engine is additive, not a breaking replacement of the caller-facing
shape.** `core/execution/protocol.py:19-24` defines `ExecutionEngine` as a
`runtime_checkable` Protocol requiring only `async def run(self, manifest) ->
ExecutionResult`. `AgentOrchestrationEngine` already exposed this exact shape
via its own `execute()`; other engines gain an **additive** `run` adapter that
conforms to the Protocol without changing existing behaviour
(`orchestration/engine.py:1966-1971`, `execute()` delegates straight to
`ParallelEngine`). Rejected alternative here, named explicitly in the
docstring: forcing every existing engine to change its call shape to match a
new canonical one, versus adding a thin conformance adapter beside what each
engine already had.

## Why the visualizer is part of this decision, not a separate one

`workflows/visualizer.py:3` ("Workflow Visualizer — Programmatically generate
beautiful Mermaid diagrams of parallel workflows") is called **directly inside
`ParallelEngine.execute()`** (`parallel_engine.py:255-`, "Generate Mermaid
diagram representing the execution topography") — not as an optional external
tool run after the fact. The mermaid diagram is a byproduct of the same
manifest+wave-schedule data structure the engine already built, so rendering
it is a property of the one execution model rather than a second concept with
its own contract. That is also why the concept id pairs "parallel-engine" and
"visualizer" in one string rather than two.

## `gather.py` — a deliberate, bounded exception to the consolidation

`graph/gather.py:1-14` centralizes the `asyncio.gather(..., return_exceptions=True)`
pattern that had been copy-pasted across 5+ files (`heavy_thinking.py`,
`hsm.py`, `reactive/dispatcher.py`). But its own docstring names one caller
that **deliberately does not** adopt it: *"`ParallelEngine._execute_wave` keeps
its own `asyncio.gather` because it has deeply integrated circuit breaker +
wave result construction that doesn't fit this generic utility."* This is the
same "one engine" instinct applied narrowly and correctly — consolidation
stops where the abstraction would have to leak the engine's own internals back
out through it.

### Pointer — `CONCEPT:AU-ORCH.execution.manifest-generators`

`agent_utilities/graph/__init__.py:58-65`. This is the package-level
re-export comment for the conversion functions defined in
`graph/manifest_generators.py` (`manifest_from_planner`,
`manifest_from_teamconfig`, `manifest_from_workflow`,
`manifest_from_heavy_thinking`, `manifest_from_preset`,
`manifest_from_department`, `manifest_for_enterprise`) — the single
conversion layer between legacy/external plan formats (HTN `GraphPlan`, KG
`TeamComposition`, a KG-stored `SwarmTemplate`, an OWL-materialized company
department) and the `ExecutionManifest` contract this decision defines.
`manifest_generators.py`'s own module docstring and every one of its function
docstrings (e.g. `manifest_from_planner`, `manifest_generators.py:52`) carry
the **same** `parallel-engine-visualizer` concept id, not a distinct one —
`manifest-generators` is a second marker naming the same module from its
package `__init__.py` export site, not an independent decision. It is
recorded as a pointer here rather than merged into the head paragraphs above
so its own file:line grounding (`graph/__init__.py:59`) stays literal and
checkable.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/parallel_engine.py`,
  `agent_utilities/models/execution_manifest.py`,
  `agent_utilities/graph/manifest_generators.py`,
  `agent_utilities/workflows/visualizer.py`,
  `agent_utilities/core/execution/protocol.py`,
  `agent_utilities/orchestration/engine.py`, `agent_utilities/graph/gather.py`.
- **Backward Compatible**: Yes — the six replaced systems are gone from the
  live path; the `ExecutionEngine` Protocol conformance is additive per engine.
- **Known weak point**: a dependency cycle in the manifest's agent graph
  degrades to fully sequential execution rather than failing
  (`parallel_engine.py:591-596`) — correct for availability, but a caller
  relying on parallel wall-clock improvement gets a silent, unflagged
  slowdown instead of an error when their manifest has a cycle.
