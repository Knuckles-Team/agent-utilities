# Design Document: Topologies are materialized per-task from the KG, not all present at once in one static graph

CONCEPT:AU-ORCH.execution.dynamic-topology-materialization

> `agent_utilities/graph/topology_engine.py` (`TopologyEngine`).

## Decision — `TopologyEngine` selects and materializes only the relevant subgraph per task, replacing a single static all-paths graph

The module docstring names both the replacement and what it replaces in its
first sentence: "Replaces static `create_graph_agent()` topology with
KG-driven dynamic graph materialization. Instead of all execution paths
existing simultaneously, the engine selects and materializes only the
relevant subgraph based on:" task domain, task complexity, KG-stored
`TopologyTemplateNode` success rates, and available specialists/tools
(`topology_engine.py:4-13`). `TopologyEngine.materialize()` (73-129)
converts a declarative `TeamComposition` (its `adaptive_agent_router` roster,
`execution_mode`, and `parallel_groups`) into an ordered `execution_plan`
the pydantic-graph runner consumes, tagged with a `topology_id` for
tracking, and — when an engine is attached — records the materialization
event back to the KG (`_record_materialization`, invoked at line 119).
`record_topology_outcome()` (349-376) closes the loop: it feeds
success/failure back as an exponential-moving-average `success_rate` on the
originating `TopologyTemplate` node (`alpha = 0.15`, lines 359-368) — which
template gets materialized next for a given task type is therefore informed
by how past materializations of that same template actually performed, not
fixed at write time.

**The rejected alternative** is the prior static `create_graph_agent()`
topology, named and rejected in the module's own opening sentence: a single
graph in which "all execution paths [exist] simultaneously" regardless of
what a given task actually needs. That approach pays the cost — in graph
complexity, prompt/tool surface, and execution branching — of every possible
path (sequential, parallel, mixed, fan-out, fan-in; the five patterns
enumerated at lines 15-20) on every run, whether or not the task uses them.
The chosen design instead treats topology as **data**: a `TopologyTemplateNode`
selected and instantiated per task, so adding a new topology shape is a KG
write, not a new code path, and competing templates are ranked by measured
`success_rate` rather than being permanently wired into one graph.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/topology_engine.py`; any reader
  of `TopologyTemplate` nodes or materialization records in the KG.
- **Backward Compatible**: Yes.
- **Known weak point**: `record_topology_outcome()`'s KG write is
  best-effort — wrapped in a broad `except Exception` that only
  `logger.debug`s on failure (`topology_engine.py:375-376`). A failed EMA
  update silently stalls that template's learning signal (its
  `success_rate` simply stops updating) without surfacing as an error to
  any caller.
