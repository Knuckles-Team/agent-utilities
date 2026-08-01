# Design Document: Solution search is a directed graph with cross-branch fusion edges, not an information-isolated tree/population

CONCEPT:AU-KG.retrieval.monte-carlo-graph-search

> `agent_utilities/harness/graph_search_evolution.py`,
> `agent_utilities/harness/agentic_evolution_engine.py:491`,
> `agent_utilities/harness/__init__.py:150`.

## Decision — nodes gain cross-branch reference edges, so a strong solution in one branch can seed another, instead of staying information-isolated

`graph_search_evolution.py:6-40` (assimilated from MLEvolve, arXiv:2606.06473)
states the prior limitation directly: "AU's prior evolutionary search (AHE-3.2
`VariantPool`, AHE-3.3 regressor evolution, KG-2.69 program synthesis)
searched *tree*- or population-structured spaces where each branch is
information-isolated: a strong solution found in one branch can never seed
another." MLEvolve's advance, ported here dependency-injected and
network-free: beyond the parent→child primary edges that carry credit
assignment, nodes gain *reference* edges importing knowledge from strong
nodes in OTHER branches; when a branch stagnates, a fusion node is created
whose reference set is the best nodes of the other branches — cross-branch
knowledge flow a tree structurally cannot express.

**The rejected alternative is the prior architecture itself**: a tree or
population where branches never communicate mid-search. That structure wastes
exactly the information a directed graph preserves — a promising partial
solution discovered in a stagnating branch is simply lost, rather than
available for another branch to build on. Three further MLEvolve pillars ride
along with the graph structure rather than being separable add-ons:
**Retrospective Memory** (a static `ColdStartKB` seeding an otherwise cold
task, plus a dynamic `GlobalCodeMemory` retrieving similar past attempts by
reward label), **Hierarchical Planning + Adaptive Code Generation** (the
planner is refined against *similar success* records from memory before
`select_coding_mode` chooses full-rewrite/stepwise/diff), and a **progressive
exploration schedule** (`exploration_schedule` decays the UCT exploration
constant from broad to focused over the search horizon). All randomness is
seeded and deterministic by step index, so a fixed seed reproduces an
identical search — a property a naive random-walk fusion policy would not
automatically have.

## Risk Assessment

- **Blast Radius**: `graph_search_evolution.py`, `agentic_evolution_engine.py`,
  `harness/__init__.py`.
- **Backward Compatible**: Yes — a new search structure alongside the
  existing tree/population evolutionary search primitives, not a replacement
  of them.
- **Known weak point**: fusion nodes are created reactively "when a branch
  stagnates" — a stagnation detector that fires too late wastes search budget
  in an already-dead branch before cross-branch knowledge is imported; too
  early, and a branch that would have recovered on its own is fused
  prematurely.
