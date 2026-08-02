# Design Document: L1 holds a capped, LRU-evicted working set — the persistent graph is never mirrored into memory

CONCEPT:AU-KG.memory.working-set-eviction

> Realised by `agent_utilities/knowledge_graph/core/working_set_manager.py:4-49`
> (module docstring: tier architecture and the governing environment variables)
> and `:97+` (`WorkingSetManager`, the LRU with hard caps, TTL and a
> configurable eviction ratio).

## Decision — the in-memory tier is a bounded cache of the graph, not a copy of it

The compute architecture is tiered: L1 is the in-memory subgraph the Rust
`GraphComputeEngine` operates on, L3 is the persistent graph. The question this
decision settles is what L1 is *allowed to contain*.

`WorkingSetManager` caps it: hard limits on node and edge count, a TTL, and a
configurable eviction ratio, all configurable by environment variable, with LRU
deciding what leaves. The module docstring states the purpose — it *"prevents
memory explosion by enforcing hard caps on the number of nodes and edges in the
working set."*

**The rejected alternative is mirroring the full persistent graph into L1**,
which is the implicit design whenever an in-memory tier has no cap. It is
attractive because it makes every compute operation a local one and removes
a whole class of cache-miss reasoning.

It fails on a property specific to this system: the persistent graph's size is
not bounded by anything the process controls. It grows with ingestion —
connectors, source syncs, memory accumulation — so "load it all" is a
memory requirement that increases without limit over the deployment's lifetime.
An unbounded L1 therefore does not have a large memory footprint; it has an
*unknowable* one, and the failure mode is an OOM that arrives at whatever point
the graph crosses the host's capacity, unrelated to the work being done at that
moment.

Capping L1 makes the memory footprint a configured constant and converts the
failure from "process dies" into "cache miss, fetch from L3" — a bounded
latency cost paid per miss, on a path that already exists.

The eviction ratio being configurable rather than fixed matters here: evicting
one entry at a time under sustained pressure produces thrashing, so the
manager evicts a *fraction* of the working set, and the right fraction depends
on the workload's locality.

**Evidence note:** the rejected alternative is stated in the code as the
failure mode being prevented ("memory explosion"), not as a quoted prior
implementation. No commit narrative was recovered — the marker was introduced
in a version-bump commit with no rationale text — so the argument above is
reconstructed from the tier architecture and the cap mechanism, not from a
documented incident.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/core/working_set_manager.py`
  and the L1 compute path.
- **Backward Compatible**: Yes — caps are configurable and the L3 fetch path
  is the pre-existing behaviour on a miss.
- **Known weak point**: correctness of *results* now depends on the working set
  containing what an operation needs. A graph algorithm over a working set that
  has silently evicted part of the relevant subgraph returns an answer computed
  over a partial graph, and nothing in the manager distinguishes "this node is
  absent because it does not exist" from "absent because it was evicted".
