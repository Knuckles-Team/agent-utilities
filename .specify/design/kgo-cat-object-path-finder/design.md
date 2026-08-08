# Design Document: The code-connectivity path-finder generalizes to any two ontology objects, reusing the one engine primitive rather than a second algorithm

CONCEPT:AU-KG.ontology.object-path-finder

> `agent_utilities/knowledge_graph/ontology/object_path.py`.

## Decision — `find_object_path` is a thin, object-type-agnostic wrapper around the EXISTING `GraphComputeEngine.get_shortest_path`

`object_path.py:1-24` names the gap directly: it closes a feature analysed
against Microsoft's Ontology-Playground, whose "Path Finder" panel runs a BFS
shortest path between two selected entities and renders the hop-by-hop chain.
The platform already had the identical underlying primitive —
`GraphComputeEngine.get_shortest_path` — but with exactly **one caller**,
`query_tools.code_connects`, scoped to `:Code` symbols only. `find_object_path`
is that same technique made object-type-agnostic: given any two object ids
(e.g. resolved via `object_set` search), it tries `source -> target` then the
reverse (so a one-directional property-graph edge is still found, mirroring
`code_connects`'s undirected-connectivity behavior) and annotates each hop with
its connecting relationship type and a friendly label via one batched Cypher
lookup (`object_path.py:64-70`).

**The rejected alternative is a second, object-generic shortest-path
implementation** — the natural approach if this were built from scratch
without noticing `code_connects` already solved the identical problem for one
narrower type. Reusing `get_shortest_path` instead means the path-finding
algorithm has exactly one implementation in the codebase; widening its
applicable domain from code symbols to arbitrary ontology objects was a
generalization of the *caller*, not a new *algorithm*. Never raising on a
missing path — returning `connected: False` instead (`object_path.py:54-60`)
— mirrors Ontology-Playground's own "no path found" UI state rather than
forcing every caller to catch an exception for the (common) case that two
objects simply aren't connected.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/object_path.py`, wired onto
  `object_set` MCP tool action `'path'` and `GET
  /api/objects/{source_id}/path/{target_id}`.
- **Backward Compatible**: Yes — new read-only capability, no existing caller
  changed.
- **Known weak point**: only the shortest path is found/annotated — if several
  shortest paths of equal length exist between the two objects, the one
  `get_shortest_path` happens to return first is the only one surfaced; there
  is no "show alternate paths" option.
