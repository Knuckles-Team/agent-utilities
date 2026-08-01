# Design Document: Object sets are a composable abstraction over existing fabric, never a second storage/retrieval path

CONCEPT:AU-KG.ontology.link-type-pivot

> `agent_utilities/knowledge_graph/ontology/object_set.py`.

## Decision — search/filter/pivot/aggregate/set-algebra all bind to the SAME facade layers everything else already uses

`object_set.py:4-55` names the Foundry provenance: the Object-Backend *Object
Set Service* and object-explorer surface, where an object set is a first-class,
composable handle over a collection of ontology objects. Three materialization
kinds (`ObjectSetKind`, `object_set.py:84-99`) mirror Foundry exactly: `STATIC`
(fixed ids), `DYNAMIC` (a predicate re-evaluated against the live graph on every
read, so membership auto-updates), and `TEMPORARY` (a TTL-bounded snapshot).
`pivot(link_type, group_by)` follows a typed link to the related set and groups
it by a target property — the concrete site the `link-type-pivot` marker
names.

**The rejected alternative is a set implementation with its own storage or
retrieval logic** — the module states this explicitly (`object_set.py:28-44`):
"the set is an abstraction *over existing fabric* — it never reinvents storage
or retrieval." Traversal/property access binds to the live `GraphComputeEngine`
authority; property/full scans go through that authority's Cypher surface when
present, falling back to the compute graph; semantic/hybrid search binds to the
existing `HybridRetriever`, degrading to a deterministic substring scan when no
embedding model is reachable "so a search always returns *something*" rather
than erroring. Interface-typed sets resolve through `interfaces.find_implementers`
via a soft import, so this module never hard-depends on the interfaces module —
absence just degrades to treating the interface name as a concrete type.
`search_around` answers "what's near X" (bounded-hop neighborhood, capped at
`DEFAULT_SEARCH_AROUND_CAP = 100_000` to bound a runaway traversal); the
adjacent `object-path-finder` module answers the different question "how does
X connect to Y" (a specific hop chain) — the two are deliberately not merged
into one primitive because they answer different questions with different cost
profiles.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/object_set.py` and every caller
  reached via the `object_set`/`of_type` factories on `OntologySystem` or the
  `kg_object_set` MCP tool.
- **Backward Compatible**: Yes — a new composable layer over existing reads.
- **Known weak point**: a `DYNAMIC` set's membership is re-evaluated on every
  read against the live graph — correct, but means two reads of the "same"
  dynamic set moments apart can return different membership with no version
  marker distinguishing which graph state produced which result.
