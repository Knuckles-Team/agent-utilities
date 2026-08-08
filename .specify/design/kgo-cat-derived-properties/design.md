# Design Document: Derived properties are never stored — computed live at read time from four coupled backings

CONCEPT:AU-KG.ontology.derived-property-registry

> `agent_utilities/knowledge_graph/ontology/derived_properties.py`.

## Decision — one dispatcher, four backings (FUNCTION/CYPHER/SPARQL/EMBEDDING), every output typed through the same PropertyType contract

`derived_properties.py:4-42` states the Foundry contract directly: Workshop's
function-backed columns / ontology computed fields declare an output type and a
backing function, computed live whenever the property is requested — never
stored on the object. `DerivedPropertyEngine.compute(obj, derived_prop, graph)`
(referenced at `derived_properties.py:39`) is the single live dispatcher for all
four `DerivedBacking` strategies (`derived_properties.py:67-79`): `FUNCTION`
invokes a registered typed function through `FunctionRuntime`; `CYPHER`
evaluates a Cypher expression through the same guarded facade path
Functions-on-Objects uses; `SPARQL` evaluates against the L2 `OWLBridge`; and
`EMBEDDING` derives a value from vector similarity via the capability index.
Every backing's output is coerced through the declared `PropertyType`
(`derived_properties.py:32-35`) so a computed value carries the identical typed
guarantee a stored one would.

**The rejected alternative is Foundry parity itself** — Foundry only
function-backs derived properties (one backing). The module explicitly frames
the other three as a deliberate "surpass-edge" (`derived_properties.py:73`):
CYPHER/SPARQL let a derived property be declared as a pure graph-query
expression with no registered function at all, and EMBEDDING lets one be
declared as a similarity computation. The cost of the wider surface is a larger
dispatch/typing contract to keep coherent (four execution paths instead of one),
accepted because a graph-native platform has cheap, safe access to
Cypher/SPARQL/embedding reads that Foundry's Functions runtime does not.
Caching is also a first-class decision here, not an afterthought: a read-through
cache keyed by `(property, object)` with explicit invalidation
(`invalidate`/`invalidate_object`/`clear`) — rather than recomputing on every
read (correctness-safe but potentially expensive for a SPARQL/EMBEDDING
backing) or caching without invalidation (fast but stale after a write).

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/derived_properties.py` and any
  caller that reads a `DerivedProperty` value.
- **Backward Compatible**: Yes — additive; a stored property is unaffected.
- **Known weak point**: cache invalidation is explicit/manual
  (`invalidate_object`), not write-triggered — a derived property computed from
  graph state that changed via a path which never calls `invalidate_object`
  will serve a stale cached value until the cache is cleared or the entry's
  cache policy expires it.
