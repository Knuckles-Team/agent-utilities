# Design Document: Ontology Interfaces — programmatic targeting resolves through an abstract shape, not a concrete type

CONCEPT:AU-KG.ontology.conformance-check

> `agent_utilities/knowledge_graph/ontology/interfaces.py`.

## Decision — an interface is a runtime-checked abstract contract, ported from the existing property/link fabric rather than a new type system

Foundry's interface model (matched explicitly in the module docstring,
`interfaces.py:6-17`) lets Functions/Actions/queries target an *interface*
(e.g. "anything Locatable") instead of one concrete object type, resolving at
runtime to whichever concrete types implement it. `interfaces.py` ports this
without inventing a parallel type system: `InterfaceProperty` is typed by the
*same* `PropertyType` Stage-A vocabulary every other ontology property uses
(`interfaces.py:84-95`, reusing `property_types.parse_type_ref`), and
`InterfaceLinkConstraint` names a required link by the existing
`RegistryEdgeType`. `InterfaceRegistry.implement()` is the conformance-check
proper: it validates a concrete type's declared shape against the interface and
collects the gaps (missing properties / unsatisfied link constraints) into an
`ImplementationReport` rather than silently accepting a partial implementation.

**The rejected alternative** is a second, interface-specific type/validation
system — the obvious path for anyone porting "interfaces" as a standalone
feature. The code explicitly reuses the OWL/SHACL namespace bindings the
existing `owl_bridge` RDF materialization already uses (`interfaces.py:64-70`,
"identical bindings ... so interface classes/shapes resolve in the same
graph") and the same type-casing convention `owl_bridge._build_rdf_graph` uses
for node-type classes (`_camel()`, `interfaces.py:74-81`), so an implementing
type's OWL class lines up with type classes minted elsewhere instead of living
in a disconnected interface-only namespace. `to_owl()` emits the interface as
both an `owl:Class` **and** a SHACL `sh:NodeShape` — one artifact serving both
the RDF-reasoning path and the write-time SHACL gate, rather than maintaining
two representations by hand.

`find_implementers()` is what makes `conforms` practically useful: it is the
concrete mechanism a Function/Action/object-query uses to resolve "target
interface X" down to the set of object types that currently implement it,
without the caller ever naming a concrete type — a soft-import degrade (absence
of `ontology.interfaces` treats the name as a concrete type rather than hard-
failing) keeps every other module that touches interfaces from taking a hard
dependency on this one.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/interfaces.py` and every module
  that resolves an interface name to concrete types (`object_set.py`'s
  interface-typed sets, `schema_graph.py`'s `implements`/`extends` edges).
- **Backward Compatible**: Yes — additive; a type that implements no interfaces
  behaves exactly as before.
- **Known weak point**: `implement()` collects gaps into a report but does not
  itself block registration — a caller that ignores the report can register a
  non-conforming implementer; conformance is advisory unless a caller checks it.
