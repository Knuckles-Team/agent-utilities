# Interfaces, Object Types, and Links

This lesson is the first stop in **Ontology School** — a short, practical tour
of the ontology model that underpins the whole Knowledge Graph. By the end
you'll be able to read (and reason about) the platform's own schema.

## Why an ontology model at all?

A Knowledge Graph without a schema is just a pile of nodes and edges — anyone
can call anything anything. This platform's ontology layer
(`agent_utilities/knowledge_graph/ontology/`) gives every object a **typed
shape**: named properties with real types, and named, cardinalitied
relationships to other objects. That's what lets a Function, an Action, or a
query be written *once* against a type and keep working as the underlying data
grows — and it's what lets tools like `ontology_interface(action='lint')`
catch a typo in a property name before it ships.

The model has three moving parts: **interfaces**, **object types**, and
**link types**. This lesson covers each in turn.

## Interfaces: abstract shape contracts

An **Interface** is an *abstract* schema element — it never has instances of
its own. It declares a **shape**: a set of required properties, plus optional
link constraints every implementing object type must satisfy. Two built-in
examples ship at import (`register_builtin_interfaces`):

- `HasProvenance` — anything that must carry `source`, `ingested_at`, and
  similar audit properties.
- `Locatable` — anything that has a physical or logical location.

Interfaces can **extend** other interfaces (an interface may have several
parents), so a sub-interface inherits its parents' required properties and
link constraints automatically.

The payoff is **programmatic targeting**: a Function, Action, or query can
name an interface instead of a concrete type. At call time the platform
resolves that interface to *every* concrete object type that implements it —
`InterfaceRegistry.find_implementers()` is exactly that resolution step. Write
a risk-scoring Function against the `HasProvenance` interface once, and it
runs unmodified against every object type that implements it, today and in
the future.

## Object types: the concrete instances

An **object type** is what actually gets instantiated — `Customer`, `Order`,
`Incident`, `CodeSymbol`. An object type **implements** zero or more
interfaces by calling `InterfaceRegistry.implement()`, which doesn't just
record the claim — it **validates** it, collecting any missing required
property or unsatisfied link constraint into an `ImplementationReport`. There
is no silent, unchecked "trust me, I implement this" path.

Note the asymmetry with a typical demo ontology tool: many such tools treat
every entity as directly concrete (no interface/implementation split). This
platform keeps the two concepts distinct on purpose — it's the same
abstract-contract-plus-concrete-implementer split used by large-scale
enterprise ontology platforms, and it's what makes interface-targeted
Functions and Actions possible.

## Link types: typed, cardinalitied edges

A **LinkType** defines a typed relationship between two object (node) types —
mirroring how an enterprise ontology platform defines a *link type* between
two object types. Three things make a link type more than "just an edge":

1. **A named, directed edge type** (`edge_type`, a real `RegistryEdgeType`),
   so the link can be written and traversed deterministically in both
   directions.
2. **A cardinality** — `ONE_TO_ONE`, `ONE_TO_MANY`, or `MANY_TO_MANY` — which
   is enforced, not decorative.
3. **A source and target object type**, so `ontology_interface(action='graph')`
   can render the whole schema (interfaces + object types + their links) as
   one interactive node/edge graph — see the **Schema View** in the web UI.

## Junction link types: many-to-many with their own data

A plain `MANY_TO_MANY` link connects two object types, but sometimes the
relationship itself carries data — an `Employee` ↔ `Project` link that also
needs a `role` and an `allocation` percentage. Modeling that as a bare edge
loses the data; modeling it as two separate one-to-many links loses the
relationship's identity.

The fix is **reification**: a `JunctionLinkType` declares an intermediary
**junction object type** (e.g. `Assignment`) that carries the link's own
properties, plus two edges connecting it to each endpoint. In graph terms,
this is exactly the standard reified-edge pattern: a junction *node* plus two
directed edges. `materialize_junction()` turns a junction link definition plus
a concrete pair of endpoint ids into real, ready-to-write graph objects — no
stub, real provenance stamped on the junction node. `endpoints_of()`,
`neighbors_via()`, and `junctions_for()` walk a materialized junction back to
its endpoints, so the reified relationship is navigable in both directions,
just like a plain link.

## Try it yourself

Everything in this lesson is live in the running platform, not just
documentation:

- `ontology_interface(action='list')` — every registered interface.
- `ontology_interface(action='implementers', name='HasProvenance')` — every
  concrete type that implements a given interface.
- `ontology_interface(action='graph')` — the whole schema (interfaces, object
  types, and their links) as one Cytoscape-style node/edge graph — the same
  payload the web UI's **Schema View** renders.
- `ontology_interface(action='summary')` — the same schema as a readable
  Markdown document.
- `ontology_interface(action='lint')` — naming-convention and typo checks
  against the live interface registry.

Next lesson: **[Querying with UQL](../querying-with-uql/01-your-first-uql-pipeline.md)** —
now that you can read the shape of the graph, learn to ask it questions.
