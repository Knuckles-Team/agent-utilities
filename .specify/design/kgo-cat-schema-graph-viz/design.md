# Design Document: The schema graph is a pure read-only projection of live registries, never a new persisted view

CONCEPT:AU-KG.ontology.schema-graph-visualization

> `agent_utilities/knowledge_graph/ontology/schema_graph.py`.

## Decision — render the EXISTING interface + link-type registries as a Cytoscape-shaped graph; add no new ontology storage

`schema_graph.py:1-46` names the gap directly: it closes a feature analysed
against Microsoft's Ontology-Playground, whose headline is "any ontology
renders as an interactive node-and-edge diagram" (Cytoscape.js) plus a
one-click Markdown export. The platform already had the schema itself, live,
at import time (`InterfaceRegistry` + `LinkTypeRegistry`) but no rendering of
it as a graph payload or human-readable document. `schema_graph.py` is
explicitly a **pure, read-only projection** over those two existing registries
— it "adds no new ontology storage and never mutates the registries"
(`schema_graph.py:12-15`), stated as directly satisfying the platform's
anti-sprawl rule ("extend the canonical ontology, never sprawl a new .ttl")
because there is no new ontology content here, only a new *view*.

**The rejected alternative is a persisted, separately-maintained "schema
diagram" data model** — the obvious path for a visualization feature, and the
one that would drift from the live registries the moment either changed without
the diagram being regenerated. Instead every call re-derives the graph fresh
from `InterfaceRegistry`/`LinkTypeRegistry` at read time, so the diagram is
never stale by construction. Two node kinds (`interface`, `object_type`) and
three edge kinds (`implements`, `extends`, `relationship`) mirror Foundry's own
Interface/Object-Type distinction, which `ontology_interface`'s existing
`list`/`implementers` actions already expose — the visualization reuses that
same distinction rather than inventing a third taxonomy. The output shape is
literally a Cytoscape.js elements list so any graph-viz frontend (agent-webui's
sigma.js `GraphView`) renders it with zero further transformation.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/schema_graph.py`, wired onto
  `ontology_interface` MCP tool actions `graph`/`summary`/`lint` and
  `GET /api/ontology/schema-graph` / `/api/ontology/schema-summary`.
- **Backward Compatible**: Yes — pure additive read projection.
- **Known weak point**: because it is a live re-derivation with no caching,
  a very large interface/link-type registry would re-walk both registries on
  every single request; there is no ETag/cache-invalidation story if that
  becomes a hot path.
