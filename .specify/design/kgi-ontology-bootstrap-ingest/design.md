# Design Document: The ingest pipeline can derive its own ontology from live graph data instead of always trusting the bundled `ontology.ttl`

> `agent_utilities/knowledge_graph/pipeline/phases/owl_reasoning.py`.

CONCEPT:AU-KG.ingest.ontology-bootstrap-ingest

## Decision — sample the graph's own records and derive classes/properties, with the bundled ontology as fallback

`owl_reasoning.py:30-40`, `80-90`.

**The rejected alternative**: always reasoning over the fixed, hand-authored
`ontology.ttl` bundled with the codebase — the simpler, status-quo choice,
and still the default when bootstrapping is disabled or yields nothing.

**The design chosen**: `bootstrap_ontology_path` applies "the self-bootstrapping
ontology agent... to ingest" — when `ctx.config.enable_ontology_bootstrap` is
set and no explicit ontology path is configured, it samples the graph's own
nodes (capped by `ontology_bootstrap_sample_limit`), lets
`OntologyBootstrapper` derive classes and typed properties (plateau-stopped —
it stops sampling once additional records stop yielding new schema), and
emits the result as Turtle to a temp file for the OWL backend to reason over.
If nothing could be derived (empty/insufficient sample), the caller falls
back to the bundled ontology rather than reasoning over an empty schema.

**Why derive from data rather than always trust the static file**: a hand
-authored ontology can drift from what the graph ACTUALLY contains,
especially after domain-pack or connector-driven schema growth (see
`.specify/design/kgi-domain-pack-claim-promotion/design.md` for how
domain packs introduce new entity types at ingest time). Bootstrapping from
live records means OWL reasoning can stay accurate to the graph's real
current shape without requiring a manual ontology edit for every new schema
introduced by a connector or domain pack, at the cost of a reasoning pass
over a possibly-incomplete sample rather than a curated, complete schema.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/pipeline/phases/owl_reasoning.py`
  (`bootstrap_ontology_path` and its call site in `execute_owl_reasoning`).
- **Backward Compatible**: Yes — disabled by default configuration behavior
  (falls back to the bundled ontology when bootstrapping is off or empty).
- **Breaking Changes**: None.
- **Known weak point**: a sampled-and-derived ontology is only as complete as
  the sample — a class/property that exists in the graph but wasn't present
  in the capped sample silently fails to appear in the derived schema for
  that run, with no explicit signal to the caller that the derived ontology
  is a partial view rather than a complete one.
