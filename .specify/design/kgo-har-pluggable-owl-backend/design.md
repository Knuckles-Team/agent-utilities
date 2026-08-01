# Design Document: OWL reasoning is backend-pluggable, so a new ontology domain updates each concrete backend's own type mapping, not one shared table

CONCEPT:AU-KG.ontology.owl-rdf-bridge

> `agent_utilities/knowledge_graph/backends/owl/base.py` (`OWLBackend` ABC),
> `agent_utilities/knowledge_graph/backends/owl/owlready2_backend.py`,
> `agent_utilities/knowledge_graph/backends/owl/stardog_backend.py`,
> `agent_utilities/knowledge_graph/core/owl_bridge.py`.

## Decision — `OWLBackend` is an ABC (`load_ontology`/`promote`/`promote_edges`/`reason`) with two independent concrete implementations, each carrying its own node/edge-type mapping table

`base.py:8-14` states the pattern directly: `OWLBackend` "mirrors the
`GraphBackend` ABC pattern used by LadybugDB/Neo4j/FalkorDB but provides
OWL-specific operations." Two concrete backends exist:
`owlready2_backend.py` — "Default in-memory + optional SQLite persistence
backend using Owlready2 and its bundled HermiT/Pellet reasoner" — and
`stardog_backend.py`, a remote-triplestore alternative. Because Owlready2's
in-process reasoner needs real local Python class objects to reason with
(`_NODE_TYPE_TO_OWL_CLASS`, `owlready2_backend.py:23-169`), while the generic
`owl_bridge.py` LPG→RDF promotion path works off namespace-qualified IRI
strings (`_NODE_TYPE_TO_OWL_CLASS` in `owl_bridge.py`, a *separately
maintained* table), the two representations are not automatically kept in
sync by any shared code — they are two hand-maintained tables that happen to
use the same domain's names. The Legal Entity & Compliance domain
(`legal_trust`, `trustee_role`, `settlor_role`, `beneficiary_role`,
`legal_entity`, `company`, `ein_application`, and the corresponding edges
`has_trustee`/`has_settlor`/`has_beneficiary`/…) is the concrete site where
this dual-registration requirement is visible: the same domain addition
appears at `owlready2_backend.py:151,275` AND `owl_bridge.py:306,570`.

**The rejected alternative is a single backend-agnostic reasoning
implementation** — plausible on paper (both backends implement the same ABC
surface), but Owlready2's in-process HermiT/Pellet reasoning and Stardog's
remote SPARQL-based reasoning have fundamentally different execution models;
collapsing them into one implementation would mean the generic `owl_bridge`
promotion path could no longer be backend-agnostic, or Owlready2 would lose
its native Python-class-based reasoning in favor of a lowest-common-denominator
interface. The accepted cost is exactly the coupling risk this file shows: a
new ontology domain must be registered in every backend it needs to reason
correctly under, with nothing structural enforcing that the tables stay in
sync beyond code review and shared naming convention.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/backends/owl/*.py`,
  `knowledge_graph/core/owl_bridge.py`.
- **Backward Compatible**: Yes — additive per-domain registration.
- **Known weak point**: nothing mechanically verifies that
  `owl_bridge.py`'s promotable-type table and each concrete backend's own
  mapping table stay in sync — a domain registered in one but not the other
  degrades silently (an object type promotable at the generic layer that the
  active backend's reasoner has no class for).
