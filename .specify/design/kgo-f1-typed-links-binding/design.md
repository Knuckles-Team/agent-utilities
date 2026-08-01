# Design Document: Model↔InferenceProfile↔TaskClass/Role are bound by FIRST-CLASS typed links, so OWL reasoning — not application code — extrapolates which profile fits which task

CONCEPT:AU-KG.ontology.typed-ontology-links-binding

> `agent_utilities/knowledge_graph/ontology/links.py:526-545`,
> `agent_utilities/models/knowledge_graph.py:990`.

## Decision — register `HAS_PROFILE`/`PROFILE_OF`/`TUNED_FOR`/`BOUND_TO_ROLE`/`USES_PROFILE` as typed `LinkType`s with declared cardinality and inverses

`links.py:526-528` names the decision: "Typed ontology links binding
Model/InferenceProfile to TaskClass/Role/Agent (HAS_PROFILE/PROFILE_OF/
TUNED_FOR/BOUND_TO_ROLE/USES_PROFILE) for profile extrapolation. First-class
typed links so OWL reasoning extrapolates which sampling profile fits a task
class / role / model from how related ones are tuned." Each `LinkType`
registration declares its source/target `RegistryNodeType`, its
`RegistryEdgeType`, a `LinkCardinality` (`model_has_profile` is
`ONE_TO_MANY`), and — where meaningful — an `inverse_edge_type`
(`HAS_PROFILE`'s inverse is `PROFILE_OF`), mirroring how `grounds`
(`CLAIM`→`EVIDENCE` via `GROUNDED_IN`, inverse `SUPPORTS`) and
`implements_claim` (`CLAIM`→`CODE_SPEC` via `IMPLEMENTED_BY`) are declared
just above in the same registry — a consistent typed-link pattern across
unrelated domains, not a bespoke mechanism invented for sampling profiles
specifically.

**The rejected alternative is an untyped, ad hoc relationship** — a generic
`"related_to"` edge (or a bare property on the Model node naming its profile
by id) that application code interprets by convention. Declaring these as
first-class `LinkType`s with real cardinality and inverses means "which
profile fits task class X" or "what models are tuned like model Y" become
ordinary graph traversals/OWL-reasoned queries rather than convention-based
lookups a caller has to already know the shape of — the reasoner extrapolates
across the link structure the same way it does for grounding/claims (see
`.specify/design/kgo-e1-ara-forensic-grounding/design.md`) or harness
inverses (see `.specify/design/kgo-har-harness-as-ontology/design.md`),
because it's the SAME `LinkTypeRegistry` mechanism, not a parallel one.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/links.py`
  (`DEFAULT_LINK_REGISTRY`), `models/knowledge_graph.py`
  (`RegistryEdgeType.HAS_PROFILE`/`PROFILE_OF`/`TUNED_FOR`/`BOUND_TO_ROLE`/
  `USES_PROFILE`), the `InferenceProfile`/`SamplingConfigurable` interfaces
  (`.specify/design/kgo-f1-inference-profile-implementers/design.md`).
- **Backward Compatible**: Yes — additive link-type registrations.
- **Known weak point**: declaring a `LinkType` here does not itself enforce
  that every `Model` actually carries the link — extrapolation quality
  depends on adoption (models actually being linked to their tuned profiles),
  which this registration cannot force.
