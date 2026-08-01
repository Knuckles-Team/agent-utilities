# Design Document: New entities join the OWL class hierarchy by embedding similarity to class prototypes, not by explicit manual classification

CONCEPT:AU-KG.ontology.semantic-subsumption

> `agent_utilities/knowledge_graph/core/semantic_subsumption.py`.

## Decision — `SemanticSubsumptionEngine` compares a discovered entity's topological embedding against OWL class PROTOTYPE vectors to auto-inject it into the correct class hierarchy

`semantic_subsumption.py:3-8` states the capability: "OWL-Driven Semantic
Subsumption. Enables zero-shot ontology alignment. When a new entity is
discovered, its topological embedding is compared against existing OWL class
prototypes to automatically inject it into the correct class hierarchy."
`SemanticSubsumptionEngine` is constructed with `owl_classes` (a map of OWL
class URI/name → prototype vector embedding) and an optional `owl_hierarchy`
(class → parent classes, "used to reconstruct full subsumption lineage") —
cosine similarity between a new entity's embedding and each class prototype
drives the classification decision.

**The rejected alternative is requiring every new entity to be manually
classified** (or classified only by an exact rule/keyword match against a
known type) before it can be placed in the OWL hierarchy — the default
behavior anywhere ontology alignment isn't automated. "Zero-shot" is the
specific claim being made: an entity that was never seen before, with no
hand-authored rule for its exact shape, can still be placed into the class
hierarchy based on embedding similarity to existing PROTOTYPES, rather than
staying unclassified until a human adds a rule for it. The trade-off accepted
is inherent to any embedding-similarity classifier: it is probabilistic, not
exact — a novel entity is placed by nearest-prototype similarity, which can
be wrong for an entity that is genuinely between two classes or unlike any
existing prototype.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/core/semantic_subsumption.py`, any
  discovery/enrichment pipeline that calls into it to classify a newly-found
  entity.
- **Backward Compatible**: Yes — an additive classification aid; entities
  classified through existing explicit paths are unaffected.
- **Known weak point**: subsumption quality is entirely bounded by the
  quality/coverage of `owl_classes`'s prototype vectors — a class with no
  prototype (or a poorly-representative one) cannot be a subsumption target
  even for an entity that genuinely belongs to it, and nothing here surfaces a
  low-confidence match distinctly from a confident one unless the caller
  inspects the raw cosine-similarity score itself.
