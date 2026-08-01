# Design Document: Enterprise "north-star" standards live in a DEDICATED interface registry, kept separate from the built-in structural shape registry

CONCEPT:AU-KG.ontology.populated-at-import-real-3

> `agent_utilities/knowledge_graph/ontology/__init__.py:295-317`
> (`OntologySystem.standards`), `agent_utilities/knowledge_graph/standardization/standards.py`
> (`ENTERPRISE_STANDARD_REGISTRY`), `agent_utilities/knowledge_graph/research/loop_controller.py:49-60`
> (`_WATERMARK_TYPES`).

## Decision — a SECOND, separately-populated `InterfaceRegistry` for enterprise north-star contracts, distinct from the structural built-in registry every other interface lives in

`ontology/__init__.py:307-314` states the separation directly: "Enterprise
standards: a DEDICATED interface registry of north-star contracts
(`ManagedApplication`/`BusinessProcess`/`DataAsset`), kept separate from the
structural interfaces above **so authoring an enterprise standard never
pollutes the built-in shape registry**. Reached from the execution plane only
through `kg.ontology.standards`." `OntologySystem` exposes BOTH registries as
distinct attributes: `self.interfaces` (the built-in `DEFAULT_INTERFACE_REGISTRY`
— `HasProvenance`, `Locatable`, `MicrostructureSignal`, `VerifiableClaim`,
etc., all import-populated at module load) and `self.standards`
(`ENTERPRISE_STANDARD_REGISTRY`, imported separately and assigned only here).
Both follow the SAME "import-populated, never an empty shell" idiom this
domain uses throughout (property types, value types, link types,
interfaces) — the concept id names that shared pattern applied to a THIRD
distinct registry, not a single occurrence.

**The rejected alternative is one shared interface registry for both
structural shapes and enterprise standards** — simpler (one registry, one
lookup path), but it would mean an org authoring/curating its own north-star
contracts (what "a compliant `ManagedApplication`" looks like for THEIR
enterprise) writes into the SAME namespace the platform's own built-in
structural interfaces live in, risking a naming collision or an accidental
edit to a platform interface while intending to add an enterprise one. A
dedicated registry, reached only through the narrow `kg.ontology.standards`
surface, keeps the two concerns — "what shape does this platform's ontology
define" versus "what standard does THIS enterprise hold itself to" —
structurally separate.

`loop_controller.py:56-60` shows a downstream consequence of this being a
distinct concern: `enterprise_resource`/`enterprise_standard` are their own
entries in the research-loop's `_WATERMARK_TYPES` set — "new harvested assets
or edited standards re-trigger the standardize stage" — a separate watermark
category from `sdd_feature`/`capability`/`article`/`requirement`/`decision`/
`concept`, because an enterprise-standard edit is a different kind of input
than the platform's own capability/decision corpus.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/__init__.py`
  (`OntologySystem.standards`), `standardization/standards.py`
  (`ENTERPRISE_STANDARD_REGISTRY`), `research/loop_controller.py`'s watermark
  set.
- **Backward Compatible**: Yes — additive registry; existing structural
  interface lookups are unaffected.
- **Known weak point**: because the two registries are reached through
  different attributes (`interfaces` vs `standards`), a caller that only knows
  to query `self.interfaces` will silently miss any enterprise standard —
  there is no single "search everything" entrypoint spanning both registries
  shown at this call site.
