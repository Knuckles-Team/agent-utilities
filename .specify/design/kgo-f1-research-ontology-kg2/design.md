# Design Document: The research-intelligence pipeline's verdicts are governed ontology Links, layered over the existing raw-edge writer rather than replacing it

CONCEPT:AU-KG.ontology.kg-2

> `agent_utilities/knowledge_graph/ontology/research_objects.py`.

## Decision — `ResearchPaper`/`ResearchConcept` become ontology Interfaces, and the ConceptMatcher's verdicts (`ADDRESSES`/`SATISFIED_BY`/`RELATES_TO`) become typed Links — while the matcher keeps writing its raw edges unchanged

`research_objects.py:4-18` states the design directly: this "makes the
unified research-intelligence pipeline *ontologically driven*: research
papers (`Article`) and ecosystem capabilities (`Concept`) become first-class
ontology **Interfaces**, and the ConceptMatcher's verdicts are first-class
typed **Links**": `ADDRESSES` (a paper addresses a research topic/concept,
inverse `ADDRESSED_BY`), `SATISFIED_BY` (the paper's contribution is a
capability already built), and `RELATES_TO` (novel-but-relevant, the gap stays
open). Registration happens into the SAME import-populated default registries
`kg.ontology` already discovers with no configuration.

The MACHINE TRIAGE TOOL flagged this id "retire" as "a bare legacy pillar
reference (kg-2) — a citation of the old KG-N.NN numbering, not a name anyone
chose." Reading the site shows this is wrong: `kg-2` here is not a bare
citation, it is the concept id actually chosen for this decision (the module's
own docstring cites it three times as the identifier for what it does, not as
a cross-reference to something else) — the same failure mode already seen
twice in this domain (`do-not-auto-merge`, `kyle-insider-stealth-surveillance`):
an id-shape heuristic mistaking a real decision for filler.

**The rejected alternative is replacing the matcher's raw-edge writes with
ontology-governed writes** — riskier, since the ConceptMatcher's existing
write path is production infrastructure. The decision explicitly keeps it:
"the matcher keeps writing raw edges for robustness; this layer gives the
SAME edges a governed ontology schema (conformance, link cardinality,
discovery)." A raw edge the matcher writes is still valid data with or
without this layer; the layer adds the ability to reason over/validate/
discover those edges through the ontology surface, without the matcher's
write path ever depending on it.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/research_objects.py`,
  `interfaces.py`/`links.py` default registries, the ConceptMatcher's raw
  write path (unaffected).
- **Backward Compatible**: Yes — additive governance layer; raw writes
  continue unchanged.
- **Known weak point**: because the raw write path is deliberately untouched,
  a raw edge that violates `ADDRESSES`/`SATISFIED_BY`/`RELATES_TO` cardinality
  constraints is not rejected at write time — conformance is advisory, the
  same trade-off documented for the finance surveillance-signal layer (see
  `.specify/design/kgo-har-kyle-surveillance/design.md`).
