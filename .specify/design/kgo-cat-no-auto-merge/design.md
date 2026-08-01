# Design Document: LLM-discovered schema extensions are always a reviewed proposal — never auto-merged into the canonical ontology

CONCEPT:AU-KG.ontology.do-not-auto-merge

> `agent_utilities/knowledge_graph/extraction/schema_discovery.py`.

## Decision — discovered classes/properties emit a `RESERVE-PENDING` proposal fragment, gated through concept-reservation + human/evolution-loop review

`schema_discovery.py:1-16` states the design directly: beyond flat-YAML
discovery, this module samples documents, asks an LLM to propose entity/relation
types, and **diffs against the existing OWL ontology** — but "we never
auto-merge: a new top-level `.ttl` is a build break (sprawl rule), so the
output is a *proposal* fragment with `RESERVE-PENDING` concept placeholders
that a human (or the evolution loop) reviews, reserves, and lands into the
existing domain module after the valid/connected/SHACL gate." Each candidate is
classified as `covered` (already an OWL class/property), `synonym` (maps to an
existing class), or `missing` (a real extension candidate) — only `missing`
candidates reach the `.ttl` fragment (`to_ttl_fragment`, `schema_discovery.py:
210-259`), and every emitted class/property is prefixed with a literal
`# CONCEPT:RESERVE-PENDING` marker rather than a synthesized concept id
(`schema_discovery.py:237, 256`) — concept ids are reserved through the flock
ledger before a class ever lands in a domain module, never hardcoded by this
generator.

**The rejected alternative is auto-applying the LLM's proposal directly** — the
faster path, and the one that would let an LLM hallucination or a
misclassified `missing` candidate silently mutate the canonical ontology.
`do-not-auto-merge` is not a naming accident (the machine triage tool's
"slugified prose fragment" heuristic misread it as filler): it is the load-
bearing guarantee of the whole discovery pipeline, restated at the second call
site too — `generate_standalone_ontology` (`schema_discovery.py:284-302`,
Ontology-Playground coverage row #13, covered separately by
`CONCEPT:AU-KG.ontology.standalone-generation`) runs the identical LLM path
against an empty base instead of a diff, and its docstring repeats "Still
`CONCEPT:AU-KG.ontology.do-not-auto-merge`: a human-reviewed proposal only,
never auto-applied/merged into the canonical ontology."

## Risk Assessment

- **Blast Radius**: `knowledge_graph/extraction/schema_discovery.py`, the
  concept-reservation flock ledger (OS-5.42), the evolution review pipeline.
- **Backward Compatible**: Yes — output is a proposal artifact, not a write.
- **Known weak point**: the guarantee is enforced by convention (this module
  simply never calls a merge/apply function) rather than by a structural
  barrier — a future caller that takes `to_ttl_fragment`'s output and writes it
  directly into a domain module without going through concept-reservation would
  violate the invariant with nothing in this module stopping it.
