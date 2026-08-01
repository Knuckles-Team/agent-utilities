# Design Document: The fact extractor is guided by the formal OWL TBox itself, not a flat parallel schema

CONCEPT:AU-KG.retrieval.mmr-diversification

> `agent_utilities/knowledge_graph/extraction/extraction_schema.py`.

## A note on the id before the decision

This concept's id (`mmr-diversification`) does not match what is actually at
its marker site — the site is the ontology-guided extraction schema loader,
not a Maximal Marginal Relevance diversification mechanism (MMR
diversification, as an actual technique, is implemented separately inside
`ContextCompiler`; see
`.specify/design/kgr-context-compiler-policy-assembly/design.md`). Per this
sweep's instruction to read the site rather than guess from the id, this
document describes what the code at `extraction_schema.py:1` actually
decides — a real, well-grounded design choice, just filed under a
mismatched-looking name. Renaming the id (or reconciling it with the
`AU-KG.query.*` MMR concept it collides with in spirit) is a follow-up for
whoever next touches concept naming in this domain; it is out of scope for
this sweep, which only writes design docs and does not edit markers.

## Decision — inject the formal OWL classes + `rdfs:domain`/`range` into the extraction prompt, not a flat YAML schema

`extraction_schema.py:1-29` states the comparison directly: "sift-kg injects
a flat YAML schema into its prompt; we inject our formal OWL classes +
`rdfs:domain/range`, then keep the post-hoc grounding
(`ontology_grounding`) and the engine's OWL reasoning downstream —
generation-time guidance *and* reasoning, which a flat schema cannot give."
The TBox (`owl:Class` + `owl:ObjectProperty` with `rdfs:domain`/`range` +
labels) is loaded from the canonical `.ttl` ontology modules into a compact,
prompt-ready `ExtractionSchema`, so the LLM fact extractor extracts
ontology-typed entities and direction-constrained relations instead of free
snake_case predicates.

**The rejected alternative is named directly**: a flat schema (sift-kg's
approach) can guide generation but gives the downstream engine nothing to
*reason* over afterward — a flat YAML schema is not an OWL model, so nothing
built on it can run OWL inference against what was extracted. Injecting the
actual formal ontology means the SAME artifact that shapes generation-time
extraction is also what the engine reasons over post-hoc, rather than
maintaining two representations (a prompt-facing flat schema and a
reasoning-facing OWL model) that could drift.

**Optionality is deliberate, not accidental**: `rdflib` lives in the `[owl]`
extra, not the serving plane (KG-2.242) — the module import-guards `rdflib`
and degrades to `None` (free-vocab extraction) when absent, so "the lean
serving image is unaffected and ontology guidance auto-activates wherever the
owl stack is installed." The docstring frames this explicitly as
auto-detection, not a flag: "the enhancement runs when the resource is
present" — the same dependency-optional pattern documented for reranking in
`.specify/design/kgr-adaptive-chunk-selection-reranking/design.md`. The `.ttl`
files themselves are always-present package data; only the rdflib *parser*
is optional, and the TBox's default namespace (`http://knuckles.team/kg#`) is
kept deliberately distinct from the engine's LPG-projection `au:` namespace —
class/property *definitions* are read from the static `.ttl` parse; *instance*
data uses the other namespace, so the two are never confused at read time.

## Risk Assessment

- **Blast Radius**: `extraction_schema.py`, `fact_extractor.py` (the
  consumer), `.ttl` ontology modules.
- **Backward Compatible**: Yes — absence of `rdflib` degrades to the
  pre-existing free-vocab extraction behavior.
- **Known weak point**: the concept id itself is misleading (see the note
  above) — anyone searching the concept registry for MMR diversification by
  this id will land here instead of the actual MMR implementation in
  `ContextCompiler`, which is a discoverability defect independent of the
  underlying design's soundness.
