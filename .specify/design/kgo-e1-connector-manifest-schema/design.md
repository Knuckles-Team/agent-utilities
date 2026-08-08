# Design Document: The Connector Manifest is pure schema — no compilation, no network, no LLM — with an explicitly DRAFT, never-auto-enforced name-crosswalk heuristic

CONCEPT:AU-KG.ontology.connector-manifest-schema

> `agent_utilities/knowledge_graph/ontology/connector_manifest.py`.

## Decision — separate the manifest's SCHEMA (this module) from its COMPILER (`manifest_compiler`) and its integrity primitives (`ontology_integrity`)

`connector_manifest.py:1-17` states the layering directly: this module
generalizes LeanIX's `ClassSpec`/`ObjectPropertySpec`/`DatatypePropertySpec`
into source-agnostic OWL primitives plus the declarative Connector Ontology
Manifest shape (`resources`/`actions`/`events`/`schema_mappings`/`sync`/
`identity`/`permissions`/`policy`/the signed `provenance` block) — but "this
module is **pure schema** (Pydantic + dataclasses); the compiler that turns a
manifest into OWL/SHACL lives in `manifest_compiler`, and the hash/signature
primitives ... live in `ontology_integrity`." Existing tables are reused, not
re-derived: "the LeanIX compiler already owns the field-value-type -> XSD map
and the ArchiMate crosswalk heuristic; every source-agnostic manifest reuses
the same tables" (`connector_manifest.py:28-31`).

**The rejected alternative is one combined schema+compiler module** — plausible
since both concerns touch the same data shape, but coupling them would mean a
caller that only needs to validate/read a manifest (e.g. a review tool) would
have to import the full compilation/OWL-emission machinery too. Splitting them
means the schema module has zero network/LLM/compilation dependencies — "No
network, no LLM — every field here is either read verbatim from an existing
connector artifact ... or a documented heuristic default flagged in
`review_todos`."

The module's one genuinely risky piece is explicitly quarantined: `HUB_NAME_HEURISTIC_CROSSWALK`
(`connector_manifest.py:57-70`) is a "D16 residue fallback" — when a resource
has neither an explicit `rdfs:subClassOf` nor an ArchiMate crosswalk hit, this
"conservative keyword table gives a best-effort DRAFT crosswalk to the
canonical hub ... class of the same/nearest *name*." **The rejected alternative
is auto-applying the name match** — the table is "deliberately small and
conservative: only common, low-ambiguity domain nouns get an entry; anything
else is left unresolved (`None`) rather than guessed," and every hit is
flagged in `review_todos` as a DRAFT requiring human sign-off — "it is NEVER
auto-enforced."

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/connector_manifest.py`,
  `manifest_compiler.py` (consumes this schema),
  `ontology_integrity.py` (signs/verifies `ProvenanceSpec`).
- **Backward Compatible**: Yes — a schema module; adding a field is additive.
- **Known weak point**: `HUB_NAME_HEURISTIC_CROSSWALK` is a hand-curated
  keyword table — its conservatism protects against false-positive crosswalks
  but means a genuinely correct name-based match for an entry NOT yet in the
  table is simply left unresolved, requiring a human to notice and add it.
