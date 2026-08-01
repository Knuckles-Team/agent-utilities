# Design Document: Ontology property types carry three coupled facts, not just a Python type

CONCEPT:AU-KG.ontology.ontology-property-types

> `agent_utilities/knowledge_graph/ontology/property_types.py`.

## Decision — every declared property type binds a validator + an XSD/OWL IRI + a storage hint, populated at import as real built-ins

`property_types.py:4-36` names the Foundry provenance ("object-link-types /
type-reference", the ontology *data-types* page documented as "inspired by
RDF, OWL and XSD") and states the three coupled facts every `PropertyType`
carries: (1) a real Pydantic validator (`coerce`/`validate` — never a
pass-through), (2) an XSD/OWL datatype IRI for RDF promotion through
`owl_bridge`, and (3) a storage hint mapped onto the *existing*
`schema_definition` column-type vocabulary (`STRING`, `DOUBLE`, `INT64`,
`TIMESTAMP`, …). `PROPERTY_TYPES` is populated at import with the full
built-in set (`property_types.py:31-36`) — the "import-populated-registry
idiom" the module explicitly names, shared with `interfaces.py` and
`value_types.py`.

**The rejected alternative is a bare type-name string** (what the LPG property
graph would default to without this module) — a property declared `"geohash"`
or `"vector"` as a free string carries no coercion guarantee, no defined RDF
serialization, and no defined storage column, so each of those three concerns
would have to be reinvented ad hoc by every caller that needs one of them
(the RDF bridge, the SHACL validator, the storage layer). Binding all three to
one declaration means a property type is defined once and is automatically
correct everywhere it is used — the cost is that adding a genuinely new
primitive type requires touching this one module's coupled triple rather than
just adding a string literal.

## Risk Assessment

- **Blast Radius**: every ontology property declaration; `value_types.py`
  (built directly on this), `interfaces.py` (`InterfaceProperty` typing),
  `owl_bridge` RDF promotion, `schema_definition` storage mapping.
- **Backward Compatible**: Yes — additive type vocabulary.
- **Known weak point**: `DEFAULT_VECTOR_DIM` resolves from
  `config.kg_embedding_dim` with a `768` fallback on parse failure
  (`property_types.py:56-61`) — a misconfigured embedding dimension silently
  falls back rather than failing loudly, which could produce a vector type
  whose declared dimension does not match the embedding model actually in use.
