# Design Document: Value types compile ONE declaration to three enforcement artifacts, not three hand-maintained ones

CONCEPT:AU-KG.ontology.value-type-shacl-load

> `agent_utilities/knowledge_graph/ontology/value_types.py`.

## Decision — a value type declared once emits a runtime validator, a SHACL shape, and an OWL datatype restriction

`value_types.py:4-44` names the Foundry provenance directly: a *value type*
(e.g. `EmailAddress`, `ISOCurrencyCode`) is a semantic wrapper around a base
`PropertyType` plus a `ValueConstraints` block (pattern, numeric/length
bounds, an allowed-value enum), reusable across many properties so the
constraint is authored once and enforced everywhere it's applied. The module
compiles one `ValueType` declaration into three coupled artifacts: (1)
`validate`/`coerce` — the runtime check, layered on top of the base
`PropertyType` coercion; (2) `to_shacl` — a `sh:NodeShape` turtle fragment that
`write_value_shapes_ttl` materializes into `shapes/value_types.shapes.ttl`, a
file the *existing* `SHACLValidator` loads exactly like `governance.shapes.ttl`
(`value_types.py:29-34`); and (3) `to_owl` — an `rdfs:Datatype` with
`owl:withRestrictions` facets over the base XSD type, so the value type
round-trips into the `owl_bridge` RDF/OWL substrate.

**The rejected alternative is enforcing the constraint at only one layer** —
e.g. a Pydantic validator alone, which would let a graph write that bypasses
the Python validation path (a direct Cypher/SPARQL write, a bulk import) land
data the value type was supposed to forbid. By compiling to a SHACL shape that
loads into the *same* validator every other write-time constraint uses, an
`ISOCurrencyCode` or `Percentage` violation is caught at the graph-write gate
regardless of which code path produced the write — not only when the value
happened to pass through this module's own `validate()` call first. `VALUE_TYPES`
is import-populated with real built-ins (`EmailAddress`, `ISOCurrencyCode`,
`Percentage`, `URL`, `E164PhoneNumber`, `Probability`) rather than shipped as an
empty registry a caller must first populate.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/value_types.py`,
  `shapes/value_types.shapes.ttl`, `core/shacl_validator.py`.
- **Backward Compatible**: Yes — additive; a property with no declared value
  type is validated only by its base `PropertyType`.
- **Known weak point**: `to_shacl`/`to_owl` are generated FROM the Python
  `ValueConstraints` declaration; the three artifacts stay coupled only as long
  as `write_value_shapes_ttl` is actually re-run after a `ValueType` changes —
  nothing forces `shapes/value_types.shapes.ttl` to be regenerated automatically
  on every registry change, so a stale shapes file could under-enforce a
  constraint the Python validator already enforces.
