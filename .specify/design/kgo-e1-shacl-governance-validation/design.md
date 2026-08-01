# Design Document: SHACL governance validation is layered (global + domain overrides), not one monolithic shapes file

CONCEPT:AU-KG.ontology.enterprise-governance-validation

> `agent_utilities/knowledge_graph/core/shacl_validator.py`.

## Decision — `SHACLValidator` composes global + domain-specific shape files rather than requiring one combined shapes document

`shacl_validator.py:1-13` states the design: "Validates the materialized RDF
graph against SHACL shapes for enterprise governance compliance. Supports
**layered shapes (global + domain overrides)** using the pyshacl library."
The class docstring restates the same capability set: "Single or multiple
shapes files (layered validation), conformance reporting with violation
details, integration with `OWLBridge` for automatic KG validation"
(`shacl_validator.py:24-32`).

**The rejected alternative is one monolithic shapes file for the whole
platform** — simpler to load, but it would force every domain (finance,
legal, harness, …) to either share one global constraint set or fork the
whole file to add a domain-specific rule. Layering means a domain module
contributes its OWN shapes (e.g. `value_types.shapes.ttl` for value-type
constraints, `governance.shapes.ttl` for action defs, `harness.shapes.ttl`
for the harness gate) that load and validate ALONGSIDE the global set, rather
than requiring every domain's rules to be merged into one file maintained by
whoever owns the platform-wide shapes. This is the SAME validator every
domain-specific SHACL consumer in this codebase already targets — the
value-types compiler (`.specify/design/kgo-cat-value-types/design.md`) and the
harness gate (`.specify/design/kgo-har-harness-as-ontology/design.md`) both
load their shapes into this one validator rather than each rolling their own
SHACL engine integration.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/core/shacl_validator.py`, every consumer
  that registers a domain shapes file (`value_types.shapes.ttl`,
  `governance.shapes.ttl`, `harness.shapes.ttl`, `shapes/*.ttl` generally),
  `OWLBridge` integration.
- **Backward Compatible**: Yes — layering is additive; a deployment with only
  the global shapes behaves as it always did.
- **Known weak point**: layered shapes files are independently authored —
  nothing in this module itself detects two domain shape files that impose
  CONTRADICTORY constraints on the same class/property; pyshacl would surface
  a conflict only as a validation failure at run time, not as a load-time
  shape-consistency check.
