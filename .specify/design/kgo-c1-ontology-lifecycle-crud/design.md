# Design Document: Unloading an ontology retracts its axioms from the live engine, not just the registry record — and validation surfaces the raw SHACL report

CONCEPT:AU-KG.ontology.ontology-lifecycle ·
CONCEPT:AU-KG.ontology.shacl-report-passthrough

> `agent_utilities/knowledge_graph/ontology/lifecycle.py`.

## Decision — `delete()` physically retracts axioms via `remove_triples`, not just a registry-record removal

`CONCEPT:AU-KG.ontology.ontology-lifecycle`

`OntologyLifecycle.delete()` (`lifecycle.py:815-827`) states the wiring
directly: "Unload an ontology: retract its axioms from the engine + drop the
registry record ... wires KG-2.265's unload to the engine's native
`remove_triples` retract op." The stored serialized turtle for each matched
version is fed back through `_retract_axioms` (`lifecycle.py:790-813`) so the
ontology's triples physically leave the engine's RDF dataset — "no longer
reasoned over / SPARQL-queryable" — before the hosted-registry record is
removed.

**The rejected alternative is registry-only deletion** — drop the bookkeeping
record and leave the axioms live in the engine's RDF dataset. That would make
`delete()` a lie: `list_ontologies()` would correctly stop showing the
ontology, but a SPARQL query or OWL reasoning pass would still see its
classes/individuals, silently contradicting what the caller was told happened.
`_retract_axioms` degrades honestly instead of pretending success: when no
engine is attached, or the engine has no `remove_triples` op, it reports
`retracted_from_engine: False` with a reason, rather than reporting a
successful retraction that didn't occur — the same fail-closed-truthfulness
discipline `activation-fails-closed` applies to `load()` (see
`.specify/design/ontology-governed-evolution/design.md`), applied here to the
opposite direction of the lifecycle.

### Pointer — `CONCEPT:AU-KG.ontology.shacl-report-passthrough`

`lifecycle.py:166-182` and `tests/unit/knowledge_graph/ontology/test_lifecycle.py:158`.
`validate_graph()` mirrors the bundled-library gate at single-candidate
granularity, and the comment states the specific decision: `shacl_report` is
"populated only when SHACL actually ran against bundled shapes ... the
literal `sh:ValidationReport`" — surfaced whole "so a caller (the
`ontology_api.py` `/ontology/validate` REST twin, the agent-webui SHACL
validation-report view) gets the **real pyshacl report** instead of just the
derived valid/errors/warnings summary." The rejected alternative is
collapsing pyshacl's structured violation report down to the module's own
`errors`/`warnings` string lists before it ever reaches a caller — which would
lose exactly the structured detail (which shape, which focus node, which
constraint) a validation-report UI needs to point a user at the actual
problem. Passing the raw report through means the summary and the full report
coexist rather than one being derived-and-discarded from the other.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/lifecycle.py`
  (`delete`/`_retract_axioms`/`validate_graph`), `ontology_api.py`
  `/ontology/validate` route.
- **Backward Compatible**: Yes — `delete()`'s prior registry-only behavior is
  now the explicit degrade path (no engine/op available), not silently
  dropped.
- **Known weak point**: `_retract_axioms` re-parses the STORED serialized
  turtle to know what to retract — if the engine's actual live axiom set ever
  diverged from that stored turtle (e.g. a direct engine write bypassing this
  lifecycle), retraction would remove what the registry believes was loaded,
  not necessarily everything actually present.
