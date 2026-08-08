# Design Document: A priority-list-then-fallback text snapshot for every typed entity's embedding, never a single hardcoded field

CONCEPT:AU-KG.ingest.entity-embedding-at-write

> `agent_utilities/knowledge_graph/enrichment/semantic.py:437-460`
> (`derive_entity_text_snapshot`), `:434-435` (`_ENTITY_FIELD_VALUE_CAP`,
> `_ENTITY_TEXT_CAP`).

## Decision — a priority list of common name/description fields, falling back to concatenating every short string leaf, with the exact selecting values captured for a CAS fence

Every typed-entity connector builds a differently-shaped `dict` (per
`ChangeEnvelope.from_connector_record`'s own docstring), so there is no
single field name that reliably holds embeddable text across connectors —
one source's entity has `title`, another has `name`, another has neither.
`derive_entity_text_snapshot` checks a priority list of common name/title
fields, then common description/body fields, and only if neither produced
anything, falls back to concatenating every short string-valued leaf
property (explicitly skipping ids/timestamps/governance fields) — so an
entity with unusual field names still gets *something* embedded rather than
silently landing with no vector at all.

The function returns not just the derived text but the exact property values
that selected it, explicitly so the snapshot can back an atomic
compare-and-set fence around the (slow) embedding call: if any field that
selected or contributed text changes between the snapshot and the write, the
CAS fails instead of persisting a vector derived from now-stale content.
Missing well-known fields are deliberately included in the snapshot as
`None` — adding a higher-priority name/summary field while the embedder is
mid-flight also changes the derived text and must also fail the CAS, not
just a change to a field already present.

**The rejected alternative is implicit in what the function refuses to do**:
picking one hardcoded field name (e.g. always embed `name`) would silently
skip embedding for any connector whose entities use a different field, and
would have no way to express "no usable text was found" — the actual
contract is `("", conditions)` in that case, which callers must treat as
"defer this record", never as "embed an empty string". A cap on both
per-field value length (`_ENTITY_FIELD_VALUE_CAP = 2000`) and total text
(`_ENTITY_TEXT_CAP = 4000`) exists so one pathological field (an inlined
document body) cannot produce an oversized embedding input via the fallback
path.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/enrichment/semantic.py`
  — consumed wherever a typed entity is embedded at write time.
- **Backward Compatible**: Yes — purely a text-selection helper; does not
  change the embedding model or storage shape.
- **Known weak point**: the fallback concatenation over string leaves is a
  best-effort heuristic, not a schema-aware selection — an entity whose only
  meaningful text lives in a non-string or deeply nested field still yields
  `("", conditions)` and is deferred, matching the module's own stated
  contract rather than silently embedding nothing.
