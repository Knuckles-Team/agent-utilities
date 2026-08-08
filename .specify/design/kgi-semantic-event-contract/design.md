# Design Document: OCEL is an interchange format, not a second event store — one validated boundary for object-centric, temporal, and neural graph proposals

> `agent_utilities/knowledge_graph/ingestion/semantic_event_model.py` (the
> canonical boundary models), `agent_utilities/knowledge_graph/ingestion/ocel_adapter.py`
> (the OCEL 2.0 JSON adapter).

CONCEPT:AU-KG.ingest.semantic-event-contract

## Decision — canonical boundary models, no LLM calls, no independent persistence

`semantic_event_model.py:1-7`, `ocel_adapter.py:1-14`.

**The rejected alternative, named directly in the `ocel_adapter` docstring**:
treating OCEL (Object-Centric Event Log, the published open standard —
`ocel-standard.org`) as a second event store living alongside the KG's own
temporal Event Knowledge Graph (tEKG) state — i.e., persisting/reasoning over
OCEL data in its own right rather than through the graph.

**The design chosen**: OCEL-shaped source truth, temporal Event Knowledge
Graph state, and neural-graph proposals all share ONE validated boundary —
the canonical models in `semantic_event_model.py`. These models perform NO
LLM calls and do not persist by themselves; every caller commits its
canonical graph slice through `ChangeEnvelope` (see
`.specify/design/kgi-change-envelope-atomic/design.md`), the same atomic
commit path every other typed source uses. `ocel_adapter.py` implements the
OCEL 2.0 four-array JSON format (`eventTypes`, `objectTypes`, `events`,
`objects`) per the published specification, validates cross-references and
declared attribute types WITHOUT an LLM (pure structural/schema validation),
and maps the result to the canonical tEKG slice — OCEL is consumed purely as
an *interchange format at the boundary*, never as the system's own event
representation.

**Why this matters**: if OCEL data were persisted as its own store, the KG
would need to reconcile two independently-evolving event representations
(its own tEKG state and a separate OCEL store) any time they needed to be
queried together. Collapsing to one canonical boundary means every event
source — OCEL-shaped, or the KG's native temporal events, or a neural-graph
proposal — converges on the same typed model before it ever reaches the
graph, so downstream consumers (process-intelligence queries, the
`ontology_process_intelligence.ttl`/`process_intelligence.shapes.ttl`
reasoning layer) reason over one representation, not several.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ingestion/semantic_event_model.py`,
  `agent_utilities/knowledge_graph/ingestion/ocel_adapter.py`,
  `agent_utilities/knowledge_graph/ontology_process_intelligence.ttl`,
  `agent_utilities/knowledge_graph/shapes/process_intelligence.shapes.ttl`.
- **Backward Compatible**: Yes — the adapter is additive; sources not
  emitting OCEL-shaped data are unaffected.
- **Breaking Changes**: None.
- **Known weak point**: the published OCEL 2.0 JSON schema "currently
  describes attribute values as strings" (per the adapter's own comment) even
  though the spec's scalar-type page documents five scalar types — the
  adapter's validation follows the spec page's types rather than the
  (looser) published schema, so a strictly schema-conformant-but-spec-violating
  OCEL file could pass schema validation elsewhere but be rejected here, or
  vice versa, depending on which authority a producer followed.
