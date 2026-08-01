# Design Document: Functions-on-Objects are read primitives composed over the live facade, not an app-defined function layer

CONCEPT:AU-KG.ontology.default-runtime-bound-import

> `agent_utilities/knowledge_graph/ontology/functions/objects.py`.

## Decision — typed read/traverse/aggregate primitives bind to the SAME live `KnowledgeGraph` facade every other read path uses

`objects.py:4-18` names the Palantir Foundry provenance directly:
*Functions-on-Objects* — typed functions that read an object's properties,
traverse its links, and aggregate over object sets. `ObjectFunctionContext`
(`objects.py:75-90`) is the real graph-read substrate those functions stand on:
every helper is a parameterized, read-only Cypher query routed through
`KnowledgeGraph.query` — the facade's *guarded* path (permission-filtered and
audited when enforcement is on, `objects.py:12-17`). When no backend is
reachable the facade returns `[]` and the helpers degrade to empty results
rather than raising — "the logic is real, the failure is graceful."

**The rejected alternative** is a bespoke Functions runtime with its own
storage/execution/permission model — the natural reading of "port Foundry's
Functions-on-Objects" as a standalone feature. Binding through the *existing*
facade instead means every Functions-on-Objects read is automatically
tenant-scoped, ACL-filtered and audit-logged for free — the same guarantees
every other KG read already carries — rather than a second permission surface
that could drift from the first. The one piece of defensive engineering unique
to this module is the relationship-type allowlist (`_REL_TYPE_RE`,
`objects.py:29-48`): a Cypher relationship type cannot be a bind parameter, so
a caller-supplied `rel_type` is validated against a strict identifier pattern
before being spliced into the query string, closing the injection surface that
free-form relationship names would otherwise open.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/functions/objects.py` and any
  `FunctionSpec` handler built as a pure callable over its results.
- **Backward Compatible**: Yes — a new read-only layer over an existing facade.
- **Known weak point**: every helper degrades silently (`[]`) on any exception
  from the facade (`objects.py:69-72`, broad `except Exception`), which is
  correct for "no backend reachable" but also masks a genuine query bug as an
  empty result — a caller cannot distinguish "truly no matches" from "the read
  itself failed" without checking logs.
