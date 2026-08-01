# Design Document: The ontology import/export REST surface is a granular typed twin of the existing MCP action dispatch, never new business logic

CONCEPT:AU-KG.ontology.import-export-rest-surface ·
CONCEPT:AU-KG.ontology.standalone-generation

> `agent_utilities/gateway/ontology_api.py`.

## Decision — `POST /ontology/load` and `GET /ontology/export` are typed REST routes over the SAME `graph_ontology` action dispatch the MCP tool already uses

`CONCEPT:AU-KG.ontology.import-export-rest-surface`

`ontology_api.py:249-320` (section header: "Import / export ...
Ontology-Playground coverage row #23") states the pattern: `load_ontology`
"Granular typed twin of `graph_ontology(action='load')` — the route the
agent-webui Import/Export modal POSTs a dropped/pasted `.ttl`/RDF file (or a
file path / URL) to. **Same core as the collapsed MCP surface; no new
business logic here.**" `export_ontology` mirrors it for the reverse
direction, re-serializing a hosted ontology to turtle.

**The rejected alternative is a separate REST-native import/export
implementation** — parse/validate/register logic duplicated for the HTTP
surface instead of routed through the identical `_call("graph_ontology", ...)`
dispatch every MCP-tool caller already uses. Duplicating it would mean a fix
or a new validation rule applied to one surface (say, the MCP tool) could
silently fail to apply to the other (the REST route) unless someone remembered
to patch both — the granular-typed-twin pattern makes that impossible by
construction, since both surfaces are typed request/response wrappers around
the one shared action dispatcher.

### Pointer — `CONCEPT:AU-KG.ontology.standalone-generation`

`ontology_api.py:151-205`, `knowledge_graph/extraction/schema_discovery.py:
284-302` (`generate_standalone_ontology`, Ontology-Playground coverage row
#13). `GET`/`POST /ontology/generate` expose the from-scratch schema-discovery
path — "Same schema-discovery LLM path as `discover_extensions`, run against
an EMPTY base instead of a live-ontology diff" (`ontology_api.py:176-179`).
The REST docstring restates the invariant covered in depth by
`.specify/design/kgo-cat-no-auto-merge/design.md`
(`CONCEPT:AU-KG.ontology.do-not-auto-merge`): "Always a human-reviewed
proposal — never auto-applied/merged." The `GET` variant exists for short
samples that fit a query string; the `POST` variant (`OntologyGenerateRequest`
body) exists for a longer `sample_text` than comfortably fits one — a real,
if small, API-ergonomics decision: two routes to the same underlying
`ontology_derive action='generate'` call rather than forcing every caller
through a URL-length-constrained `GET`.

## Risk Assessment

- **Blast Radius**: `gateway/ontology_api.py`
  (`/ontology/load`, `/ontology/export`, `/ontology/generate`), the
  agent-webui Import/Export modal.
- **Backward Compatible**: Yes — additive typed REST routes over an existing
  dispatch surface.
- **Known weak point**: because both REST and MCP route through
  `_call("graph_ontology", ...)`, a bug in that shared dispatcher affects both
  surfaces simultaneously with no independent fallback — the granular-twin
  pattern trades duplication risk for shared-failure risk.
