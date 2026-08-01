# Design Document: Ontologies are generic (iri, version)-keyed registrations, browsable as a curated gallery

CONCEPT:AU-KG.ontology.manage-arbitrary ·
CONCEPT:AU-KG.ontology.catalogue-browse

> `agent_utilities/knowledge_graph/ontology/lifecycle.py` (`OntologyLifecycle`,
> the storage/CRUD decision) and `agent_utilities/gateway/ontology_api.py` +
> `agent_utilities/mcp/tools/ontology_tools.py` (the browse/search REST+MCP
> surface built on it).

## Decision — host ANY ontology the caller loads, keyed generically, not a fixed enumerated set

`CONCEPT:AU-KG.ontology.manage-arbitrary`

`OntologyLifecycle` (`lifecycle.py:430-459`) is documented directly as "CRUD
lifecycle for ontologies hosted in the running KG." `load()` (`lifecycle.py:554`)
accepts an arbitrary turtle/URL/text `source`, parses and SHACL-validates it,
resolves an IRI (from the parsed graph, an explicit `iri=`, or a synthesized
`urn:hosted-ontology:<hash>` when neither is given), and stores the record keyed
by `(resolved_iri, resolved_version)` — idempotent on that key unless `force`.

**The rejected alternative is a fixed, enumerated ontology set** — the "one
canonical bundled TBox" model most graph platforms ship, where hosting a new
ontology means a code change to register it. `manage-arbitrary` instead makes
hosting self-service: any caller (a fleet package, an operator, a workspace
provider) can `load()` a new ontology at runtime with no code change, scoped to
a per-tenant dedicated graph name (`_ontology_graph_name`, reusing the same
`tenant_graph_name` convention every other tenant-scoped engine access uses —
see `.specify/design/ontology-governed-evolution/design.md` for the dedicated-
graph rationale this builds on). The cost accepted: there is no fixed schema the
platform can assume is always present, so every consumer that reasons over
"the ontology" must resolve which one(s) are active rather than hard-coding an
IRI.

### Pointer — `CONCEPT:AU-KG.ontology.catalogue-browse`

`ontology_api.py:325-364`, `lifecycle.py:571-574` and `lifecycle.py:648-679`.
Once hosting is arbitrary/open, the hosted set needs a way to be found again —
this is the curated-library **browse** surface: `GET /ontology/catalogue` and
`list_ontologies(search=, category=, source_type=, tag=)` layer case-insensitive
substring/facet filters over the existing single registry. The docstring is
explicit about the rejected framing: this is Ontology-Playground's "gallery
of many interchangeable demo ontologies" concept, **narrowed** to "one
continuously-extended ontology library" — filtering, not a second index. No new
storage is added; `category`/`tags` are optional curation metadata stored
directly on the existing lifecycle record (`lifecycle.py:632-633`), and every
filter defaults to unset so a plain `list_ontologies()` call is byte-for-byte
unchanged from before the browse surface existed. `deprecated_only` (built on
`CONCEPT:AU-KG.ontology.deprecation-workflow`, already covered by
`.specify/design/ontology-governed-evolution/design.md`) is an independent
facet axis alongside `active_only` — a version can be both deprecated and
still active mid-migration, so neither filter implies the other.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/lifecycle.py`,
  `gateway/ontology_api.py`, `mcp/tools/ontology_tools.py`.
- **Backward Compatible**: Yes — `catalogue-browse`'s filters are additive and
  default to unset; `manage-arbitrary` is the pre-existing storage contract, not
  a new one.
- **Known weak point**: nothing enforces that `category`/`tags` curation
  metadata is populated — an ontology loaded without them is fully hosted and
  reasoned-over but invisible to facet filtering (falls back to being found only
  by `search` substring match).
