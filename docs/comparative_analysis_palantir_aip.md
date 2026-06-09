# Comparative Analysis: agent-utilities Ontology vs. Palantir Foundry

> Capability-by-capability comparison of the **agent-utilities ontology layer**
> (`agent_utilities/knowledge_graph/ontology/` + the `actions/` extension) against
> **Palantir Foundry's Ontology**. Grounded in (a) a code audit of the live
> `OntologySystem` composition root and its registries, and (b) the marketing-
> stripped Foundry reference captures in
> [`reference/palantir-foundry/`](reference/palantir-foundry/README.md).
>
> **Bottom line:** the ontology layer now implements first-class equivalents of
> every Foundry ontology primitive in the capability matrix below — object/link/
> property/value types, interfaces, functions, derived properties, actions, edits,
> indexing, the object-set service, permissioning, document processing, and the
> explorer/views/vertex operator surfaces. Because the substrate is a formal
> OWL2+SHACL knowledge graph over a Rust epistemic engine, several capabilities
> Foundry charges for fall out **structurally** (reasoned ACLs, validation,
> embedding/Cypher/SPARQL-backed derived properties, bitemporal history).
>
> This supersedes the prior AIP "12-capability" framing; that material is retained
> in §6 because it is still accurate at the platform level. This document's primary
> contribution is the **ontology capability gap matrix** (§3).

---

## 1. Scope of the comparison

Foundry's pitch is that the **Ontology** — a governed, executable model unifying
*data + logic + actions + security* — is what stops enterprise agents from
hallucinating and lets them act on real operational state under the same governance
as humans (see [`why-ontology.md`](reference/palantir-foundry/why-ontology.md)).
That is precisely the agent-utilities thesis (the Epistemic Knowledge Graph,
Pillar 2). So the comparison is apples-to-apples. The interesting questions: where
do we match each ontology primitive, and what do we get that a closed commercial
platform structurally cannot?

## 2. How the ontology layer is composed

`OntologySystem` (`knowledge_graph/ontology/__init__.py`) is the composition root
the execution plane reaches through `KnowledgeGraph.ontology`. It binds six
import-populated registries (property types, value types, interfaces, links,
functions, derived properties) plus the durable edit ledger, the object-index
funnel, object-set factories, permissioning, and document processing — and, when
constructed with a live graph, resolves Functions-on-Objects / derived compute /
interface targeting against the real store + L2 semantic (OWL/SHACL) + retrieval
(HNSW) layers. The Action System (`knowledge_graph/actions/`) supplies the verbs.

## 3. Ontology capability gap matrix

Status legend: **FULL** = first-class equivalent implemented and wired;
**FULL+** = implemented *and* exceeds Foundry via the substrate; **PARTIAL** =
implemented core with a scoped backlog. Each row cites the concrete module +
`CONCEPT` id, and the Foundry reference capture it is measured against.

| Foundry capability | Foundry reference | agent-utilities implementation (module · CONCEPT) | Status |
|---|---|---|---|
| **Object types** (entity schema, properties, user edits) | type-reference, ontologies-overview | Object types are KG node types; properties typed/coerced via `ontology/property_types.py` (`PropertyType`, `KG-2.47`); edits via the ledger (`KG-2.43`). | **FULL** |
| **Property types** (base + composite vocabulary) | type-reference | `ontology/property_types.py` (`KG-2.47`): scalars, `decimal`, `date`/`timestamp`, geo (`geohash`/`geoshape`/`geo_point`), `timeseries`/`geotimeseries`, `struct`, `attachment`, `media_reference`, `marking`, `bytes`, `array<…>`, `vector<dim>`/`embedding`. `column_type_for()` bridges to node-table DDL; `coerce_value`/`validate_value` gate the write path. | **FULL+** (vector/embedding native) |
| **Value types** (constrained semantic wrappers) | type-reference | `ontology/value_types.py` (`KG-2.39`): `ValueType`+`ValueConstraints` compiling to **SHACL `PropertyShape` + OWL `rdfs:Datatype`**; built-ins `EmailAddress`, `URL`, `ISOCurrencyCode`, `Percentage`, … ; gates writes via the SHACL validator. | **FULL+** (OWL/SHACL-backed) |
| **Link types** (typed relationships) + many-to-many | type-reference | `ontology/links.py` (`KG-2.26`): `LinkType` (cardinality), `JunctionLinkType` reifying M:N links as first-class junction **objects** + role-keyed edges; reverse traversal (`endpoints_of`, `junctions_for`, `neighbors_via`). | **FULL+** (reified junction = first-class) |
| **Interfaces** (shared shape, polymorphism, inheritance) | interface-overview | `ontology/interfaces.py` (`KG-2.38`): `Interface` w/ shared properties + link constraints, multi-level inheritance, conformance check, `resolve_target()` programmatic targeting, **OWL `rdfs:subClassOf` + `sh:node` projection**. | **FULL+** (reasoning-aware targeting) |
| **Functions** (code-authored ontology logic) | functions-overview | `ontology/functions/` (`KG-2.41`): `FunctionSpec`/`FunctionParameter`, three Foundry kinds (`PLAIN \| ON_OBJECTS \| QUERY`), `ObjectFunctionContext` (Functions-on-Objects graph reads), single audited `FunctionRuntime` w/ typed I/O validation + versioned release. | **FULL** |
| **Derived properties** (computed attributes) | ontologies-overview | `ontology/derived_properties.py` (`KG-2.40`): `DerivedBacking` = `FUNCTION` **+ `CYPHER` + `SPARQL` + `EMBEDDING`** (Foundry has FUNCTION only); read-time compute + audit via the bound engine. | **FULL+** (graph/semantic/vector backings) |
| **Action types** (transactional verbs) | action-types-overview | `knowledge_graph/actions/` (`KG-2.25` core + `KG-2.42` action-type extension): `OntologyAction` w/ typed parameters, submission criteria, function-backing, side effects (notification/webhook), batched + permission-gated + audited + KG-persisted execution. | **FULL+** (reasoned eligibility) |
| **Object edits** (property-set / link / create-delete, history, revert) | object-edits-overview (stub), action-types | `ontology/edits/` (`KG-2.43`): `Edit`+`EditLedger` as durable `object_edit` nodes with before/after snapshots, per-object `history()`, point-in-time `as_of()`, inverse `revert_edit`, and `WriteBackRouter`/`JsonlEditSink` writeback. | **FULL+** (bitemporal, durable) |
| **Object indexing** (batch + streaming, restrictions, staleness) | object-indexing-overview, object-backend | `ontology/indexing/` (`KG-2.44`): `ObjectIndexFunnel` (batch full-rebuild + incremental/streaming deltas) gated by composable `DataRestriction`; `StalenessLedger` (content-hash + watermark drift) — drives the **same** `CapabilityIndex` the router ranks against. | **FULL** |
| **Object Set Service** (static/dynamic, search-around, aggregate) | object-backend | `ontology/object_set.py` (`KG-2.45`): `ObjectSet` (`STATIC`/`DYNAMIC`), `filter`, `search`, `search_around` (typed N-hop traversal), `aggregate`, `pivot`; factories `object_set_from_ids`/`object_set_of_type`/`dynamic_object_set`. | **FULL** |
| **Object permissioning** (markings, row/col security) | object-permissioning-overview | `ontology/permissioning.py` (`KG-2.46`): `Marking`, per-property classification, `redact_object`/`restricted_view`, `build_acl`, `enforce`, and **entailment-aware `propagate_markings`/`propagate_over_edges`**. Backed by the `permissions_kernel`. | **FULL+** (entailment propagation) |
| **Document processing** (media → chunk → embed → objects) | document-processing | `ontology/document_processing.py` (`KG-2.48`): recursive separator-priority `chunk_text` w/ overlap, explode, embed, materialize as first-class `Chunk` objects (`HAS_CHUNK`/`CHUNK_OF` links) with a clear `DocumentExtractionError` degradation path. | **FULL** |
| **Object Explorer** (search/filter/aggregate, bulk actions, pivot, saved) | object-explorer-overview | webui `ObjectExplorerView.tsx` + `/api/enhanced/ontology/object-set/{search,search-around,pivot,aggregate,save,list}` + `…/actions` bulk apply. | **FULL** |
| **Object Views** (standard/full/panel, props/links/actions) | object-views-overview | webui `ObjectView.tsx` + `/api/enhanced/ontology/object/{id}`, `…/object-view/{type}` (configurable), edit/revert endpoints. | **FULL** |
| **Vertex** (graph viz, expand, scenarios) | vertex-overview | webui `VertexView.tsx` over `…/object-set/search-around` + object/link reads. | **PARTIAL** (graph viz + traversal; what-if scenarios are backlog) |
| **Object backend / OMS** (schema service) | object-backend | Ontology Metadata role played by the import-populated registries + the KG node/link-type catalog; OSS role by `object_set.py`; funnel by `indexing/`. | **FULL** |

## 4. Unique value-adds — what we get *for free* that Foundry structurally cannot

These are emergent advantages of the substrate, not catch-up items.

1. **OWL2 + SHACL behind the type system, not just typing.** Value types compile to
   SHACL `PropertyShape`s + OWL `rdfs:Datatype` (`value_types.py`) and interfaces
   project to `rdfs:subClassOf` + `sh:node` (`interfaces.py`). Foundry's value types
   are validation metadata; ours are *reasoning + validation* — invalid nodes are
   quarantined by the SHACL gate before they persist, and interface conformance is a
   logical entailment, not a hand-maintained tag.
2. **Reasoned, entailment-aware ACL marking propagation.** `permissioning.py`
   propagates markings *over the graph's edges* (`propagate_over_edges`), so a
   marking on a sensitive node flows to derived/linked nodes by inference rather than
   manual re-marking. Foundry markings are applied per resource.
3. **Derived properties backed by embedding / Cypher / SPARQL — not only Functions.**
   `derived_properties.py` adds `CYPHER`, `SPARQL`, and `EMBEDDING` (nearest-concept
   vector similarity) backings alongside Foundry's lone `FUNCTION` backing.
4. **Reified junction links are first-class objects.** `JunctionLinkType` materializes
   a M:N relationship as a queryable node with role-keyed edges and its own
   properties/history — the relationship itself can be reasoned over and edited.
5. **Bitemporal, durable edit history with point-in-time reconstruction.** The edit
   ledger persists every edit as a graph node and supports `as_of(ts)` reconstruction
   and inverse-edit revert (`edits/`), going beyond Foundry's action-log/undo.
6. **Self-evolving ontology over a Rust epistemic engine.** The same KG that hosts the
   ontology drives the golden-loop self-evolution and reward-weighted routing; the
   ontology is not a static schema artifact but a live, improvable structure, and all
   compute (PageRank/spectral/VF2/quant) runs in the `epistemic-graph` engine over
   MessagePack/UDS rather than a backend DB round-trip.
7. **Vendor-neutral, MCP-first, multi-tenant cohabitation.** Humans and any MCP client
   share one governed KG under the same policy fabric; the ontology surfaces are
   reachable via MCP tools (`ontology_*`) and `/api/enhanced/ontology/*` alike.

## 5. Genuine gaps (and disposition)

| Gap | Severity | Disposition |
|---|---|---|
| Vertex **what-if scenario** modeling (staged alternate graphs) | MEDIUM | Backlog — graph viz + traversal shipped; scenario staging not yet. |
| Interface **aggregation** across implementers in the object-set service | LOW | Backlog (Foundry itself lists this "in development"). |
| Ontology **branching / proposal-review** workflow as a UI primitive | MEDIUM | Backlog — edits are versioned/durable; a branch-and-merge review surface is future. |
| Release **channels** for functions (beta/stable, canary) | MEDIUM | Backlog — functions are versioned/released; a channel system is infra. |

None of the above are half-implemented in the tree; each is a scoped future concept.

## 6. Prior platform-level framing (still accurate)

The earlier AIP 12-capability / 9-layer mapping remains valid at the *platform*
altitude (it compares agent-utilities to Palantir AIP broadly, not the ontology
primitives specifically). It is summarized here; the ontology matrix in §3 is the
authoritative comparison for the ontology layer.

- agent-utilities implements ~10/12 AIP capabilities in full, exceeds on the formal
  OWL+SHACL ontology, graph-native Rust compute, closed-loop self-evolution, and
  reward-weighted routing axes, with packaging/release-channels and a formal HITL
  escalation matrix as documented backlog.
- Anchors: `core/model_factory.py`, `security/permissions_kernel.py`,
  `observability/`, `knowledge_graph/core/{owl_bridge,shacl_validator,ogm}.py`,
  `retrieval/capability_index.py`, `harness/`, `orchestration/`.

---

*Source of truth for the Foundry-ontology comparison. Capability claims are anchored
to concrete modules + CONCEPT ids above and to the captures in
[`reference/palantir-foundry/`](reference/palantir-foundry/README.md). Re-audit when
the ontology layer changes.*
