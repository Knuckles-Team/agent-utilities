# Palantir Foundry Ontology — Reference Capture

Clean, marketing-stripped captures of the canonical Palantir Foundry Ontology
documentation, used as the grounding reference for the agent-utilities
ontology-parity effort. Each file records the **source URL** at its top.

> Capture provenance: these pages were fetched and distilled **during the
> ontology-parity effort** (no exact date asserted). Where a page was a
> navigation/landing stub with no substantive body, that is flagged inline and the
> fetch was retried once.

The comparative analysis that uses these captures lives at
[`../../comparative_analysis_palantir_aip.md`](../../comparative_analysis_palantir_aip.md);
the implemented system architecture is at
[`../../architecture/ontology_system.md`](../../architecture/ontology_system.md).

## Index

| File | One-line summary | Source URL |
|---|---|---|
| [why-ontology.md](why-ontology.md) | Decision-centric data + logic + action + security framing; lifecycle (define → branch → consume). | <https://www.palantir.com/docs/foundry/ontology/why-ontology/> |
| [ontologies-overview.md](ontologies-overview.md) | The six ontology resource types, property system, branching/migration lifecycle, private/shared access. | <https://www.palantir.com/docs/foundry/ontologies/ontologies-overview/> |
| [functions-overview.md](functions-overview.md) | Code-authored ontology logic; TS v1/v2 + Python; function types, ontology binding, versioned publish. | <https://www.palantir.com/docs/foundry/functions/overview/> |
| [type-reference.md](type-reference.md) | Object/property/shared-property/link/action types; value types; type-level vs instance-level. | <https://www.palantir.com/docs/foundry/object-link-types/type-reference/> |
| [action-types-overview.md](action-types-overview.md) | Action = one transaction of edits; rules-based vs function-backed; submission criteria; side effects; revert. | <https://www.palantir.com/docs/foundry/action-types/overview/> |
| [interface-overview.md](interface-overview.md) | Interface shape + shared props + link constraints; multi-level inheritance; programmatic targeting. | <https://www.palantir.com/docs/foundry/interfaces/interface-overview/> |
| [document-processing.md](document-processing.md) | Media → text-extract → chunk(overlap) → explode → embed → Chunk objects linked to source; search. | <https://www.palantir.com/docs/foundry/ontology/document-processing/> |
| [object-explorer-overview.md](object-explorer-overview.md) | Search/filter, charts/SQL, object-set compare, bulk actions, pivot (link traversal), saved explorations. | <https://www.palantir.com/docs/foundry/object-explorer/overview/> |
| [object-views-overview.md](object-views-overview.md) | Standard vs configured views; full vs panel form factors; property/link/action/related-app components. | <https://www.palantir.com/docs/foundry/object-views/overview/> |
| [vertex-overview.md](vertex-overview.md) | Graph visualization + templates; scenarios/what-if; time-series + media/annotation; action invocation. | <https://www.palantir.com/docs/foundry/vertex/overview/> |
| [object-backend-overview.md](object-backend-overview.md) | OSv2 storage; Object Data Funnel; Object Set Service (static/dynamic/temp/permanent); search-around; OMS; MDOs. | <https://www.palantir.com/docs/foundry/object-backend/overview/> |
| [object-permissioning-overview.md](object-permissioning-overview.md) | Two-level authz (schema resources vs object/link data); RV-backed row/col controls; security policies. | <https://www.palantir.com/docs/foundry/object-permissioning/overview/> |
| [object-indexing-overview.md](object-indexing-overview.md) | Batch vs streaming pipelines; Object Data Funnel; data restrictions; freshness/reindex lifecycle. | <https://www.palantir.com/docs/foundry/object-indexing/overview/> |
| [object-edits-overview.md](object-edits-overview.md) | Edit = property-set / link add-remove / object create-delete via an Action; landing page (stub) + reconstructed model. | <https://www.palantir.com/docs/foundry/object-edits/overview/> |

## Capture note

`object-edits-overview.md` is the only page whose URL is a navigation/landing
stub with no substantive feature body. It was fetched twice (one retry); both
returned only the landing facts (recorded verbatim in that file), with the detailed
edit/undo model documented inline within the Action Types and Object Backend pages.
All other captures are distilled directly from their source pages.
