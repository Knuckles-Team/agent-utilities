# Object Permissioning Overview

> Source: <https://www.palantir.com/docs/foundry/object-permissioning/overview/>
> Captured during the ontology-parity effort. Concrete feature taxonomy only.
> Note: the overview page is a hub; mechanism detail lives in the linked sub-pages noted below.

## Two-level authorization model

### Level 1 — Ontology resources (schema)
Schema definitions for structural components: object types (display names, properties, data types, descriptions), link types, action types. **These resources do not refer to actual property/primary-key values** — they define the framework, not the data.

### Level 2 — Objects and links (data)
Actual values: objects with primary keys + property values, and links with concrete data (e.g. an Airplane object with `Plane ID = my_plane_id1`, `Maximum Occupancy = 240`).

## Referenced security capabilities (sub-pages)

| Feature | Purpose |
|---|---|
| Ontology Permissions | Control access to schema resources |
| Managing Object Security | Data-level access controls |
| Restricted-View-Backed Object Types | Granular row/column access |
| Multi-Datasource Objects (MDOs) | Cross-source security integration |
| Object Security Policies | Policy-based enforcement |

Mechanism specifics (marking propagation, discretionary/mandatory controls, column-level restriction) are documented in those linked sub-pages, not the overview.
