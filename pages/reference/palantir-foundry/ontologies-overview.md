# Ontologies Overview

> Source: <https://www.palantir.com/docs/foundry/ontologies/ontologies-overview/>
> Captured during the ontology-parity effort. Concrete feature taxonomy only.

## Core concept

An **ontology** is the foundational artifact organizing ontological resources across a Foundry instance, with a 1:1 mapping to spaces and organization-based access control.

## Primary resource types

1. **Object types** — fundamental data-entity definitions; properties, metadata, behavioral config; can integrate with external systems; user-editable.
2. **Link types** — relationships between object types; directional and semantic; metadata-reference config.
3. **Action types** — function-backed or rules-based operations; parameters with filtering/defaults; submission criteria; side effects (notifications, webhooks, schedule triggers); inline edits, batched execution, permissions.
4. **Interfaces** — abstract type definitions for polymorphic modeling; implementation + extension; interface-specific link types.
5. **Shared properties** — reusable property definitions across object types; struct-compatible; metadata-reference support.
6. **Object type groups** — organizational containers for related object types (logical categorization, search/explore taxonomy).

## Property system

- **Base types**: primitive and composite structures.
- **Property reducers**: aggregation/transformation logic.
- **Derived properties**: computed attributes.
- **Edit-only properties**: restricted modification scope.
- **Required/mandatory properties**: integrity constraints.
- **Value & conditional formatting**: display customization.
- **Structs**: composite property types with a main-field designation.

## Lifecycle management

- **Branching** — versioning + proposal-review workflows.
- **Change persistence** — save / review / restore change history.
- **Migration** — cross-ontology data movement.
- **Versioning** — function and value-type version management.

## Access & organization

- **Private ontologies** — single-org, org-marked resources.
- **Shared ontologies** — multi-org pools with security isolation.
- **Project-based permissions** and **ontology permissions** (legacy + current models).

## Search & discovery

- Semantic search with document processing; multimodal + embedding-model integration; ontology-augmented generation; SQL-based analysis and text-search syntax.

## Usage monitoring

- Volume tracking, indexing-compute measurement, query-compute usage monitoring.
