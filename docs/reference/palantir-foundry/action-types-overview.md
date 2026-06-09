# Action Types Overview

> Source: <https://www.palantir.com/docs/foundry/action-types/overview/>
> Captured during the ontology-parity effort. Concrete feature taxonomy only.

## Definition

An **action type** is "the definition of a set of changes or edits to objects, property values, and links that a user can take at once" — a single transaction.

## Architectural components

### Parameters
- User-input mechanism; dropdown selections with filtering; default values; contextual override config; documented performance considerations.

### Execution models
- **Rules-based** — declarative logic for automatic state transitions; condition evaluation + property-modification rules.
- **Function-backed** — programmatic execution via deployed functions; batched execution for bulk ops; TS v1/v2 + Python.

### Submission & validation
- **Submission criteria** — conditional validation before commit; user-authorization checks; data-consistency enforcement.
- **Permission controls** — role-based restrictions; object-level security integration; edit-only property constraints.

## Side effects & downstream actions
- Notification delivery; webhook invocation; schedule/build triggers; cascading edits to related objects.

## Data mutation capabilities
- Direct property modification; link create/modify; attachment/media uploads; inline edits; ontology materialization updates.

## Platform integration
Actions are callable from Object Explorer, Object Views (standard/full/panel), Workshop, interfaces/structs, and external systems via API.

## Observability & governance
- Execution logging + metrics; undo/revert; user edit-history tracking; monitoring dashboards.
