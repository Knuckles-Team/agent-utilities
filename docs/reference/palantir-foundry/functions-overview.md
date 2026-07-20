# Functions Overview

> Source: <https://www.palantir.com/docs/foundry/functions/overview/>
> Captured during the ontology-parity effort. Concrete feature taxonomy only.

## Definition

Functions are "code-authored logic that can be executed quickly in operational contexts," with first-class support for authoring logic *based on the Ontology*.

## Languages & runtimes

- **TypeScript**: v1 (legacy) and v2 (current).
- **Python**: full support with a dedicated toolchain.
- Feature support varies by language/version; language choice is feature-driven.

## Function types by execution context

- **Data transformation** — return object sets / variable values for dashboards; compute derived columns in Workshop; chart aggregations.
- **Ontology operations** — read object properties, traverse links, execute flexible Ontology edits across multiple objects, back function-backed actions for complex mutations.
- **Integration** — query external systems via webhooks; enrich objects with external data; API calls from the function body.
- **Domain-specific** — Pipeline Builder sidecars (Python); query functions published via an API gateway; custom aggregations on object types; functions-on-objects (relationship-aware); LLM integrations (TS v2 + Python).

## Ontology binding mechanisms

- **Object access**: direct property access on typed objects; link traversal; object-set queries/filtering; search + aggregation.
- **Edit capabilities**: user-facing mutations through actions; staged writes (TS v2); schema-aware object creation with ID generation; attachment/media handling.

## Lifecycle & deployment

- **Development**: unit-testing framework with stub objects; local dev (Python); debug modes + logging.
- **Publishing/versioning**: versioned releases with dependency ranges; Marketplace product integration; permission-based execution control.
- **Monitoring**: function metrics/perf tracking; execution telemetry; action logs for edit operations.

## Server-side execution model

Functions execute server-side in an isolated environment, enabling operational automation without client-side processing overhead.
