# Object Indexing Overview

> Source: <https://www.palantir.com/docs/foundry/object-indexing/overview/>
> Captured during the ontology-parity effort. Concrete feature taxonomy only.

## Overview

Indexing makes tabular (or other) datasource data available through specialized databases for faster retrieval. The **Object Data Funnel** service orchestrates this for Object Storage V2.

## Pipeline types

- **Batch pipelines** — create/modify object instances on a schedule; for higher-latency-tolerant sources; cost-conscious.
- **Streaming pipelines** — real-time availability; low-latency; continuous ingestion/updates.

## Key capabilities

- **Data processing** — transforms varied formats into queryable object instances; handles metadata alongside primary data.
- **Service architecture** — Funnel orchestrates pipelines, ensures freshness/consistency, maintains metadata alongside indexed objects.

## Data restrictions

- Documented constraints on indexable data types; config controls for sensitive info; access patterns defined at the indexing layer.

## Maintenance & lifecycle

- Schema-change management for evolving object types; data-freshness considerations between batch schedules; pipeline reconfiguration on datasource changes.

> Legacy Object Storage V1 (Phonograph) indexing docs remain for existing implementations.
