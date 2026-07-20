# Object Backend Overview

> Source: <https://www.palantir.com/docs/foundry/object-backend/overview/>
> Captured during the ontology-parity effort. Concrete feature taxonomy only.

## Core storage & indexing

### Object databases
- **Object Storage V1 (Phonograph)** — legacy canonical DB; consolidated indexing + querying; tracks user edits; sunset 2026-06-30.
- **Object Storage V2** — current; decoupled indexing/querying; horizontal scaling; up to 2,000 properties per object type; incremental indexing on by default; tens of billions of objects per type.

### Object Data Funnel
Orchestration service that ingests from datasources (datasets, restricted views, streaming), processes Action-driven user edits, maintains sync with upstream changes, and supports low-latency streaming.

## Query & access layer

### Object Set Service (OSS)
Read-serving component for filtering/search, aggregation, object loading/retrieval.

**Object sets** (resource type):
- **Static** — fixed lists of primary keys.
- **Dynamic** — filter-based, auto-updating.
- **Temporary** — 24-hour expiring inter-service transfers.
- **Permanent** — persistent saved resources.

### Search & aggregation backend
- Spark-based query execution (OSv2); **Search Around** operations (100,000-object default limit); accuracy via distributed computation.

## Write & modification

### Actions service
- Structured edits; complex permission enforcement; historical action logging; up to 10,000 objects per action (configurable higher).

### Functions on objects
- Operational code execution for custom aggregations, decision-support computations, rapid in-application execution.

## Schema & metadata

### Ontology Metadata Service (OMS)
Definitional layer for object-type schemas, link types, action-type specs, entity metadata.

## Multi-datasource capability

**Multi-Datasource Objects (MDOs)** — property-level permission granularity, column-level access controls, composite objects from multiple sources.

**Key shift (V1→V2):** decoupling indexing from querying enables independent scaling and improved throughput across edit + aggregation workloads.
