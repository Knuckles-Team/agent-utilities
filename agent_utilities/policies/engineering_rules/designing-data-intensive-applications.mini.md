---
name: "Designing Data Intensive Applications"
description: "Data ownership, consistency semantics, and distributed system contracts"
author: "Martin Kleppmann"
version: "1.0.0"
tier: "mini"
---
# Designing Data-Intensive Applications: Mini

## When to use

Use for systems where correctness depends on data ownership, consistency semantics, event flow, replay, schema evolution, or partitioning.

## Primary bias to correct

Do not let storage, messaging, or retries smuggle in hidden correctness assumptions.

## Decision rules

- Give each fact a clear source of truth and owner. Avoid systems where the same truth is written independently in multiple places without one authoritative path.
- Make consistency explicit per write path. Protect immediate invariants inside one local transaction or aggregate by default, and document where staleness or reconciliation is acceptable.
- Require idempotency and replay safety for retried, queued, or event-driven writes. Never rely on wishful exactly-once semantics.
- Preserve ordering only where the product needs it, and encode the ordering contract explicitly with keys, versions, sequence rules, or partition ownership.
- Treat events, streams, and service APIs as contracts. Their schemas must evolve compatibly, and consumers must be able to survive mixed-version traffic.
- Treat caches, indexes, projections, and search views as secondary derived data that may lag and must be rebuildable.
- Partition around locality and access patterns. Make cross-partition or cross-service coordination an explicit design choice, not an accidental side effect.

## Trigger rules

- When touching a write path, state the source of truth, consistency boundary, and failure semantics.
- When adding retries, background jobs, or event consumers, prove the handler is idempotent under duplicate delivery and replay.
- When changing a schema or event, decide how older readers, newer writers, and rebuild paths will work.
- When the design writes one business fact in multiple places, make the coordination contract explicit or collapse to one authoritative write.

## Final checklist

- One clear owner per fact?
- Explicit consistency and staleness semantics?
- Safe under retry, replay, and duplicate delivery?
- Secondary views rebuildable?
