---
name: "Release It"
description: "Production stability patterns and anti-patterns for resilient systems"
author: "Michael T. Nygard"
version: "1.0.0"
tier: "mini"
---
# Release It!: Mini

## When to use

Use for services, APIs, jobs, queues, and critical paths that must survive production reality rather than just pass the happy path.

## Primary bias to correct

A correct happy path is not a production design.

## Decision rules

- Assume every dependency can fail, stall, or return bad data. Put timeouts on network and external calls by default.
- Retry only when the operation is safe under duplication, and always bound retries with backoff or jitter. Never create retry storms.
- Isolate failure with circuit breakers, bulkheads, and fast failure so one dependency does not drag down the rest of the system.
- Design for load explicitly: backpressure, queue limits, admission control, rate limiting, and load shedding are architecture, not cleanup.
- Protect state with idempotency and protect the system with validation, quotas, and resource limits.
- Make logs, metrics, and traces sufficient to diagnose degradation, overload, and partial outages.
- Make startup, deployment, caches, APIs, and background jobs fail safely and recover predictably instead of silently wedging.

## Trigger rules

- When adding an external call, choose timeout, retry, and fallback behavior deliberately.
- When adding a queue or async job, define capacity limits, duplicate-safety, and failure handling.
- When adding a cache, decide what happens on miss, stampede, staleness, and invalidation failure.
- When adding a new endpoint or critical path, define overload behavior before traffic defines it for you.

## Final checklist

- Timeout?
- Duplicate-safe retry?
- Failure isolated?
- Overload behavior explicit?
- Observable under stress?
