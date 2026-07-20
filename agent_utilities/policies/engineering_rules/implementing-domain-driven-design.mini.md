---
name: "Implementing Domain Driven Design"
description: "Tactical DDD implementation with aggregates, repositories, and events"
author: "Vaughn Vernon"
version: "1.0.0"
tier: "mini"
---
# Implementing Domain-Driven Design: Mini

## When to use

Use when you already know DDD is warranted and need implementation rules for aggregates, repositories, events, and context integration.

## Primary bias to correct

Do not let object graphs or ORM convenience erase consistency boundaries.

## Decision rules

- Treat bounded contexts as real implementation boundaries. Keep context maps explicit enough that code structure and integration choices reflect them.
- Protect the core domain and keep simpler areas simpler instead of spreading heavyweight modeling uniformly.
- Treat aggregates as consistency boundaries. Enforce aggregate-root discipline and default to one aggregate per transaction.
- Reference other aggregates by identity, not by direct object graph traversal. Use application services and domain events to coordinate work across boundaries.
- Use entities for continuity and value objects for descriptive concepts; keep value objects immutable unless a stronger reason exists.
- Keep repositories focused on aggregate access, not on ad hoc query dumping or generic CRUD abstractions detached from the model.
- Keep application services thin: they coordinate use cases, but business rules and invariants live in the domain.
- Use anti-corruption layers and explicit identity translation when integrating different bounded contexts.

## Trigger rules

- When a transaction wants to update multiple aggregates synchronously, re-check whether the invariant is truly immediate.
- When code starts navigating across aggregate internals, replace graph coupling with identity references and coordination.
- When names drift to generic technical placeholders or one term starts carrying multiple meanings inside a context, rename back to one domain term per concept across code, tests, commands, events, and repositories.
- When one context starts reusing another context's terms or model types directly, add translation instead of sharing the model.
- When a repository API starts looking generic and pattern-driven, narrow it back to aggregate needs.

## Final checklist

- Clear bounded context?
- One term per concept inside the context?
- One aggregate per immediate invariant?
- Cross-context translation explicit?
- Application services coordinating rather than deciding?
