---
name: "Domain Driven Design Distilled"
description: "Compact DDD patterns for bounded contexts and aggregates"
author: "Vaughn Vernon"
version: "1.0.0"
tier: "mini"
---
# Domain-Driven Design Distilled: Mini

## When to use

Use when a problem domain has real business complexity but you want a practical DDD core without heavyweight ceremony.

## Primary bias to correct

Do not apply DDD patterns everywhere just because they exist.

## Decision rules

- Start by identifying subdomains and bounded contexts. Apply the richest modeling only where the business complexity justifies it.
- Use one ubiquitous language per bounded context, and let that language drive names in code, use cases, and conversations.
- Use entities for continuity, value objects for descriptive concepts, and domain services only when behavior does not belong naturally on an entity or value object.
- Keep aggregates small and built around true invariants and transactional consistency, not around object graph convenience.
- Use repositories to load and persist aggregates, not as generic query bags or thin ORM facades.
- Keep application services thin. They coordinate use cases, but the domain model owns business rules and decisions.
- Keep infrastructure outside the model and translate foreign shapes at the edge.
- Stay practical: when the domain is simple, use simpler patterns instead of ceremonial DDD.

## Trigger rules

- When business language is fuzzy or overloaded, pause and sharpen the vocabulary before adding more code.
- When a transaction wants to mutate many roots at once, revisit the aggregate boundary or the consistency requirement.
- When a service starts holding domain decisions, move that logic into the model or make the missing concept explicit.
- When DDD jargon appears without a real domain reason, simplify.

## Final checklist

- Clear bounded context?
- One language per concept?
- Small aggregates around real invariants?
- Domain rules living in the domain?
