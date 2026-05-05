---
name: "Patterns Of Enterprise Application Architecture"
description: "Enterprise integration patterns for data access and domain logic"
author: "Martin Fowler"
version: "1.0.0"
tier: "mini"
---
# Patterns of Enterprise Application Architecture: Mini

## When to use

Use when designing business-layer, persistence, and integration patterns in a traditional enterprise application.

## Primary bias to correct

Do not default to one enterprise pattern everywhere.

## Decision rules

- Choose the business-logic pattern deliberately. Use transaction scripts for simple flows, table-oriented patterns for table-centric logic, and domain models for rich invariants and behavior.
- Use service layer for application workflow and keep controllers or handlers thin. Do not let the delivery layer become the business layer.
- Use remote facades and DTOs only when crossing expensive or unstable boundaries where coarse-grained contracts beat chatty coupling.
- Match the persistence pattern to the problem. Active Record is acceptable for simple CRUD-centric areas; Data Mapper or Repository becomes useful when domain behavior and persistence concerns must stay separate.
- Use Unit of Work, Identity Map, and Lazy Load deliberately so lifecycle, caching, and load behavior stay understandable.
- Make transaction boundaries and lock strategy explicit around real consistency and contention needs.
- Keep integration and distribution coarse-grained and explicit. Remote calls are not local calls with slower syntax.

## Trigger rules

- When adding a repository, ask whether it hides meaningful complexity or merely restates the ORM.
- When a simple use case starts accumulating domain rules, revisit whether transaction script is still enough.
- When remote calls become chatty, redesign the boundary as a coarse-grained contract.
- When contention or duplicate updates matter, choose an explicit lock or conflict strategy instead of hoping the ORM saves you.

## Final checklist

- Right business-logic pattern for the domain complexity?
- Right persistence pattern for the separation need?
- Explicit transaction boundary?
- Remote boundary coarse and intentional?
