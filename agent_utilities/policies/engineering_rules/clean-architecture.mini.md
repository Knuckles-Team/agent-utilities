---
name: "Clean Architecture"
description: "Dependency management and architectural boundaries for maintainable systems"
author: "Robert C. Martin"
version: "1.0.0"
tier: "mini"
---
# Clean Architecture: Mini

## When to use

Use for systems where business rules must stay stable while frameworks, databases, transports, and vendors change.

## Primary bias to correct

Do not let framework structure or adapter convenience become the architecture.

## Decision rules

- Enforce the dependency rule: source dependencies point toward policy, never toward frameworks or delivery mechanisms.
- Keep the domain pure. Business rules must not read HTTP objects, ORM models, framework context, queue SDK types, or vendor payloads directly.
- Let the application layer coordinate use cases, transactions, and ports. It should orchestrate business work, not become a dumping ground for domain rules.
- Keep interface adapters responsible for translation: requests to use-case input, domain results to responses, and persistence/transport models to domain-friendly shapes.
- Treat infrastructure as replaceable detail. Put databases, frameworks, SDKs, mailers, queues, and storage behind ports or gateways.
- Define the use case and its boundary models before wiring details. The shape of the framework code should follow the use case, not the reverse.
- Prefer feature-oriented structure so the boundary around a use case stays visible instead of being split across controller/service/repository buckets.
- Test the core without real infrastructure. Add focused adapter tests where translation or integration behavior is the risk.

## Trigger rules

- When framework or ORM types appear in domain or use-case code, stop and move translation to an adapter.
- When a controller, handler, or job starts carrying business rules, move policy inward to the use case or domain.
- When adding a new external dependency, introduce a port if the core would otherwise learn vendor details.
- When a cross-layer call bypasses the intended path, fix the boundary instead of normalizing the shortcut.

## Final checklist

- Could the business rules survive a database, UI, or framework replacement?
- Is the use case visible as a unit, with plain input and output models?
- Are dependency arrows still pointing toward policy?
- Are risky integrations isolated to adapters or infrastructure code?
