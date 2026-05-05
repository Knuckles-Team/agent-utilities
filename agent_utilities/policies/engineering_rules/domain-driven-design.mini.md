---
name: "Domain Driven Design"
description: "Strategic domain modeling and bounded context architecture"
author: "Eric Evans"
version: "1.0.0"
tier: "mini"
---
# Domain-Driven Design: Mini

## When to use

Use when the main challenge is understanding and protecting a complex business domain rather than merely organizing technical code.

## Primary bias to correct

Do not let tables, controllers, or ORM convenience define the model.

## Decision rules

- Use DDD only where the domain needs a deep model. Start with knowledge crunching, scenario walkthroughs, and model discovery before choosing patterns.
- Enforce one ubiquitous language per bounded context, and make implicit concepts explicit when the current language causes awkwardness or ambiguity.
- Model bounded contexts deliberately. Use explicit context maps and do not share one model across different meanings just to reduce translation code.
- Put business invariants inside expressive entities, value objects, and small aggregates with valid construction and clear consistency boundaries.
- Use repositories, factories, domain services, specifications, and domain events only when they express domain meaning. Do not add them as generic ceremony.
- Keep the application layer focused on orchestration. Keep infrastructure outside the model and translate foreign models explicitly at boundaries.
- Distill and protect the core domain. Use richer modeling there, and allow simpler styles in supporting or generic areas.
- Prefer supple, model-driven design over anemic domain models, ORM-driven shapes, or pattern-driven obscurity.

## Trigger rules

- When business terms are overloaded or teams argue about meaning, pause and refine the language before extending the model.
- When a transaction wants to touch many objects, decide whether the invariant is local, eventual, or split across contexts.
- When persistence or transport shapes start dictating the model, reassert the domain boundary and add translation.
- When a service accumulates decisions that sound like business policy, make the missing concept or invariant explicit in the model.

## Final checklist

- Is the model driven by domain language instead of storage shape?
- Are context boundaries explicit where meanings diverge?
- Are invariants protected inside the model?
- Is the core domain getting the richest design attention?
