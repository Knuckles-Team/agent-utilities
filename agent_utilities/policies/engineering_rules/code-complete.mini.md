---
name: "Code Complete"
description: "Software construction best practices and defensive programming"
author: "Steve McConnell"
version: "1.0.0"
tier: "mini"
---
# Code Complete: Mini

## When to use

Use when you want disciplined construction choices during implementation, especially around defensive coding, control flow, and code shape.

## Primary bias to correct

Do not let local cleverness outrun construction discipline.

## Decision rules

- Treat construction quality as defect prevention, not as cleanup after the fact.
- Choose clarity, locality, and explicitness over clever compactness or novelty.
- Keep routines with one clear purpose and hard-to-misuse interfaces. Separate setup, validation, computation, and side effects when they are different concerns.
- Keep data meaning explicit: small scope, deliberate initialization, clear units, strong types, constants, and no magic sentinels.
- Validate inputs at trust boundaries and use assertions or contracts for programmer assumptions. Make invalid or impossible states visible instead of silently tolerated.
- Handle errors at the right level, preserve diagnostic context, and keep the normal path readable.
- Prefer straightforward control flow. Replace stable condition forests with table-driven or data-driven logic when that makes the rules plainer.
- Keep classes and modules cohesive, hide incidental detail, and reduce coupling through clear contracts and limited knowledge.
- Build in small, verifiable increments. Review code for defect risk, consistency, and rising complexity while constructing it, not only afterward.

## Trigger rules

- When input crosses a trust boundary, decide what is validated, asserted, rejected, or recovered from.
- When a routine accumulates several phases or a long signature, split the responsibility or model the data more clearly.
- When a variable carries units, status, or lifecycle implicitly, turn that meaning into a clearer type, constant, or name.
- When branching grows around a stable mapping, check whether a table or policy object would make the logic more inspectable.
- When a clever shortcut saves lines but costs inspection effort, prefer the boring form.
- When a comment explains obvious code, rewrite or delete it; keep comments for intent, rationale, contracts, and non-obvious facts.

## Final checklist

- Defect risk lower?
- Data meaning explicit?
- Normal path readable?
- Comments explain something the code should not carry alone?
- Small reviewable step?
