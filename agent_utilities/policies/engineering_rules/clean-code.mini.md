---
name: "Clean Code"
description: "Code readability, naming, functions, and testing discipline"
author: "Robert C. Martin"
version: "1.0.0"
tier: "mini"
---
# Clean Code: Mini

## When to use

Use when readability, local reasoning, and maintainable code shape are the main concerns, especially during everyday implementation and review.

## Primary bias to correct

Working code is not automatically clean code.

## Decision rules

- Write for local reasoning. A reader should understand the path without reconstructing hidden state, wide jumps, or naming trivia.
- Use precise names and one term per concept. Rename code when vocabulary hides intent, overloads meaning, or forces comments to compensate.
- Keep functions small, focused, and at one level of abstraction. Tell the story top-down so intent appears before detail.
- Keep parameters few and meaningful. Avoid boolean flags, output parameters, and grab-bag argument lists; model the concept instead.
- Separate commands from queries and eliminate hidden side effects. A function that answers should not also mutate behind the reader's back.
- Keep the happy path readable. Isolate error handling, invalid-state handling, and boundary adaptation so the main behavior stays obvious.
- Expose behavior rather than raw representation. Avoid train-wreck access, utility dumping grounds, and classes or modules with mixed responsibilities.
- Make public APIs small, explicit, and hard to misuse. Organize code so likely changes stay local.
- Use comments only for rationale, constraints, warnings, or external contracts. Do not narrate code instead of improving it.
- Treat tests as production code: readable, deterministic, and aligned with the behavior or contract they protect.
- When touching code, remove the smell that most increases change cost: duplication, deep nesting, misleading names, primitive obsession, hidden dependency, or accidental complexity.

## Trigger rules

- When a function mixes setup, validation, computation, and side effects, split the phases.
- When a comment explains control flow, simplify names or structure before keeping the comment.
- When a function both mutates and answers, or hides a mode switch behind a flag, separate the responsibilities.
- When a boundary leaks framework, vendor, or persistence quirks inward, add or strengthen a local adapter.
- When fixing a bug or changing behavior, add or update the test that protects the intended contract.

## Final checklist

- Can a reader follow the change locally?
- Are names and APIs carrying the meaning without narration?
- Is mutation explicit and the happy path still clear?
- Did I remove at least one smell from the touched area?
- Do tests protect the changed behavior or contract?
