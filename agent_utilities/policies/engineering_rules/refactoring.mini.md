---
name: "Refactoring"
description: "Systematic code improvement through safe, incremental transformations"
author: "Martin Fowler"
version: "1.0.0"
tier: "mini"
---
# Refactoring: Mini

## When to use

Use when improving structure without intending to change observable behavior, or when preparatory cleanup makes a requested change safer.

## Primary bias to correct

Do not smuggle redesign, rewrite, and behavior change into one uncontrolled patch.

## Decision rules

- Refactoring preserves observable behavior. If behavior must change, keep the behavioral delta distinct from structural cleanup whenever practical.
- Use small, verifiable steps. Prefer a sequence of safe transformations over a dramatic rewrite or broad structural leap.
- Get a safety net first: tests, characterization checks, types, assertions, logs, or another credible verification path.
- Refactor the smell that blocks the current change: duplication, long functions, long parameter lists, hidden dependencies, shotgun surgery, primitive obsession, sprawling conditionals, or speculative abstraction.
- Prefer known moves such as rename, extract, move, inline, simplify, split phase, and data-shape improvements over improvised structural surgery.
- Make preparatory refactors only when they directly lower the risk of the requested change.
- Stop when the requested change is safe and clear. Do not keep refactoring just because more cleanup is imaginable.

## Trigger rules

- When a patch mixes behavior change and structural cleanup, split the intent unless the context makes that impossible.
- When a risky area lacks tests, characterize current behavior before major structural edits.
- When a conditional tree keeps growing, consider extraction, split phase, polymorphism, or data-driven structure.
- When the urge is to rewrite, first ask what smaller transformation would recover control.

## Final checklist

- Observable behavior preserved?
- Small verified step?
- Current smell addressed?
- Cleanup stopped at the useful point?
