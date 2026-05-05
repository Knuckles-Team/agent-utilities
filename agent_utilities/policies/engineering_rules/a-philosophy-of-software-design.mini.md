---
name: "A Philosophy Of Software Design"
description: "Complexity reduction through deep modules and clean abstractions"
author: "John Ousterhout"
version: "1.0.0"
tier: "mini"
---
# A Philosophy of Software Design: Mini

## When to use

Use for module design, APIs, refactoring shallow abstractions, and changes that feel awkward or spread complexity across files.

## Primary bias to correct

Do not confuse more layers, more helpers, or smaller pieces with lower complexity.

## Decision rules

- Treat complexity as the main design metric. Prefer the change that reduces cognitive load, change amplification, and hidden dependencies.
- Prefer deep modules: a small interface that hides meaningful internal complexity is better than extra pass-through layers.
- Hide volatile details, internal data representation, bookkeeping, normalization, and messy edge cases behind stable interfaces.
- Design interfaces around client knowledge, not implementation mechanics. Avoid staged call sequences, flag arguments, and APIs that leak how the work gets done.
- When a feature feels awkward, fix the design strategically instead of adding local exceptions, helper scatter, or tactical wrappers.
- Pull complexity downward. The lower layer that owns the detail should absorb it instead of pushing it onto every caller.
- Make temporal dependencies explicit and minimize order-sensitive behavior that readers must reconstruct mentally.
- At function and variable level, keep meaning packed: avoid jump-heavy micro-functions and pass-through variables that expose internal steps without hiding complexity.
- Use comments only for contracts, invariants, or rationale that code cannot express cleanly.

## Trigger rules

- When adding a module, layer, service, or helper, prove that it hides real complexity instead of renaming or forwarding it.
- When touching an API, check whether callers need too much context, too many steps, or knowledge of internals.
- When adding a special case, flag, or exception path, first ask whether the abstraction boundary is wrong.
- When one change spreads across many files, look for missing information hiding, missing module depth, or temporal coupling.
- When splitting a function or introducing a temporary only for style, check whether it improves understanding or only adds jumps and visible intermediate state.

## Final checklist

- Did the change reduce the number of facts a reader must hold at once?
- Did complexity move downward into the owning module instead of outward into callers?
- Did the interface get simpler or more semantic rather than more configurable?
- Did I remove special cases instead of normalizing them in more places?
