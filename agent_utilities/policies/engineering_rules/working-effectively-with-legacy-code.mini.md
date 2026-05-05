---
name: "Working Effectively With Legacy Code"
description: "Safe refactoring techniques for untested and legacy codebases"
author: "Michael Feathers"
version: "1.0.0"
tier: "mini"
---
# Working Effectively with Legacy Code: Mini

## When to use

Use when code is hard to change safely because behavior is unclear, tests are weak, or dependencies are entangled.

## Primary bias to correct

The first move in legacy code is control, not elegance.

## Decision rules

- Define the job as making change safe, not as rewriting old code into your preferred shape.
- Get characterization tests or equivalent observations before changing behavior you do not fully understand.
- Find or create seams so you can observe and change behavior without editing the entire dependency tangle at once.
- Break dependencies explicitly: hidden inputs, hard outputs, construction problems, globals, and static calls should become visible points of control.
- Prefer sprout, wrap, and extract-and-override style techniques when direct edits would require too much risky surgery.
- Handle risky areas with specific moves: isolate database-heavy code, peel UI/framework code away, shrink constructors, and split large methods gradually.
- Reject no-safety changes, rewrite-first instincts, and cosmetic cleanup that leaves the dangerous structure untouched.

## Trigger rules

- When a method is too large to reason about safely, create a seam before making a semantic change.
- When globals, static calls, constructors, or framework callbacks control the flow, break one dependency at a time until the behavior is testable.
- When behavior is uncertain, characterize first even if the tests are ugly.
- When the urge is to rewrite, ask what smaller sprout or wrap move would let you change the code safely today.

## Final checklist

- Safer change path?
- Behavior characterized?
- Seam created?
- Hidden dependency reduced?
