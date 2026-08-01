# Design Document: One facade + one result contract over five disparate engineering-discipline patterns

CONCEPT:AU-AHE.harness.agentic-engineering-patterns

> `agent_utilities/harness/engineering.py`, dispatching to
> `agent_utilities/patterns/{tdd,first_run_tests,manual_testing,walkthroughs,interactive_explanations}.py`.

## Decision — a single `EngineeringPatternOrchestrator.execute(PatternType, ...)` call, never a per-pattern entrypoint

The module docstring states the shape directly (`engineering.py:4-30`): TDD
cycles, first-run baselines, manual testing, code walkthroughs and interactive
explanations each live as their own module under `agent_utilities/patterns/`,
but the harness exposes exactly ONE entrypoint — an enum-typed `PatternType`
dispatched by `EngineeringPatternOrchestrator.execute` — rather than five
separate call sites a caller has to know about individually.

Two concrete mechanics enforce this:

1. **Uniform result contract.** Every pattern, regardless of what it actually
   does internally (a TDD cycle returns pass/fail cycle output; a walkthrough
   returns a document), is coerced into the same `PatternResult` shape
   (`pattern`, `success`, `output`, `artifacts`, `metadata`, `error`) by the
   dispatcher's private `_execute_*` wrappers (`engineering.py:178-206` for the
   TDD case, mirrored per pattern).
2. **Exceptions never escape the facade.** `execute()` wraps the entire
   dispatch in one `try/except Exception` (`engineering.py:148-176`) and
   converts any failure into a `PatternResult(success=False, error=str(exc))` —
   a caller driving multiple patterns in a loop never needs a pattern-specific
   except clause.

**The rejected alternative is letting each pattern module stay its own
integration point** — a caller imports `run_tdd_cycle` here, a walkthrough
generator there, each with its own signature, its own return type, and its own
failure mode (some raise, some return sentinels). That's what existed before
this facade: five independently-shaped tools instead of one. The cost of
unification is indirection — the facade adds a dispatch hop and a lazy
per-call import (`from agent_utilities.patterns.tdd import run_tdd_cycle`
inside `_execute_tdd`, not at module load) before every pattern actually runs,
so `engineering.py` itself stays free of the patterns' own heavier
dependencies until one is actually invoked.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/engineering.py` only; the five
  `agent_utilities/patterns/*` modules keep their own public APIs unchanged.
- **Backward Compatible**: Yes — the facade is an additive entrypoint, not a
  replacement for direct pattern-module imports.
- **Known weak point**: the uniform `PatternResult.error = str(exc)` swallows
  the original exception type and traceback at the facade boundary; a caller
  that needs structured error handling per pattern (not just "did it
  succeed") has to inspect the string message.
