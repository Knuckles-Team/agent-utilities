# Design Document: Structured-output exhaustion falls back across *models* at the caller, catching only the one exception that means "this model cannot produce this schema"

CONCEPT:AU-ORCH.routing.model-fallback-chain

> Realised by `agent_utilities/capabilities/model_fallback.py:1-90`
> (`run_fallback_chain`, `model_fallback_chain`). Introduced by commit
> `6683eca0` ("feat(D-47): caller-level model/schema fallback chain for
> exhausted output repair").

## Decision — two separate pieces: a generic ordered-attempt primitive, and a config-driven builder that sources its model order from the registry's own tier ranking

`StructuredOutputRepairExhausted` is raised when a model has failed repeatedly
to produce output matching a required schema, even after repair attempts. The
introducing commit records the gap bluntly: that exception *"was raised but had
zero `except` sites anywhere in the package"* — it propagated to the caller and
the turn simply failed, even when a stronger model in the same registry would
have satisfied the schema on the first try.

The fix is deliberately split in two (`model_fallback.py:16` calls the split
out explicitly). `run_fallback_chain` is a generic primitive: it runs an ordered
sequence of attempts and moves to the next one *only* on
`StructuredOutputRepairExhausted`. `model_fallback_chain` is the builder that
produces that ordered sequence, and it does not carry its own model list —
it sources the order from `ModelRegistry.explain_pick_for_task`, reusing the
tier-fallback ranking the router already computes.

**Two alternatives were rejected, and one of them is pinned by a test.**

First, a hardcoded fallback model list. It was rejected because the registry
already ranks candidates for exactly this purpose; a second, separately-
maintained list would drift from the first and would not respect the tags,
tiers or availability the registry knows about. Sourcing from
`explain_pick_for_task` means the fallback order is the *same* ordering the
router would have used, one rung further down its own ranking.

Second — and this is the one with a regression test guarding it — a blanket
`except Exception` retry. The commit demonstrates the choice empirically rather
than asserting it: *"broadening the except clause to `Exception` makes
`test_non_repair_exception_propagates_without_fallback` fail ... confirming the
test pins real behavior."* A catch-all would convert every genuine error —
an auth failure, a network partition, a bug in the tool being called — into a
silent, expensive re-run against a second model, hiding the real fault and
multiplying its cost. Narrowing the catch to the single exception that actually
means "this model cannot produce this schema" keeps every other failure loud.

## Risk Assessment

- **Blast Radius**: `agent_utilities/capabilities/model_fallback.py`; callers
  that opt into the chain. Nothing changes for callers that do not.
- **Backward Compatible**: Yes — before this, `StructuredOutputRepairExhausted`
  propagated to the caller; it still does when no fallback chain is configured
  or when the chain is exhausted.
- **Known weak point**: the chain re-runs the *whole* attempt against the next
  model, so a schema failure discovered late in an expensive turn is paid for
  twice. There is no partial-result reuse, and no cap here on how much total
  cost a chain may consume — that bound comes from whatever budget the caller
  is already operating under.
