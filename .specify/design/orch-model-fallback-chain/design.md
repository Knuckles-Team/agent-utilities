# Design Document: Structured-output exhaustion falls back across *models* at the caller, catching only the one exception that means "this model cannot produce this schema"

CONCEPT:AU-ORCH.routing.model-fallback-chain

> Realised by `agent_utilities/capabilities/model_fallback.py`
> (`run_fallback_chain`). Introduced by commit
> `6683eca0` ("feat(D-47): caller-level model/schema fallback chain for
> exhausted output repair").

## Decision — one generic ordered-attempt primitive

`StructuredOutputRepairExhausted` is raised when a model has failed repeatedly
to produce output matching a required schema, even after repair attempts. The
introducing commit records the gap bluntly: that exception *"was raised but had
zero `except` sites anywhere in the package"* — it propagated to the caller and
the turn simply failed, even when a stronger model in the same registry would
have satisfied the schema on the first try.

`run_fallback_chain` is a generic primitive: it runs an ordered
sequence of attempts and moves to the next one *only* on
`StructuredOutputRepairExhausted`. The caller supplies the governed attempt
order directly; this capability does not duplicate model selection or registry
policy.

**Two alternatives were rejected, and one of them is pinned by a test.**

First, any second model-order builder in this capability. It was rejected
because the routing authority already ranks candidates; duplicating that policy
would drift from the current selection contract. Callers therefore pass the
ordered attempts produced by their routing authority.

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
- **Current contract**: callers either invoke the primitive with their governed
  ordered attempts or let `StructuredOutputRepairExhausted` propagate.
- **Known weak point**: the chain re-runs the *whole* attempt against the next
  model, so a schema failure discovered late in an expensive turn is paid for
  twice. There is no partial-result reuse, and no cap here on how much total
  cost a chain may consume — that bound comes from whatever budget the caller
  is already operating under.
