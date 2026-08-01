# Design Document: A classified, bounded, visible repair loop replaces pydantic-ai's one undifferentiated retry bounce

CONCEPT:AU-AHE.harness.structured-output-repair

> `agent_utilities/capabilities/output_repair.py`.

## Decision — classify every structured-output failure into an explicit taxonomy and drive a bounded, attempt-logged repair loop

The module docstring (`output_repair.py:4-33`) grounds the decision in two
concrete production failures that motivated it: a delegated `pydantic_graph`
run against ServiceNow exceeded its token budget at 6,982 tokens *after* a
successful tool call, and separately a fleet tool silently ignored an
unknown `limit` argument and returned 212 KB from one call. Neither was a
malformed-output problem, but both exposed the same underlying gap: when a
model DOES produce bad structured output, pydantic-ai's own retry machinery
"treats every cause identically — one undifferentiated `ValidationError`/
`ModelRetry` bounce" — and a caller can't tell *why* it failed or *that a
repair was even attempted* once the run succeeds or gives up.

**The rejected alternative is pydantic-ai's default behavior itself: an
opaque, undifferentiated retry with no visibility into cause or attempt
history.** This module classifies each failure (budget-exhausted-mid-output,
invalid JSON, wrong shape, outright refusal, ...) and re-asks the model with
a targeted, classification-specific instruction, appending every attempt to
`StructuredOutputRepair.attempts`. On success, the attempts are stamped onto
`AgentRunResult` as `output_repair_attempts`, extending the truthfulness
contract (`AU-ORCH.execution.messaging-orchestration-transparency`) to
output repair specifically: a caller can never read a repaired run back as
an indistinguishable clean first-try success. Exhausting the bound raises a
plain `StructuredOutputRepairExhausted` (not a `ModelRetry`) carrying the
full attempt list — **the rejected alternative there is either an unbounded
retry storm or pydantic-ai's own generic "Exceeded maximum output retries"**
— instead the run fails closed with a typed, inspectable error. A budget
exhausted mid-output or an upstream content-filter refusal is recorded and
never blindly retried, since retrying either wastes budget or repeats a
refusal the model isn't going to reverse.

## Risk Assessment

- **Blast Radius**: `agent_utilities/capabilities/output_repair.py`,
  `agent_utilities/capabilities/composition.py`.
- **Backward Compatible**: Yes — wraps pydantic-ai's retry path; a caller
  not reading `output_repair_attempts` sees no behavior change.
- **Known weak point**: named directly in the module's own tracking
  (D-W15-13 in `reports/deferred/waves1-5-gate.md`) — this is visibility to
  the *direct caller only*. `orchestration/agent_runner.py::_record_execution_trace`,
  which writes the persisted RunTrace/KG/OTel record, does not yet read
  `output_repair_attempts`, so a repaired run is still indistinguishable
  from a clean first-try success in the persisted trace, even though the
  direct caller can tell the difference.
