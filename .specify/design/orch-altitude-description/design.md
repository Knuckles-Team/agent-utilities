# Design Document: The deferred-turn acknowledgement describes the turn's ACTUAL execution altitude, and the focused-tools budget is computed from its own shape

CONCEPT:AU-ORCH.routing.altitude-description

> Realised by `agent_utilities/messaging/router.py:572-588` — the branch that
> selects `"focused-tools turn (N tools in parallel)"` when
> `shape.tool_servers` is non-empty, and `"full multi-agent turn"` otherwise.
> The shape itself is computed in
> `agent_utilities/orchestration/execution_profile.py` (`plan_execution_shape`).
> Introduced by commit `1a2c387b` ("fix(orch): focused-tools reply budget +
> accurate ack").

## Decision — the acknowledgement text and the reply budget are both derived from the execution shape, not from a single blanket profile

When a messaging turn will take a while, the router sends the user an
acknowledgement naming what it is doing and roughly how long it will take. That
acknowledgement was wrong for a whole class of turns, and it was wrong because
it was reading the wrong thing.

The commit records the mechanism: *"`reply_budget_s` computed ~190s for a
focused-tools turn (it carries `_FULL_FIELDS`), so the messaging ack mislabeled
it as a 'full multi-agent turn (~190s)'."* A focused-tools turn — one that runs
a handful of tools in parallel and returns — was carrying the full-graph
field set, so both the budget and the label it drove described a full
multi-agent turn that was never going to happen.

The fix changes both halves together, which is the point. The label branches on
the actual shape (`shape.tool_servers` non-empty ⇒ focused tools), and the
budget for the focused case is computed from that shape's own cost drivers:
`35 + 20 × len(tool_servers)`, capped at 190s. The number of tool servers is
what a focused-tools turn's duration actually scales with, so the budget is
derived from the work rather than inherited from a profile that does not apply.

**The rejected alternative is what shipped before: one blanket full-graph
budget field used for every deferred turn, with the acknowledgement rendered
from it.** It is simpler and it is never *under*-estimating, which is why it
survived — an over-long estimate produces no error. It was rejected because
the acknowledgement is a user-facing promise: telling someone a 40-second
operation will take ~190 seconds is a worse answer than not telling them
anything, and it makes the system look slower than it is on exactly its
cheaper path.

## Scope note

This concept covers the description and the focused-tools budget formula. It
does not cover `plan_execution_shape` itself, which computes the shape and
carries its own concept.

## Risk Assessment

- **Blast Radius**: `agent_utilities/messaging/router.py`; reads the shape
  produced by `agent_utilities/orchestration/execution_profile.py`.
- **Backward Compatible**: The user-visible acknowledgement text and timing
  estimate change for focused-tools turns. That is the fix.
- **Known weak point**: `35 + 20 × len(tool_servers)` is an unvalidated linear
  model. It assumes tool servers contribute independently and roughly equally,
  which is wrong for a slow tool paired with fast ones, and the 190s cap means
  a genuinely wide fan-out is under-estimated — reintroducing, at the top end,
  the same class of inaccurate promise this decision set out to remove.
